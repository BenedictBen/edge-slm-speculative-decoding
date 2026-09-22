"""Core Speculative Decoding Engine with Exact Rejection Sampling and Adaptive Lookahead."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.core.adaptive_policy import EntropyGatedPolicy
from src.core.memory_profiler import BenchmarkProfiler, ProfileResult


@dataclass
class StepTrace:
    """Detailed record of an individual speculative verification step."""
    step_index: int
    draft_tokens: List[int]
    draft_tokens_text: List[str]
    num_accepted: int
    accepted_tokens: List[int]
    bonus_token: Optional[int]
    stopped_by_entropy: bool
    entropies: List[float] = field(default_factory=list)


@dataclass
class GenerationOutput:
    """Output structure returned by the generation methods."""
    token_ids: torch.Tensor
    text: str
    profile: ProfileResult
    traces: List[StepTrace] = field(default_factory=list)


class SpeculativeEngine:
    """Implements standard Autoregressive and Speculative Decoding with adaptive lookahead."""

    def __init__(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.profiler = BenchmarkProfiler(device=self.device)
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    @torch.no_grad()
    def generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationOutput:
        """Standard autoregressive generation using the target model as baseline."""
        input_ids = input_ids.to(self.target_model.device)
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]

        self.profiler.start()

        for _ in range(max_new_tokens):
            outputs = self.target_model(generated)
            next_token_logits = outputs.logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / max(temperature, 1e-5)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.eos_token_id:
                break

        num_gen = generated.shape[1] - prompt_len
        profile = self.profiler.stop(num_tokens_generated=num_gen)
        text = self.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

        return GenerationOutput(token_ids=generated, text=text, profile=profile)

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        gamma: int = 5,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        adaptive_policy: Optional[EntropyGatedPolicy] = None,
    ) -> GenerationOutput:
        """Speculative decoding with exact distribution alignment and optional EG-Spec gating."""
        input_ids = input_ids.to(self.target_model.device)
        draft_device = self.draft_model.device
        current_seq = input_ids.clone()
        prompt_len = input_ids.shape[1]

        total_draft_tokens = 0
        total_accepted_tokens = 0
        speculative_steps = 0
        traces: List[StepTrace] = []

        self.profiler.start()

        while (current_seq.shape[1] - prompt_len) < max_new_tokens:
            speculative_steps += 1
            step_entropy_values = []
            stopped_by_entropy = False

            # -------------------------------------------------------------
            # 1. DRAFT PHASE: Generate up to gamma candidates using draft model
            # -------------------------------------------------------------
            draft_seq = current_seq.to(draft_device).clone()
            draft_tokens: List[int] = []
            draft_probs_list: List[torch.Tensor] = []

            max_lookahead = gamma if adaptive_policy is None else adaptive_policy.max_gamma

            for k in range(max_lookahead):
                draft_outputs = self.draft_model(draft_seq)
                d_logits = draft_outputs.logits[:, -1, :]

                # Check Entropy-Gated policy if enabled
                if adaptive_policy is not None:
                    should_cont, meta = adaptive_policy.evaluate_draft_step(d_logits, k + 1)
                    step_entropy_values.append(meta.get("entropy", 0.0))
                    if not should_cont:
                        stopped_by_entropy = True
                        # If policy says stop and we haven't drafted anything, draft at least 1 token
                        if k == 0:
                            pass
                        else:
                            break

                if temperature == 0.0:
                    d_next_token = torch.argmax(d_logits, dim=-1, keepdim=True)
                    d_prob = F.softmax(d_logits.float(), dim=-1)
                else:
                    d_prob = F.softmax(d_logits / max(temperature, 1e-5), dim=-1)
                    d_next_token = torch.multinomial(d_prob, num_samples=1)

                draft_tokens.append(d_next_token.item())
                draft_probs_list.append(d_prob)
                draft_seq = torch.cat([draft_seq, d_next_token], dim=1)

                if d_next_token.item() == self.eos_token_id:
                    break

            k_actual = len(draft_tokens)
            if k_actual == 0:
                # Fallback to single target autoregressive token
                t_out = self.target_model(current_seq)
                single_tok = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
                current_seq = torch.cat([current_seq, single_tok], dim=1)
                continue

            total_draft_tokens += k_actual

            # -------------------------------------------------------------
            # 2. VERIFICATION PHASE: Target model forward pass over prefix + draft
            # -------------------------------------------------------------
            candidate_tensor = torch.tensor(
                [draft_tokens], dtype=torch.long, device=self.target_model.device
            )
            verification_seq = torch.cat([current_seq, candidate_tensor], dim=1)

            target_outputs = self.target_model(verification_seq)
            target_all_logits = target_outputs.logits[0]  # [seq_len, vocab_size]

            prefix_len = current_seq.shape[1]
            # Target distributions at candidate positions:
            # Position (prefix_len - 1) predicts candidate 0
            # Position (prefix_len + j - 1) predicts candidate j
            target_logits_slice = target_all_logits[prefix_len - 1 : prefix_len + k_actual, :]

            # -------------------------------------------------------------
            # 3. REJECTION SAMPLING / ACCEPTANCE CHECK
            # -------------------------------------------------------------
            n_accepted = 0
            accepted_tokens: List[int] = []
            bonus_token: Optional[int] = None

            for j in range(k_actual):
                cand_id = draft_tokens[j]
                t_logits = target_logits_slice[j]

                if temperature == 0.0:
                    # Deterministic greedy matching
                    target_pred = torch.argmax(t_logits).item()
                    if target_pred == cand_id:
                        n_accepted += 1
                        accepted_tokens.append(cand_id)
                    else:
                        # Rejection: accept target's correction and abort draft horizon
                        bonus_token = target_pred
                        break
                else:
                    # Stochastic exact rejection sampling
                    t_prob = F.softmax(t_logits / max(temperature, 1e-5), dim=-1)
                    q_cand = draft_probs_list[j][0, cand_id].item()
                    p_cand = t_prob[cand_id].item()

                    ratio = p_cand / max(q_cand, 1e-9)
                    u = torch.rand(1).item()

                    if u < min(1.0, ratio):
                        n_accepted += 1
                        accepted_tokens.append(cand_id)
                    else:
                        # Resample from normalized residual max(0, p - q)
                        residual = torch.clamp(t_prob - draft_probs_list[j][0].to(t_prob.device), min=0.0)
                        res_sum = residual.sum()
                        if res_sum > 1e-9:
                            residual_prob = residual / res_sum
                            resampled = torch.multinomial(residual_prob, num_samples=1).item()
                        else:
                            resampled = torch.multinomial(t_prob, num_samples=1).item()
                        bonus_token = resampled
                        break

            # If all k_actual candidates were accepted, sample the bonus token at k_actual position
            if n_accepted == k_actual:
                final_t_logits = target_logits_slice[k_actual]
                if temperature == 0.0:
                    bonus_token = torch.argmax(final_t_logits).item()
                else:
                    bonus_prob = F.softmax(final_t_logits / max(temperature, 1e-5), dim=-1)
                    bonus_token = torch.multinomial(bonus_prob, num_samples=1).item()

            total_accepted_tokens += n_accepted

            # -------------------------------------------------------------
            # 4. UPDATE CURRENT SEQUENCE
            # -------------------------------------------------------------
            tokens_to_append = accepted_tokens.copy()
            if bonus_token is not None:
                tokens_to_append.append(bonus_token)

            # Cap tokens_to_append so we do not overshoot max_new_tokens
            cur_gen_count = current_seq.shape[1] - prompt_len
            remaining = max_new_tokens - cur_gen_count
            if len(tokens_to_append) > remaining:
                tokens_to_append = tokens_to_append[:remaining]

            if tokens_to_append:
                append_tensor = torch.tensor(
                    [tokens_to_append], dtype=torch.long, device=self.target_model.device
                )
                current_seq = torch.cat([current_seq, append_tensor], dim=1)

            # Record step trace for logging and visualization
            draft_texts = [self.tokenizer.decode([t]) for t in draft_tokens]
            traces.append(
                StepTrace(
                    step_index=speculative_steps,
                    draft_tokens=draft_tokens,
                    draft_tokens_text=draft_texts,
                    num_accepted=n_accepted,
                    accepted_tokens=accepted_tokens,
                    bonus_token=bonus_token,
                    stopped_by_entropy=stopped_by_entropy,
                    entropies=step_entropy_values,
                )
            )

            # Check EOS termination
            if self.eos_token_id in tokens_to_append:
                break

        num_gen = current_seq.shape[1] - prompt_len
        profile = self.profiler.stop(
            num_tokens_generated=num_gen,
            num_draft_tokens=total_draft_tokens,
            num_accepted_tokens=total_accepted_tokens,
            num_speculative_steps=speculative_steps,
        )
        text = self.tokenizer.decode(current_seq[0, prompt_len:], skip_special_tokens=True)

        return GenerationOutput(
            token_ids=current_seq, text=text, profile=profile, traces=traces
        )
