"""Entropy-Gated Adaptive Speculation (EG-Spec) Policy.

Dynamically halts draft token generation when draft model uncertainty exceeds
threshold boundaries, preventing wasted target verification passes on low-fidelity
quantized generations.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


class EntropyGatedPolicy:
    """Policy for dynamically gating speculative draft length based on distribution entropy and confidence."""

    def __init__(
        self,
        entropy_threshold: float = 1.45,
        confidence_margin: float = 0.25,
        min_gamma: int = 1,
        max_gamma: int = 5,
        enabled: bool = True,
    ):
        """Initialize the entropy gating policy.
        
        Args:
            entropy_threshold: Maximum Shannon entropy (in nats) before early stopping.
            confidence_margin: Minimum margin between top-1 and top-2 probabilities.
            min_gamma: Minimum number of draft tokens before early termination can trigger.
            max_gamma: Maximum speculative lookahead horizon.
            enabled: If False, acts as a static policy running for full max_gamma.
        """
        self.entropy_threshold = float(entropy_threshold)
        self.confidence_margin = float(confidence_margin)
        self.min_gamma = int(min_gamma)
        self.max_gamma = int(max_gamma)
        self.enabled = bool(enabled)

    @torch.no_grad()
    def evaluate_draft_step(
        self,
        logits: torch.Tensor,
        current_step: int,
    ) -> Tuple[bool, Dict[str, float]]:
        """Evaluate whether the draft model should continue proposing additional tokens.
        
        Args:
            logits: 1D or 2D tensor of logits for the latest predicted token [vocab_size] or [1, vocab_size].
            current_step: 1-indexed count of how many draft tokens have been generated so far.
            
        Returns:
            Tuple of (should_continue, metadata_dict).
        """
        if not self.enabled:
            should_cont = current_step < self.max_gamma
            return should_cont, {"entropy": 0.0, "margin": 1.0, "reason": "static_horizon"}

        if current_step >= self.max_gamma:
            return False, {"entropy": 0.0, "margin": 0.0, "reason": "max_gamma_reached"}

        # Normalize logits to float32 for numerical stability in entropy calculation
        if logits.dim() == 2:
            logits = logits[-1]
        probs = F.softmax(logits.float(), dim=-1)

        # 1. Compute Shannon Entropy: H(p) = -sum(p * log(p))
        eps = 1e-9
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs).item()

        # 2. Compute Top-1 vs Top-2 Confidence Margin
        top2_vals, _ = torch.topk(probs, k=2, dim=-1)
        margin = (top2_vals[0] - top2_vals[1]).item()

        # Check minimum guard: always draft at least min_gamma tokens
        if current_step < self.min_gamma:
            return True, {
                "entropy": entropy,
                "margin": margin,
                "reason": "min_gamma_guarantee",
            }

        # Check gating conditions:
        # High entropy indicates uniform uncertainty; low margin indicates strong competition
        is_uncertain = (entropy > self.entropy_threshold) or (margin < self.confidence_margin)

        if is_uncertain:
            return False, {
                "entropy": entropy,
                "margin": margin,
                "reason": "entropy_threshold_exceeded" if entropy > self.entropy_threshold else "margin_too_low",
            }

        return True, {
            "entropy": entropy,
            "margin": margin,
            "reason": "confident_to_continue",
        }
