"""Quantization and model loading abstractions for Small Language Models (SLMs).

Supports FP16, INT8 (LLM.int8()), and INT4 (NF4 NormalFloat4) via BitsAndBytes,
with a robust fallback for CPU simulation and dry-run testing.
"""

import logging
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility for loading target and draft models under different quantization strategies."""

    @staticmethod
    def get_quantization_config(quant_mode: str, device: str):
        """Construct BitsAndBytesConfig for the requested quantization level."""
        quant_mode = quant_mode.lower()
        if device == "cpu" or not torch.cuda.is_available():
            if quant_mode in ["int8", "int4"]:
                logger.warning(
                    f"Quantization mode '{quant_mode}' requested on CPU. "
                    "BitsAndBytes requires CUDA; falling back to standard CPU precision."
                )
            return None, torch.float32

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("bitsandbytes not installed. Falling back to fp16.")
            return None, torch.float16

        if quant_mode == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            return bnb_config, None
        elif quant_mode == "int8":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            return bnb_config, None
        elif quant_mode in ["fp16", "float16"]:
            return None, torch.float16
        elif quant_mode in ["bf16", "bfloat16"]:
            return None, torch.bfloat16
        else:
            return None, torch.float32

    @classmethod
    def load_model_and_tokenizer(
        cls,
        model_name: str,
        quant_mode: str = "fp16",
        device: str = "cuda",
        trust_remote_code: bool = True,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load causal LM and its tokenizer with appropriate quantization and device placement."""
        resolved_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        bnb_config, torch_dtype = cls.get_quantization_config(quant_mode, resolved_device)

        logger.info(f"Loading '{model_name}' in '{quant_mode}' on {resolved_device}...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            if resolved_device == "cuda":
                load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if bnb_config is None and resolved_device == "cpu":
            model = model.to("cpu")

        model.eval()
        return model, tokenizer

    @classmethod
    def build_synthetic_pair(
        cls,
        vocab_size: int = 1000,
        target_dim: int = 128,
        draft_dim: int = 64,
    ) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
        """Build a pair of miniature synthetic transformer models for fast unit tests without network downloads."""
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

        # Synthetic target configuration (~500k params)
        target_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=target_dim,
            n_layer=4,
            n_head=4,
            n_positions=256,
            bos_token_id=1,
            eos_token_id=1,
        )
        target_model = GPT2LMHeadModel(target_config)
        target_model.eval()

        # Synthetic draft configuration (~100k params)
        draft_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=draft_dim,
            n_layer=2,
            n_head=2,
            n_positions=256,
            bos_token_id=1,
            eos_token_id=1,
        )
        draft_model = GPT2LMHeadModel(draft_config)
        draft_model.eval()

        SAMPLE_WORDS = [
            "<pad>", "<eos>", "def", "calculate_sphere_volume", "(", "radius", ":", "float", ")", "->",
            "float", ":", "\"\"\"Calculates", "volume", "of", "a", "sphere.", "\"\"\"", "import", "math",
            "if", "radius", "<", "0", ":", "raise", "ValueError", "(", "\"Radius", "must", "be", "positive\"", ")",
            "return", "(", "4.0", "/", "3.0", ")", "*", "math.pi", "*", "(",
            "radius", "**", "3", ")", "#", "Unit", "test", "cases", "assert", "round", "(", "calculate_sphere_volume",
            "(", "1.0", ")", ",", "2", ")", "==", "4.19", "assert", "calculate_sphere_volume", "(", "0", ")", "==", "0.0",
            "print", "(", "\"All", "tests", "passed!\"", ")", "result", "=", "volume", "formula", "computation"
        ]

        class MockTokenizer:
            def __init__(self, v_size: int):
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.vocab_size = v_size

            def encode(self, text, return_tensors="pt"):
                tokens = [abs(hash(w)) % (self.vocab_size - 2) + 2 for w in text.split()]
                if not tokens:
                    tokens = [2]
                tensor = torch.tensor([tokens], dtype=torch.long)
                return tensor

            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.squeeze().tolist()
                if not isinstance(token_ids, list):
                    token_ids = [token_ids]
                words = []
                for t in token_ids:
                    if t in [0, 1] and skip_special_tokens:
                        continue
                    idx = t % len(SAMPLE_WORDS)
                    words.append(SAMPLE_WORDS[idx])
                return " ".join(words)

            def __call__(self, text, return_tensors="pt"):
                input_ids = self.encode(text)
                return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        mock_tokenizer = MockTokenizer(vocab_size)
        return target_model, draft_model, mock_tokenizer
