"""Benchmark suite orchestrating evaluations across mathematical, code, and summarization tasks."""

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Optional
try:
    import pandas as pd
except Exception:
    pd = None
import torch

from src.core.adaptive_policy import EntropyGatedPolicy
from src.core.speculative_engine import GenerationOutput, SpeculativeEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkPrompt:
    id: str
    category: str
    prompt: str


STANDARD_BENCHMARK_PROMPTS: List[BenchmarkPrompt] = [
    # 1. Mathematical Reasoning (GSM8k-style)
    BenchmarkPrompt(
        id="math_01",
        category="mathematical_reasoning",
        prompt="Question: A store owner bought 15 boxes of pencils. Each box contains 24 pencils. He gave 45 pencils to a local school and sold 2/3 of the remaining pencils. How many pencils does he have left? Let's solve this step by step.",
    ),
    BenchmarkPrompt(
        id="math_02",
        category="mathematical_reasoning",
        prompt="Question: A train travels from Town A to Town B at 60 mph and returns at 40 mph. If the distance between the towns is 120 miles, what is the average speed for the entire round trip? Work through the calculation carefully.",
    ),
    BenchmarkPrompt(
        id="math_03",
        category="mathematical_reasoning",
        prompt="Question: If a rectangular garden has a perimeter of 72 meters and the length is 6 meters longer than twice the width, what are the dimensions and total area of the garden?",
    ),

    # 2. Code Generation (HumanEval / MBPP style)
    BenchmarkPrompt(
        id="code_01",
        category="code_generation",
        prompt="Write a Python function `find_longest_substring_without_repeating(s: str) -> int` that returns the length of the longest substring without repeating characters using the sliding window technique. Include docstrings and unit test examples.",
    ),
    BenchmarkPrompt(
        id="code_02",
        category="code_generation",
        prompt="Implement a thread-safe LRU Cache class in Python with `get(key)` and `put(key, value, capacity)` methods using a doubly linked list and a hash map.",
    ),
    BenchmarkPrompt(
        id="code_03",
        category="code_generation",
        prompt="Write an efficient Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping numerical intervals in O(N log N) time complexity.",
    ),

    # 3. Summarization / Long-form Context (CNN/DM style)
    BenchmarkPrompt(
        id="sum_01",
        category="summarization",
        prompt="Summarize the key architectural differences between dense Transformer attention and Linear State Space Models (SSMs) like Mamba. Discuss memory complexity during autoregressive generation and throughput tradeoffs.",
    ),
    BenchmarkPrompt(
        id="sum_02",
        category="summarization",
        prompt="Explain the core mechanism of Low-Rank Adaptation (LoRA) and QLoRA for parameter-efficient fine-tuning of neural networks. Highlight weight freezing and rank decomposition math.",
    ),
    BenchmarkPrompt(
        id="sum_03",
        category="summarization",
        prompt="Provide a concise technical overview of post-training quantization techniques: INT8 weight-activation quantization (SmoothQuant) versus INT4 weight-only NormalFloat4 (NF4).",
    ),
]


class BenchmarkSuite:
    """Orchestrates comprehensive benchmark sweeps across decoding paradigms and task domains."""

    def __init__(
        self,
        engine: SpeculativeEngine,
        prompts: Optional[List[BenchmarkPrompt]] = None,
    ):
        self.engine = engine
        self.prompts = prompts or STANDARD_BENCHMARK_PROMPTS

    def run_evaluation_suite(
        self,
        gamma: int = 5,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        entropy_threshold: float = 1.45,
        confidence_margin: float = 0.25,
    ) -> Any:
        """Run standard Autoregressive, Static Speculative, and Adaptive EG-Spec across all benchmark prompts."""
        records = []

        logger.info(f"Initiating benchmark suite across {len(self.prompts)} prompts...")

        eg_policy = EntropyGatedPolicy(
            entropy_threshold=entropy_threshold,
            confidence_margin=confidence_margin,
            min_gamma=1,
            max_gamma=gamma,
            enabled=True,
        )

        for item in self.prompts:
            logger.info(f"Evaluating prompt {item.id} [{item.category}]...")
            encoded = self.engine.tokenizer(item.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]

            # Method 1: Standard Autoregressive Baseline
            out_ar = self.engine.generate_autoregressive(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            # Method 2: Static Speculative Decoding (Fixed gamma)
            out_static = self.engine.generate_speculative(
                input_ids=input_ids,
                gamma=gamma,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                adaptive_policy=None,
            )

            # Method 3: Adaptive Entropy-Gated Speculative Decoding (EG-Spec)
            out_eg = self.engine.generate_speculative(
                input_ids=input_ids,
                gamma=gamma,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                adaptive_policy=eg_policy,
            )

            # Record metrics
            ar_ms = out_ar.profile.ms_per_token
            static_ms = out_static.profile.ms_per_token
            eg_ms = out_eg.profile.ms_per_token

            static_speedup = ar_ms / max(static_ms, 1e-5)
            eg_speedup = ar_ms / max(eg_ms, 1e-5)

            records.append({
                "prompt_id": item.id,
                "category": item.category,
                "ar_tokens_per_sec": out_ar.profile.tokens_per_sec,
                "ar_ms_per_token": ar_ms,
                "ar_tokens_gen": out_ar.profile.num_tokens_generated,
                "static_tokens_per_sec": out_static.profile.tokens_per_sec,
                "static_ms_per_token": static_ms,
                "static_speedup": round(static_speedup, 3),
                "static_acc_rate": out_static.profile.acceptance_rate,
                "eg_tokens_per_sec": out_eg.profile.tokens_per_sec,
                "eg_ms_per_token": eg_ms,
                "eg_speedup": round(eg_speedup, 3),
                "eg_acc_rate": out_eg.profile.acceptance_rate,
                "peak_vram_mb": out_eg.profile.peak_vram_mb,
            })

        if pd is not None:
            return pd.DataFrame(records)
        return records
