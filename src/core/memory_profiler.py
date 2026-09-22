"""High-precision memory, latency, and throughput profiling utilities."""

import os
import time
from dataclasses import asdict, dataclass
from typing import Optional
import psutil
import torch


@dataclass
class ProfileResult:
    """Dataclass holding performance benchmarks for a generation run."""
    total_time_sec: float
    num_tokens_generated: int
    tokens_per_sec: float
    ms_per_token: float
    peak_vram_mb: float
    ram_usage_mb: float
    acceptance_rate: Optional[float] = None
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_speculative_steps: int = 0

    def to_dict(self):
        return asdict(self)


class BenchmarkProfiler:
    """Profiles GPU VRAM, CPU memory, and precise wall-clock execution time."""

    def __init__(self, device: str = "cuda"):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._start_time = 0.0
        self._process = psutil.Process(os.getpid())

    def start(self):
        """Begin timing and reset peak memory statistics."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(
        self,
        num_tokens_generated: int,
        num_draft_tokens: int = 0,
        num_accepted_tokens: int = 0,
        num_speculative_steps: int = 0,
    ) -> ProfileResult:
        """Stop timing, collect memory footprint, and compute throughput metrics."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_sec = time.perf_counter() - self._start_time

        # Calculate memory metrics
        if self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        else:
            peak_vram = 0.0

        ram_mb = self._process.memory_info().rss / (1024.0 ** 2)

        # Calculate rate metrics
        tokens_per_sec = num_tokens_generated / elapsed_sec if elapsed_sec > 0 else 0.0
        ms_per_token = (elapsed_sec * 1000.0) / num_tokens_generated if num_tokens_generated > 0 else 0.0

        acceptance_rate = (
            (num_accepted_tokens / num_draft_tokens) if num_draft_tokens > 0 else None
        )

        return ProfileResult(
            total_time_sec=round(elapsed_sec, 4),
            num_tokens_generated=num_tokens_generated,
            tokens_per_sec=round(tokens_per_sec, 2),
            ms_per_token=round(ms_per_token, 2),
            peak_vram_mb=round(peak_vram, 2),
            ram_usage_mb=round(ram_mb, 2),
            acceptance_rate=round(acceptance_rate, 4) if acceptance_rate is not None else None,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            num_speculative_steps=num_speculative_steps,
        )
