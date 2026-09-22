"""Unit tests validating mathematical exactness, entropy gating, and profiling."""

import pytest
import torch
from src.core.adaptive_policy import EntropyGatedPolicy
from src.core.memory_profiler import BenchmarkProfiler
from src.core.quant_loader import ModelLoader
from src.core.speculative_engine import SpeculativeEngine


@pytest.fixture
def synthetic_engine():
    """Fixture providing initialized engine with lightweight synthetic models."""
    target_m, draft_m, tok = ModelLoader.build_synthetic_pair(
        vocab_size=100,
        target_dim=64,
        draft_dim=32,
    )
    return SpeculativeEngine(
        target_model=target_m,
        draft_model=draft_m,
        tokenizer=tok,
        device="cpu",
    )


def test_exactness_greedy_decoding(synthetic_engine):
    """Verify that greedy speculative decoding produces output identical to autoregressive decoding."""
    prompt = "testing speculative decoding exactness"
    input_ids = synthetic_engine.tokenizer(prompt)["input_ids"]

    # Autoregressive generation
    ar_out = synthetic_engine.generate_autoregressive(
        input_ids=input_ids,
        max_new_tokens=15,
        temperature=0.0,
    )

    # Speculative generation (fixed gamma = 3)
    spec_out = synthetic_engine.generate_speculative(
        input_ids=input_ids,
        gamma=3,
        max_new_tokens=15,
        temperature=0.0,
        adaptive_policy=None,
    )

    # Assert exact token matching
    assert torch.equal(ar_out.token_ids, spec_out.token_ids), (
        f"Mismatch between AR tokens: {ar_out.token_ids.tolist()} "
        f"and Speculative tokens: {spec_out.token_ids.tolist()}"
    )


def test_exactness_with_entropy_gating(synthetic_engine):
    """Verify that enabling EG-Spec preserves output exactness under greedy decoding."""
    prompt = "testing entropy gating exactness"
    input_ids = synthetic_engine.tokenizer(prompt)["input_ids"]

    ar_out = synthetic_engine.generate_autoregressive(
        input_ids=input_ids,
        max_new_tokens=15,
        temperature=0.0,
    )

    policy = EntropyGatedPolicy(
        entropy_threshold=1.2,
        confidence_margin=0.2,
        min_gamma=1,
        max_gamma=4,
        enabled=True,
    )

    spec_out = synthetic_engine.generate_speculative(
        input_ids=input_ids,
        gamma=4,
        max_new_tokens=15,
        temperature=0.0,
        adaptive_policy=policy,
    )

    assert torch.equal(ar_out.token_ids, spec_out.token_ids)


def test_entropy_gated_policy_logic():
    """Verify threshold triggering in EntropyGatedPolicy."""
    policy = EntropyGatedPolicy(
        entropy_threshold=1.0,
        confidence_margin=0.3,
        min_gamma=1,
        max_gamma=5,
        enabled=True,
    )

    # Uniform distribution -> maximum entropy (log(10) ~= 2.30 nats)
    uniform_logits = torch.zeros(10)
    should_cont, meta = policy.evaluate_draft_step(uniform_logits, current_step=2)
    assert not should_cont
    assert meta["reason"] == "entropy_threshold_exceeded"
    assert meta["entropy"] > 1.0

    # Highly peaked distribution -> near-zero entropy
    peaked_logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    should_cont, meta = policy.evaluate_draft_step(peaked_logits, current_step=2)
    assert should_cont
    assert meta["entropy"] < 0.1
    assert meta["margin"] > 0.9


def test_profiler_statistics():
    """Verify that the profiler captures valid non-negative metrics."""
    profiler = BenchmarkProfiler(device="cpu")
    profiler.start()
    res = profiler.stop(
        num_tokens_generated=20,
        num_draft_tokens=30,
        num_accepted_tokens=24,
        num_speculative_steps=6,
    )

    assert res.total_time_sec >= 0.0
    assert res.tokens_per_sec > 0.0
    assert res.acceptance_rate == pytest.approx(0.8, 0.01)
    assert res.num_accepted_tokens == 24
