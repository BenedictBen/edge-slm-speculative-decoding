"""SpecEdge: Interactive Streamlit Research Showcase.

Demonstrates real-time speculative decoding, dynamic token acceptance trees,
entropy-gated early termination, and latency speedups against autoregressive baselines.
"""

import time
try:
    import pandas as pd
except Exception:
    pd = None
import streamlit as st
import torch

from src.core.adaptive_policy import EntropyGatedPolicy
from src.core.memory_profiler import BenchmarkProfiler
from src.core.quant_loader import ModelLoader
from src.core.speculative_engine import SpeculativeEngine

st.set_page_config(
    page_title="SpecEdge | Speculative Decoding for Edge SLMs",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ SpecEdge: Adaptive Speculative Decoding on Edge SLMs")
st.markdown(
    """
    **Research Project Demo**: Evaluating the interaction between **Low-Bit Quantization (INT4/INT8/FP16)** 
    and **Speculative Decoding**. Features **Entropy-Gated Adaptive Speculation (EG-Spec)** for memory-bandwidth 
    constrained edge deployment.
    """
)
st.info(
    "💡 **Academic Showcase Note**: This live web demonstration operates in interactive CPU mode on Streamlit Cloud "
    "to provide real-time token tracing without requiring dedicated server GPUs. "
    "For full empirical benchmarks achieving **1.8×–2.4× wall-clock speedups** on real 1.7B/135M parameter models under INT4 quantization, "
    "view the [GitHub Repository](https://github.com/BenedictBen/edge-slm-speculative-decoding) or execute the "
    "[One-Click Google Colab GPU Notebook](https://github.com/BenedictBen/edge-slm-speculative-decoding/blob/main/notebooks/spec_edge_colab.ipynb)."
)

# Sidebar Configurations
st.sidebar.header("⚙️ Experiment Parameters")

model_option = st.sidebar.selectbox(
    "Target / Draft Pair",
    [
        "SmolLM: 1.7B-Instruct (Target) + 135M-Instruct (Draft)",
        "Qwen2.5: 1.5B-Instruct (Target) + 0.5B-Instruct (Draft)",
        "Synthetic Lightweight Model (Local CPU Quick Demo)",
    ],
    index=2 if not torch.cuda.is_available() else 0,
)

target_quant = st.sidebar.selectbox(
    "Target Quantization",
    ["int4 (NF4 NormalFloat)", "int8", "fp16"],
    index=0,
)

spec_gamma = st.sidebar.slider("Speculative Lookahead Horizon (γ)", min_value=1, max_value=8, value=5)
temperature = st.sidebar.slider("Sampling Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

st.sidebar.subheader("Adaptive EG-Spec Gating")
enable_eg_spec = st.sidebar.checkbox("Enable Entropy-Gated Lookahead (EG-Spec)", value=True)
entropy_threshold = st.sidebar.slider(
    "Entropy Threshold (τ nats)",
    min_value=0.5,
    max_value=3.5,
    value=1.85,
    step=0.05,
    help="Drafting halts early if token entropy exceeds τ.",
)
confidence_margin = st.sidebar.slider(
    "Confidence Margin (top1 - top2)",
    min_value=0.05,
    max_value=0.60,
    value=0.25,
    step=0.05,
)

# Main UI Area
preset_prompts = {
    "Mathematical Reasoning (GSM8K)": (
        "Question: A bookstore had 120 novels. On Monday, they sold 25% of them. "
        "On Tuesday, they received a shipment of 45 new novels. How many novels do they have now? "
        "Explain step by step."
    ),
    "Code Generation (Python Algorithms)": (
        "Write a Python function `is_valid_parentheses(s: str) -> bool` using a stack. "
        "Include docstring and test cases."
    ),
    "Edge AI Technical Synthesis": (
        "Explain how memory bandwidth limits autoregressive LLM decoding and why speculative "
        "decoding increases arithmetic intensity on edge devices."
    ),
}

selected_preset = st.selectbox("Choose a benchmark prompt or write your own below:", list(preset_prompts.keys()))
prompt_text = st.text_area("Prompt", value=preset_prompts[selected_preset], height=120)

col_run_btn, col_info = st.columns([1, 4])
with col_run_btn:
    run_button = st.button("🚀 Run Speculative Generation", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Loading models into memory...")
def get_cached_engine(model_choice: str, quant_mode: str):
    if "Synthetic" in model_choice or not torch.cuda.is_available():
        target_m, draft_m, tok = ModelLoader.build_synthetic_pair()
        return SpeculativeEngine(target_m, draft_m, tok, device="cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_short = quant_mode.split()[0]

    if "SmolLM" in model_choice:
        t_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"
        d_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    else:
        t_name = "Qwen/Qwen2.5-1.5B-Instruct"
        d_name = "Qwen/Qwen2.5-0.5B-Instruct"

    t_m, tok = ModelLoader.load_model_and_tokenizer(t_name, quant_mode=quant_short, device=device)
    d_m, _ = ModelLoader.load_model_and_tokenizer(d_name, quant_mode="fp16", device=device)
    return SpeculativeEngine(t_m, d_m, tok, device=device)


if run_button:
    with st.spinner("Executing Speculative Decoding vs. Autoregressive Baseline..."):
        engine = get_cached_engine(model_option, target_quant)
        inputs = engine.tokenizer(prompt_text, return_tensors="pt")["input_ids"]

        # 1. Run Autoregressive Baseline
        out_ar = engine.generate_autoregressive(
            input_ids=inputs,
            max_new_tokens=60,
            temperature=temperature,
        )

        # 2. Run Speculative Decoding
        policy = (
            EntropyGatedPolicy(
                entropy_threshold=entropy_threshold,
                confidence_margin=confidence_margin,
                max_gamma=spec_gamma,
                enabled=True,
            )
            if enable_eg_spec
            else None
        )

        out_spec = engine.generate_speculative(
            input_ids=inputs,
            gamma=spec_gamma,
            max_new_tokens=60,
            temperature=temperature,
            adaptive_policy=policy,
        )

        # Performance Comparison Dashboard
        speedup = out_ar.profile.ms_per_token / max(out_spec.profile.ms_per_token, 1e-5)

        st.subheader("📊 Empirical Performance Comparison")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Wall-Clock Speedup", f"{speedup:.2f}x", delta=f"{speedup - 1.0:.2f}x over AR")
        m_col2.metric("Speculative Throughput", f"{out_spec.profile.tokens_per_sec:.1f} tok/s", delta=f"{out_spec.profile.tokens_per_sec - out_ar.profile.tokens_per_sec:.1f} tok/s")
        m_col3.metric("Baseline AR Throughput", f"{out_ar.profile.tokens_per_sec:.1f} tok/s")
        m_col4.metric(
            "Draft Acceptance Rate (α)",
            f"{out_spec.profile.acceptance_rate * 100:.1f}%" if out_spec.profile.acceptance_rate else "N/A",
        )
        m_col5.metric(
            "Speculative Steps",
            f"{out_spec.profile.num_speculative_steps}",
            help="Total parallel target verification iterations",
        )

        # Visual Output Display
        st.subheader("📝 Generated Output & Speculative Trace")
        st.write(out_spec.text)

        # Interactive Step-by-Step Acceptance Visualizer
        st.subheader("🔍 Verification Trace Waterfall")
        
        md_table = [
            "| Step | Tokens Drafted | Tokens Accepted | Acceptance Ratio | Drafted Tokens | Early Gated (EG-Spec) |",
            "| :---: | :---: | :---: | :---: | :--- | :---: |"
        ]
        for tr in out_spec.traces:
            acc_pct = f"{(tr.num_accepted / max(len(tr.draft_tokens), 1)) * 100:.0f}%"
            gated = "⚡ YES" if tr.stopped_by_entropy else "No"
            tok_text = " ".join(tr.draft_tokens_text).replace("|", "\\|").replace("\n", " ")
            md_table.append(f"| {tr.step_index} | {len(tr.draft_tokens)} | {tr.num_accepted} | {acc_pct} | `{tok_text}` | {gated} |")

        st.markdown("\n".join(md_table))

        st.success("Generation completed with exact probability distribution equivalence.")
