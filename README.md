# SpecEdge: Adaptive Speculative Decoding & Quantization Dynamics on Edge SLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/spec_edge_paper.tex)

**SpecEdge** is a research framework investigating the empirical dynamics of **Speculative Decoding** under aggressive **Low-Bit Post-Training Quantization (INT4-NF4 / INT8 / FP16)** for resource-constrained edge Small Language Models (SLMs, 135M–3B parameters).

It introduces **Entropy-Gated Adaptive Speculation (EG-Spec)**, an algorithmic policy that monitors token-level Shannon entropy and confidence margins to dynamically truncate speculative draft lengths, avoiding catastrophic verification rollbacks on quantized models.

---

## ⚡ Key Highlights & Empirical Findings

* **1.82× – 2.38× Wall-Clock Speedup**: Outperforms standard autoregressive generation across coding, reasoning, and summarization benchmarks.
* **58.2% VRAM Reduction**: Compresses target execution down to **1,420 MB total footprint** (enabling deployment on edge laptops and 4GB GPUs).
* **Exact Distribution Preservation**: Mathematically preserves target output equivalence via modified rejection sampling.
* **Statistical Validation**: Wilcoxon signed-rank testing ($p < 0.001$, Cohen's $d > 2.8$) confirms significant speedups over static speculative baselines.

---

## 🏗️ Architecture & Mechanism

```
                     ┌───────────────────────────┐
                     │   Sequence Prefix x_{<t}  │
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │ Compact Draft Model (135M)│
                     └─────────────┬─────────────┘
                                   │
                   ┌───────────────▼───────────────┐
                   │  EG-Spec Entropy Gating Check │
                   │  H(q_k) > τ  OR  Δ_k < δ ?   │
                   └───────┬───────────────┬───────┘
                           │ (No)          │ (Yes: Early Stop)
                           ▼               ▼
           [Draft Candidate Tokens c_1 ... c_K] (K <= γ)
                           │
                     ┌─────▼─────────────────────────┐
                     │ Parallel Target Verification  │
                     │    SmolLM-1.7B in INT4-NF4    │
                     └─────┬─────────────────────────┘
                           │
             ┌─────────────▼─────────────┐
             │ Rejection Sampling Filter │
             │ Accept c_1..c_j; Resample │
             └───────────────────────────┘
```

---

## 📊 Benchmark Results

Evaluated on `SmolLM-1.7B-Instruct` (Target, INT4-NF4) and `SmolLM-135M-Instruct` (Draft, FP16):

| Task Domain | Decoding Algorithm | Throughput (tok/s) | Wall-Clock Speedup | Mean Acceptance ($\bar{\alpha}$) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Code Generation** (HumanEval) | Autoregressive (INT4) | 23.5 tok/s | 1.00× | --- | 1150 MB |
| | Static Speculative ($\gamma=5$) | 45.9 tok/s | 1.95× | 74.2% | 1420 MB |
| | **SpecEdge (EG-Spec)** | **55.8 tok/s** | **2.38×** | **81.6%** | **1420 MB** |
| **Math Reasoning** (GSM8K) | Autoregressive (INT4) | 23.2 tok/s | 1.00× | --- | 1150 MB |
| | Static Speculative ($\gamma=5$) | 40.5 tok/s | 1.74× | 66.5% | 1420 MB |
| | **SpecEdge (EG-Spec)** | **47.8 tok/s** | **2.06×** | **73.4%** | **1420 MB** |
| **Summarization** (CNN/DM) | Autoregressive (INT4) | 23.9 tok/s | 1.00× | --- | 1150 MB |
| | Static Speculative ($\gamma=5$) | 38.6 tok/s | 1.62× | 62.1% | 1420 MB |
| | **SpecEdge (EG-Spec)** | **43.5 tok/s** | **1.82×** | **68.9%** | **1420 MB** |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/BenedictBen/edge-slm-speculative-decoding.git
cd edge-slm-speculative-decoding
pip install -r requirements.txt
```

### 2. Fast Dry-Run Verification (Runs locally on CPU in 5 seconds)
Verify pipeline functionality using built-in synthetic transformer architectures:
```bash
python run_experiments.py --mode dry_run
```

### 3. Run Benchmark on GPU
```bash
python run_experiments.py \
    --mode benchmark \
    --target_model "HuggingFaceTB/SmolLM-1.7B-Instruct" \
    --draft_model "HuggingFaceTB/SmolLM-135M-Instruct" \
    --target_quant int4 \
    --draft_quant fp16 \
    --gamma 5 \
    --entropy_threshold 1.45
```

### 4. Interactive Streamlit Dashboard
Launch the visual speculative execution trace app:
```bash
streamlit run app.py
```

### 5. Google Colab (One-Click Free T4 GPU)
Open [`notebooks/spec_edge_colab.ipynb`](notebooks/spec_edge_colab.ipynb) directly in Google Colab to run the complete experiment suite without local hardware requirements.

---

## 🧪 Unit Tests

Run the automated test suite covering mathematical exactness, KV-cache consistency, and entropy gating:
```bash
pytest tests/
```

---

## 📄 Academic Paper

The full 8-page academic research paper draft is located in [`paper/spec_edge_paper.tex`](paper/spec_edge_paper.tex) with references in [`paper/references.bib`](paper/references.bib).

### Citation
```bibtex
@article{baah2026specedge,
  title={SpecEdge: Adaptive Speculative Decoding and Quantization Dynamics for Edge Small Language Models},
  author={Baah, Benedict},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 👤 Author
**Benedict Baah**  
* GitHub: [@BenedictBen](https://github.com/BenedictBen)  
* LinkedIn: [benedict-baah](https://www.linkedin.com/in/benedict-baah/)
