import logging
import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import seaborn as sns
except Exception:
    sns = None

logger = logging.getLogger(__name__)


class PublicationVisualizer:
    """Generates camera-ready figures formatted for NeurIPS/IEEE/ACM manuscripts."""

    def __init__(self, output_dir: str = "./paper/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            if sns is not None:
                sns.set_theme(style="whitegrid", font="DejaVu Sans")
            else:
                plt.style.use("default")
            plt.rcParams.update({
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 13,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 14,
            })
        except Exception as e:
            logger.warning(f"Theme setup notice: {e}")

    def plot_speedup_by_category(
        self,
        df,
        filename: str = "fig1_speedup_by_task",
    ):
        """Figure 1: Comparison of Wall-Clock Speedup across Task Categories."""
        try:
            plt.figure(figsize=(8, 4.5), dpi=300)

            if pd is not None and isinstance(df, pd.DataFrame):
                grouped = df.groupby("category")[["static_speedup", "eg_speedup"]].agg(["mean", "sem"]).reset_index()
                categories = grouped["category"].tolist()
                static_means = grouped[("static_speedup", "mean")]
                static_errs = grouped[("static_speedup", "sem")].fillna(0)
                eg_means = grouped[("eg_speedup", "mean")]
                eg_errs = grouped[("eg_speedup", "sem")].fillna(0)
            else:
                records = df if isinstance(df, list) else []
                cat_data = {}
                for r in records:
                    cat = r["category"]
                    cat_data.setdefault(cat, {"static": [], "eg": []})
                    cat_data[cat]["static"].append(r["static_speedup"])
                    cat_data[cat]["eg"].append(r["eg_speedup"])
                categories = list(cat_data.keys())
                static_means = [float(np.mean(cat_data[c]["static"])) if cat_data[c]["static"] else 1.0 for c in categories]
                static_errs = [float(np.std(cat_data[c]["static"]) / np.sqrt(len(cat_data[c]["static"]))) if len(cat_data[c]["static"]) > 1 else 0.0 for c in categories]
                eg_means = [float(np.mean(cat_data[c]["eg"])) if cat_data[c]["eg"] else 1.0 for c in categories]
                eg_errs = [float(np.std(cat_data[c]["eg"]) / np.sqrt(len(cat_data[c]["eg"]))) if len(cat_data[c]["eg"]) > 1 else 0.0 for c in categories]

            pretty_cats = [c.replace("_", " ").title() for c in categories]
            x = np.arange(len(categories))
            width = 0.35

            bars1 = plt.bar(
                x - width / 2,
                static_means,
                width,
                yerr=static_errs,
                capsize=4,
                label="Static Speculative (γ=5)",
                color="#4A90E2",
                edgecolor="black",
                linewidth=0.8,
            )

            bars2 = plt.bar(
                x + width / 2,
                eg_means,
                width,
                yerr=eg_errs,
                capsize=4,
                label="SpecEdge (EG-Spec Adaptive)",
                color="#2ECC71",
                edgecolor="black",
                linewidth=0.8,
            )

            plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="AR Baseline (1.0x)")
            plt.ylabel("Wall-Clock Speedup (x over AR)")
            plt.title("Empirical Speedup across Task Domains under INT4 Quantization")
            plt.xticks(x, pretty_cats)
            plt.ylim(0.5, max(max(eg_means) * 1.25, 2.5))
            plt.legend(loc="upper left", frameon=True)
            plt.tight_layout()

            png_path = os.path.join(self.output_dir, f"{filename}.png")
            pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
            plt.savefig(png_path, dpi=300)
            plt.savefig(pdf_path)
            plt.close()
            return png_path
        except Exception as e:
            logger.warning(f"Plotting '{filename}' deferred (graphics backend notification: {e})")
            return None

    def plot_quantization_acceptance_frontier(
        self,
        quant_regimes: List[str],
        gamma_values: List[int],
        acc_matrix: np.ndarray,
        filename: str = "fig2_quant_acceptance_frontier",
    ):
        """Figure 2: Token Acceptance Rate across Quantization Regimes & Lookahead Horizons."""
        try:
            plt.figure(figsize=(7.5, 4.5), dpi=300)
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

            for i, regime in enumerate(quant_regimes):
                plt.plot(
                    gamma_values,
                    acc_matrix[i],
                    marker="o",
                    linewidth=2.0,
                    markersize=6,
                    label=regime.replace("_", " ").upper(),
                    color=palette[i % len(palette)],
                )

            plt.xlabel("Speculative Lookahead Horizon (γ)")
            plt.ylabel("Mean Token Acceptance Rate (α)")
            plt.title("Acceptance Dynamics under Quantization Bit-Width Shift")
            plt.xticks(gamma_values)
            plt.ylim(0.4, 0.95)
            plt.legend(loc="upper right", frameon=True)
            plt.tight_layout()

            png_path = os.path.join(self.output_dir, f"{filename}.png")
            pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
            plt.savefig(png_path, dpi=300)
            plt.savefig(pdf_path)
            plt.close()
            return png_path
        except Exception as e:
            logger.warning(f"Plotting '{filename}' deferred (graphics backend notification: {e})")
            return None

    def plot_pareto_memory_throughput(
        self,
        configurations: List[dict],
        filename: str = "fig3_pareto_memory_throughput",
    ):
        """Figure 3: Pareto Frontier of VRAM Footprint vs. Generation Throughput."""
        try:
            plt.figure(figsize=(7.5, 4.5), dpi=300)

            for cfg in configurations:
                plt.scatter(
                    cfg["vram_mb"],
                    cfg["throughput"],
                    s=120,
                    color=cfg.get("color", "#34495E"),
                    label=cfg["name"],
                    edgecolor="black",
                    alpha=0.9,
                )
                plt.annotate(
                    cfg["name"],
                    (cfg["vram_mb"], cfg["throughput"]),
                    textcoords="offset points",
                    xytext=(8, -4),
                    fontsize=9,
                )

            plt.xlabel("Peak VRAM Allocation (MB)")
            plt.ylabel("Generation Throughput (Tokens / Sec)")
            plt.title("Edge Deployment Pareto Frontier: Memory vs. Generation Rate")
            plt.tight_layout()

            png_path = os.path.join(self.output_dir, f"{filename}.png")
            pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
            plt.savefig(png_path, dpi=300)
            plt.savefig(pdf_path)
            plt.close()
            return png_path
        except Exception as e:
            logger.warning(f"Plotting '{filename}' deferred (graphics backend notification: {e})")
            return None
