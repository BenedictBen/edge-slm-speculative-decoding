"""Statistical significance testing and confidence interval generation for empirical benchmarks."""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """Performs rigorous statistical hypothesis testing on latency, throughput, and acceptance rates."""

    @staticmethod
    def bootstrap_confidence_interval(
        data: List[float],
        n_bootstraps: int = 2000,
        ci: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Compute mean and bootstrap confidence intervals."""
        arr = np.array(data, dtype=np.float64)
        if len(arr) == 0:
            return 0.0, 0.0, 0.0

        mean = float(np.mean(arr))
        boot_means = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstraps):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(np.mean(sample))

        low_p = (1.0 - ci) / 2.0 * 100.0
        high_p = (1.0 + ci) / 2.0 * 100.0
        ci_lower = float(np.percentile(boot_means, low_p))
        ci_upper = float(np.percentile(boot_means, high_p))

        return mean, ci_lower, ci_upper

    @classmethod
    def compare_paired_methods(
        cls,
        baseline_values: List[float],
        candidate_values: List[float],
        metric_name: str = "latency",
    ) -> Dict[str, float]:
        """Perform paired t-test and Wilcoxon signed-rank test between two methods across identical prompts."""
        base = np.array(baseline_values, dtype=np.float64)
        cand = np.array(candidate_values, dtype=np.float64)

        diffs = cand - base
        base_mean, base_low, base_high = cls.bootstrap_confidence_interval(baseline_values)
        cand_mean, cand_low, cand_high = cls.bootstrap_confidence_interval(candidate_values)

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(cand, base)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(cand, base)
        except Exception:
            wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan

        # Cohen's d effect size for paired samples
        diff_std = np.std(diffs, ddof=1) if len(diffs) > 1 else 1e-9
        cohen_d = float(np.mean(diffs) / max(diff_std, 1e-9))

        return {
            "metric": metric_name,
            "baseline_mean": round(base_mean, 3),
            "baseline_ci95": (round(base_low, 3), round(base_high, 3)),
            "candidate_mean": round(cand_mean, 3),
            "candidate_ci95": (round(cand_low, 3), round(cand_high, 3)),
            "mean_difference": round(float(np.mean(diffs)), 3),
            "t_statistic": round(float(t_stat), 4),
            "t_pvalue": float(t_pvalue),
            "wilcoxon_statistic": round(float(wilcoxon_stat), 4) if not np.isnan(wilcoxon_stat) else None,
            "wilcoxon_pvalue": float(wilcoxon_pvalue) if not np.isnan(wilcoxon_pvalue) else None,
            "cohens_d": round(cohen_d, 3),
            "is_significant_p01": bool(t_pvalue < 0.01 or (not np.isnan(wilcoxon_pvalue) and wilcoxon_pvalue < 0.01)),
        }

    @staticmethod
    def format_latex_table(results_dict: Dict[str, Dict[str, float]]) -> str:
        """Format hypothesis test results into LaTeX table syntax for inclusion in paper."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Statistical Significance of SpecEdge Speedups across Benchmark Tasks (Paired Wilcoxon Signed-Rank Test and Cohen's $d$ Effect Size).}",
            r"\label{tab:statistical_significance}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Benchmark Task} & \textbf{AR Latency (ms)} & \textbf{EG-Spec Latency (ms)} & \textbf{Speedup} & \textbf{$p$-value} & \textbf{Cohen's $d$} \\",
            r"\midrule",
        ]
        for task, res in results_dict.items():
            base_m = res["baseline_mean"]
            cand_m = res["candidate_mean"]
            speedup = round(base_m / max(cand_m, 1e-5), 2)
            pval = res.get("wilcoxon_pvalue", res.get("t_pvalue", 0.0))
            pval_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
            stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            cohen_d = res.get("cohens_d", 0.0)
            lines.append(
                f"{task} & {base_m} & {cand_m} & {speedup}$\\times$ & {pval_str}$^{{{stars}}}$ & {cohen_d} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)
