#!/usr/bin/env python3
"""Statistical aggregation and LaTeX/Markdown reporting utilities (M-06)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class BootstrapSummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_seeds: int


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    mean_diff: float
    significant: bool


def _as_1d_array(scores: Sequence[float] | np.ndarray, name: str = "scores") -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D array with at least 2 elements.")
    return arr


def bootstrap_mean_ci(
    scores: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapSummary:
    """Compute mean, std, and percentile bootstrap CI across random seeds."""
    arr = _as_1d_array(scores)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))

    if np.allclose(arr, arr[0]):
        return BootstrapSummary(mean_val, 0.0, mean_val, mean_val, int(arr.size))

    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_resamples, arr.size), replace=True)
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.percentile(samples.mean(axis=1), [50.0 * alpha, 100.0 - 50.0 * alpha])
    return BootstrapSummary(mean_val, std_val, float(ci_low), float(ci_high), int(arr.size))


def paired_wilcoxon(
    quantum_scores: Sequence[float] | np.ndarray,
    classical_scores: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> WilcoxonResult:
    """Run a paired Wilcoxon signed-rank test on seed-matched metric arrays."""
    q = _as_1d_array(quantum_scores, "quantum_scores")
    c = _as_1d_array(classical_scores, "classical_scores")
    if q.shape != c.shape:
        raise ValueError(f"Length mismatch: {q.size} vs {c.size}.")

    diffs = q - c
    if np.allclose(diffs, 0.0):
        return WilcoxonResult(0.0, 1.0, 0.0, False)

    res = wilcoxon(q, c, alternative=alternative)
    p_val = float(res.pvalue)
    return WilcoxonResult(float(res.statistic), p_val, float(np.mean(diffs)), bool(p_val < alpha))


def compare_best_quantum_vs_classical(
    quantum_results: Mapping[str, Sequence[float] | np.ndarray],
    classical_results: Mapping[str, Sequence[float] | np.ndarray],
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Select highest-mean quantum and classical models and run paired Wilcoxon."""
    if not quantum_results or not classical_results:
        raise ValueError("Both quantum_results and classical_results must be non-empty.")

    q_stats = {
        k: bootstrap_mean_ci(v, n_resamples=n_resamples, seed=seed)
        for k, v in quantum_results.items()
    }
    c_stats = {
        k: bootstrap_mean_ci(v, n_resamples=n_resamples, seed=seed)
        for k, v in classical_results.items()
    }

    best_q = max(q_stats, key=lambda k: q_stats[k].mean)
    best_c = max(c_stats, key=lambda k: c_stats[k].mean)

    return {
        "best_quantum_model": best_q,
        "best_classical_model": best_c,
        "quantum_summary": q_stats[best_q],
        "classical_summary": c_stats[best_c],
        "wilcoxon": paired_wilcoxon(quantum_results[best_q], classical_results[best_c], alpha=alpha),
    }


def _escape_latex(text: str) -> str:
    return re.sub(r"([_%&$#{}])", r"\\\1", str(text))


def _render_table(
    results: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    fmt: Literal["markdown", "latex"],
    columns: Sequence[str] | None = None,
    decimals: int = 3,
    highlight_best: bool = True,
    caption: str = "Q-Spy results (mean and 95\\% bootstrap CI).",
    label: str = "tab:qspy_results",
    n_resamples: int = 10_000,
    seed: int = 42,
) -> str:
    if not results:
        raise ValueError("results must not be empty.")

    summaries = {
        model: {m: bootstrap_mean_ci(s, n_resamples=n_resamples, seed=seed) for m, s in metrics.items()}
        for model, metrics in results.items()
    }
    cols = (
        list(columns)
        if columns is not None
        else list(dict.fromkeys(c for m in summaries.values() for c in m))
    )
    best_per_col = {
        c: max((m[c].mean for m in summaries.values() if c in m), default=-np.inf) for c in cols
    }

    def fmt_cell(s: BootstrapSummary, is_best: bool) -> str:
        txt = f"{s.mean:.{decimals}f} [{s.ci_low:.{decimals}f}, {s.ci_high:.{decimals}f}]"
        if not is_best:
            return txt
        return f"**{txt}**" if fmt == "markdown" else f"\\textbf{{{txt}}}"

    rows: list[tuple[str, list[str]]] = []
    for model, metrics in summaries.items():
        row_label = model if fmt == "markdown" else _escape_latex(model)
        cells = [
            fmt_cell(metrics[c], highlight_best and np.isclose(metrics[c].mean, best_per_col[c]))
            if c in metrics
            else ("—" if fmt == "markdown" else "--")
            for c in cols
        ]
        rows.append((row_label, cells))

    if fmt == "markdown":
        return "\n".join([
            "| Model | " + " | ".join(cols) + " |",
            "| :--- | " + " | ".join(":---:" for _ in cols) + " |",
            *(f"| {model} | " + " | ".join(cells) + " |" for model, cells in rows),
        ])

    header_cols = " & ".join(_escape_latex(c) for c in cols)
    body = [f"{model} & " + " & ".join(cells) + " \\\\" for model, cells in rows]
    return "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{l{'c' * len(cols)}}}",
        "\\toprule",
        f"Model & {header_cols} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def format_markdown_table(
    results: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    columns: Sequence[str] | None = None,
    decimals: int = 3,
    highlight_best: bool = True,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> str:
    """Render a `{model: {metric: seed_scores}}` summary table in Markdown."""
    return _render_table(
        results, "markdown", columns, decimals, highlight_best, n_resamples=n_resamples, seed=seed
    )


def format_latex_table(
    results: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    columns: Sequence[str] | None = None,
    caption: str = "Q-Spy results (mean and 95\\% bootstrap CI).",
    label: str = "tab:qspy_results",
    decimals: int = 3,
    highlight_best: bool = True,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> str:
    """Render a `{model: {metric: seed_scores}}` summary table in LaTeX (`booktabs`)."""
    return _render_table(
        results, "latex", columns, decimals, highlight_best, caption, label, n_resamples, seed
    )