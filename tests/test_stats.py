#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap
from scipy.stats import wilcoxon as scipy_wilcoxon

from stats import (
    bootstrap_mean_ci,
    compare_best_quantum_vs_classical,
    format_latex_table,
    format_markdown_table,
    paired_wilcoxon,
)


def test_bootstrap_mean_ci_against_scipy() -> None:
    scores = np.linspace(0.79, 0.88, 10)
    res = bootstrap_mean_ci(scores, n_resamples=20_000, seed=42)
    ref = scipy_bootstrap(
        (scores,), np.mean, n_resamples=20_000, method="percentile", random_state=42
    )

    assert res.mean == pytest.approx(float(np.mean(scores)))
    assert res.std == pytest.approx(float(np.std(scores, ddof=1)))
    assert res.ci_low == pytest.approx(float(ref.confidence_interval.low), abs=5e-3)
    assert res.ci_high == pytest.approx(float(ref.confidence_interval.high), abs=5e-3)

    const = bootstrap_mean_ci(np.full(10, 0.75))
    assert (const.mean, const.std, const.ci_low, const.ci_high) == (0.75, 0.0, 0.75, 0.75)

    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.8])


def test_paired_wilcoxon_against_scipy() -> None:
    q = np.linspace(0.84, 0.91, 10)
    c = np.linspace(0.78, 0.84, 10)

    res = paired_wilcoxon(q, c)
    ref = scipy_wilcoxon(q, c, alternative="two-sided")
    assert res.statistic == pytest.approx(float(ref.statistic))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.significant is True

    zero = paired_wilcoxon(c, c)
    assert (zero.statistic, zero.p_value, zero.significant) == (0.0, 1.0, False)

    with pytest.raises(ValueError, match="Length mismatch"):
        paired_wilcoxon(q[:5], c)


def test_compare_best_and_tables() -> None:
    q_scores = np.linspace(0.86, 0.92, 10)
    c_scores = np.linspace(0.79, 0.85, 10)

    comp = compare_best_quantum_vs_classical(
        {"q_prod": q_scores - 0.1, "q_ent": q_scores},
        {"knn": c_scores - 0.05, "rbf_svm": c_scores},
        n_resamples=1_000,
    )
    assert comp["best_quantum_model"] == "q_ent"
    assert comp["best_classical_model"] == "rbf_svm"
    assert comp["wilcoxon"].p_value == pytest.approx(
        float(scipy_wilcoxon(q_scores, c_scores).pvalue)
    )

    results = {"fewshot_svc": {"macro_f1": q_scores}, "rbf_svm": {"macro_f1": c_scores}}
    md = format_markdown_table(results, decimals=3, n_resamples=1_000)
    tex = format_latex_table(results, decimals=3, n_resamples=1_000)

    assert "| fewshot_svc | **0.890 [" in md
    assert "fewshot\\_svc & \\textbf{0.890 [" in tex
    assert "\\bottomrule" in tex