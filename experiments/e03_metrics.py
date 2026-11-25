"""E-03 metrics and paired event-bootstrap uncertainty helpers."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


METRIC_NAMES = ("auroc", "ap", "tpr_at_1pct_fpr")


def anomaly_metrics(y, scores, fpr_limit: float = 0.01):
    """Return novelty metrics and the attained ROC points.

    `1` is novel and larger scores mean more anomalous. The low-FPR metric is
    the greatest TPR at an actually attained ROC point with FPR no greater
    than `fpr_limit`; it does not interpolate an operating point.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("need aligned one-dimensional binary targets with both classes")
    if not np.isfinite(scores).all() or not 0 < fpr_limit < 1:
        raise ValueError("scores must be finite and FPR limit in (0,1)")
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "tpr_at_1pct_fpr": float(tpr[fpr <= fpr_limit].max()),
    }, (fpr, tpr)


def event_bootstrap_seed(held_out: str, seed: int, base_seed: int = 29) -> int:
    """Stable per-fold seed independent of fold execution or aggregation order."""
    payload = f"E03:event-bootstrap:{base_seed}:{held_out}:{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def paired_event_bootstrap(
    y,
    scores_by_model: dict[str, np.ndarray],
    *,
    n_resamples: int = 1000,
    seed: int = 29,
    fpr_limit: float = 0.01,
):
    """Compute class-stratified paired bootstrap intervals for one fitted fold.

    The same resampled known and novel event indices are used for every model,
    so projected-minus-classical differences remain paired. Resampling within
    each target class fixes the selected test prevalence and estimates
    event-sample uncertainty conditional on the observed class counts. Results
    are returned as metric and paired-difference record lists.
    """
    y = np.asarray(y)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("event bootstrap requires both binary outcome classes")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not scores_by_model or len(scores_by_model) != len(set(scores_by_model)):
        raise ValueError("scores_by_model must have unique model names")
    scores = {name: np.asarray(values, dtype=np.float64) for name, values in scores_by_model.items()}
    if any(values.shape != y.shape or not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("each model must provide one finite score per event")

    known = np.flatnonzero(y == 0)
    novel = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    values = {name: {metric: [] for metric in METRIC_NAMES} for name in scores}
    deltas = {name: {metric: [] for metric in METRIC_NAMES} for name in list(scores)[1:]}
    point_values = {name: anomaly_metrics(y, values_)[0] for name, values_ in scores.items()}

    for _ in range(n_resamples):
        sampled = np.concatenate((
            rng.choice(known, size=len(known), replace=True),
            rng.choice(novel, size=len(novel), replace=True),
        ))
        sampled_y = y[sampled]
        replicate = {}
        for name, score in scores.items():
            replicate[name] = anomaly_metrics(sampled_y, score[sampled], fpr_limit)[0]
            for metric in METRIC_NAMES:
                values[name][metric].append(replicate[name][metric])
        projected = next(iter(scores))
        for baseline in list(scores)[1:]:
            for metric in METRIC_NAMES:
                deltas[baseline][metric].append(replicate[projected][metric] - replicate[baseline][metric])

    metric_rows = []
    for model, metric_values in values.items():
        for metric, samples in metric_values.items():
            low, high = np.quantile(samples, [0.025, 0.975])
            metric_rows.append({
                "model": model,
                "metric": metric,
                "value": point_values[model][metric],
                "ci_low": float(low),
                "ci_high": float(high),
                "n_bootstrap": n_resamples,
                "n_test": len(y),
                "n_known": len(known),
                "n_novel": len(novel),
                "positive_fraction": float(len(novel) / len(y)),
                "ci_method": "paired_stratified_event_bootstrap_fixed_prevalence",
            })

    difference_rows = []
    projected = next(iter(scores))
    for baseline, metric_values in deltas.items():
        for metric, samples in metric_values.items():
            low, high = np.quantile(samples, [0.025, 0.975])
            difference_rows.append({
                "comparison": f"{projected} minus {baseline}",
                "metric": metric,
                "value": point_values[projected][metric] - point_values[baseline][metric],
                "ci_low": float(low),
                "ci_high": float(high),
                "n_bootstrap": n_resamples,
                "n_test": len(y),
                "n_known": len(known),
                "n_novel": len(novel),
                "ci_method": "paired_stratified_event_bootstrap_fixed_prevalence",
            })
    return metric_rows, difference_rows
