"""Independent E-03 checks of the shared surrogate-LOCO selection receipt."""
from __future__ import annotations

import json
import math


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _validate_calibration(calibration, params):
    if calibration.get("target_fpr") != .05:
        raise ValueError("nu calibration must target 5% known false alarms")
    trials = calibration.get("nu_trials", [])
    if not trials or any(not math.isfinite(t["fpr"]) or not 0 <= t["fpr"] <= 1 for t in trials):
        raise ValueError("missing/nonfinite known-only nu calibration trials")
    chosen = min(trials, key=lambda t: (abs(t["fpr"]-.05), float(t["nu"] or 0)))
    if calibration.get("observed_fpr") != chosen["fpr"] or params.get("nu") != chosen["nu"]:
        raise ValueError("nu was not selected from known-only 5% calibration")
    if "gamma_factor" in params:
        if [t["nu"] for t in trials] != [.01, .05, .1, .15, .2]:
            raise ValueError("OCSVM nu calibration trial grid differs from shared policy")
        median, gamma = calibration.get("median_squared_distance"), calibration.get("effective_gamma")
        if (not isinstance(median, (float, int)) or not math.isfinite(median) or median <= 0
                or not isinstance(gamma, (float, int)) or not math.isfinite(gamma)
                or not math.isclose(gamma, params["gamma_factor"]/median, rel_tol=1e-12)):
            raise ValueError("gamma must equal c / median squared training distance")


def validate_hyperparameters(report, expected_candidates, expected_grid, expected_folds):
    """Reconstruct candidate means, calibration and winner from saved inner folds.

    Expected grid/candidates must be reconstructed by a fresh shared M-04 model,
    and expected folds by its splitter using only selected known-training labels.
    A receipt check is distinct from rerunning its estimator fits on raw data.
    """
    if report.get("tuning_strategy") != "nested_surrogate_loco_auroc":
        raise ValueError("H3 requires nested surrogate-LOCO discrimination, not FPR-only tuning")
    audit = report.get("selection_audit", {})
    rows = report.get("inner_cv", [])
    if (not rows or len(rows) != len(expected_candidates)
            or report["n_configs_evaluated"] != len(rows)
            or _canonical(audit.get("grid")) != _canonical(expected_grid)
            or audit.get("objective") != "mean_surrogate_auroc"
            or audit.get("inner_splits") != len(expected_folds)):
        raise ValueError("hyperparameter budget/grid/objective/fold coverage mismatch")
    means = []
    for row, candidate in zip(rows, expected_candidates):
        if _canonical(row["params"]) != _canonical(candidate) or len(row["folds"]) != len(expected_folds):
            raise ValueError("candidate generation or surrogate fold coverage differs from shared policy")
        scores = []
        for fold, expected in zip(row["folds"], expected_folds):
            if any(fold.get(key) != value for key, value in expected.items()):
                raise ValueError("surrogate selection used incompatible fold membership/support")
            score = fold["auroc"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid surrogate AUROC")
            params = fold["calibrated_params"]
            if any(params.get(key) != value for key, value in candidate.items()):
                raise ValueError("calibration changed the gamma/model candidate")
            _validate_calibration(fold["calibration"], params)
            scores.append(score)
        if not math.isclose(row["mean_surrogate_auroc"], math.fsum(scores)/len(scores), abs_tol=1e-12):
            raise ValueError("candidate mean does not reproduce from surrogate AUROCs")
        means.append(row["mean_surrogate_auroc"])
    best = min(range(len(rows)), key=lambda i: (-means[i], expected_candidates[i].get("gamma_factor", float("inf")), i))
    selected = report["best_params"]
    if (any(selected.get(key) != value for key, value in expected_candidates[best].items())
            or report["best_cv_score"] != means[best] or audit["best_candidate"] != best):
        raise ValueError("selected hyperparameters violate surrogate AUROC/tie rule")
    _validate_calibration(audit["final_nu_calibration"], selected)
    return dict(status="pass", candidates=len(rows), surrogate_folds=len(expected_folds),
                best_candidate=best, objective="mean_surrogate_auroc")
