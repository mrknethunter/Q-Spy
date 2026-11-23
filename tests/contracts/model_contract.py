from __future__ import annotations

import numpy as np


def check_fewshot_model(model, Z_train, y_train, Z_test) -> None:
    pred = model.fit(Z_train, y_train).predict(Z_test)
    assert pred.shape == (len(Z_test),) and set(pred) <= set(y_train)
    assert len(model.score_samples(Z_test)) == len(Z_test)
    _check_budget(model)


def check_novelty_model(model, Z_train, Z_inliers, Z_outliers) -> None:
    model.fit(Z_train)
    s_in, s_out = model.score_samples(Z_inliers), model.score_samples(Z_outliers)
    assert s_in.shape == (len(Z_inliers),) and s_out.shape == (len(Z_outliers),)
    assert np.median(s_out) > np.median(s_in), "novelty scores must be HIGHER for anomalies"
    _check_budget(model)


def _check_budget(model) -> None:
    r = model.get_budget_report()
    assert 1 <= r["n_configs_evaluated"] <= r["budget_limit"], r
