#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from models.baselines import build_baseline

# fixture  su 6 classi, k=5 esempi per classe, d=7 feature Omicron
def test_rbf_svm_fewshot_fixture():
    rng = np.random.default_rng(0)
    X_train = np.vstack([rng.normal(loc=c, scale=0.8, size=(5, 7)) for c in range(6)])
    y_train = np.repeat(np.arange(6), 5)

    X_test = np.vstack([rng.normal(loc=c, scale=0.8, size=(20, 7)) for c in range(6)])
    y_test = np.repeat(np.arange(6), 20)

    model = build_baseline("rbf_svm", budget=16, seed=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = model.get_budget_report()

    assert y_pred.shape == (120,)
    assert report["n_configs_evaluated"] <= 16
    assert f1_score(y_test, y_pred, average="macro") > 0.7
    assert balanced_accuracy_score(y_test, y_pred) > 0.7