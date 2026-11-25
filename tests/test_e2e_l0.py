from pathlib import Path

import numpy as np
import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(not any(RAW.glob("*")), reason="needs the D-01 download in data/raw")


def test_e2e_l0_quick_runs_one_p1_and_one_p2_config_through_the_store(tmp_path):
    from experiments.e2e_l0 import run

    res = run(seed=0, quick=True, store_root=tmp_path, out=None)

    assert set(zip(res.protocol, res.model)) == {("P1", "projected_svc"), ("P1", "rbf_svm"),
                                                 ("P2", "projected_ocsvm"), ("P2", "rbf_ocsvm")}
    for protocol in ("P1", "P2"):
        assert res[res.protocol == protocol].n_configs.nunique() == 1
    metrics = res[["macro_f1", "balanced_accuracy", "auroc", "average_precision"]].to_numpy(dtype=float)
    assert np.all((metrics[~np.isnan(metrics)] >= 0) & (metrics[~np.isnan(metrics)] <= 1))
    assert len(list(tmp_path.glob("e2e-*-P1-*"))) == 2 and len(list(tmp_path.glob("e2e-*-P2-*"))) == 2
