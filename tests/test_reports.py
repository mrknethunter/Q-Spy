import numpy as np

from experiments.h01_pilot_plot import plot_pilot
from experiments.q06_noise_prediction import plot, predict
from store.io import write_bloch
from tests.fixtures.synthetic import synthetic_bloch, synthetic_curated, synthetic_features

df = synthetic_curated(n_per_class=4)
X, y = synthetic_features(df), df["label"].to_numpy()


def test_q06_prediction_has_one_row_per_config_and_tier(tmp_path):
    res = predict(X, y, configs=[(0.5, 0.1), (1.0, 0.1)], n_qubits=3, layers=1)
    assert len(res) == 2 * 3 and set(res["tier"]) == {"L0", "L1", "L2"}
    assert (res.loc[res["tier"] == "L0", "rmse_vs_L0"] == 0).all() and res["kta"].notna().all()
    plot(res, tmp_path / "q06.png")
    assert (tmp_path / "q06.png").stat().st_size > 0


def test_pilot_plot_reads_the_store(tmp_path):
    B0 = synthetic_bloch(5, 2, seed=1)
    for tier, B in {"L0": B0, "L1": B0 + 0.01, "L2": B0 - 0.01, "HW": B0 + 0.02}.items():
        write_bloch(f"run-{tier}", [str(i) for i in range(5)], np.clip(B, -1, 1), {"tier": tier}, root=tmp_path)
    plot_pilot("run", tmp_path / "pilot.png", root=tmp_path)
    assert (tmp_path / "pilot.png").stat().st_size > 0
