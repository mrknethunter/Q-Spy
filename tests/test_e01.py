import numpy as np

from experiments.e01_concentration import plot, run_grid
from tests.fixtures.synthetic import synthetic_curated, synthetic_features

df = synthetic_curated(n_per_class=10)
X, y = synthetic_features(df), df["label"].to_numpy()


def test_grid_has_one_row_per_config_and_kernel(tmp_path):
    res = run_grid(X, y, qubits=[2, 4], layers=[1], s_values=[0.5, 1.0], fidelity_max_qubits=2)
    # projected for every config, fidelity only where n <= fidelity_max_qubits
    assert len(res) == 4 + 2 and set(res["kernel"]) == {"projected", "fidelity"}
    assert (res["var_offdiag"] >= 0).all() and res["effective_rank"].between(1, len(X)).all()
    assert np.isfinite(res["kta"]).all()


def test_plot_writes_figure(tmp_path):
    res = run_grid(X, y, qubits=[2, 3], layers=[1, 2], s_values=[0.5, 1.0], fidelity_max_qubits=3)
    plot(res, tmp_path / "e01.png")
    assert (tmp_path / "e01.png").stat().st_size > 0
