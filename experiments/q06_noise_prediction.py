from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python experiments/q06_noise_prediction.py`
from diagnostics.alignment import kernel_target_alignment  # noqa: E402
from kernels.projected import projected_kernel  # noqa: E402
from quantum.embed import EmbedConfig, embed  # noqa: E402

CONFIGS = [(0.1778, 0.03), (0.1778, 0.05), (0.1778, 0.1), (1.0, 0.03)]  # (bandwidth s, precision)


def predict(X, y, configs=CONFIGS, n_qubits=8, layers=2, seed=7):
    rows = []
    for s, precision in configs:
        cfg = EmbedConfig(n_qubits, layers, s)
        B = {"L0": embed(X, cfg, "L0").B, "L1": embed(X, cfg, "L1", precision=precision, seed=seed).B,
             "L2": embed(X, cfg, "L2", precision=precision, seed=seed).B}
        for tier, b in B.items():
            rows.append({"s": s, "precision": precision, "tier": tier,
                         "kta": kernel_target_alignment(projected_kernel(b, b, 1.0), y),
                         "rmse_vs_L0": float(np.sqrt(np.mean((b - B["L0"]) ** 2))),
                         "spread_L0": float(B["L0"].std(axis=0).mean())})
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"s={s:g}\nprec={p:g}" for s, p in res[["s", "precision"]].drop_duplicates().itertuples(index=False)]
    colors = {"L0": "#475569", "L1": "#0E7C86", "L2": "#C2415D"}
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, (tier, c) in enumerate(colors.items()):
        d = res[res["tier"] == tier]
        ax[0].bar(x + (i - 1) * 0.27, d["kta"], 0.27, color=c, label=tier)
        if tier != "L0":
            ax[1].bar(x + (i - 1.5) * 0.35, d["rmse_vs_L0"], 0.35, color=c, label=f"{tier} vs L0")
    ax[1].scatter(x, res[res["tier"] == "L0"]["precision"], color="black", marker="_", s=400, label="requested precision", zorder=3)
    ax[0].set(xticks=x, xticklabels=labels, ylabel="kernel-target alignment", title="(a) label alignment by tier")
    ax[1].set(xticks=x, xticklabels=labels, ylabel="RMSE of Bloch vectors", title="(b) deviation from the exact result")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_main_subset():
    from data.curate import load_curated
    from data.hardware_subsets import load_hardware_subsets
    from scripts.qpu_campaign import gravityspy_features
    df = load_curated()
    ids = load_hardware_subsets(allow_draft=True)["subsets"]["main"]["ids"]
    features = gravityspy_features({"scaler_protocol": "P2", "scaler_seed": 0}, df)
    return features(ids), df.set_index("id").loc[ids, "label"].to_numpy()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q-06: predicted effect of shot and device noise on the 120 main hardware points.")
    p.add_argument("--out", default="results/q06")
    a = p.parse_args()
    X, y = load_main_subset()
    res = predict(X, y)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "q06_main.csv", index=False)
    plot(res, out / "q06_main.png")
    print(res.round(4).to_string(index=False))
