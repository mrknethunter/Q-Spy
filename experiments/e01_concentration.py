from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python experiments/e01_concentration.py`
from diagnostics.alignment import kernel_target_alignment  # noqa: E402
from diagnostics.concentration import effective_rank, off_diagonal_values  # noqa: E402
from kernels.fidelity import fidelity_kernel  # noqa: E402
from kernels.projected import projected_kernel  # noqa: E402
from quantum.embed import EmbedConfig, embed  # noqa: E402

QUBITS = [4, 6, 8, 10, 12, 14, 16]
LAYERS = [1, 2, 4, 8]
S_VALUES = list(np.round(np.logspace(-1.5, 0.5, 9), 4))


def load_points(points: int, seed: int, synthetic: bool):
    if synthetic:
        from tests.fixtures.synthetic import synthetic_curated, synthetic_features
        df = synthetic_curated(seed=seed)
        rows = np.random.default_rng(seed).choice(len(df), min(points, len(df)), replace=False)
        return synthetic_features(df)[rows], df["label"].to_numpy()[rows]
    from data.curate import load_curated
    from data.splits import load_split
    from scripts.qpu_campaign import gravityspy_features
    df = load_curated().set_index("id", drop=False)
    features = gravityspy_features({"scaler_protocol": "P2", "scaler_seed": 0}, df)  # same scaling as the campaign
    train = df.loc[list(load_split("P2", 0, held_out="Fast_Scattering").train_ids)]
    sample = train.groupby("label").sample(n=points // train["label"].nunique(), random_state=seed)
    return features(sample["id"]), sample["label"].to_numpy()


def _measures(K, y):
    try:
        kta = kernel_target_alignment(K, y)
    except ValueError:  # fully degenerate kernel: record it instead of losing the whole grid
        kta = np.nan
    off = off_diagonal_values(K)
    return {"var_offdiag": float(off.var()), "mean_offdiag": float(off.mean()),
            "effective_rank": float(effective_rank(K)), "kta": kta}


def run_grid(X, y, qubits=QUBITS, layers=LAYERS, s_values=S_VALUES, fidelity_max_qubits=14, gamma=1.0):
    rows = []
    for n in qubits:
        for L in layers:
            for s in s_values:
                cfg = EmbedConfig(n, L, float(s))
                B = embed(X, cfg, "L0").B
                rows.append({"n": n, "L": L, "s": s, "kernel": "projected", **_measures(projected_kernel(B, B, gamma), y)})
                if n <= fidelity_max_qubits:
                    rows.append({"n": n, "L": L, "s": s, "kernel": "fidelity", **_measures(fidelity_kernel(X, X, cfg), y)})
            print(f"n={n} L={L} done", flush=True)
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"projected": "#0E7C86", "fidelity": "#C2415D"}
    s_ref = res["s"].iloc[(res["s"] - 1.0).abs().argmin()]
    n_ref = res.loc[res["kernel"] == "fidelity", "n"].max()
    L_ref = 2 if 2 in set(res["L"]) else res["L"].min()
    n_kta = 8 if 8 in set(res["n"]) else res["n"].min()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for kern, c in colors.items():
        for L, style in zip(sorted(res["L"].unique()), ["-", "--", ":", "-."]):
            d = res[(res["kernel"] == kern) & (res["L"] == L) & (res["s"] == s_ref)].sort_values("n")
            ax[0].plot(d["n"], d["var_offdiag"], style, color=c, marker="o", ms=3, label=f"{kern}, L={L}")
            ax[2].plot(d["n"], d["kta"], style, color=c, marker="o", ms=3)
        d = res[(res["kernel"] == kern) & (res["L"] == L_ref) & (res["n"] == n_ref)].sort_values("s")
        ax[1].plot(d["s"], d["var_offdiag"], color=c, marker="o", ms=3, label=kern)
        d = res[(res["kernel"] == kern) & (res["L"] == L_ref) & (res["n"] == n_kta)].sort_values("s")
        ax[3].plot(d["s"], d["kta"], color=c, marker="o", ms=3, label=kern)
    ax[0].set(yscale="log", xlabel="qubits n", ylabel="Var of off-diagonal k", title=f"(a) concentration vs n, s={s_ref}")
    ax[1].set(xscale="log", yscale="log", xlabel="bandwidth s", title=f"(b) vs bandwidth, n={n_ref}, L={L_ref}")
    ax[2].set(xlabel="qubits n", ylabel="kernel-target alignment", title=f"(c) label alignment, s={s_ref}")
    ax[3].set(xscale="log", xlabel="bandwidth s", ylabel="kernel-target alignment", title=f"(d) label alignment vs bandwidth, n={n_kta}, L={L_ref}")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].legend(fontsize=8)
    ax[3].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="E-01: kernel concentration map over (n, L, s), projected vs fidelity, exact L0.")
    p.add_argument("--points", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", help="synthetic features instead of the curated Gravity Spy data")
    p.add_argument("--fidelity-max-qubits", type=int, default=14)
    p.add_argument("--quick", action="store_true", help="small grid for a smoke run")
    p.add_argument("--out", default="results/e01")
    a = p.parse_args()
    X, y = load_points(a.points, a.seed, a.synthetic)
    grid = dict(qubits=[4, 8], layers=[1, 2], s_values=[0.3, 1.0]) if a.quick else {}
    res = run_grid(X, y, fidelity_max_qubits=a.fidelity_max_qubits, **grid)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "synthetic" if a.synthetic else "gravityspy"
    res.to_csv(out / f"e01_{tag}.csv", index=False)
    plot(res, out / f"e01_{tag}.png")
    print(f"wrote {out}/e01_{tag}.csv and .png ({len(res)} rows)")
