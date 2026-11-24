from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python experiments/h01_pilot_plot.py`
from store.io import read_bloch  # noqa: E402

COLORS = {"L1": "#0E7C86", "L2": "#D98E04", "HW": "#C2415D"}


def plot_pilot(run_prefix: str, path, root=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = {t: read_bloch(f"{run_prefix}-{t}", root=root)[1] for t in ("L0", "L1", "L2", "HW")}
    meta = read_bloch(f"{run_prefix}-HW", root=root)[2]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot([-1, 1], [-1, 1], color="#94A3B8", lw=1)
    for t in ("L2", "HW"):
        ax[0].scatter(B["L0"].ravel(), B[t].ravel(), s=10, alpha=0.6, color=COLORS[t], label=t)
    ax[0].set(xlabel="exact value (L0)", ylabel="measured / emulated value", title="(a) every <X>, <Y>, <Z> vs exact")
    for t in ("L1", "L2", "HW"):
        ax[1].hist((B[t] - B["L0"]).ravel(), bins="auto", histtype="step", lw=1.8, color=COLORS[t], label=t)
    ax[1].set(xlabel="error vs exact", ylabel="count", title="(b) error distribution by tier")
    per_qubit = np.sqrt(((B["HW"] - B["L0"]) ** 2).mean(axis=(0, 2)))
    layout = meta.get("layout") or list(range(len(per_qubit)))
    ax[2].bar(range(len(per_qubit)), per_qubit, color=COLORS["HW"], label="HW RMSE vs exact")
    ro = [(meta.get("calibration") or {}).get("qubits", {}).get(str(q), {}).get("readout_error") for q in layout]
    if all(r is not None for r in ro):
        ax[2].plot(range(len(ro)), ro, "o--", color="#475569", label="readout error (calibration)")
    ax[2].set(xticks=range(len(layout)), xticklabels=layout, xlabel="physical qubit",
              title="(c) hardware error per qubit vs calibration")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="H-01: figures of the pilot run stored in store/runs/<run>-<tier>.")
    p.add_argument("--run", default="hw-20261124T005515Z")
    p.add_argument("--out", default="results/pilot/h01_pilot.png")
    a = p.parse_args()
    plot_pilot(a.run, a.out)
    print(f"wrote {a.out}")
