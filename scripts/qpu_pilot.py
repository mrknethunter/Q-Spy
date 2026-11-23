"""Embed a few points on every tier and compare each with the exact result; results go to results/pilot/.

    python scripts/qpu_pilot.py             dry run: L0, L1, L2 locally + estimated QPU seconds
    python scripts/qpu_pilot.py --submit    real IBM run (needs "QPU approved", docs/qpu_policy.md)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/qpu_pilot.py`
from quantum.embed import DEFAULT_DEVICE, EmbedConfig, embed  # noqa: E402
from tests.fixtures.synthetic import synthetic_curated, synthetic_features  # noqa: E402


def _compare(B, ref):
    return {"rmse": float(np.sqrt(np.mean((B - ref) ** 2))), "corr": float(np.corrcoef(B.ravel(), ref.ravel())[0, 1])}


def run_pilot(points=10, qubits=8, layers=2, s=1.0, precision=0.03, submit=False, device=DEFAULT_DEVICE,
              seed=7, out_dir="results/pilot"):
    X = synthetic_features(synthetic_curated(seed=seed))  # shortcut: synthetic until D-02/D-03 are in main
    X = X[np.random.default_rng(seed).choice(len(X), points, replace=False)]
    cfg = EmbedConfig(qubits, layers, s)
    B = {"L0": embed(X, cfg, "L0").B}
    for tier in ("L1", "L2"):
        B[tier] = embed(X, cfg, tier, precision=precision, seed=seed).B
    hw = embed(X, cfg, "HW", precision=precision, seed=seed, submit=submit, device=device)
    if hw.B is not None:
        B["HW"] = hw.B
    run_id = ("hw-" if submit else "dry-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {"run_id": run_id, "embed_config": hw.meta["embed_config"], "points": points, "precision": precision,
               "hw": hw.meta, "vs_L0": {t: _compare(b, B["L0"]) for t, b in B.items() if t != "L0"}}
    if "HW" in B:
        summary["HW_vs_L2"] = _compare(B["HW"], B["L2"])  # does the emulator predict the hardware noise?
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{run_id}.npz", X=X, **B)
    (out / f"{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=10)
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--s", type=float, default=1.0)
    p.add_argument("--precision", type=float, default=0.03)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--submit", action="store_true", help="really run on IBM hardware (consumes QPU time)")
    a = p.parse_args()
    print(json.dumps(run_pilot(a.points, a.qubits, a.layers, a.s, a.precision, a.submit, a.device), indent=2, default=str))
