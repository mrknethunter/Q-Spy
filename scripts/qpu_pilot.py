from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/qpu_pilot.py`
from quantum.embed import DEFAULT_DEVICE, EmbedConfig, embed  # noqa: E402
from store.io import write_bloch  # noqa: E402
from tests.fixtures.synthetic import synthetic_curated, synthetic_features  # noqa: E402

USAGE = (
    "Embed a few points on every tier, compare each with the exact result, store every tier.\n"
    "  python scripts/qpu_pilot.py            dry run: L0, L1, L2 locally + estimated QPU seconds\n"
    "  python scripts/qpu_pilot.py --submit   real IBM run (needs 'QPU approved', docs/qpu_policy.md)"
)


def _compare(B, ref):
    return {"rmse": float(np.sqrt(np.mean((B - ref) ** 2))), "corr": float(np.corrcoef(B.ravel(), ref.ravel())[0, 1])}


def run_pilot(points=10, qubits=8, layers=2, s=1.0, precision=0.03, submit=False, device=DEFAULT_DEVICE,
              seed=7, out_dir="results/pilot", store_root=None):
    df = synthetic_curated(seed=seed)  # shortcut: synthetic until D-02/D-03 are in main
    rows = np.random.default_rng(seed).choice(len(df), points, replace=False)
    ids, X = df["id"].to_numpy()[rows], synthetic_features(df)[rows]
    cfg = EmbedConfig(qubits, layers, s)
    res = {"L0": embed(X, cfg, "L0")}
    for tier in ("L1", "L2"):
        res[tier] = embed(X, cfg, tier, precision=precision, seed=seed)
    hw = embed(X, cfg, "HW", precision=precision, seed=seed, submit=submit, device=device)
    if hw.B is not None:
        res["HW"] = hw
    run_id = ("hw-" if submit else "dry-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for tier, r in res.items():
        write_bloch(f"{run_id}-{tier}", ids, r.B, r.meta, root=store_root)
    B0 = res["L0"].B
    summary = {"run_id": run_id, "embed_config": hw.meta["embed_config"], "points": points, "precision": precision,
               "hw": hw.meta, "vs_L0": {t: _compare(r.B, B0) for t, r in res.items() if t != "L0"}}
    if "HW" in res:
        summary["HW_vs_L2"] = _compare(res["HW"].B, res["L2"].B)  # does the emulator predict the hardware noise?
        m = hw.meta
        summary["ledger_row"] = (f"| {datetime.now(timezone.utc):%Y-%m-%d} | H-01 | | {m['backend_name']} | pilot, {points} points, "
                                 f"{qubits} qubits, L={layers} | {m['estimated_qpu_s']:.1f} | {m['usage_s']} | {', '.join(m['job_ids'])} |")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=10)
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--s", type=float, default=1.0)
    p.add_argument("--precision", type=float, default=0.03)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--submit", action="store_true", help="really run on IBM hardware (consumes QPU time)")
    a = p.parse_args()
    print(json.dumps(run_pilot(a.points, a.qubits, a.layers, a.s, a.precision, a.submit, a.device), indent=2, default=str))
