from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/qpu_campaign.py`
from diagnostics.alignment import kernel_target_alignment  # noqa: E402
from kernels.projected import projected_kernel  # noqa: E402
from quantum.embed import DEFAULT_DEVICE, EmbedConfig, embed  # noqa: E402
from store.io import write_bloch  # noqa: E402

USAGE = (
    "H-02 main hardware campaign from the frozen configuration (L-05) and the D-05 hardware subsets.\n"
    "  python scripts/qpu_campaign.py            dry run: plan, QPU estimate, L0/L2 emulation in the store\n"
    "  python scripts/qpu_campaign.py --submit   real IBM run in one Batch (needs hw-freeze and 'QPU approved')"
)


def experiments(frozen, subsets):
    n, sub = frozen["n_qubits"], subsets["subsets"]
    conc = [(f"concentration-n{q}", c["ids"], q, True) for q, c in zip(frozen["concentration_qubits"], sub["concentration"])]
    return [("drift-start", sub["drift_references"]["ids"], n, True), ("main", sub["main"]["ids"], n, True),
            ("control", sub["control"]["ids"], n, False), *conc, ("drift-end", sub["drift_references"]["ids"], n, True)]


def _compare(B, ref):
    return {"rmse": float(np.sqrt(np.mean((B - ref) ** 2))), "corr": float(np.corrcoef(B.ravel(), ref.ravel())[0, 1])}


def run_campaign(frozen, subsets, features, submit=False, device=DEFAULT_DEVICE, seed=7, out_dir="results/campaign",
                 store_root=None, labels=None):
    precision = frozen["precision"]
    run_id = ("hw-" if submit else "dry-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan, data = [], {}
    for name, ids, n, mitigation in experiments(frozen, subsets):
        X, cfg = np.asarray(features(ids)), EmbedConfig(n, frozen["layers"], frozen["s"])
        dry = embed(X, cfg, "HW", precision=precision, mitigation=mitigation).meta
        plan.append({"experiment": name, "points": len(ids), "n_qubits": n, "mitigation": mitigation,
                     "n_circuits": dry["n_circuits"], "estimated_qpu_s": dry["estimated_qpu_s"],
                     "two_qubit_depth": dry["two_qubit_depth"]})
        data[name] = (ids, X, cfg, mitigation)
    total = sum(e["estimated_qpu_s"] for e in plan)
    if submit and subsets.get("status") != "bound_to_hw_freeze":  # D-05 status once L-05 is tagged
        raise RuntimeError("hardware subsets are not frozen: tag hw-freeze (L-05) before submitting")
    if submit and total > frozen["budget_s"]:
        raise RuntimeError(f"estimated {total:.0f} s exceeds the budget of {frozen['budget_s']} s: nothing submitted")

    tiers = {}
    for e in plan:
        ids, X, cfg, _ = data[e["experiment"]]
        t = tiers[e["experiment"]] = {"L0": embed(X, cfg, "L0"), "L2": embed(X, cfg, "L2", precision=precision, seed=seed)}
        for tier, r in t.items():
            write_bloch(f"{run_id}-{e['experiment']}-{tier}", ids, r.B, r.meta, root=store_root)
        e["L2_vs_L0"] = _compare(t["L2"].B, t["L0"].B)  # Q-06: predicted hardware degradation (no mitigation)
        if labels is not None and len(set(labels(ids))) > 1:
            e["kta"] = {k: kernel_target_alignment(projected_kernel(r.B, r.B, 1.0), labels(ids)) for k, r in t.items()}
    summary = {"run_id": run_id, "frozen": frozen, "estimated_qpu_s": total, "experiments": plan}

    if submit:
        from qiskit_ibm_runtime import Batch, QiskitRuntimeService
        dev = QiskitRuntimeService().backend(device) if isinstance(device, str) else device
        summary["ledger_rows"] = []
        with Batch(backend=dev) as batch:
            for e in plan:
                ids, X, cfg, mitigation = data[e["experiment"]]
                hw = embed(X, cfg, "HW", precision=precision, seed=seed, submit=True, device=dev, mode=batch,
                           mitigation=mitigation)
                write_bloch(f"{run_id}-{e['experiment']}-HW", ids, hw.B, hw.meta, root=store_root)
                tiers[e["experiment"]]["HW"] = hw
                e.update(usage_s=hw.meta["usage_s"], job_ids=hw.meta["job_ids"],
                         vs_L0=_compare(hw.B, tiers[e["experiment"]]["L0"].B), vs_L2=_compare(hw.B, tiers[e["experiment"]]["L2"].B))
                summary["ledger_rows"].append(
                    f"| {datetime.now(timezone.utc):%Y-%m-%d} | H-02 | | {hw.meta['backend_name']} | {e['experiment']}, "
                    f"{e['points']} points, {cfg.n_qubits} qubits | {e['estimated_qpu_s']:.1f} | {hw.meta['usage_s']} | "
                    f"{', '.join(hw.meta['job_ids'])} |")
        summary["drift"] = _compare(tiers["drift-end"]["HW"].B, tiers["drift-start"]["HW"].B)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def gravityspy_features(frozen, curated=None):
    from data.curate import load_curated
    from data.splits import load_split
    from features.transforms import make_features
    df = (load_curated() if curated is None else curated).set_index("id", drop=False)
    train = load_split(frozen["scaler_protocol"], frozen["scaler_seed"], held_out="Fast_Scattering").train_ids
    _, scaler = make_features(df.loc[list(train)])  # same scaling as the L0 experiments, fitted on training ids only
    return lambda ids: make_features(df.loc[list(ids)], scaler)[0]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/hardware_frozen.yaml")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--submit", action="store_true", help="really run on IBM hardware (consumes QPU time)")
    a = p.parse_args()
    from data.hardware_subsets import load_hardware_subsets
    frozen = yaml.safe_load(Path(a.config).read_text())
    subsets = load_hardware_subsets(allow_draft=not a.submit)
    s = run_campaign(frozen, subsets, gravityspy_features(frozen), submit=a.submit, device=a.device)
    print(json.dumps({k: v for k, v in s.items() if k != "frozen"}, indent=2, default=str))
