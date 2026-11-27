#!/usr/bin/env python3
# H-02 / Q-06 / E-04 preview: what the ibm_kingston campaign measured, read back from the store (no QPU use).
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.curate import HELD_OUT, load_curated  # noqa: E402
from data.hardware_subsets import load_hardware_subsets  # noqa: E402
from data.splits import load_split  # noqa: E402
from diagnostics.alignment import kernel_target_alignment  # noqa: E402
from diagnostics.concentration import off_diagonal_variance  # noqa: E402
from kernels.fidelity import fidelity_kernel  # noqa: E402
from kernels.projected import projected_kernel  # noqa: E402
from quantum.embed import EmbedConfig, embed  # noqa: E402
from scripts.qpu_campaign import gravityspy_features  # noqa: E402
from store.io import read_bloch  # noqa: E402

RUN = "hw-20261125T124523Z"
REPEAT = "hw-20261125T113624Z"  # first submission, interrupted after main: an independent repeat of drift-start and main
EXPERIMENTS = ("drift-start", "main", "control", "concentration-n4", "concentration-n8", "concentration-n12", "drift-end")


def load(run, exp, tier):
    ids, B, meta = read_bloch(f"{run}-{exp}-{tier}")
    return list(ids), np.asarray(B, dtype=np.float64), meta


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def delta_kta_ci(B_ref, B, y, reps=2000, seed=0):
    # H4: KTA(hardware) - KTA(exact) with a bootstrap 95% CI over points
    K0, K1 = projected_kernel(B_ref, B_ref, 1.0), projected_kernel(B, B, 1.0)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            d.append(kernel_target_alignment(K1[np.ix_(i, i)], y[i]) - kernel_target_alignment(K0[np.ix_(i, i)], y[i]))
    return float(kernel_target_alignment(K1, y) - kernel_target_alignment(K0, y)), [float(v) for v in np.percentile(d, [2.5, 97.5])]


def novelty_preview(frozen, df, features, main_ids, B_main, n_train=2048, c=31.6, seed=0):
    # shortcut: E-03's most frequent projected choice (c = 31.6, gamma = c / median) and nu = 0.05; E-04 proper re-tunes
    train_ids = load_split(frozen["scaler_protocol"], frozen["scaler_seed"], held_out=HELD_OUT).train_ids
    rng = np.random.default_rng(seed)
    train_ids = sorted(rng.choice(np.array(train_ids), size=n_train, replace=False).tolist())
    cfg = EmbedConfig(frozen["n_qubits"], frozen["layers"], frozen["s"])
    Z = embed(features(train_ids), cfg, "L0").B.reshape(n_train, -1) / np.sqrt(frozen["n_qubits"])
    X = features(train_ids)
    y = (df.loc[main_ids, "label"] == HELD_OUT).to_numpy().astype(int)
    out = {}
    for name, train, tests in (("projected", Z, {t: B.reshape(len(B), -1) / np.sqrt(frozen["n_qubits"]) for t, B in B_main.items()}),
                               ("rbf_features", X, {"features": features(main_ids)})):
        d = ((train[:, None, :] - train[None, :, :]) ** 2).sum(-1)
        median = float(np.median(d[np.triu_indices(n_train, 1)]))
        model = OneClassSVM(kernel="rbf", gamma=c / median, nu=0.05).fit(train)
        for tier, T in tests.items():
            out[f"{name}:{tier}"] = float(roc_auc_score(y, -model.decision_function(T)))
    return out, int(y.sum()), int(len(y) - y.sum())


def analyse(config="configs/hardware_frozen.yaml", out="results/h02"):
    frozen = yaml.safe_load(Path(config).read_text())
    subsets = load_hardware_subsets()
    df = load_curated()
    features = gravityspy_features(frozen, df)
    df = df.set_index("id")
    rows, conc = [], []
    tiers = {}
    for exp in EXPERIMENTS:
        L0 = load(RUN, exp, "L0")
        L2 = load(RUN, exp, "L2")
        HW = load(RUN, exp, "HW")
        assert L0[0] == L2[0] == HW[0]
        ids = L0[0]
        tiers[exp] = dict(ids=ids, L0=L0[1], L2=L2[1], HW=HW[1])
        row = dict(experiment=exp, points=len(ids), n_qubits=L0[1].shape[1], mitigation=HW[2].get("mitigation"),
                   usage_s=HW[2].get("usage_s"), estimated_qpu_s=HW[2].get("estimated_qpu_s"),
                   rmse_HW_L0=rmse(HW[1], L0[1]), rmse_L2_L0=rmse(L2[1], L0[1]), bias_HW=float(np.mean(HW[1] - L0[1])),
                   job_id=(HW[2].get("job_ids") or [None])[0])
        if exp.startswith(("main", "control", "concentration")):
            y = df.loc[ids, "label"].to_numpy()
            for t in ("L0", "L2", "HW"):
                row[f"kta_{t}"] = kernel_target_alignment(projected_kernel(tiers[exp][t], tiers[exp][t], 1.0), y)
        rows.append(row)
        if exp.startswith("concentration"):
            n = L0[1].shape[1]
            cfg = EmbedConfig(n, frozen["layers"], frozen["s"])
            for t in ("L0", "HW"):
                conc.append(dict(n_qubits=n, kernel="projected", tier=t,
                                 variance=off_diagonal_variance(projected_kernel(tiers[exp][t], tiers[exp][t], 1.0))))
            Xc = features(ids)
            conc.append(dict(n_qubits=n, kernel="fidelity", tier="L0", variance=off_diagonal_variance(fidelity_kernel(Xc, Xc, cfg))))
    table = pd.DataFrame(rows)
    conc = pd.DataFrame(conc)

    rep_ids, rep_main, _ = load(REPEAT, "main", "HW")
    assert rep_ids == tiers["main"]["ids"]
    y_main = df.loc[rep_ids, "label"].to_numpy()
    s, e = tiers["drift-start"], tiers["drift-end"]
    assert s["ids"] == e["ids"]
    novelty, n_pos, n_neg = novelty_preview(frozen, df, features, tiers["main"]["ids"],
                                            {"L0": tiers["main"]["L0"], "HW": tiers["main"]["HW"]})
    summary = dict(
        run=RUN, repeat=REPEAT, device=frozen["device"],
        usage_s_campaign=int(table.usage_s.sum()), usage_s_repeat=8 + 124,
        shot_noise_rmse=frozen["precision"],
        main_rmse_HW_L0=table.set_index("experiment").loc["main", "rmse_HW_L0"],
        control_rmse_HW_L0=table.set_index("experiment").loc["control", "rmse_HW_L0"],
        device_error_without_mitigation=float(np.sqrt(max(0.0, table.set_index("experiment").loc["control", "rmse_HW_L0"] ** 2 - frozen["precision"] ** 2))),
        repeat_main_rmse_HW_HW=rmse(rep_main, tiers["main"]["HW"]),
        repeat_main_kta_HW=kernel_target_alignment(projected_kernel(rep_main, rep_main, 1.0), y_main),
        independent_shot_noise_rmse=float(np.sqrt(2) * frozen["precision"]),
        drift_rmse_start_end_HW=rmse(s["HW"], e["HW"]),
        main_kernel_corr_L0_HW=float(np.corrcoef(*(projected_kernel(B, B, 1.0)[np.triu_indices(len(B), 1)]
                                                   for B in (tiers["main"]["L0"], tiers["main"]["HW"])))[0, 1]),
        h4_delta_kta=delta_kta_ci(tiers["main"]["L0"], tiers["main"]["HW"], y_main),
        h4_delta_kta_repeat=delta_kta_ci(tiers["main"]["L0"], rep_main, y_main),
        novelty_preview_auroc=novelty, novelty_preview_counts=dict(fast_scattering=n_pos, known=n_neg),
    )
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    table.to_csv(o / "h02_experiments.csv", index=False)
    conc.to_csv(o / "h02_concentration.csv", index=False)
    (o / "h02_summary.json").write_text(json.dumps(summary, indent=2))
    plot(tiers, table, conc, summary, frozen, o / "h02_campaign.png")
    return table, conc, summary


def plot(tiers, table, conc, summary, frozen, path):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    m, c = tiers["main"], tiers["control"]
    ax[0].scatter(c["L0"].ravel(), c["HW"].ravel(), s=4, alpha=.35, color="#9aa5b1", label="control, no mitigation")
    ax[0].scatter(m["L0"].ravel(), m["HW"].ravel(), s=4, alpha=.5, color="#c0392b", label="main, mitigated")
    ax[0].plot([-1, 1], [-1, 1], color="k", lw=.8)
    ax[0].set(xlabel="exact (L0)", ylabel="ibm_kingston", title="(a) measured vs exact Bloch components")
    ax[0].legend(fontsize=9, loc="upper left", markerscale=3)
    bins = np.linspace(-.15, .15, 61)
    ax[1].hist((m["L2"] - m["L0"]).ravel(), bins, histtype="step", lw=1.6, color="#e67e22", label=f"L2 emulator, rmse {rmse(m['L2'], m['L0']):.3f}")
    ax[1].hist((m["HW"] - m["L0"]).ravel(), bins, histtype="step", lw=1.6, color="#c0392b", label=f"HW mitigated, rmse {summary['main_rmse_HW_L0']:.3f}")
    ax[1].hist((c["HW"] - c["L0"]).ravel(), bins, histtype="step", lw=1.6, color="#5d6d7e", label=f"HW no mitigation, rmse {summary['control_rmse_HW_L0']:.3f}")
    ax[1].axvline(0, color="k", lw=.6)
    ax[1].set(xlabel="measured − exact", ylabel="count", title="(b) error distribution")
    ax[1].legend(fontsize=9)
    for (kernel, tier), d in conc.groupby(["kernel", "tier"]):
        style = dict(projected=dict(color="#1e8449"), fidelity=dict(color="#c0392b"))[kernel]
        ax[2].plot(d.n_qubits, d.variance, marker="o" if tier == "L0" else "s", ls="-" if tier == "L0" else "--",
                   label=f"{kernel}, {tier}", **style)
    ax[2].set(yscale="log", xticks=[4, 8, 12], xlabel="qubits", ylabel="off-diagonal kernel variance",
              title="(c) concentration on hardware (30 glitches)")
    ax[2].legend(fontsize=9)
    t = table.set_index("experiment").loc["main"]
    vals = [t.kta_L0, t.kta_L2, t.kta_HW, summary["repeat_main_kta_HW"]]
    ax[3].bar(range(4), vals, color=["#1e8449", "#e67e22", "#c0392b", "#e6b0aa"])
    for i, v in enumerate(vals):
        ax[3].text(i, v + .005, f"{v:.3f}", ha="center", fontsize=10)
    ax[3].set(xticks=range(4), xticklabels=["exact", "emulator", "HW", "HW repeat"], ylim=(0, .62),
              ylabel="kernel-target alignment", title="(d) class information on 120 glitches")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyse the H-02 ibm_kingston campaign from the store (no QPU use).")
    p.add_argument("--out", default="results/h02")
    a = p.parse_args()
    table, conc, summary = analyse(out=a.out)
    print(table.round(4).to_string(index=False))
    print(conc.round(6).to_string(index=False))
    print(json.dumps(summary, indent=2))
