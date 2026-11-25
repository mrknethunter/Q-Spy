#!/usr/bin/env python3
# L-04: curated data -> features -> L0 embedding -> store -> kernel models -> metrics, one P1 and one P2 config.
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.curate import HELD_OUT, load_curated  # noqa: E402
from data.splits import load_split, verify_dataset  # noqa: E402
from features.transforms import make_features  # noqa: E402
from models import FewShotSVC, NoveltyOCSVM, build_baseline, evaluate_fewshot, evaluate_novelty  # noqa: E402
from quantum.embed import EmbedConfig, embed  # noqa: E402
from store.io import read_bloch, write_bloch  # noqa: E402

CFG = EmbedConfig(n_qubits=8, layers=2, s=0.1778)
FULL = dict(k=5, p1_test=700, p2_train=400, p2_test=800, budget=16)
QUICK = dict(k=5, p1_test=70, p2_train=80, p2_test=160, budget=4)


def stratified(df, ids, n, seed):
    part = df.loc[list(ids)]
    if len(part) <= n:
        return part
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in part.groupby("label"):
        take = max(1, round(n * len(group) / len(part)))
        picks.extend(rng.choice(group.index.to_numpy(), size=min(take, len(group)), replace=False))
    return part.loc[sorted(picks)]


def bloch_via_store(run_id, part, X, root):
    write_bloch(run_id, part.index, embed(X, CFG, "L0").B, {"tier": "L0", "embed_config": asdict(CFG)}, root)
    ids, B, _ = read_bloch(run_id, root)
    assert list(ids) == [str(i) for i in part.index]
    return np.asarray(B, dtype=np.float64)


def p1_rows(df, seed, k, n_test, budget, tag, store_root=None):
    split = load_split("P1", seed, k)
    train = df.loc[list(split.train_ids)]
    test = stratified(df, split.test_ids, n_test, seed)
    X_tr, scaler = make_features(train.reset_index())
    X_te, _ = make_features(test.reset_index(), scaler)
    B_tr = bloch_via_store(f"{tag}-P1-k{k}-s{seed}-train", train, X_tr, store_root)
    B_te = bloch_via_store(f"{tag}-P1-k{k}-s{seed}-test", test, X_te, store_root)
    models = {"projected_svc": (FewShotSVC(kernel="projected", n_qubits=8, budget=budget, seed=seed), B_tr, B_te),
              "rbf_svm": (build_baseline("rbf_svm", budget=budget, seed=seed), X_tr, X_te)}
    rows = []
    for name, (model, a, b) in models.items():
        model.fit(a, train["label"].to_numpy())
        m = evaluate_fewshot(test["label"].to_numpy(), model.predict(b))
        rows.append(dict(protocol="P1", seed=seed, k=k, model=name, n_train=len(train), n_test=len(test),
                         n_configs=model.get_budget_report()["n_configs_evaluated"],
                         macro_f1=m["macro_f1"], balanced_accuracy=m["balanced_accuracy"]))
    return rows


def sweep(ks=(5, 10, 20, 50), seeds=(0, 1, 2, 3, 4), n_test=700, budget=16, out="results/e2e"):
    # shortcut: preview of E-02 (P1 few-shot); E-02 proper uses 20 seeds and all baselines
    df = load_curated().set_index("id")
    tag = f"e2e-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    path = Path(out) / "fewshot_sweep.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for k in ks:
        for seed in seeds:
            if len(done) and ((done.k == k) & (done.seed == seed)).any():
                continue
            done = pd.concat([d for d in (done, pd.DataFrame(p1_rows(df, seed, k, n_test, budget, tag))) if len(d)],
                             ignore_index=True)
            Path(out).mkdir(parents=True, exist_ok=True)
            done.to_csv(path, index=False)
    return done


def run(seed=0, quick=False, store_root=None, out="results/e2e"):
    size = QUICK if quick else FULL
    df = load_curated()
    verify_dataset(df)
    df = df.set_index("id")
    tag = f"e2e-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    rows = []

    rows += p1_rows(df, seed, size["k"], size["p1_test"], size["budget"], tag, store_root)

    split = load_split("P2", seed)
    train = stratified(df, split.train_ids, size["p2_train"], seed)
    test = stratified(df, split.test_ids, size["p2_test"], seed)
    X_tr, scaler = make_features(train.reset_index())
    X_te, _ = make_features(test.reset_index(), scaler)
    B_tr = bloch_via_store(f"{tag}-P2-train", train, X_tr, store_root)
    B_te = bloch_via_store(f"{tag}-P2-test", test, X_te, store_root)
    y_novel = (test["label"] == HELD_OUT).to_numpy().astype(int)
    models = {"projected_ocsvm": (NoveltyOCSVM(kernel="projected", n_qubits=8, budget=size["budget"], seed=seed), B_tr, B_te),
              "rbf_ocsvm": (build_baseline("rbf_ocsvm", budget=size["budget"], seed=seed), X_tr, X_te)}
    for name, (model, a, b) in models.items():
        model.fit(a)
        m = evaluate_novelty(y_novel, model.score_samples(b))
        rows.append(dict(protocol="P2", seed=seed, held_out=HELD_OUT, model=name, n_train=len(train), n_test=len(test),
                         n_configs=model.get_budget_report()["n_configs_evaluated"], **m))

    res = pd.DataFrame(rows)
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        res.to_csv(Path(out) / "e2e_l0.csv", index=False)
        (Path(out) / "e2e_l0.json").write_text(json.dumps(dict(run_tag=tag, quick=quick, embed_config=asdict(CFG), **size), indent=2))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="L-04 end-to-end L0 pipeline on real data (one P1, one P2 config).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="small subsets for a smoke run")
    p.add_argument("--out", default="results/e2e")
    p.add_argument("--sweep", action="store_true", help="P1 few-shot preview over k and 5 seeds; resumable")
    a = p.parse_args()
    res = sweep(out=a.out) if a.sweep else run(a.seed, a.quick, out=a.out)
    print(res.round(4).to_string(index=False))
