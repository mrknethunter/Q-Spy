"""E-03: auditable P2 novelty detection at exact L0; see docs/E03.md."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from data.curate import HELD_OUT, KNOWN_CLASSES, load_curated
from data.splits import DEFAULT_SPLIT_DIR, load_split, verify_dataset
from features.transforms import FEATURE_COLUMNS, make_features
from models.baselines import build_baseline
from quantum.embed import EmbedConfig, embed
from store.io import write_bloch

MODELS = ("projected_ocsvm", "rbf_ocsvm", "isolation_forest", "lof")
REPRESENTATIONS = {"projected_ocsvm": "R3", "rbf_ocsvm": "R1",
                   "isolation_forest": "X_tilde", "lof": "X_tilde"}
ROC_GRID = np.unique(np.r_[0., np.geomspace(1e-4, .01, 40), np.linspace(.01, 1., 200)])
ROOT = Path(__file__).resolve().parent.parent


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _id_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def _write_gzip(path, value):
    with Path(path).open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                if isinstance(value, pd.DataFrame):
                    value.to_csv(text, index=False, float_format="%.17g")
                else:
                    text.write(_json(value))


def load_data(curated_csv=None, raw_dir=None):
    """I-03 loader. CSV parsing preserves the D-04 binary-float fingerprint."""
    if curated_csv is not None and raw_dir is not None:
        raise ValueError("choose curated_csv OR raw_dir")
    frame = (pd.read_csv(curated_csv, float_precision="round_trip")
             if curated_csv is not None else load_curated(raw_dir=raw_dir))
    return (frame,)


def stratified_subset(frame, limit, namespace):
    """A stable subset of an existing partition, proportional to (label, ifo).

    Reserve one row per nonempty stratum, then use largest remainders for the
    remaining capacity. SHA-256 ranking is independent of input order and RNG.
    limit=0 means all rows; partitions are never redrawn or moved.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("sample limits must be nonnegative integers")
    if frame.empty or frame["id"].duplicated().any():
        raise ValueError("partition must be nonempty with unique IDs")
    if not limit or len(frame) <= limit:
        return frame.reset_index(drop=True).sort_values("id").copy()
    groups = list(frame.groupby(["label", "ifo"], sort=True))
    if limit < len(groups):
        raise ValueError("sample limit cannot preserve every label/detector stratum")
    capacity = np.array([len(group) - 1 for _, group in groups], dtype=np.int64)
    remaining = limit - len(groups)
    numerators = capacity * remaining
    allocations = 1 + numerators // capacity.sum()
    remainders = numerators % capacity.sum()
    for position in sorted(range(len(groups)), key=lambda i: (-remainders[i], i))[:limit - int(allocations.sum())]:
        allocations[position] += 1
    selected = []
    for (_, group), count in zip(groups, allocations):
        ids = group["id"].astype(str).tolist()
        ranking = sorted(range(len(group)), key=lambda i: (
            hashlib.sha256((namespace + "\0" + ids[i]).encode()).digest(), ids[i]))
        selected.append(group.iloc[ranking[:int(count)]])
    return pd.concat(selected, ignore_index=True).sort_values("id").copy()


def assert_no_leakage(indexed, split, unknown):
    """Check FULL frozen partitions before any cap, scaling, fit or scoring."""
    ids = [tuple(part) for part in (split.train_ids, split.val_ids, split.test_ids)]
    sets = [set(part) for part in ids]
    if any(not part or len(part) != len(unique) for part, unique in zip(ids, sets)):
        raise ValueError("empty partition or duplicate partition IDs")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("partition ID leakage")
    if not set.union(*sets).issubset(set(indexed.index)):
        raise ValueError("partition contains IDs absent from curated data")
    for partition in ids[:2]:
        if indexed.loc[list(partition), "label"].isin(unknown).any():
            raise ValueError("held-out class leakage into training or validation")
    labels = indexed.loc[list(ids[2]), "label"]
    if not labels.isin(unknown).any() or labels.isin(unknown).all():
        raise ValueError("test must contain both normal and novel examples")
    if not set(unknown).issubset(set(labels)):
        raise ValueError("test is missing a declared unknown family")


def anomaly_metrics(y, scores, fpr_limit=.01):
    """Novel=1, higher score=more anomalous. AP is not trapezoidal PR-AUC.

    TPR@1%FPR is max TPR among attained ROC points with FPR <= .01;
    no linear interpolation or test-selected deployment threshold is used.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("need aligned one-dimensional binary targets with both classes")
    if not np.isfinite(scores).all() or not 0 < fpr_limit < 1:
        raise ValueError("scores must be finite and FPR limit in (0,1)")
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    return {"auroc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "tpr_at_1pct_fpr": float(tpr[fpr <= fpr_limit].max())}, (fpr, tpr)


def calibration_threshold(known_scores, fpr=.01):
    """Threshold fitted ONLY on independent known validation, decision score > t."""
    scores = np.asarray(known_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 < fpr < 1:
        raise ValueError("invalid known validation scores or FPR")
    return float(np.quantile(scores, 1 - fpr, method="higher"))


def _counts(frame):
    return {str(label): {str(ifo): int(count) for ifo, count in group.groupby("ifo").size().items()}
            for label, group in frame.groupby("label", sort=True)}


def exact_bloch(X, cfg, batch_size=512):
    """Bound Aer PUB expansion memory without changing circuits or row order."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("embed_batch_size must be a positive integer")
    blocks = []
    for start in range(0, len(X), batch_size):
        block = embed(X[start:start + batch_size], cfg, "L0").B
        if block is None:
            raise ValueError("L0 embedding returned no Bloch vectors")
        blocks.append(block)
    return np.concatenate(blocks)


def summarize(metrics, bootstrap=5000, seed=29):
    """Paired seed bootstrap, conditional on fixed test IDs and held-out family.

    Does NOT bootstrap rows from overlapping folds as independent observations.
    The same bootstrap seed-index draws are shared across all four models.
    """
    summaries, differences = [], []
    for key, group in metrics.groupby(["held_out", "scope", "metric"], sort=True):
        table = group.pivot(index="seed", columns="model", values="value").sort_index()
        if set(table.columns) != set(MODELS) or table.isna().any().any():
            raise ValueError("incomplete paired results for confidence intervals")
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(table), size=(bootstrap, len(table)))
        for model in MODELS:
            values = table[model].to_numpy()
            samples = values[draws].mean(axis=1)
            lower, upper = np.quantile(samples, [.025, .975]) if len(table) > 1 else (None, None)
            summaries.append(dict(zip(("held_out", "scope", "metric"), key)) |
                             dict(model=model, mean=float(values.mean()), ci_low=lower, ci_high=upper,
                                  n_seeds=len(table), ci_method="paired_seed_bootstrap_conditional_test"))
        for baseline in MODELS[1:]:
            delta = (table[MODELS[0]] - table[baseline]).to_numpy()
            lower, upper = np.quantile(delta[draws].mean(axis=1), [.025, .975]) if len(table) > 1 else (None, None)
            differences.append(dict(zip(("held_out", "scope", "metric"), key)) |
                               dict(comparison=f"{MODELS[0]} minus {baseline}", mean=float(delta.mean()),
                                    ci_low=lower, ci_high=upper, n_seeds=len(table)))
    return pd.DataFrame(summaries), pd.DataFrame(differences)


def _plot_rocs(curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(MODELS, ("#087E8B", "#C44E52", "#6C5B9B", "#D48526")))
    folds = list(curves["held_out"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
    for axis, held_out in zip(axes.flat, folds):
        for model in MODELS:
            subset = curves[(curves["held_out"] == held_out) & (curves["model"] == model)
                            & (curves["scope"] == "target_family")]
            table = subset.pivot(index="seed", columns="fpr", values="tpr").sort_index(axis=1)
            x = table.columns.to_numpy(dtype=float)
            axis.plot(x, table.mean(axis=0), color=colors[model], label=model)
            axis.fill_between(x, table.quantile(.025), table.quantile(.975), color=colors[model], alpha=.12)
        axis.plot([0, 1], [0, 1], "--", color="grey", lw=.8)
        axis.set(title=held_out, xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=.2)
    for axis in list(axes.flat)[len(folds):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("E-03 / L0: held-out family vs known test families; mean ROC, seed 2.5–97.5 percentiles")
    fig.tight_layout()
    fig.savefig(out / "roc_rotations.png", dpi=150)
    plt.close(fig)
    if HELD_OUT in folds:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for model in MODELS:
            subset = curves[(curves["held_out"] == HELD_OUT) & (curves["model"] == model)
                            & (curves["scope"] == "target_family")]
            table = subset.pivot(index="seed", columns="fpr", values="tpr").sort_index(axis=1)
            x = table.columns.to_numpy(dtype=float)
            for axis in axes:
                axis.plot(x, table.mean(axis=0), label=model, color=colors[model])
                axis.fill_between(x, table.quantile(.025), table.quantile(.975), color=colors[model], alpha=.12)
        for axis in axes:
            axis.set(xlabel="False positive rate", ylabel="True positive rate", ylim=(0, 1))
            axis.grid(alpha=.2)
            axis.axvline(.01, color="grey", ls="--", lw=.8)
        axes[0].set(xlim=(0, 1), title="Fast_Scattering: full ROC")
        axes[1].set(xlim=(0, .05), title="Low-FPR region (descriptive ROC)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "roc_fast_scattering.png", dpi=160)
        plt.close(fig)


def run_experiment(frame, *, seeds=None, held_outs=None, budget=8, cv_folds=3,
                   n_qubits=8, layers=2, s=.1778, max_train=2048, max_val=2048,
                   max_test=8192, bootstrap=5000, split_dir=None,
                   artifact_dir="results/e03", store_bloch=True, embed_batch_size=512):
    """I-03 entry point. Return contract-compatible long-form metric rows.

    Caps are explicit resource limits, not the uncapped full-data experiment.
    The frozen partitions are validated before selecting within-partition subsets.
    """
    seeds = list(range(10)) if seeds is None else list(seeds)
    held_outs = [HELD_OUT, *KNOWN_CLASSES] if held_outs is None else list(held_outs)
    for name, value, minimum in (("budget", budget, 1), ("cv_folds", cv_folds, 2),
                                  ("n_qubits", n_qubits, 7), ("layers", layers, 1),
                                  ("bootstrap", bootstrap, 1), ("embed_batch_size", embed_batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if budget > 30:
        raise ValueError("budget <= 30 ensures the same number of candidates for every model")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct nonnegative integers from the split bundle")
    if not held_outs or len(set(held_outs)) != len(held_outs) or not set(held_outs) <= {HELD_OUT, *KNOWN_CLASSES}:
        raise ValueError("invalid held-out family selectors")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")
    for limit in (max_train, max_val, max_test):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("sample limits must be nonnegative integers")
    directory = Path(split_dir) if split_dir is not None else DEFAULT_SPLIT_DIR
    verify_dataset(frame, split_dir=directory)
    split_manifest = json.loads((directory / "manifest.json").read_text())
    entries = {(entry["seed"], entry["held_out"]): entry for entry in split_manifest["splits"]
               if entry["protocol"] == "P2"}
    if any((seed, family) not in entries for family in held_outs for seed in seeds):
        raise ValueError("requested P2 seed/family not present in frozen bundle")
    indexed = frame.set_index("id", drop=False, verify_integrity=True)
    out = Path(artifact_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out}: use a new output directory to preserve previous results")
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    cfg = EmbedConfig(n_qubits, layers, float(s), True)
    settings = dict(seeds=seeds, held_outs=held_outs, budget=budget, cv_folds=cv_folds,
                    embed_config=asdict(cfg), max_train=max_train, max_val=max_val,
                    max_test=max_test, bootstrap=bootstrap, bootstrap_seed=29,
                    embed_batch_size=embed_batch_size,
                    sampling="proportional_label_detector_min_one_SHA256", test_seed=2026)
    source_files = ("data/curate.py", "data/splits.py", "data/manifest.json", "features/transforms.py",
                    "models/baselines.py", "kernels/projected.py", "quantum/embed.py",
                    "quantum/feature_map.py", "quantum/observables.py", "experiments/e03_novelty.py",
                    "requirements.txt")
    manifest = dict(experiment="E-03", protocol="P2", tier="L0", status="running", settings=settings,
                    input_rows=len(frame), data_kind=("gravityspy_d02" if split_manifest["dataset_sha256"] ==
                    "c8bb3d84efa5b583ee41d671ae195b73bb5cdf901e3af03e351e1e3eac0260f1" else "alternative_dataset"),
                    dataset_sha256=split_manifest["dataset_sha256"], split_manifest_sha256=_hash(directory / "manifest.json"),
                    source_sha256={file: _hash(ROOT / file) for file in source_files},
                    python=platform.python_version(), versions={name: importlib.metadata.version(name) for name in
                    ("numpy", "pandas", "scikit-learn", "matplotlib", "qiskit", "qiskit-aer")},
                    started_at=datetime.now(timezone.utc).isoformat(), completed_cases=0)
    (out / "manifest.json").write_text(_json(manifest))
    rows, rocs, audit, selections, family_counts, neighbors = [], [], [], [], [], []
    started = time.monotonic()
    try:
        for family in held_outs:
            fold_predictions = []
            for seed in seeds:
                entry = entries[seed, family]
                unknown = tuple(entry["unknown_classes"])
                if HELD_OUT not in unknown or family not in unknown:
                    raise ValueError("manifest does not reserve the required unknown families")
                split = load_split("P2", seed, held_out=family, split_dir=directory)
                assert_no_leakage(indexed, split, unknown)
                full = [indexed.loc[list(ids)] for ids in (split.train_ids, split.val_ids, split.test_ids)]
                train, val, test = [stratified_subset(part, limit, namespace) for part, limit, namespace in zip(
                    full, (max_train, max_val, max_test),
                    (f"E03:{family}:train:{seed}", f"E03:{family}:val:{seed}", f"E03:{family}:test:2026"))]
                X_train, scaler = make_features(train)
                X_val, _ = make_features(val, scaler)
                X_test, _ = make_features(test, scaler)
                X = np.concatenate([X_train, X_val, X_test])
                B = exact_bloch(X, cfg, embed_batch_size)
                if B is None or B.shape != (len(X), n_qubits, 3) or not np.isfinite(B).all():
                    raise ValueError("L0 embedding returned invalid Bloch vectors")
                # Exact k_P = RBF(B_flat / sqrt(n), gamma), not RBF(B_flat, gamma).
                Z = B.reshape(len(B), -1).astype(np.float64) / np.sqrt(n_qubits)
                train_end, val_end = len(train), len(train) + len(val)
                Z_parts = (Z[:train_end], Z[train_end:val_end], Z[val_end:])
                config_hash = hashlib.sha256(_json(settings).encode()).hexdigest()[:8]
                case = f"e03-{family}-seed{seed}-{_id_hash(train['id'])[:10]}-{config_hash}"
                embed_run_id = None
                if store_bloch:
                    embed_run_id = case
                    write_bloch(case, pd.concat([train, val, test])["id"].tolist(), B,
                                {"tier": "L0", "embed_config": asdict(cfg), "protocol": "P2",
                                 "held_out": family, "seed": seed,
                                 "scaler_center": scaler.center_.tolist(), "scaler_scale": scaler.scale_.tolist()})
                partitions = dict(zip(("train", "val", "test"), (train, val, test)))
                case_audit = dict(held_out=family, seed=seed, unknown_classes=list(unknown),
                                  full_counts={name: _counts(part) for name, part in zip(partitions, full)},
                                  selected_counts={name: _counts(part) for name, part in partitions.items()},
                                  id_sha256={name: _id_hash(part["id"]) for name, part in partitions.items()},
                                  full_partition_leakage_checked=True, scaler_fit_partition="selected_train_only",
                                  feature_order=list(FEATURE_COLUMNS), scaler_center=scaler.center_.tolist(),
                                  scaler_scale=scaler.scale_.tolist(), embed_config=asdict(cfg),
                                  bloch_sha256=hashlib.sha256(B.tobytes()).hexdigest(), models={})
                selections.append(dict(held_out=family, seed=seed,
                                       **{name: part["id"].tolist() for name, part in partitions.items()}))
                prediction = test[["id", "label", "ifo", "run"]].reset_index(drop=True).copy()
                prediction.insert(0, "seed", seed)
                prediction.insert(0, "held_out", family)
                prediction["is_unknown"] = test["label"].isin(unknown).to_numpy(dtype=int)
                known = ~test["label"].isin(unknown).to_numpy()
                nearest = None
                for model_name in MODELS:
                    model = build_baseline("rbf_ocsvm" if model_name == "projected_ocsvm" else model_name,
                                           budget=budget, seed=seed, cv_folds=cv_folds)
                    train_rep, val_rep, test_rep = Z_parts if model_name == "projected_ocsvm" else (X_train, X_val, X_test)
                    fit_started = time.monotonic()
                    model.fit(train_rep)
                    validation_scores = model.score_samples(val_rep)
                    scores = model.score_samples(test_rep)
                    threshold = calibration_threshold(validation_scores)
                    predicted = scores > threshold
                    report = model.get_budget_report()
                    if report["n_configs_evaluated"] != budget:
                        raise ValueError("unequal hyperparameter configuration budgets")
                    case_audit["models"][model_name] = report | dict(
                        threshold=threshold, validation_fpr=float(np.mean(validation_scores > threshold)),
                        score_std=float(np.std(scores)), collapsed_scores=bool(np.std(scores) < 1e-12),
                        fit_and_score_seconds=time.monotonic() - fit_started,
                        effective_gamma=getattr(model.best_estimator_, "_gamma", None),
                        inner_cv=model.cv_results_)
                    prediction[model_name + "_score"] = scores
                    prediction[model_name + "_threshold"] = threshold
                    for scope in ("target_family", "protocol_unknown"):
                        mask = (known | test["label"].eq(family).to_numpy()) if scope == "target_family" else np.ones(len(test), bool)
                        y = (~known[mask]).astype(int)
                        values, (fpr, tpr) = anomaly_metrics(y, scores[mask])
                        values["tpr_validation_threshold"] = float(predicted[mask][y == 1].mean())
                        values["fpr_validation_threshold"] = float(predicted[mask][y == 0].mean())
                        common = dict(experiment="E-03", protocol="P2", seed=seed, k=None, held_out=family,
                                      scope=scope, representation=REPRESENTATIONS[model_name], model=model_name,
                                      tuning="known_only_inner_cv", embed_run_id=embed_run_id if model_name == "projected_ocsvm" else None,
                                      budget=budget, n_configs_evaluated=report["n_configs_evaluated"],
                                      best_params=json.dumps(report["best_params"], sort_keys=True),
                                      n_train=len(train), n_val=len(val), n_test=int(mask.sum()),
                                      positive_fraction=float(y.mean()))
                        rows.extend(common | dict(metric=metric, value=value) for metric, value in values.items())
                        display_tpr = np.interp(ROC_GRID, fpr, tpr)
                        rocs.extend(dict(held_out=family, seed=seed, scope=scope, model=model_name, fpr=float(x), tpr=float(z))
                                    for x, z in zip(ROC_GRID, display_tpr))
                    for (label, ifo, run), positions in test.reset_index(drop=True).groupby(["label", "ifo", "run"]).indices.items():
                        family_counts.append(dict(held_out=family, seed=seed, model=model_name, label=label,
                                                  ifo=ifo, run=run, is_unknown=label in unknown,
                                                  n=len(positions), flagged=int(predicted[positions].sum())))
                    if family == HELD_OUT:
                        misses = test["label"].eq(HELD_OUT).to_numpy() & ~predicted
                        if misses.any():
                            if nearest is None:
                                nearest = NearestNeighbors(n_neighbors=1).fit(X_train)
                            distances, positions = nearest.kneighbors(X_test[misses])
                            labels = train.iloc[positions[:, 0]]["label"].to_numpy()
                            for label in np.unique(labels):
                                selected = labels == label
                                neighbors.append(dict(seed=seed, model=model_name, nearest_known_family=label,
                                                      missed_fast=int(selected.sum()),
                                                      mean_distance=float(distances[selected, 0].mean())))
                fold_predictions.append(prediction)
                audit.append(case_audit)
                manifest["completed_cases"] += 1
                (out / "manifest.json").write_text(_json(manifest))
                print(f"E03 {manifest['completed_cases']}/{len(seeds)*len(held_outs)}: {family}, seed={seed}, "
                      f"train/val/test={len(train)}/{len(val)}/{len(test)}", flush=True)
            _write_gzip(out / "predictions" / (family + ".csv.gz"), pd.concat(fold_predictions, ignore_index=True))
        metrics = pd.DataFrame(rows)
        curves = pd.DataFrame(rocs)
        summary, paired = summarize(metrics, bootstrap=bootstrap)
        metrics.to_csv(out / "e03_metrics.csv", index=False)
        summary.to_csv(out / "e03_summary.csv", index=False)
        paired.to_csv(out / "e03_paired_differences.csv", index=False)
        _write_gzip(out / "roc_curves.csv.gz", curves)
        pd.DataFrame(family_counts).to_csv(out / "family_errors.csv", index=False)
        pd.DataFrame(neighbors, columns=["seed", "model", "nearest_known_family", "missed_fast", "mean_distance"]).to_csv(
            out / "neighbors_fast_misses.csv", index=False)
        _write_gzip(out / "selected_ids.json.gz", selections)
        (out / "audit.json").write_text(_json(audit))
        _plot_rocs(curves, out)
        manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                        runtime_seconds=time.monotonic()-started, metric_rows=len(metrics),
                        files={str(path.relative_to(out)): _hash(path) for path in sorted(out.rglob("*"))
                               if path.is_file() and path.name != "manifest.json"})
        (out / "manifest.json").write_text(_json(manifest))
        return metrics
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        (out / "manifest.json").write_text(_json(manifest))
        raise
