"""Process-parallel E-03 rotations; identical per-case procedure and seed pairing."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import inspect
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from data.curate import HELD_OUT, KNOWN_CLASSES
from data.splits import DEFAULT_SPLIT_DIR, verify_dataset
from experiments.e03_novelty import ROOT, _hash, _json, _plot_rocs, _write_gzip, run_experiment, summarize
from experiments.e03_integrity import (SOURCE_FILES, atomic_json, verify_complete_artifacts,
                                       verify_restart_identity)


def _run_fold(frame, family, arguments):
    with threadpool_limits(limits=1):
        return run_experiment(frame, held_outs=[family], **arguments)


def run_parallel(frame, *, workers=4, held_outs=None, artifact_dir="results/e03", resume=False, **parameters):
    """Run independent rotations; resume only byte-verified matching completed workers.

    Incomplete workers are preserved in ``failed_attempts`` before recomputation.
    A complete but corrupt/incompatible worker is rejected, never silently reused.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    families = [HELD_OUT, *KNOWN_CLASSES] if held_outs is None else list(held_outs)
    if not families or len(families) != len(set(families)) or not set(families) <= {HELD_OUT, *KNOWN_CLASSES}:
        raise ValueError("invalid held-out families")
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    bound = inspect.signature(run_experiment).bind(frame, held_outs=families, **parameters)
    bound.apply_defaults()
    effective = {key: value for key, value in bound.arguments.items()
                 if key not in {"frame", "artifact_dir", "held_outs"}}
    effective["seeds"] = list(range(10)) if effective["seeds"] is None else list(effective["seeds"])
    splits = Path(effective["split_dir"] or DEFAULT_SPLIT_DIR).resolve()
    effective["split_dir"] = str(splits)
    verify_dataset(frame, split_dir=splits)
    split_manifest = json.loads((splits / "manifest.json").read_text())
    identity = dict(parameters=effective, families=families, workers=workers,
                    dataset_sha256=split_manifest["dataset_sha256"],
                    split_manifest_sha256=_hash(splits / "manifest.json"),
                    source_sha256={path: _hash(ROOT / path) for path in SOURCE_FILES},
                    python=platform.python_version(),
                    versions={name: importlib.metadata.version(name) for name in
                              ("numpy", "pandas", "scikit-learn", "matplotlib", "qiskit", "qiskit-aer")})
    out = Path(artifact_dir)
    populated = out.exists() and any(out.iterdir())
    if populated and not resume:
        raise FileExistsError(f"{out}: use a new output directory")
    if resume and not populated:
        raise ValueError("resume requires an existing E-03 run manifest")
    previous = None
    if populated:
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        verify_restart_identity(previous.get("restart_identity", {}), identity)
        if previous.get("status") == "complete":
            verify_complete_artifacts(out, repository=ROOT)
            return pd.read_csv(out / "e03_metrics.csv", float_precision="round_trip")
        if previous.get("status") not in {"running", "failed"}:
            raise ValueError("unsupported restart state")
    stage = out / "partial_rotations"
    stage.mkdir(parents=True, exist_ok=resume)
    started = time.monotonic()
    running = dict(status="running", experiment="E-03", protocol="P2", tier="L0",
                   started_at=(previous["started_at"] if previous else datetime.now(timezone.utc).isoformat()),
                   completed_cases=0, restart_identity=identity)
    atomic_json(out / "manifest.json", running)
    try:
        pending = []
        for family in families:
            child_dir = stage / family
            child_path = child_dir / "manifest.json"
            if child_path.is_file():
                try:
                    child = json.loads(child_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeError):
                    child = {"status": "failed"}  # preserve and recompute a torn partial manifest
                if child.get("status") == "complete":
                    _verify_worker(child_dir, family, identity)
                    running["completed_cases"] += child["completed_cases"]
                    continue
            if child_dir.exists():
                # Preserve partial bytes. No unrelated path is moved or removed.
                attempts = out / "failed_attempts" / family
                attempts.mkdir(parents=True, exist_ok=True)
                destination = attempts / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                if not child_dir.resolve().is_relative_to(out.resolve()):
                    raise ValueError("worker path escapes E-03 run directory")
                shutil.move(str(child_dir), str(destination))
            pending.append(family)
        atomic_json(out / "manifest.json", running)
        # Spawn avoids forking an initialized Aer/OpenMP runtime, and matches Windows.
        with ProcessPoolExecutor(max_workers=min(workers, len(families)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            tasks = {pool.submit(_run_fold, frame, family,
                                 dict(effective, artifact_dir=str(stage / family))): family for family in pending}
            for future in as_completed(tasks):
                family = tasks[future]
                future.result()
                child = _verify_worker(stage / family, family, identity)
                running["completed_cases"] += child["completed_cases"]
                atomic_json(out / "manifest.json", running)
                print(f"E03 completed rotation: {family}", flush=True)
        children = [stage / family for family in families]
        manifests = [_verify_worker(child, family, identity) for child, family in zip(children, families)]
        if any(child["status"] != "complete" for child in manifests):
            raise ValueError("a rotation did not complete")
        expected_seeds = set(manifests[0]["settings"]["seeds"])
        for family, child_dir, child_manifest in zip(families, children, manifests):
            if child_manifest["completed_cases"] != len(expected_seeds):
                raise ValueError(f"rotation {family} has incomplete seed coverage")
            if child_manifest["settings"]["held_outs"] != [family]:
                raise ValueError(f"rotation {family} manifest identifies a different fold")
            for relative, digest in child_manifest["files"].items():
                path = child_dir / relative
                if not path.is_file() or _hash(path) != digest:
                    raise ValueError(f"rotation {family} output hash mismatch: {relative}")
            audit = json.loads((child_dir / "audit.json").read_text())
            cases = {(case["held_out"], case["seed"]) for case in audit}
            if cases != {(family, seed) for seed in expected_seeds}:
                raise ValueError(f"rotation {family} has missing or duplicated audit cases")
            if (not (child_dir / "predictions" / f"{family}.csv.gz").is_file()
                    or not (child_dir / "validation_scores" / f"{family}.csv.gz").is_file()):
                raise ValueError(f"rotation {family} has no saved prediction file")
        # Shared source, input, settings and seeds must agree; only the family differs.
        reference = manifests[0]
        for child in manifests[1:]:
            for key in ("dataset_sha256", "split_manifest_sha256", "source_sha256", "versions"):
                if child[key] != reference[key]:
                    raise ValueError(f"rotation provenance differs: {key}")
            for key, value in reference["settings"].items():
                if key != "held_outs" and child["settings"][key] != value:
                    raise ValueError(f"rotation settings differ: {key}")
        metric_frames = [pd.read_csv(child / "e03_metrics.csv", float_precision="round_trip") for child in children]
        metrics = pd.concat(metric_frames, ignore_index=True)
        summary, paired = summarize(metrics, bootstrap=reference["settings"]["bootstrap"])
        metrics.to_csv(out / "e03_metrics.csv", index=False)
        summary.to_csv(out / "e03_summary.csv", index=False)
        paired.to_csv(out / "e03_paired_differences.csv", index=False)
        pd.concat([pd.read_csv(child / "e03_event_bootstrap.csv", float_precision="round_trip")
                   for child in children], ignore_index=True).to_csv(
                       out / "e03_event_bootstrap.csv", index=False)
        pd.concat([pd.read_csv(child / "e03_event_paired_differences.csv", float_precision="round_trip")
                   for child in children], ignore_index=True).to_csv(
                       out / "e03_event_paired_differences.csv", index=False)
        curves = pd.concat([pd.read_csv(child / "roc_curves.csv.gz", float_precision="round_trip")
                            for child in children], ignore_index=True)
        _write_gzip(out / "roc_curves.csv.gz", curves)
        for name in ("family_errors.csv", "neighbors_fast_misses.csv", "missed_fast_events.csv"):
            pd.concat([pd.read_csv(child / name, float_precision="round_trip") for child in children],
                      ignore_index=True).to_csv(out / name, index=False)
        audit, selections = [], []
        (out / "predictions").mkdir(exist_ok=resume)
        (out / "validation_scores").mkdir(exist_ok=resume)
        for child in children:
            audit.extend(json.loads((child / "audit.json").read_text()))
            with gzip.open(child / "selected_ids.json.gz", "rt") as file:
                selections.extend(json.load(file))
            for path in (child / "predictions").glob("*.csv.gz"):
                shutil.copyfile(path, out / "predictions" / path.name)
            for path in (child / "validation_scores").glob("*.csv.gz"):
                shutil.copyfile(path, out / "validation_scores" / path.name)
            if (child / "bloch").exists():
                shutil.copytree(child / "bloch", out / "bloch", dirs_exist_ok=True)
        (out / "audit.json").write_text(_json(audit))
        _write_gzip(out / "selected_ids.json.gz", selections)
        _plot_rocs(curves, out)
        manifest = reference | dict(status="complete", started_at=running["started_at"],
                                     finished_at=datetime.now(timezone.utc).isoformat(),
                                     completed_cases=sum(child["completed_cases"] for child in manifests),
                                     metric_rows=len(metrics), runtime_seconds=time.monotonic()-started,
                                     worker_seconds_sum=sum(child["runtime_seconds"] for child in manifests),
                                     execution="independent_family_processes", restart_identity=identity)
        manifest["settings"] = reference["settings"] | dict(held_outs=families, workers=workers)
        manifest["source_sha256"]["experiments/e03_parallel.py"] = _hash(ROOT / "experiments/e03_parallel.py")
        manifest["rotations"] = [{"family": family, "cases": child["completed_cases"],
                                  "runtime_seconds": child["runtime_seconds"]}
                                 for family, child in zip(families, manifests)]
        manifest["files"] = {path.relative_to(out).as_posix(): _hash(path) for path in sorted(out.rglob("*"))
                             if path.is_file() and path != out / "manifest.json"
                             and not any(part in {"partial_rotations", "failed_attempts"}
                                         for part in path.relative_to(out).parts)}
        # Keep worker data until after the complete aggregate has passed its integrity checks.
        atomic_json(out / "manifest.json", manifest)
        verify_complete_artifacts(out, repository=ROOT)
        if not stage.resolve().is_relative_to(out.resolve()):
            raise ValueError("staging directory escapes E-03 run directory")
        shutil.rmtree(stage)
        return metrics
    except BaseException as error:
        running.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(out / "manifest.json", running)
        raise


def _verify_worker(directory, family, identity):
    """Bind a completed worker to the parent's exact data, software and request."""
    manifest = verify_complete_artifacts(directory, repository=ROOT)
    for key in ("dataset_sha256", "split_manifest_sha256", "source_sha256", "versions", "python"):
        if manifest[key] != identity[key]:
            raise ValueError(f"rotation provenance differs: {key}")
    settings = manifest["settings"]
    requested = identity["parameters"]
    for key in ("seeds", "budget", "cv_folds", "max_train", "max_val", "max_test",
                "bootstrap", "event_bootstrap", "embed_batch_size", "store_bloch"):
        if settings[key] != requested[key]:
            raise ValueError(f"rotation settings differ: {key}")
    expected_embed = {key: requested[key] for key in ("n_qubits", "layers", "s")}
    expected_embed["entangle"] = True
    if settings["embed_config"] != expected_embed or settings["held_outs"] != [family]:
        raise ValueError("rotation embedding or family differs")
    if requested["store_bloch"] and not (Path(directory) / "bloch").is_dir():
        raise ValueError("rotation is missing its requested E-03-local Bloch store")
    return manifest
