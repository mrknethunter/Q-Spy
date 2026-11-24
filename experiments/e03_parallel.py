"""Process-parallel E-03 rotations; identical per-case procedure and seed pairing."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import pandas as pd

from data.curate import HELD_OUT, KNOWN_CLASSES
from experiments.e03_novelty import ROOT, _hash, _json, _plot_rocs, _write_gzip, run_experiment, summarize


def _run_fold(frame, family, arguments):
    return run_experiment(frame, held_outs=[family], **arguments)


def run_parallel(frame, *, workers=4, held_outs=None, artifact_dir="results/e03", **parameters):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    families = [HELD_OUT, *KNOWN_CLASSES] if held_outs is None else list(held_outs)
    if not families or len(families) != len(set(families)) or not set(families) <= {HELD_OUT, *KNOWN_CLASSES}:
        raise ValueError("invalid held-out families")
    out = Path(artifact_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out}: use a new output directory")
    stage = out / "partial_rotations"
    stage.mkdir(parents=True)
    started = time.monotonic()
    running = dict(status="running", experiment="E-03", protocol="P2", tier="L0",
                   started_at=datetime.now(timezone.utc).isoformat(), completed_cases=0)
    (out / "manifest.json").write_text(_json(running))
    try:
        # Spawn avoids forking an initialized Aer/OpenMP runtime, and matches Windows.
        with ProcessPoolExecutor(max_workers=min(workers, len(families)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            tasks = {pool.submit(_run_fold, frame, family,
                                 dict(parameters, artifact_dir=str(stage / family))): family for family in families}
            for future in as_completed(tasks):
                family = tasks[future]
                future.result()
                child = json.loads((stage / family / "manifest.json").read_text())
                running["completed_cases"] += child["completed_cases"]
                (out / "manifest.json").write_text(_json(running))
                print(f"E03 completed rotation: {family}", flush=True)
        children = [stage / family for family in families]
        manifests = [json.loads((child / "manifest.json").read_text()) for child in children]
        if any(child["status"] != "complete" for child in manifests):
            raise ValueError("a rotation did not complete")
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
        curves = pd.concat([pd.read_csv(child / "roc_curves.csv.gz", float_precision="round_trip")
                            for child in children], ignore_index=True)
        _write_gzip(out / "roc_curves.csv.gz", curves)
        for name in ("family_errors.csv", "neighbors_fast_misses.csv"):
            pd.concat([pd.read_csv(child / name, float_precision="round_trip") for child in children],
                      ignore_index=True).to_csv(out / name, index=False)
        audit, selections = [], []
        (out / "predictions").mkdir()
        for child in children:
            audit.extend(json.loads((child / "audit.json").read_text()))
            with gzip.open(child / "selected_ids.json.gz", "rt") as file:
                selections.extend(json.load(file))
            for path in (child / "predictions").glob("*.csv.gz"):
                shutil.copyfile(path, out / "predictions" / path.name)
        (out / "audit.json").write_text(_json(audit))
        _write_gzip(out / "selected_ids.json.gz", selections)
        _plot_rocs(curves, out)
        manifest = reference | dict(status="complete", started_at=running["started_at"],
                                     finished_at=datetime.now(timezone.utc).isoformat(),
                                     completed_cases=sum(child["completed_cases"] for child in manifests),
                                     metric_rows=len(metrics), runtime_seconds=time.monotonic()-started,
                                     worker_seconds_sum=sum(child["runtime_seconds"] for child in manifests),
                                     execution="independent_family_processes")
        manifest["settings"] = reference["settings"] | dict(held_outs=families, workers=workers)
        manifest["source_sha256"]["experiments/e03_parallel.py"] = _hash(ROOT / "experiments/e03_parallel.py")
        manifest["rotations"] = [{"family": family, "cases": child["completed_cases"],
                                  "runtime_seconds": child["runtime_seconds"]}
                                 for family, child in zip(families, manifests)]
        shutil.rmtree(stage)
        manifest["files"] = {str(path.relative_to(out)): _hash(path) for path in sorted(out.rglob("*"))
                             if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(_json(manifest))
        return metrics
    except BaseException as error:
        running.update(status="failed", error=f"{type(error).__name__}: {error}")
        (out / "manifest.json").write_text(_json(running))
        raise
