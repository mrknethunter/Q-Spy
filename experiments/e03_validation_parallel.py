"""Verify genuine completed E-03 workers, then check their exact aggregation.

This postprocessing orchestrator does not change experiment inputs or scoring.
It records its own source hash separately from the generation-source manifest.
Hard-linked snapshots preserve original worker bytes when staging is removed;
snapshots are read-only inputs to the canonical e03_report verifier.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.e03_integrity import atomic_json, sha256_file, verify_complete_artifacts
from experiments.e03_novelty import ROOT
from experiments.e03_report import verify_results


def snapshot_worker(source: Path, destination: Path) -> bool:
    """Link immutable outputs as they appear; copy the real manifest at completion."""
    if not source.is_dir():
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or path.name == "manifest.json" or path.name.startswith(".e03-"):
            continue
        target = destination / path.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except PermissionError:
                shutil.copyfile(path, target)  # Windows sandboxes can forbid hard links
            except FileNotFoundError:
                continue
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        return False
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("status") != "complete":
        return False
    # Completion makes all named files immutable; link any final files now.
    for relative, digest in manifest["files"].items():
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source / relative, target)
            except PermissionError:
                shutil.copyfile(source / relative, target)
        if sha256_file(target) != digest:
            original = source / relative
            if sha256_file(original) != digest or os.path.samefile(original, target):
                raise ValueError(f"completed worker hash mismatch: {relative}")
            # An early byte copy may have observed an open writer. Refresh it
            # only after the source has completed and its hash is verified.
            shutil.copyfile(original, target)
    (destination / "manifest.json").write_bytes(payload)
    verify_complete_artifacts(destination, repository=ROOT)
    return True


def _verify_worker(snapshot, curated_csv):
    with threadpool_limits(limits=1):
        return verify_results(snapshot, curated_csv=curated_csv)


def verify_aggregate(directory, snapshots, receipts):
    """Require verified coverage and exact equality to the canonical worker outputs."""
    directory, snapshots = Path(directory), Path(snapshots)
    manifest = verify_complete_artifacts(directory, repository=ROOT)
    families = manifest["settings"]["held_outs"]
    if set(receipts) != set(families):
        raise ValueError("incomplete independently verified worker coverage")
    children = [snapshots / family for family in families]
    child_manifests = []
    for family, child in zip(families, children):
        sub = verify_complete_artifacts(child, repository=ROOT)
        receipt = receipts[family]
        if (receipt["status"] != "pass" or receipt["manifest_sha256"] != sha256_file(child / "manifest.json")
                or sub["settings"]["held_outs"] != [family]
                or sub["settings"]["seeds"] != manifest["settings"]["seeds"]):
            raise ValueError("worker receipt does not bind the actual experiment cases")
        for key in ("dataset_sha256", "split_manifest_sha256", "source_sha256", "versions"):
            if sub[key] != manifest[key]:
                raise ValueError(f"aggregate/verified-worker provenance mismatch: {key}")
        for key, value in sub["settings"].items():
            if key != "held_outs" and manifest["settings"][key] != value:
                raise ValueError(f"aggregate/worker setting mismatch: {key}")
        for relative, digest in sub["files"].items():
            if relative.startswith(("predictions/", "validation_scores/", "bloch/")):
                if manifest["files"].get(relative) != digest:
                    raise ValueError(f"aggregate changed verified worker bytes: {relative}")
        child_manifests.append(sub)
    # Canonical numerical verification already recalculated each child. Here
    # prove every aggregate table contains precisely those verified values.
    tables = ("e03_metrics.csv", "e03_summary.csv", "e03_paired_differences.csv",
              "e03_event_bootstrap.csv", "e03_event_paired_differences.csv",
              "roc_curves.csv.gz", "family_errors.csv", "neighbors_fast_misses.csv", "missed_fast_events.csv")
    for name in tables:
        frames = [pd.read_csv(child / name, float_precision="round_trip") for child in children]
        populated = [frame for frame in frames if not frame.empty]
        expected = pd.concat(populated, ignore_index=True) if populated else frames[0].iloc[:0].copy()
        actual = pd.read_csv(directory / name, float_precision="round_trip")
        columns = list(expected.columns)
        expected = expected.sort_values(columns, na_position="last").reset_index(drop=True)
        actual = actual.sort_values(columns, na_position="last").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)
    combined_audit, combined_ids = [], []
    for child in children:
        combined_audit.extend(json.loads((child / "audit.json").read_text()))
        with gzip.open(child / "selected_ids.json.gz", "rt") as stream:
            combined_ids.extend(json.load(stream))
    if json.loads((directory / "audit.json").read_text()) != combined_audit:
        raise ValueError("aggregate audit differs from verified worker audits")
    with gzip.open(directory / "selected_ids.json.gz", "rt") as stream:
        if json.load(stream) != combined_ids:
            raise ValueError("aggregate IDs differ from verified worker IDs")
    cases = sum(receipt["full_frozen_cases_checked"] for receipt in receipts.values())
    values = sum(receipt["metric_values_recalculated"] for receipt in receipts.values())
    if cases != manifest["completed_cases"] or values != manifest["metric_rows"]:
        raise ValueError("aggregate coverage differs from independently recalculated cases")
    return dict(status="pass", full_frozen_cases_checked=cases, metric_values_recalculated=values,
        models_per_case=4, all_result_hashes_checked=True, source_hashes_checked=True,
        full_partition_class_exclusion_checked=True, selected_ids_and_labels_checked=True,
        fixed_test_across_seeds_checked=True, equal_budgets_checked=True,
        validation_thresholds_recalculated=True, paired_bootstrap_recalculated=True,
        event_bootstrap_recalculated=True,
        hyperparameter_candidates_folds_calibration_and_winner_checked=True,
        scalers_refitted_from_selected_training=all(r["scalers_refitted_from_selected_training"] for r in receipts.values()),
        manifest_sha256=sha256_file(directory / "manifest.json"),
        verifier_sha256=receipts[families[0]]["verifier_sha256"],
        orchestration_sha256=sha256_file(__file__), verification="canonical_worker_recalculation_and_exact_aggregate_equality")


def watch_and_verify(directory, snapshots, receipts_dir, curated_csv, workers=4):
    directory, snapshots, receipts_dir = map(Path, (directory, snapshots, receipts_dir))
    receipts_dir.mkdir(parents=True, exist_ok=True)
    initial = json.loads((directory / "manifest.json").read_text())
    families = initial["restart_identity"]["families"]
    receipts, submitted = {}, {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while len(receipts) < len(families):
            for family in families:
                if family not in submitted and snapshot_worker(
                        directory / "partial_rotations" / family, snapshots / family):
                    submitted[family] = pool.submit(_verify_worker, snapshots / family, curated_csv)
                    print(f"Independent verification started: {family}", flush=True)
            for family, future in submitted.items():
                if family not in receipts and future.done():
                    receipts[family] = future.result()
                    atomic_json(receipts_dir / f"{family}.json", receipts[family])
                    print(f"Independent verification passed: {family}", flush=True)
            status = json.loads((directory / "manifest.json").read_text())["status"]
            if status == "failed":
                raise ValueError("the experiment failed; no complete verification may be issued")
            if status == "complete" and set(submitted) != set(families):
                raise ValueError("missing original worker snapshots; use the canonical aggregate verifier")
            if len(receipts) < len(families):
                time.sleep(.5)
    while json.loads((directory / "manifest.json").read_text())["status"] == "running":
        time.sleep(.5)
    result = verify_aggregate(directory, snapshots, receipts)
    atomic_json(receipts_dir / "aggregate.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--curated-csv", type=Path, required=True)
    args = parser.parse_args()
    watch_and_verify(args.directory, args.snapshots, args.receipts, args.curated_csv)
