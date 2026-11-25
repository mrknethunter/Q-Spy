"""E-03-only artifact and restart checks, runnable without the science stack."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable


SOURCE_FILES = (
    "data/curate.py", "data/splits.py", "data/manifest.json", "data/download.py", "data/profile.py", "features/transforms.py",
    "models/base.py", "models/baselines.py", "kernels/projected.py", "quantum/embed.py",
    "quantum/feature_map.py", "quantum/observables.py", "store/io.py", "experiments/run.py",
    "experiments/e03_novelty.py", "experiments/e03_preprocessing.py", "experiments/e03_metrics.py",
    "experiments/e03_parallel.py", "experiments/e03_report.py", "experiments/e03_integrity.py",
    "experiments/e03_tuning.py",
    "experiments/rq4/e03_gravityspy.yaml", "configs/hardware_frozen.yaml",
    "docs/contracts.md", "requirements.txt",
)
REQUIRED_OUTPUTS = (
    "e03_metrics.csv", "e03_summary.csv", "e03_paired_differences.csv",
    "e03_event_bootstrap.csv", "e03_event_paired_differences.csv",
    "roc_curves.csv.gz", "roc_rotations.png", "selected_ids.json.gz", "audit.json",
    "family_errors.csv", "neighbors_fast_misses.csv", "missed_fast_events.csv",
)


def sha256_file(path: str | Path) -> str:
    """Hash without materializing large prediction/Bloch files in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_path(root: str | Path, relative: str) -> Path:
    """Accept canonical relative POSIX paths contained in the run directory."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"unsafe artifact path: {relative!r}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or parts.as_posix() != relative:
        raise ValueError(f"unsafe artifact path: {relative!r}")
    base = Path(root).resolve()
    result = (base / relative).resolve()
    if result == base or not result.is_relative_to(base):
        raise ValueError(f"artifact escapes run directory: {relative!r}")
    return result


def verify_hashes(root: str | Path, hashes: dict[str, str]) -> None:
    """Reject omitted mappings, missing outputs, path traversal and changed bytes."""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("empty or invalid fingerprint mapping")
    for relative, expected in hashes.items():
        path = artifact_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"hash mismatch or missing file: {relative}")


def case_index(records: list[dict], expected: set[tuple[str, int]]) -> dict:
    """Require exactly one record per family/seed, including duplicate detection."""
    indexed = {(case["held_out"], case["seed"]): case for case in records}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("incomplete or duplicated case coverage")
    return indexed


def verify_complete_artifacts(directory: str | Path, *, repository: str | Path | None = None) -> dict:
    """Check a complete E-03 run before aggregation, resume reuse or reporting.

    This verifies byte integrity and exact case coverage, not scientific metrics.
    Only ``e03_report.verify_results`` can recalculate those metrics from scores.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("experiment") != "E-03"
            or manifest.get("protocol") != "P2" or manifest.get("tier") != "L0"):
        raise ValueError("only complete E-03 P2 L0 experiments may be reused or reported")
    settings = manifest["settings"]
    families, seeds = settings["held_outs"], settings["seeds"]
    if (not families or not seeds or len(set(families)) != len(families)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("invalid or duplicated family/seed selectors")
    keys = {(family, seed) for family in families for seed in seeds}
    if manifest["completed_cases"] != len(keys):
        raise ValueError("completed-case count differs from requested cases")
    required = set(REQUIRED_OUTPUTS)
    for family in families:
        required.update((f"predictions/{family}.csv.gz", f"validation_scores/{family}.csv.gz"))
    if "Fast_Scattering" in families:
        required.add("roc_fast_scattering.png")
    if not required <= set(manifest["files"]):
        raise ValueError("manifest omits required outputs: " + ", ".join(sorted(required-set(manifest["files"]))))
    verify_hashes(directory, manifest["files"])
    if repository is not None:
        if set(manifest["source_sha256"]) != set(SOURCE_FILES):
            raise ValueError("incomplete source fingerprint set")
        verify_hashes(repository, manifest["source_sha256"])
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    case_index(audit, keys)
    with gzip.open(directory / "selected_ids.json.gz", "rt", encoding="utf-8") as stream:
        case_index(json.load(stream), keys)
    return manifest


def verify_restart_identity(previous: dict, requested: dict) -> None:
    """No changed data, split, code, software, seeds, budget or settings on restart."""
    if previous != requested:
        changed = sorted(key for key in set(previous) | set(requested)
                         if previous.get(key) != requested.get(key))
        raise ValueError("incompatible restart identity: " + ", ".join(changed))


def atomic_json(path: str | Path, value: dict) -> None:
    """Replace a manifest atomically; a crash cannot leave a half-written JSON."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".e03-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def conservative_threshold(scores: Iterable[float], fpr: float = .01) -> float:
    """99th percentile with 'higher' order statistic and strict score > threshold.

    No interpolation, labels or score-dependent sign changes are used. The
    empirical exceedance count is checked even for tiny samples and ties.
    """
    values = sorted(float(value) for value in scores)
    if not values or not all(math.isfinite(value) for value in values) or not 0 < fpr < 1:
        raise ValueError("invalid known validation scores or FPR")
    threshold = values[math.ceil((len(values)-1) * (1-fpr))]
    if sum(value > threshold for value in values) / len(values) > fpr:
        raise ValueError("validation calibration exceeds its FPR limit")
    return threshold
