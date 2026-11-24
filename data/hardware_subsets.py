"""D-05 hardware IDs, with an explicit pending state until L-05 is tagged.

Generate: ``python -m data.hardware_subsets``. Inspect pending IDs with
``load_hardware_subsets(allow_draft=True)``. Hardware consumers use the default
loader, which rejects a draft. No hardware job or quantum config is created.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from numbers import Integral
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from .splits import DEFAULT_SPLIT_DIR, HELD_OUT, SplitError, load_split, verify_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "hardware_subsets.json"
SCHEMA = "q-spy-hardware-subsets-v1"
DRAFT = "draft_pending_l05"
BOUND = "bound_to_hw_freeze"
FREEZE_TAG = "hw-freeze"
DETECTORS = ("H1", "L1")
FIXED_SOURCE = dict(protocol="P2", seed=0, k=None, held_out=HELD_OUT, partition="test",
                    filter_run="O3b", exclude_all_d04_train_validation=True)


class HardwareSubsetError(ValueError):
    """Invalid subset data, integrity, source or missing L-05 freeze."""


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _rank(items, namespace: str) -> list[str]:
    prefix = namespace.encode("utf-8") + b"\0"
    return sorted(items, key=lambda item: (hashlib.sha256(prefix + item.encode("utf-8")).digest(), item))


def _balanced_counts(labels: list[str], size: int, offset: int = 0) -> dict[str, int]:
    base, extra = divmod(size, len(labels))
    counts = {label: base for label in labels}
    for i in range(extra):
        counts[labels[(offset + i) % len(labels)]] += 1
    return counts


def _select(pool: pd.DataFrame, counts: dict[str, int], *, namespace: str,
            odd_offset: int = 0) -> list[str]:
    """Balance classes and total detectors; rotate ties for odd class quotas."""
    labels = sorted(counts)
    odd = [label for label in labels if counts[label] % 2]
    if odd:
        odd = odd[odd_offset % len(odd):] + odd[:odd_offset % len(odd)]
    target_h1 = (sum(counts.values()) + 1) // 2
    extra_h1 = target_h1 - sum(count // 2 for count in counts.values())
    h1_extra_labels = set(odd[:extra_h1])
    selected = []
    for label in labels:
        h1 = counts[label] // 2 + int(label in h1_extra_labels)
        for ifo, count in (("H1", h1), ("L1", counts[label] - h1)):
            candidates = pool.loc[(pool.label == label) & (pool.ifo == ifo), "id"].tolist()
            if len(candidates) < count:
                raise HardwareSubsetError(f"Insufficient source support for {namespace}: {label}/{ifo} needs {count}, has {len(candidates)}")
            selected.extend(_rank(candidates, namespace + "/" + label + "/" + ifo)[:count])
    return sorted(selected)


def _describe(ids: list[str], catalog: pd.DataFrame) -> dict:
    selected = catalog.set_index("id").loc[ids]
    labels = sorted(set(catalog.label))
    counts = selected.groupby(["label", "ifo"]).size()
    return dict(
        ids=ids, size=len(ids),
        counts_by_class={label: int((selected.label == label).sum()) for label in labels},
        counts_by_detector={ifo: int((selected.ifo == ifo).sum()) for ifo in DETECTORS},
        counts_by_class_detector={label: {ifo: int(counts.get((label, ifo), 0)) for ifo in DETECTORS} for label in labels},
        counts_by_run={run: int((selected.run == run).sum()) for run in ("O1", "O2", "O3a", "O3b")},
    )


def _source(split_dir: Path) -> tuple[dict, pd.DataFrame, bytes]:
    """Use the public frozen loader to verify both catalog and P2 assets."""
    try:
        chosen = load_split("P2", 0, None, HELD_OUT, split_dir=split_dir)
        manifest_bytes = (split_dir / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        with np.load(split_dir / "catalog.npz", allow_pickle=False) as source:
            catalog = pd.DataFrame({column: source[column] for column in ("id", "label", "ifo", "run")})
        exposed = np.zeros(len(catalog), dtype=bool)
        for protocol in ("P1", "P2", "P3"):
            name = protocol + ".npz"
            content = (split_dir / name).read_bytes()
            spec = manifest["files"][name]
            if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
                raise HardwareSubsetError(f"Frozen source asset failed integrity: {name}")
            with np.load(io.BytesIO(content), allow_pickle=False) as source:
                for entry in manifest["splits"]:
                    if entry["protocol"] != protocol:
                        continue
                    member = source[entry["array_key"]]
                    if member.shape != (len(catalog),) or member.dtype != np.uint8 or not np.isin(member, (0, 1, 2, 3)).all():
                        raise HardwareSubsetError("Invalid source split membership")
                    exposed |= (member == 1) | (member == 2)
    except (OSError, ValueError, KeyError) as error:
        raise HardwareSubsetError(f"Cannot use frozen D-04 source: {error}") from error
    test = catalog.loc[catalog.id.isin(chosen.test_ids) & catalog.run.eq("O3b") & ~exposed].sort_values("id").reset_index(drop=True)
    return manifest, test, manifest_bytes


def _selection_digest(payload: dict) -> str:
    # Freeze metadata and lifecycle status cannot change the selected IDs.
    keys = ("schema", "selection_seed", "source", "subsets", "overlap_policy")
    return hashlib.sha256(_json({key: payload[key] for key in keys})).hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    try:
        process = subprocess.run(["git", "-C", str(repo), *args], check=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process.stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise HardwareSubsetError("Cannot verify local hw-freeze tag and required files; complete L-05 first") from error


def _read_freeze(repo: Path, split_manifest: bytes) -> dict:
    commit = _git(repo, "rev-parse", "--verify", "refs/tags/" + FREEZE_TAG + "^{commit}").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise HardwareSubsetError("Invalid resolved hw-freeze commit")
    files = {}
    for path in ("configs/hardware_frozen.yaml", "docs/decisions.md", "data/splits/v1/manifest.json"):
        content = _git(repo, "show", commit + ":" + path)
        if not content.strip():
            raise HardwareSubsetError(f"Tagged file is empty: {path}")
        if path == "data/splits/v1/manifest.json" and content != split_manifest:
            raise HardwareSubsetError("hw-freeze refers to a different D-04 split manifest")
        files[path] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    return dict(tag=FREEZE_TAG, commit=commit, files=files,
                verification="local_git_tag_and_blob_hashes",
                group_approval="L-05 responsibility; not independently inferred from Git tag")


def generate_hardware_subsets(frame: pd.DataFrame, *, split_dir: str | Path | None = None,
                              seed: int = 2026, freeze_repo: str | Path | None = None) -> dict:
    """Prepare fixed balanced IDs; pending unless linked to the actual L-05 tag.

    All candidates come from O3b in the frozen seed-0 P2 test, with every
    committed D-04 training/validation ID excluded across all protocols. No raw
    feature, model output or quantum/hardware result influences selection.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise HardwareSubsetError("selection seed must be an integer in [0, 2**32)")
    directory = Path(split_dir) if split_dir is not None else DEFAULT_SPLIT_DIR
    try:
        verify_dataset(frame, split_dir=directory)
    except SplitError as error:
        raise HardwareSubsetError(str(error)) from error
    manifest, pool, manifest_bytes = _source(directory)
    labels = sorted(set(pool.label))
    if labels != sorted([*manifest["known_classes"], HELD_OUT]):
        raise HardwareSubsetError("P2 test pool must include all known classes and Fast_Scattering")
    main_ids = _select(pool, _balanced_counts(labels, 120), namespace=f"D05/main/{seed}")
    main = pool.loc[pool.id.isin(main_ids)]
    control_ids = _select(pool.loc[~pool.id.isin(main_ids)], _balanced_counts(labels, 40),
                          namespace=f"D05/control/{seed}", odd_offset=len(labels) // 2)
    available = main.copy()
    concentration = []
    remainder = 30 % len(labels)
    for group in range(3):
        counts = _balanced_counts(labels, 30, offset=group * remainder)
        ids = _select(available, counts, namespace=f"D05/concentration/{seed}/{group}", odd_offset=group)
        concentration.append(dict(name=f"concentration_{group + 1}", **_describe(ids, pool)))
        available = available.loc[~available.id.isin(ids)]
    reference_labels = [HELD_OUT, *_rank([label for label in labels if label != HELD_OUT], f"D05/reference-families/{seed}")[:4]]
    reference_ids = _select(main, {label: 1 for label in reference_labels}, namespace=f"D05/drift/{seed}")
    payload = dict(
        schema=SCHEMA, status=DRAFT, selection_seed=int(seed), l05_freeze=None,
        source=dict(**FIXED_SOURCE, dataset_sha256=manifest["dataset_sha256"],
                    split_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    split_manifest_path="data/splits/v1/manifest.json", candidate_rows=len(pool)),
        overlap_policy=dict(
            main_control="disjoint",
            concentration="three pairwise-disjoint subsets of main; 30 main IDs unused by concentration",
            drift_references="subset of main, intentionally reused for repeated drift measurements; may overlap concentration",
            source="all IDs from frozen P2 seed-0 test and O3b; no training or validation ID from any committed D-04 split",
        ),
        subsets=dict(main=_describe(main_ids, pool), control=_describe(control_ids, pool),
                     concentration=concentration, drift_references=_describe(reference_ids, pool)),
    )
    payload["selection_sha256"] = _selection_digest(payload)
    if freeze_repo is not None:
        payload["l05_freeze"] = _read_freeze(Path(freeze_repo), manifest_bytes)
        payload["status"] = BOUND
    validate_hardware_subsets(payload, split_dir=directory, allow_draft=True)
    return payload


def validate_hardware_subsets(payload: dict, *, split_dir: str | Path | None = None,
                              allow_draft: bool = False, freeze_repo: str | Path | None = None) -> None:
    """Check real IDs, balance, provenance, overlap, counts and lifecycle state."""
    directory = Path(split_dir) if split_dir is not None else DEFAULT_SPLIT_DIR
    manifest, pool, manifest_bytes = _source(directory)
    try:
        if payload["schema"] != SCHEMA or payload["status"] not in (DRAFT, BOUND):
            raise HardwareSubsetError("Unsupported hardware-subset schema or status")
        if payload["selection_sha256"] != _selection_digest(payload):
            raise HardwareSubsetError("Hardware ID selection digest mismatch")
        source = payload["source"]
        if (any(source[key] != value for key, value in FIXED_SOURCE.items())
                or source["dataset_sha256"] != manifest["dataset_sha256"]
                or source["split_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
                or source["split_manifest_path"] != "data/splits/v1/manifest.json"
                or source["candidate_rows"] != len(pool)):
            raise HardwareSubsetError("Hardware source differs from the frozen P2 test source")
        sets = payload["subsets"]
        concentration = sets["concentration"]
        if len(concentration) != 3 or [group["name"] for group in concentration] != ["concentration_1", "concentration_2", "concentration_3"]:
            raise HardwareSubsetError("Expected three named concentration groups")
        named = [("main", sets["main"], 120), ("control", sets["control"], 40),
                 *[(group["name"], group, 30) for group in concentration],
                 ("drift_references", sets["drift_references"], 5)]
        universe = set(pool.id)
        for name, group, size in named:
            ids = group["ids"]
            if (not isinstance(ids, list) or not all(isinstance(x, str) for x in ids)
                    or len(ids) != size or len(set(ids)) != size or ids != sorted(ids)
                    or not set(ids) <= universe):
                raise HardwareSubsetError(f"Invalid IDs or cardinality in {name}")
            expected = _describe(ids, pool)
            if any(group[key] != value for key, value in expected.items()):
                raise HardwareSubsetError(f"Incorrect counts in {name}")
            if name != "drift_references":
                counts = list(group["counts_by_class"].values())
                if min(counts) == 0 or max(counts) - min(counts) > 1:
                    raise HardwareSubsetError(f"Class balance failed in {name}")
                if group["counts_by_detector"] != {"H1": size // 2, "L1": size // 2}:
                    raise HardwareSubsetError(f"Detector balance failed in {name}")
        main = set(sets["main"]["ids"])
        control = set(sets["control"]["ids"])
        parts = [set(group["ids"]) for group in concentration]
        if main & control:
            raise HardwareSubsetError("Main and control must be disjoint")
        if any(not group <= main for group in parts) or len(set.union(*parts)) != 90:
            raise HardwareSubsetError("Concentration groups must be pairwise-disjoint subsets of main")
        references = sets["drift_references"]
        if not set(references["ids"]) <= main:
            raise HardwareSubsetError("Drift references must be main-set IDs")
        ref_counts = references["counts_by_class"]
        if ref_counts[HELD_OUT] != 1 or sum(count > 0 for count in ref_counts.values()) != 5:
            raise HardwareSubsetError("Drift references need five distinct families including Fast_Scattering")
        if references["counts_by_detector"] != {"H1": 3, "L1": 2}:
            raise HardwareSubsetError("Drift references must cover both detectors (3 H1, 2 L1)")
        if payload["status"] == DRAFT:
            if payload["l05_freeze"] is not None:
                raise HardwareSubsetError("A draft cannot claim an L-05 binding")
            if not allow_draft:
                raise HardwareSubsetError("D-05 is pending L-05: no hw-freeze binding; draft IDs are for review only")
        else:
            freeze = payload["l05_freeze"]
            if (freeze["tag"] != FREEZE_TAG or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", freeze["commit"])
                    or set(freeze["files"]) != {"configs/hardware_frozen.yaml", "docs/decisions.md", "data/splits/v1/manifest.json"}
                    or freeze["files"]["data/splits/v1/manifest.json"]["sha256"] != source["split_manifest_sha256"]):
                raise HardwareSubsetError("Malformed L-05 freeze binding")
            for spec in freeze["files"].values():
                if (not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
                        or isinstance(spec["bytes"], bool) or not isinstance(spec["bytes"], int) or spec["bytes"] <= 0):
                    raise HardwareSubsetError("Malformed tagged-file hash or size")
            if freeze_repo is not None and freeze != _read_freeze(Path(freeze_repo), manifest_bytes):
                raise HardwareSubsetError("Stored binding differs from local hw-freeze tag or tagged files")
    except (KeyError, TypeError, AttributeError) as error:
        raise HardwareSubsetError(f"Malformed hardware-subset document: {error}") from error


def write_hardware_subsets(payload: dict, *, output: str | Path | None = None) -> None:
    """Only an unchanged draft -> tagged binding can update an existing file."""
    path = Path(output) if output is not None else DEFAULT_OUTPUT
    content = _json(payload)
    if path.exists():
        previous_bytes = path.read_bytes()
        if previous_bytes == content:
            return
        previous = json.loads(previous_bytes)
        if (previous.get("status") != DRAFT or payload["status"] != BOUND
                or previous.get("selection_sha256") != payload["selection_sha256"]
                or _selection_digest(previous) != payload["selection_sha256"]):
            raise HardwareSubsetError("Existing hardware ID selection differs or is already bound; choose a new output version")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".hardware-subsets-", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            target.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_hardware_subsets(path: str | Path | None = None, *, split_dir: str | Path | None = None,
                          allow_draft: bool = False, freeze_repo: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else DEFAULT_OUTPUT
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HardwareSubsetError(f"Cannot read hardware subsets: {error}") from error
    validate_hardware_subsets(payload, split_dir=split_dir, allow_draft=allow_draft, freeze_repo=freeze_repo)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--curated-csv", type=Path)
    sources.add_argument("--raw-dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bind-freeze-from", type=Path, help="Local Git repository with actual L-05 hw-freeze tag")
    args = parser.parse_args(argv)
    try:
        if args.curated_csv is not None:
            frame = pd.read_csv(args.curated_csv, dtype={name: str for name in ("id", "ifo", "label", "run")}, float_precision="round_trip")
        else:
            from .curate import load_curated
            frame = load_curated(raw_dir=args.raw_dir)
        payload = generate_hardware_subsets(frame, split_dir=args.split_dir, seed=args.seed, freeze_repo=args.bind_freeze_from)
        write_hardware_subsets(payload, output=args.output)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"D-05 status: {payload['status']}; main=120, control=40, concentration=3x30, drift=5")
    print(f"Selection SHA-256: {payload['selection_sha256']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
