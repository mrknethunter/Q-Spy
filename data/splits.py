"""D-04: immutable, deterministic P1/P2/P3 split bundles.

``load_split(protocol, seed, k=None, held_out=None)`` reads committed indices;
it never generates a new split or loads the raw dataset. Build a bundle with
``python -m data.splits``. See docs/splits.md for explicit protocol choices.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from numbers import Integral
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits" / "v1"
SCHEMA = "q-spy-splits-v1"
HELD_OUT = "Fast_Scattering"
K_VALUES = (5, 10, 20, 50)
DETECTORS = ("H1", "L1")
FEATURES = ("duration", "peak_frequency", "central_freq", "bandwidth", "amplitude", "snr", "q_value")
STRING_COLUMNS = ("id", "ifo", "label", "run")
FLOAT_COLUMNS = ("gps", "confidence", *FEATURES)
PARTITIONS = {"unused": 0, "train": 1, "val": 2, "test": 3}
FILES = ("catalog.npz", "P1.npz", "P2.npz", "P3.npz")


class SplitError(ValueError):
    """A dataset, protocol selector or frozen bundle is invalid."""


@dataclass(frozen=True)
class Split:
    """Stable ordered IDs; tuples prevent accidental mutation after loading."""

    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


@dataclass(frozen=True)
class SplitConfig:
    seeds: tuple[int, ...] = tuple(range(10))
    test_per_class: int = 100
    val_per_class: int = 20
    test_pool_seed: int = 2026
    p2_test_fraction: float = 0.2
    p2_val_fraction: float = 0.2
    p3_val_fraction: float = 0.2

    def __post_init__(self) -> None:
        if (not isinstance(self.seeds, tuple) or not self.seeds
                or len(set(self.seeds)) != len(self.seeds)):
            raise SplitError("seeds must be a nonempty tuple of distinct integers")
        for name, value in [("seed", x) for x in self.seeds] + [
            ("test_pool_seed", self.test_pool_seed),
            ("test_per_class", self.test_per_class), ("val_per_class", self.val_per_class),
        ]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise SplitError(f"{name} must be a nonnegative integer")
        if min(self.test_per_class, self.val_per_class) < 2:
            raise SplitError("P1 pools must contain at least two examples per class to retain both detectors")
        for name in ("p2_test_fraction", "p2_val_fraction", "p3_val_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1:
                raise SplitError(f"{name} must be finite and between 0 and 1")
        if self.p2_test_fraction + self.p2_val_fraction >= 1:
            raise SplitError("P2 validation and test fractions must leave training examples")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set((*STRING_COLUMNS, *FLOAT_COLUMNS)) - set(frame.columns)
    if missing:
        raise SplitError(f"Curated dataset is missing columns: {sorted(missing)}")
    out = frame.loc[:, [*STRING_COLUMNS, *FLOAT_COLUMNS]].copy()
    for name in STRING_COLUMNS:
        if out[name].isna().any():
            raise SplitError(f"Missing {name} values")
        out[name] = out[name].astype(str)
        if out[name].str.strip().ne(out[name]).any() or out[name].eq("").any():
            raise SplitError(f"Empty or whitespace-padded {name} values")
    if not out.id.is_unique or out.empty:
        raise SplitError("Curated IDs must be unique and the dataset nonempty")
    if not out.ifo.isin(DETECTORS).all() or not out.run.isin(("O1", "O2", "O3a", "O3b")).all():
        raise SplitError("Invalid detector or observing-run provenance")
    for name in FLOAT_COLUMNS:
        try:
            out[name] = pd.to_numeric(out[name], errors="raise").astype("float64")
        except (TypeError, ValueError) as error:
            raise SplitError(f"Invalid numeric column: {name}") from error
    if not np.isfinite(out[list(FLOAT_COLUMNS)].to_numpy()).all():
        raise SplitError("Curated numeric values must be finite")
    if not out.confidence.between(0, 1).all() or not (out[list(FEATURES)] > 0).all().all() or (out.gps < 0).any():
        raise SplitError("Invalid curated confidence, GPS or positive raw features")
    known = sorted(set(out.label) - {HELD_OUT})
    if not 6 <= len(known) <= 8 or set(known) & {"No_Glitch", "None_of_the_Above"}:
        raise SplitError("Expected 6–8 specific known glitch classes plus Fast_Scattering")
    held = out.loc[out.label == HELD_OUT]
    if held.empty or not held.run.isin(("O3a", "O3b")).all():
        raise SplitError("Fast_Scattering must be present and confined to O3a/O3b")
    if "is_held_out" in frame:
        actual = frame["is_held_out"]
        if actual.isna().any() or not actual.isin((True, False)).all() or not np.array_equal(actual.to_numpy(), frame.label.eq(HELD_OUT).to_numpy()):
            raise SplitError("is_held_out disagrees with the reserved Fast_Scattering label")
    for label in (*known, HELD_OUT):
        if set(out.loc[out.label == label, "ifo"]) != set(DETECTORS):
            raise SplitError(f"Class {label} must occur in both detectors")
    return out.sort_values("id", kind="stable").reset_index(drop=True)


def _fingerprint(frame: pd.DataFrame) -> str:
    """Bind ordered IDs, provenance, targets and every unscaled input value."""
    digest = hashlib.sha256(SCHEMA.encode("ascii"))
    for name in STRING_COLUMNS:
        payload = _json_bytes({name: frame[name].tolist()})
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    for name in FLOAT_COLUMNS:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(frame[name].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Fixed zip timestamps, key order, endian and compression; no pickle."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            array = arrays[name]
            if array.dtype.hasobject:
                raise SplitError("Object arrays are not allowed")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _rank(indices: np.ndarray, ids: np.ndarray, namespace: str) -> np.ndarray:
    prefix = namespace.encode("utf-8") + b"\0"
    ranked = sorted(indices.tolist(), key=lambda i: (hashlib.sha256(prefix + ids[i].encode("utf-8")).digest(), ids[i]))
    return np.array(ranked, dtype="<i8")


def _stratified_sample(indices: np.ndarray, count: int, ids: np.ndarray,
                       ifos: np.ndarray, namespace: str,
                       cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Proportional detector allocation with one reserved example per detector."""
    groups = {ifo: indices[ifos[indices] == ifo] for ifo in DETECTORS}
    if any(len(group) == 0 for group in groups.values()) or count < 2 or count > len(indices):
        raise SplitError(f"Insufficient detector support for {namespace}: requested {count}, available {len(indices)}")
    capacities = {ifo: len(group) - 1 for ifo, group in groups.items()}
    total = sum(capacities.values())
    allocation = {ifo: 1 for ifo in DETECTORS}
    if total:
        remainders = {}
        for ifo in DETECTORS:
            base, remainder = divmod((count - 2) * capacities[ifo], total)
            allocation[ifo] += base
            remainders[ifo] = remainder
        for ifo in sorted(DETECTORS, key=lambda x: (-remainders[x], x))[:count - sum(allocation.values())]:
            allocation[ifo] += 1
    sampled = []
    for ifo in DETECTORS:
        key = namespace + "/" + ifo
        if cache is None:
            ranked = _rank(groups[ifo], ids, key)
        else:
            if key not in cache:
                cache[key] = _rank(groups[ifo], ids, key)
            ranked = cache[key]
        sampled.append(ranked[:allocation[ifo]])
    return np.sort(np.concatenate(sampled))


def _metadata(stats: tuple, membership: np.ndarray, *, protocol: str, seed: int,
              k: int | None, held_out: str | None, unknown: list[str], key: str) -> dict:
    labels, stratum, run_codes = stats
    counts = {}
    for partition, code in PARTITIONS.items():
        mask = membership == code
        cells = np.bincount(stratum[mask], minlength=len(labels) * 2).reshape(-1, 2)
        counts[partition] = {
            "total": int(mask.sum()),
            "by_class_detector": {label: {ifo: int(cells[i, j]) for j, ifo in enumerate(DETECTORS)} for i, label in enumerate(labels)},
            "by_run": dict(zip(("O1", "O2", "O3a", "O3b"), map(int, np.bincount(run_codes[mask], minlength=4)))),
        }
    return dict(protocol=protocol, seed=int(seed), k=k, held_out=held_out,
                unknown_classes=unknown, array_key=key, counts=counts)


def _build_arrays(frame: pd.DataFrame, config: SplitConfig) -> tuple[dict, list[dict]]:
    ids = frame.id.to_numpy(dtype=str)
    labels = frame.label.to_numpy(dtype=str)
    ifos = frame.ifo.to_numpy(dtype=str)
    runs = frame.run.to_numpy(dtype=str)
    known = sorted(set(labels) - {HELD_OUT})
    indices = np.arange(len(frame), dtype="<i8")
    by_label = {label: indices[labels == label] for label in (*known, HELD_OUT)}
    all_labels = sorted(set(labels))
    stats = (all_labels,
             pd.Categorical(labels, categories=all_labels).codes * 2 + pd.Categorical(ifos, categories=DETECTORS).codes,
             pd.Categorical(runs, categories=("O1", "O2", "O3a", "O3b")).codes)
    rank_cache = {}
    files: dict[str, dict[str, np.ndarray]] = {protocol: {} for protocol in ("P1", "P2", "P3")}
    records = []

    # P1: reserve one fixed balanced test pool before any experimental seed.
    fixed_test = np.concatenate([
        _stratified_sample(by_label[label], config.test_per_class, ids, ifos,
                           f"P1/test/{config.test_pool_seed}/{label}", rank_cache) for label in known
    ])
    test_mask = np.zeros(len(frame), dtype=bool)
    test_mask[fixed_test] = True
    p1_remaining = {label: group[~test_mask[group]] for label, group in by_label.items() if label in known}
    for seed in sorted(config.seeds):
        validation = np.concatenate([
            _stratified_sample(p1_remaining[label], config.val_per_class, ids, ifos,
                               f"P1/validation/{seed}/{label}", rank_cache) for label in known
        ])
        reserved = test_mask.copy()
        reserved[validation] = True
        for k in K_VALUES:
            training = np.concatenate([
                _stratified_sample(by_label[label][~reserved[by_label[label]]], k, ids, ifos,
                                   f"P1/training/{seed}/{label}", rank_cache) for label in known
            ])
            member = np.zeros(len(frame), dtype=np.uint8)
            member[fixed_test], member[validation], member[training] = 3, 2, 1
            key = f"seed{seed:02d}_k{k:02d}"
            files["P1"][key] = member
            records.append(_metadata(stats, member, protocol="P1", seed=seed, k=k,
                                     held_out=None, unknown=[], key=key))

    # P2: all held-out rows are test; known rows supply train, val and normals in test.
    p2_base = np.zeros(len(frame), dtype=np.uint8)
    p2_candidates = {}
    for label in known:
        for ifo in DETECTORS:
            group = by_label[label][ifos[by_label[label]] == ifo]
            n_test = max(1, math.floor(len(group) * config.p2_test_fraction))
            n_val = max(1, math.floor(len(group) * config.p2_val_fraction))
            if n_test + n_val >= len(group):
                raise SplitError(f"P2 needs train/val/test support in {label}/{ifo}")
            ranked = _rank(group, ids, f"P2/test/{config.test_pool_seed}/{label}/{ifo}")
            p2_base[ranked[:n_test]] = 3
            p2_candidates[(label, ifo)] = (ranked[n_test:], n_val)
    for seed in sorted(config.seeds):
        normal = p2_base.copy()
        for (label, ifo), (candidates, count) in p2_candidates.items():
            ranked = _rank(candidates, ids, f"P2/validation/{seed}/{label}/{ifo}")
            normal[ranked[:count]], normal[ranked[count:]] = 2, 1
        for fold, held_out in enumerate((HELD_OUT, *known)):
            unknown = [HELD_OUT] if held_out == HELD_OUT else [HELD_OUT, held_out]
            member = normal.copy()
            member[np.isin(labels, unknown)] = 3
            key = f"fold{fold:02d}_seed{seed:02d}"
            files["P2"][key] = member
            records.append(_metadata(stats, member, protocol="P2", seed=seed, k=None,
                                     held_out=held_out, unknown=unknown, key=key))

    # P3: O3a known families only in train/validation, all O3b rows in test.
    for seed in sorted(config.seeds):
        member = np.zeros(len(frame), dtype=np.uint8)
        member[runs == "O3b"] = 3
        for label in known:
            for ifo in DETECTORS:
                group = indices[(labels == label) & (ifos == ifo) & (runs == "O3a")]
                n_val = max(1, math.floor(len(group) * config.p3_val_fraction))
                if n_val >= len(group):
                    raise SplitError(f"P3 needs O3a train/validation support in {label}/{ifo}")
                ranked = _rank(group, ids, f"P3/validation/{seed}/{label}/{ifo}")
                member[ranked[:n_val]], member[ranked[n_val:]] = 2, 1
        key = f"seed{seed:02d}"
        files["P3"][key] = member
        records.append(_metadata(stats, member, protocol="P3", seed=seed, k=None,
                                 held_out=None, unknown=[HELD_OUT], key=key))
    return files, records


def generate_splits(frame: pd.DataFrame, *, output_dir: str | Path | None = None,
                    config: SplitConfig | None = None) -> dict:
    """Build all protocols; an existing version is reused only byte-for-byte.

    Changing data or settings requires another output version directory. No
    generation timestamp or absolute source path enters the frozen assets.
    """
    config = config if config is not None else SplitConfig()
    data = _canonical_frame(frame)
    output = Path(output_dir) if output_dir is not None else DEFAULT_SPLIT_DIR
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays, records = _build_arrays(data, config)
    settings = asdict(config)
    settings.update(seeds=[int(x) for x in sorted(config.seeds)],
                    test_per_class=int(config.test_per_class), val_per_class=int(config.val_per_class),
                    test_pool_seed=int(config.test_pool_seed))
    manifest = dict(schema=SCHEMA, dataset_sha256=_fingerprint(data), rows=len(data),
                    feature_columns=list(FEATURES), known_classes=sorted(set(data.label) - {HELD_OUT}),
                    reserved_held_out=HELD_OUT, k_values=list(K_VALUES), config=settings,
                    partition_codes=dict(PARTITIONS), files={}, splits=records)
    with tempfile.TemporaryDirectory(prefix=".splits-build-", dir=output.parent) as directory:
        stage = Path(directory)
        catalog = {name: data[name].to_numpy(dtype=str) for name in STRING_COLUMNS}
        _write_npz(stage / "catalog.npz", catalog)
        for protocol, contents in arrays.items():
            _write_npz(stage / f"{protocol}.npz", contents)
        for name in FILES:
            content = (stage / name).read_bytes()
            manifest["files"][name] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        if output.exists():
            expected = {*FILES, "manifest.json"}
            if (not output.is_dir() or {p.name for p in output.iterdir()} != expected
                    or any((output / name).read_bytes() != (stage / name).read_bytes() for name in expected)):
                raise SplitError("Split version already exists with different data/settings/content; choose a new --output-dir (for example data/splits/v2)")
        else:
            os.replace(stage, output)
    return manifest


def _read_manifest(directory: Path) -> dict:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["schema"] != SCHEMA or set(manifest["files"]) != set(FILES)
                or manifest["partition_codes"] != PARTITIONS):
            raise SplitError("Unsupported or malformed split manifest")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SplitError(f"Cannot read split bundle in {directory}: {error}") from error


def _asset(directory: Path, manifest: dict, name: str) -> io.BytesIO:
    try:
        content = (directory / name).read_bytes()
        spec = manifest["files"][name]
        if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise SplitError(f"Split asset failed integrity check: {name}")
    except (OSError, KeyError, TypeError) as error:
        raise SplitError(f"Cannot read split asset {name}: {error}") from error
    return io.BytesIO(content)


def load_split(protocol: str, seed: int, k: int | None = None,
               held_out: str | None = None, *, split_dir: str | Path | None = None) -> Split:
    """Read one committed split. No sampling, raw data or ML fit occurs here."""
    if not isinstance(protocol, str) or protocol.upper() not in ("P1", "P2", "P3"):
        raise SplitError("protocol must be P1, P2 or P3")
    protocol = protocol.upper()
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise SplitError("seed must be an integer from the committed seed list")
    if protocol == "P1":
        if isinstance(k, bool) or not isinstance(k, Integral) or k not in K_VALUES or held_out is not None:
            raise SplitError("P1 requires k in {5, 10, 20, 50} and held_out=None")
    elif k is not None or (protocol == "P3" and held_out is not None):
        raise SplitError("P2/P3 require k=None; P3 also requires held_out=None")
    if protocol == "P2" and held_out is None:
        held_out = HELD_OUT
    directory = Path(split_dir) if split_dir is not None else DEFAULT_SPLIT_DIR
    manifest = _read_manifest(directory)
    found = [entry for entry in manifest["splits"] if
             (entry["protocol"], entry["seed"], entry["k"], entry["held_out"]) == (protocol, seed, k, held_out)]
    if len(found) != 1:
        raise SplitError("Requested split is not in this frozen version; check seed, k and held_out")
    entry = found[0]
    try:
        with np.load(_asset(directory, manifest, "catalog.npz"), allow_pickle=False) as archive:
            ids = archive["id"]
        with np.load(_asset(directory, manifest, protocol + ".npz"), allow_pickle=False) as archive:
            member = archive[entry["array_key"]]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SplitError(f"Cannot decode split assets: {error}") from error
    if (ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != manifest["rows"]
            or len(set(ids)) != len(ids) or member.dtype != np.uint8 or member.shape != ids.shape
            or not np.isin(member, (0, 1, 2, 3)).all()):
        raise SplitError("Invalid ID catalog or partition membership array")
    parts = [tuple(ids[member == code].tolist()) for code in (1, 2, 3)]
    if any(len(part) != entry["counts"][name]["total"] for part, name in zip(parts, ("train", "val", "test"))):
        raise SplitError("Partition counts disagree with the split manifest")
    return Split(*parts)


def verify_dataset(frame: pd.DataFrame, *, split_dir: str | Path | None = None) -> None:
    """Reject a changed curated table even when its set of IDs is unchanged."""
    directory = Path(split_dir) if split_dir is not None else DEFAULT_SPLIT_DIR
    manifest = _read_manifest(directory)
    canonical = _canonical_frame(frame)
    if len(canonical) != manifest["rows"] or _fingerprint(canonical) != manifest["dataset_sha256"]:
        raise SplitError("Curated dataset differs from the dataset bound to this split version")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--curated-csv", type=Path, help="Existing D-02 CSV; read with round-trip float parsing")
    source.add_argument("--raw-dir", type=Path, help="Pass a local raw directory to D-02 load_curated")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--test-pool-seed", type=int, default=2026)
    parser.add_argument("--p2-test-fraction", type=float, default=.2)
    parser.add_argument("--p2-val-fraction", type=float, default=.2)
    parser.add_argument("--p3-val-fraction", type=float, default=.2)
    args = parser.parse_args(argv)
    try:
        config = SplitConfig(test_per_class=args.test_per_class, val_per_class=args.val_per_class,
                             test_pool_seed=args.test_pool_seed, p2_test_fraction=args.p2_test_fraction,
                             p2_val_fraction=args.p2_val_fraction, p3_val_fraction=args.p3_val_fraction)
        if args.curated_csv is not None:
            frame = pd.read_csv(args.curated_csv, dtype={name: str for name in STRING_COLUMNS}, float_precision="round_trip")
        else:
            from .curate import load_curated
            frame = load_curated(raw_dir=args.raw_dir)
        manifest = generate_splits(frame, output_dir=args.output_dir, config=config)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    counts = {protocol: sum(entry["protocol"] == protocol for entry in manifest["splits"]) for protocol in ("P1", "P2", "P3")}
    print(f"Frozen {len(manifest['splits'])} splits for {manifest['rows']:,} rows: {counts}")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
    print(f"Split directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
