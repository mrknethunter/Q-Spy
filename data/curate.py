"""D-02: reproducible curation of the checksum-verified D-01 release.

Run ``python -m data.curate --help`` from the project root. Importing this
module performs no I/O. ``load_curated()`` returns a pandas DataFrame.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from .download import DEFAULT_OUTPUT, load_manifest, verified
from .profile import FEATURES

ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS = tuple(FEATURES)
KNOWN_CLASSES = (
    "Blip", "Extremely_Loud", "Koi_Fish", "Low_Frequency_Burst",
    "Low_Frequency_Lines", "Scattered_Light", "Tomte",
)
HELD_OUT = "Fast_Scattering"
DETECTORS = ("H1", "L1")
RUNS = ("O1", "O2", "O3a", "O3b")
RAW_COLUMNS = (
    "gravityspy_id", "ifo", "peak_time", "peak_time_ns",
    "ml_label", "ml_confidence", *FEATURE_COLUMNS,
)
OUTPUT_COLUMNS = (
    "id", "gps", "ifo", "label", "confidence", *FEATURE_COLUMNS,
    "run", "is_held_out",
)
CARD_BEGIN = "<!-- BEGIN D-02 CURATION -->"
CARD_END = "<!-- END D-02 CURATION -->"


class CurationError(ValueError):
    """The input cannot produce the requested complete curated dataset."""


@dataclass(frozen=True)
class CurateConfig:
    confidence_threshold: float = 0.9
    known_classes: tuple[str, ...] = KNOWN_CLASSES
    coincidence_window_s: float = 0.1
    max_per_class: int | None = None
    seed: int = 42
    drop_extreme_q: bool = True

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("confidence_threshold", 0.0, 1.0),
            ("coincidence_window_s", 0.0, 1e6),
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not lower <= value <= upper):
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        names = self.known_classes
        if (not isinstance(names, tuple) or not all(isinstance(x, str) and x.strip() == x and x for x in names)
                or not 6 <= len(names) <= 8 or len(set(names)) != len(names)):
            raise ValueError("known_classes must be a tuple of 6–8 distinct nonempty labels")
        if set(names) & {HELD_OUT, "No_Glitch", "None_of_the_Above"}:
            raise ValueError("Known classes must exclude held-out and non-morphology labels")
        if (isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
                or not 0 <= self.seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.max_per_class is not None and (
            isinstance(self.max_per_class, bool)
            or not isinstance(self.max_per_class, Integral) or self.max_per_class < 2
        ):
            raise ValueError("max_per_class must be None or an integer >= 2")
        if not isinstance(self.drop_extreme_q, bool):
            raise ValueError("drop_extreme_q must be boolean")


def _detector_counts(frame: pd.DataFrame, labels: tuple[str, ...]) -> dict:
    counts = frame.groupby(["label", "ifo"]).size()
    return {label: {ifo: int(counts.get((label, ifo), 0)) for ifo in DETECTORS}
            for label in labels}


def _allocate(counts: dict[str, int], target: int) -> dict[str, int]:
    """Largest remainders on remaining capacity, retaining every detector."""
    active = sorted(ifo for ifo, count in counts.items() if count)
    if target < len(active):
        raise CurationError("Class cap is too small to preserve detector strata")
    allocation = {ifo: 1 for ifo in active}
    remaining = target - len(active)
    capacity = {ifo: counts[ifo] - 1 for ifo in active}
    total = sum(capacity.values())
    if not total:
        return allocation
    remainders = {}
    for ifo in active:
        base, remainder = divmod(remaining * capacity[ifo], total)
        allocation[ifo] += base
        remainders[ifo] = remainder
    left = target - sum(allocation.values())
    for ifo in sorted(active, key=lambda x: (-remainders[x], x))[:left]:
        allocation[ifo] += 1
    return allocation


def _stratified_cap(frame: pd.DataFrame, config: CurateConfig) -> pd.DataFrame:
    if config.max_per_class is None:
        return frame
    selected = []
    for _, group in frame.groupby("label", sort=True):
        if len(group) <= config.max_per_class:
            selected.append(group)
            continue
        counts = {ifo: int((group["ifo"] == ifo).sum()) for ifo in DETECTORS}
        allocation = _allocate(counts, config.max_per_class)
        for ifo, count in allocation.items():
            stratum = group.loc[group["ifo"] == ifo].sort_values("id")
            selected.append(stratum.sample(n=count, random_state=int(config.seed)))
    return pd.concat(selected, ignore_index=True)


def curate_frame(raw: pd.DataFrame, config: CurateConfig | None = None) -> pd.DataFrame:
    """Curate an in-memory raw table; ``run`` must identify each source file.

    Filtering never changes the caller's frame. Bad measurements are counted
    and removed; missing columns or invalid detector/run provenance fail loudly.
    Time deduplication is detector-local and label-independent. Connected
    clusters use consecutive gaps <= the configured window (including chains).
    Representatives maximize confidence, then SNR, with deterministic ties.
    """
    config = config if config is not None else CurateConfig()
    required = set(RAW_COLUMNS) | {"run"}
    missing = required - set(raw.columns)
    if missing:
        raise CurationError(f"Missing columns: {sorted(missing)}")
    frame = raw.loc[:, [*RAW_COLUMNS, "run"]].copy()
    for column in ("gravityspy_id", "ifo", "ml_label", "run"):
        frame[column] = frame[column].astype("string").str.strip()
    if not frame["ifo"].isin(DETECTORS).all() or not frame["run"].isin(RUNS).all():
        raise CurationError("Invalid detector or observing-run provenance")
    settings = asdict(config)
    settings.update(confidence_threshold=float(config.confidence_threshold),
                    coincidence_window_s=float(config.coincidence_window_s), seed=int(config.seed),
                    max_per_class=None if config.max_per_class is None else int(config.max_per_class))
    report = {"input_rows": len(frame), "config": settings, "removed": {}}

    def keep(mask: pd.Series, reason: str) -> None:
        nonlocal frame
        mask = mask.fillna(False)
        report["removed"][reason] = int((~mask).sum())
        frame = frame.loc[mask].copy()

    frame["confidence"] = pd.to_numeric(frame["ml_confidence"], errors="coerce").astype("float64")
    keep(frame["confidence"].between(config.confidence_threshold, 1, inclusive="both"),
         "confidence_below_threshold_or_invalid")
    keep(~(frame["ml_label"].eq(HELD_OUT).fillna(False) & ~frame["run"].isin(("O3a", "O3b"))),
         "held_out_outside_o3")
    for column in (*FEATURE_COLUMNS, "peak_time", "peak_time_ns"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    valid = pd.Series(np.isfinite(frame[list(FEATURE_COLUMNS)]).all(axis=1), index=frame.index)
    valid &= (frame[list(FEATURE_COLUMNS)] > 0).all(axis=1)
    for name, upper in (("peak_time", 9_000_000_000), ("peak_time_ns", 999_999_999)):
        numeric = frame[name]
        valid &= np.isfinite(numeric) & numeric.between(0, upper) & (numeric % 1 == 0)
    valid &= frame["gravityspy_id"].notna() & frame["gravityspy_id"].ne("")
    valid &= frame["ml_label"].notna() & frame["ml_label"].ne("")
    keep(valid.fillna(False), "invalid_id_label_time_or_feature")
    keep(frame["q_value"].ne(1e20) if config.drop_extreme_q else pd.Series(True, index=frame.index),
         "q_value_equals_1e20")
    if frame.empty:
        raise CurationError("No usable rows remain after quality filtering")

    frame["_peak_ns"] = (frame["peak_time"].astype("int64") * 1_000_000_000
                          + frame["peak_time_ns"].astype("int64"))
    frame["gps"] = frame["peak_time"] + frame["peak_time_ns"] / 1_000_000_000
    frame["label"] = frame["ml_label"]
    frame["id"] = frame["ifo"] + ":" + frame["gravityspy_id"]
    # Including every feature resolves ties without depending on source row order.
    ranking = ["confidence", "snr", "id", "_peak_ns", "run", "label",
               *(name for name in FEATURE_COLUMNS if name != "snr")]
    ascending = [False, False, *([True] * (len(ranking) - 2))]
    frame = frame.sort_values(ranking, ascending=ascending, kind="stable")
    before = len(frame)
    frame = frame.drop_duplicates("id", keep="first").copy()
    report["removed"]["repeated_id"] = before - len(frame)
    temporal = frame.sort_values(["ifo", "_peak_ns", "id"], kind="stable").copy()
    times = temporal["_peak_ns"].to_numpy(dtype=np.int64)
    ifos = temporal["ifo"].to_numpy()
    window_ns = int(round(config.coincidence_window_s * 1_000_000_000))
    starts = np.r_[True, (ifos[1:] != ifos[:-1]) | (np.diff(times) > window_ns)]
    temporal["_cluster"] = np.cumsum(starts)
    before = len(temporal)
    frame = temporal.sort_values(ranking, ascending=ascending, kind="stable").drop_duplicates("_cluster").copy()
    report["removed"]["coincident_trigger"] = before - len(frame)
    report["eligible_counts_after_dedup"] = {
        str(label): int(count) for label, count in frame["label"].value_counts().sort_index().items()
    }

    labels = (*config.known_classes, HELD_OUT)
    keep(frame["label"].isin(labels), "unselected_class")
    before_counts = _detector_counts(frame, labels)
    absent = [f"{label}/{ifo}" for label in labels for ifo in DETECTORS
              if before_counts[label][ifo] == 0]
    if absent:
        raise CurationError("Selected classes must occur in both detectors; absent: " + ", ".join(absent))
    before = len(frame)
    frame = _stratified_cap(frame, config)
    report["removed"]["stratified_cap"] = before - len(frame)
    frame["is_held_out"] = frame["label"].eq(HELD_OUT).astype(bool)
    result = frame.loc[:, OUTPUT_COLUMNS].sort_values(["ifo", "gps", "id"], kind="stable").reset_index(drop=True)
    report.update(
        output_rows=len(result), known_classes=list(config.known_classes), held_out=HELD_OUT,
        counts_before_cap=before_counts, counts_by_detector=_detector_counts(result, labels),
        counts_by_run={run: _detector_counts(result.loc[result["run"] == run], labels) for run in RUNS},
        feature_columns=list(FEATURE_COLUMNS), feature_selection_status="provisional_in_D01",
        coincidence_window_ns=window_ns,
    )
    result.attrs["curation"] = report
    return result


def load_curated(*, raw_dir: str | Path | None = None,
                 config: CurateConfig | None = None) -> pd.DataFrame:
    """Return curated rows from all eight local, verified D-01 source files.

    The zero-argument call uses ``data/raw`` and the documented defaults.
    There is no automatic download, scaling, train/test split or disk write.
    """
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_OUTPUT
    manifest = load_manifest()
    specs = manifest["files"]
    expected = {f"{ifo}_{run}.csv" for ifo in DETECTORS for run in RUNS}
    if len(specs) != 8 or {spec["name"] for spec in specs} != expected:
        raise CurationError("Manifest must describe all eight detector/run files exactly once")
    for spec in specs:
        if not verified(directory / spec["name"], spec):
            raise CurationError(f"{spec['name']}: missing/corrupt; run python data/download.py")
    frames = []
    for spec in sorted(specs, key=lambda item: item["name"]):
        match = re.fullmatch(r"(H1|L1)_(O1|O2|O3a|O3b)\.csv", spec["name"])
        assert match is not None  # The complete manifest-name check above guarantees this.
        ifo, run = match.groups()
        try:
            frame = pd.read_csv(
                directory / spec["name"], usecols=list(RAW_COLUMNS),
                dtype={name: "string" for name in ("gravityspy_id", "ifo", "ml_label", "peak_time", "peak_time_ns")},
                float_precision="round_trip", encoding="utf-8-sig",
            )
        except (ValueError, OSError) as error:
            raise CurationError(f"{spec['name']}: {error}") from error
        if not frame["ifo"].eq(ifo).fillna(False).all():
            raise CurationError(f"{spec['name']}: detector column disagrees with source filename")
        frame["run"] = run
        frames.append(frame)
    result = curate_frame(pd.concat(frames, ignore_index=True), config)
    result.attrs["curation"]["source"] = {
        "doi": manifest["doi"], "version": manifest["version"],
        "files": [{key: spec[key] for key in ("name", "size_bytes", "md5")} for spec in specs],
    }
    return result


def render_curation_card(report: dict) -> str:
    config = report["config"]
    lines = [CARD_BEGIN, "## D-02 curated dataset", "",
             f"Input rows: **{report['input_rows']:,}**. Curated rows: **{report['output_rows']:,}**.", "",
             ("These counts describe the checksum-verified D-01 release." if "source" in report
              else "These counts describe an in-memory input; upstream source verification is not asserted."), "",
             "### Selection and interpretation", "",
             f"Keep confidence >= {config['confidence_threshold']} and <= 1 (inclusive). Labels are ML predictions, not ground truth.",
             "The default seven known labels are Blip, Extremely_Loud, Koi_Fish, Low_Frequency_Burst, Low_Frequency_Lines, Scattered_Light and Tomte.",
             "They are seven of the most frequent named morphology labels in D-01, with representation in both detectors across the observing runs.",
             "No_Glitch and None_of_the_Above are not specific glitch morphologies. Blip_Low_Frequency is O3-only; it is not added as another known label.",
             "1080Lines is concentrated in H1/O2 (31,684 of 33,624 raw rows); it is omitted to reduce detector/run confounding. This is a documented selection choice, not a claim that the chosen morphologies are statistically independent.",
             "Fast_Scattering is retained separately as held-out, only from O3a/O3b. Its rows remain in the returned table with is_held_out=True; downstream novelty training must exclude them.",
             f"This run's known-class selection: {', '.join(report['known_classes'])}.", "",
             "### Quality and duplicate policy", "",
             "Discard non-finite/nonpositive feature values, missing labels/IDs and invalid integer peak timestamps.",
             f"Drop exactly q_value=1e20: {config['drop_extreme_q']}. D-01 flagged this extreme value; treating it as unusable is an explicit conservative choice, not an established upstream sentinel definition. --keep-extreme-q reverses this choice.",
             "First collapse repeated published IDs within a detector. Then cluster triggers independently for H1 and L1, ignoring labels and run boundaries.",
             f"Consecutive exact peak timestamps within {config['coincidence_window_s']} s belong to one connected cluster, including transitive chains. The 0.1 s default is a configurable engineering choice; 0 selects exact-time duplicates only.",
             "Keep the greatest confidence, then greatest SNR; resolve ties by stable ID, exact time, run, label and feature values. Deduplication precedes class selection.",
             "Coincident H1/L1 observations are distinct measurements and are not collapsed across detectors. Split construction must consider cross-detector coincidences if independence is required.", "",
             "### Detector strata", "",
             "Curation preserves detector strata without forcing equal class sizes. Every selected label, including held-out, must have at least one surviving H1 and L1 row; otherwise loading fails.",
             f"Optional per-class cap: {config['max_per_class']}; seed: {config['seed']}. Caps sample without replacement within (label, ifo), approximately preserving detector proportions and retaining both detectors. No training/validation/test split is made here.", "",
             "### Class counts after all curation and optional sampling", "",
             "| Label | Role | H1 | L1 | Total |", "| --- | --- | --- | --- | --- |"]
    for label, counts in report["counts_by_detector"].items():
        role = "held-out" if label == HELD_OUT else "known"
        lines.append(f"| {label} | {role} | {counts['H1']} | {counts['L1']} | {sum(counts.values())} |")
    lines += ["", "### Counts by observing run and detector", "",
              "| Label | H1 O1 | L1 O1 | H1 O2 | L1 O2 | H1 O3a | L1 O3a | H1 O3b | L1 O3b |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label in report["counts_by_detector"]:
        values = [str(report["counts_by_run"][run][label][ifo]) for run in RUNS for ifo in DETECTORS]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "### Sequential exclusion counts", "", "| Reason | Rows |", "| --- | --- |"]
    lines += [f"| {reason} | {count} |" for reason, count in report["removed"].items()]
    lines += ["", "Counts are sequential, so each removed row is counted once. Raw counts in the D-01 section remain unfiltered.", "",
              "### Interface and reproducibility", "",
              "load_curated() returns id (string, detector-prefixed published ID), gps (float64 GPS seconds), ifo, label, confidence (float64), the seven raw float64 features in D-01 order, plus run and is_held_out (bool). GPS is reconstructed from integer peak seconds/nanoseconds; clustering uses exact int64 nanoseconds, not rounded GPS floats.",
              "The seven-feature selection remains provisional as documented in D-01. No feature scaling or fitted preprocessing is performed, and raw data are never modified.",
              "Reproduce: python -m data.curate. Customize with --confidence, --known-classes, --coincidence-window, --max-per-class, --seed and --keep-extreme-q. Source names, MD5s, sizes, configuration and detector/run counts are retained in docs/curation_summary.json.",
              "The L-02 implementation and its actual shared tests were absent from the supplied branch. Local contract tests cover the interface supplied in the issue; compatibility with the merged L-02 suite still needs verification.", CARD_END]
    return "\n".join(lines) + "\n"


def write_outputs(frame: pd.DataFrame, *, output_dir: Path, report_path: Path, card_path: Path) -> None:
    report = frame.attrs["curation"]
    block = render_curation_card(report)
    old = card_path.read_text(encoding="utf-8") if card_path.exists() else "# Q-Spy data card\n"
    if CARD_BEGIN in old or CARD_END in old:
        if old.count(CARD_BEGIN) != 1 or old.count(CARD_END) != 1 or old.index(CARD_BEGIN) > old.index(CARD_END):
            raise CurationError("Malformed D-02 data-card markers; no outputs written")
        start = old.index(CARD_BEGIN)
        end = old.index(CARD_END) + len(CARD_END)
        card = old[:start] + block.rstrip() + old[end:]
    else:
        card = old.rstrip() + "\n\n" + block
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "curated.csv", index=False, float_format="%.17g")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    card_path.write_text(card, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--known-classes", nargs="+", default=list(KNOWN_CLASSES))
    parser.add_argument("--coincidence-window", type=float, default=0.1, help="Detector-local time window in seconds")
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-extreme-q", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "curated")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "curation_summary.json")
    parser.add_argument("--data-card", type=Path, default=ROOT / "docs" / "data_card.md")
    args = parser.parse_args(argv)
    try:
        config = CurateConfig(
            confidence_threshold=args.confidence, known_classes=tuple(args.known_classes),
            coincidence_window_s=args.coincidence_window, max_per_class=args.max_per_class,
            seed=args.seed, drop_extreme_q=not args.keep_extreme_q,
        )
        frame = load_curated(raw_dir=args.raw_dir, config=config)
        write_outputs(frame, output_dir=args.output_dir, report_path=args.report, card_path=args.data_card)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Curated {len(frame):,} rows; known classes={len(config.known_classes)}, held-out={HELD_OUT}")
    print(pd.DataFrame(frame.attrs["curation"]["counts_by_detector"]).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
