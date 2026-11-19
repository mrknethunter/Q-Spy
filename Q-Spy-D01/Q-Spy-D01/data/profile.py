"""Rebuild the D-01 data card from all eight checksum-verified raw CSV files.

Usage: python data/profile.py
Only the Python 3.10+ standard library is required. No rows are filtered.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

try:
    from .download import DEFAULT_OUTPUT, load_manifest, verified
except ImportError:
    from download import DEFAULT_OUTPUT, load_manifest, verified

ROOT = Path(__file__).resolve().parent.parent
# Provisional: neither issue screenshot nor the supplied repository identifies
# the agreed seven features. Confirm this list against the project proposal.
FEATURES = {
    "duration": ("s", "Duration reported by Omicron"),
    "peak_frequency": ("Hz", "Frequency of the peak Omicron tile"),
    "central_freq": ("Hz", "Central frequency reported by Omicron"),
    "bandwidth": ("Hz", "Frequency extent reported by Omicron"),
    "amplitude": ("pipeline units (see note)", "Omicron amplitude, retained without conversion"),
    "snr": ("dimensionless", "Omicron signal-to-noise ratio"),
    "q_value": ("dimensionless", "Omicron quality factor Q"),
}
CLASS_COLUMNS = (
    "1400Ripples", "1080Lines", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation",
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch",
    "None_of_the_Above", "Paired_Doves", "Power_Line", "Repeating_Blips",
    "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode", "Wandering_Line", "Whistle",
)
CONF_EDGES = (0.5, 0.8, 0.9, 0.95, 0.99)
CONF_LABELS = ("[0,0.5)", "[0.5,0.8)", "[0.8,0.9)", "[0.9,0.95)", "[0.95,0.99)", "[0.99,1]")


@dataclass
class NumericStats:
    finite: int = 0
    missing: int = 0
    invalid: int = 0
    nonfinite: int = 0
    nonpositive: int = 0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, raw: str | None) -> float | None:
        if raw is None or not raw.strip():
            self.missing += 1
            return None
        try:
            value = float(raw)
        except ValueError:
            self.invalid += 1
            return None
        if not math.isfinite(value):
            self.nonfinite += 1
            return None
        self.finite += 1
        self.nonpositive += value <= 0
        self.minimum = value if self.minimum is None else min(value, self.minimum)
        self.maximum = value if self.maximum is None else max(value, self.maximum)
        return value


def quantile(values: list[float], probability: float) -> float | None:
    """Linear interpolation at (n-1)*p; values must already be sorted."""
    if not values:
        return None
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def scan_csv(path: Path, expected_ifo: str) -> dict:
    numeric = {name: NumericStats() for name in FEATURES}
    confidence = NumericStats()
    values = []
    histogram = [0] * len(CONF_LABELS)
    classes: Counter = Counter()
    channels: Counter = Counter()
    quality_flags = {"q_value_equals_1e20": 0, "confidence_zero": 0, "snr_equals_7_5": 0}
    rows = out_of_range = 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        required = set(FEATURES) | {"ifo", "ml_label", "ml_confidence"}
        if not required.issubset(columns):
            raise ValueError(f"{path.name}: missing columns {sorted(required - set(columns))}")
        for row in reader:
            rows += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}:{reader.line_num}: malformed CSV row")
            if row["ifo"] != expected_ifo:
                raise ValueError(f"{path.name}:{reader.line_num}: unexpected ifo {row['ifo']!r}")
            classes[row["ml_label"] or "<missing>"] += 1
            channels[row.get("channel", "")] += 1
            for name, stats in numeric.items():
                value = stats.add(row[name])
                if name == "q_value" and value == 1e20:
                    quality_flags["q_value_equals_1e20"] += 1
                if name == "snr" and value == 7.5:
                    quality_flags["snr_equals_7_5"] += 1
            value = confidence.add(row["ml_confidence"])
            if value == 0:
                quality_flags["confidence_zero"] += 1
            if value is not None:
                values.append(value)
                if 0 <= value <= 1:
                    histogram[bisect_right(CONF_EDGES, value)] += 1
                else:
                    out_of_range += 1
    values.sort()
    return {
        "rows": rows,
        "columns": columns,
        "classes": dict(sorted(classes.items())),
        "channels": dict(sorted(channels.items())),
        "quality_flags": quality_flags,
        "features": {name: vars(stats) for name, stats in numeric.items()},
        "confidence": {
            **vars(confidence),
            "mean": math.fsum(values) / len(values) if values else None,
            "p10": quantile(values, 0.1),
            "p50": quantile(values, 0.5),
            "p90": quantile(values, 0.9),
            "p99": quantile(values, 0.99),
            "out_of_range": out_of_range,
            "histogram": dict(zip(CONF_LABELS, histogram)),
        },
    }


def build_summary(raw_dir: Path) -> dict:
    manifest = load_manifest()
    # Fail before producing a plausible-looking partial data card.
    for spec in manifest["files"]:
        if not verified(raw_dir / spec["name"], spec):
            raise ValueError(f"{spec['name']}: missing/corrupt; run python data/download.py")
    files = {}
    for spec in manifest["files"]:
        print(f"PROFILE {spec['name']}", flush=True)
        files[spec["name"]] = {
            **spec,
            **scan_csv(raw_dir / spec["name"], spec["name"].split("_")[0]),
        }
    return {
        "record_id": manifest["record_id"],
        "doi": manifest["doi"],
        "license": manifest["license"],
        "feature_selection_status": "provisional; confirm against project proposal",
        "features": list(FEATURES),
        "rows": sum(f["rows"] for f in files.values()),
        "files": files,
    }


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.12g}"


def table(headers: list[str], rows: list[list]) -> str:
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def render_card(summary: dict) -> str:
    files = summary["files"]
    names = list(files)
    sections = [
        "# Gravity Spy raw data card (D-01)",
        "## Scope and provenance\n\n"
        "Pinned release: [Zenodo 5649212, v1.0.0](https://zenodo.org/records/5649212), "
        "DOI [10.5281/zenodo.5649212](https://doi.org/10.5281/zenodo.5649212). "
        "Published in 2021; companion paper: Glanzer et al. (2023). "
        "These are tabular machine-learning classifications and Omicron metadata, "
        "not detector strain time series, volunteer votes, or downloaded spectrogram images. "
        "H1 = LIGO Hanford; L1 = LIGO Livingston. O3 is split into O3a and O3b.\n\n"
        f"This card profiles **{summary['rows']:,} unfiltered rows** in all eight CSVs. "
        "Counts include low-confidence events, No_Glitch and None_of_the_Above wherever present. "
        "No deduplication, label remapping, normalization or class selection is applied (D-02 handles curation).",
        "## Reproduce\n\nPython 3.10+; no third-party packages. From the repository root:\n\n"
        "```sh\npython data/download.py\n```\n\n"
        "This one command downloads all eight original CSVs into `data/raw/` (742,195,211 bytes total), "
        "checks size and the publisher's pinned MD5 checksums, and skips valid local files on subsequent runs. "
        "A damaged file is replaced only after a verified download; failed transfers exit nonzero. "
        "Allow about 1 GB free space for the dataset and a temporary file. Raw files are git-ignored. "
        "The pinned `data/manifest.json` avoids silently following future release changes. "
        "MD5 detects accidental corruption; HTTPS and the pinned release supply provenance, "
        "and MD5 is not a cryptographic authenticity guarantee.\n\n"
        "```sh\n# Offline integrity check\npython data/download.py --verify-only\n"
        "# Regenerate this card and exact machine-readable statistics\npython data/profile.py\n"
        "# Download a single file for a smoke check\npython data/download.py --files H1_O1.csv\n```\n\n"
        "Default paths are relative to the scripts, so invocation also works from another working directory. "
        "`--output-dir` selects a raw download location; pass the same path to `profile.py --raw-dir`. "
        "The profiler refuses incomplete or checksum-mismatched datasets. "
        "All ranges below are observed finite min/max, not imposed selection limits. "
        "Markdown values use 12 significant digits; `docs/data_summary.json` retains Python float precision.",
        "## Files and row counts\n\n" + table(
            ["File", "Rows", "Bytes", "MD5"],
            [[name, files[name]["rows"], files[name]["size_bytes"], files[name]["md5"]] for name in names]
        ),
        "## Seven numeric features\n\n"
        "**Selection is provisional.** The supplied D-01 screenshot requires seven features but does not name them, "
        "and the supplied repository snapshot contains no proposal or feature contract. "
        "The following seven non-timestamp Omicron parameters form a proposed input vector, in this exact order. "
        "The team must confirm it against the proposal before D-01's seven-feature acceptance criterion is signed off. "
        "All source columns are retained regardless of this choice. "
        "`ml_label` is the target; confidence scores, identifiers and image URLs are excluded from this proposed input vector.",
    ]
    feature_rows = []
    detailed_rows = []
    for feature, (unit, description) in FEATURES.items():
        stats = [files[name]["features"][feature] for name in names]
        minima = [s["minimum"] for s in stats if s["minimum"] is not None]
        maxima = [s["maximum"] for s in stats if s["maximum"] is not None]
        feature_rows.append([
            f"`{feature}`", unit, description,
            number(min(minima) if minima else None), number(max(maxima) if maxima else None),
            sum(s["missing"] for s in stats), sum(s["invalid"] for s in stats),
            sum(s["nonfinite"] for s in stats), sum(s["nonpositive"] for s in stats),
        ])
        for name in names:
            s = files[name]["features"][feature]
            detailed_rows.append([name, feature, number(s["minimum"]), number(s["maximum"])])
    sections.append(table(
        ["Column", "Unit", "Meaning", "Raw min", "Raw max", "Empty", "Invalid", "NaN/Inf", "<=0"], feature_rows
    ))
    sections.append(
        "Amplitude caveat: the release identifies `amplitude` as Omicron output but does not specify its "
        "physical normalization/unit. The underlying channel is calibrated gravitational-wave strain; "
        "that alone does not establish whether this derived amplitude is strain, an integrated quantity, "
        "or another normalization. This card therefore keeps the published values and marks the unit as "
        "pipeline units pending confirmation from the release authors or the exact pipeline configuration. "
        "Do not substitute a strain spectral-density unit without evidence.\n\n"
        "`central_freq` and `bandwidth` describe the reported event extent and can exceed the release's "
        "stated 10–2048 Hz peak-frequency selection window. Keep these distinct from `peak_frequency`. "
        "Nonpositive and non-finite inputs are reported, not silently removed. "
        "Feature transforms and scaling must be fitted on training data in later tasks."
    )
    sections.append("## Observed quality flags\n\n" + table(
        ["File", "q_value == 1e20", "ml_confidence == 0", "snr == 7.5"],
        [[name, *(files[name]["quality_flags"][key] for key in
                   ("q_value_equals_1e20", "confidence_zero", "snr_equals_7_5"))] for name in names]
    ) + "\n\nThe repeated Q value `1e20` is an extreme raw value that may be a sentinel; "
        "its meaning is not established by the release description. It is finite and therefore "
        "is not counted as NaN/Inf above. Preserve it in raw data and decide its treatment in D-02 "
        "before scaling or fitting a model. Zero confidence is also retained. One or more exact "
        "boundary SNR values, if listed, should not be silently removed on the basis of the "
        "release's textual SNR > 7.5 selection statement.")
    schema = list(dict.fromkeys(column for f in files.values() for column in f["columns"]))
    descriptions = {
        "event_time": ("GPS s", "Event timestamp, including a fractional part"),
        "ifo": ("category", "Detector identifier H1 or L1"),
        "peak_time": ("GPS s", "Integer seconds of trigger peak"),
        "peak_time_ns": ("ns", "Nanosecond component of peak time"),
        "start_time": ("GPS s", "Integer seconds of trigger start"),
        "start_time_ns": ("ns", "Nanosecond component of start time"),
        "channel": ("text", "Analysed strain channel"),
        "gravityspy_id": ("identifier", "Published Gravity Spy event identifier"),
        "ml_label": ("category", "Predicted morphology label; count these strings as published"),
        "ml_confidence": ("dimensionless [0,1]", "Confidence attached to ml_label; not calibrated certainty"),
        **{name: (unit, description) for name, (unit, description) in FEATURES.items()},
        **{name: ("dimensionless [0,1]", "Original 22-class classifier score") for name in CLASS_COLUMNS},
        **{f"url{i}": ("URL", f"Omega-scan image link, {duration} s window")
           for i, duration in enumerate((0.5, 1, 2, 4), 1)},
    }
    sections.append("## Complete observed schema\n\n" + table(
        ["Column", "Unit / type", "Meaning", "Present in files"],
        [[f"`{col}`", *descriptions.get(col, ("not specified", "Additional raw column; retained unchanged")),
          sum(col in f["columns"] for f in files.values())] for col in schema]
    ))
    sections.append(
        "Times are GPS, not Unix/UTC. Keep the separate integer seconds and nanoseconds when exact "
        "time matching matters; a floating-point `event_time` can lose subsecond precision. "
        "The 22 class-score columns are the original network outputs even in O3; they must not be "
        "assumed to be a score vector aligned with all O3 labels. The upstream description says "
        "None_of_the_Above was removed for O3 but also lists it among O3 labels; the observed counts "
        "below are authoritative for this release. `ml_label`/`ml_confidence` are used as published, "
        "without reconstructing them from the original score columns."
    )
    labels = sorted(set().union(*(f["classes"] for f in files.values())))
    sections.append("## Class counts by detector and run\n\n" + table(
        ["ml_label", *(name.removesuffix(".csv") for name in names), "Total"],
        [[label, *(files[name]["classes"].get(label, 0) for name in names),
          sum(f["classes"].get(label, 0) for f in files.values())] for label in labels]
        + [["TOTAL", *(files[name]["rows"] for name in names), summary["rows"]]]
    ))
    sections.append("## ML confidence distribution\n\n"
        "Quantiles use sorted finite values and linear interpolation at `(n-1)*p`. "
        "These are classifier scores, not independently calibrated probabilities.\n\n" + table(
        ["File", "Min", "P10", "Median", "P90", "P99", "Max", "Mean", "Empty", "Invalid", "NaN/Inf", "Outside [0,1]"],
        [[name, *(number(files[name]["confidence"][key]) for key in
                   ("minimum", "p10", "p50", "p90", "p99", "maximum", "mean")),
          *(files[name]["confidence"][key] for key in ("missing", "invalid", "nonfinite", "out_of_range"))]
         for name in names]
    ))
    sections.append(table(
        ["File", *CONF_LABELS],
        [[name, *(files[name]["confidence"]["histogram"][label] for label in CONF_LABELS)] for name in names]
    ))
    sections.append("## Feature ranges by detector and run\n\n" + table(
        ["File", "Feature", "Raw min", "Raw max"], detailed_rows
    ))
    sections.append("## Limitations and downstream use\n\n"
        "Labels are model predictions; rare classes, imperfect labels and class imbalance need explicit treatment. "
        "Different detectors/runs have different noise populations and observing conditions, so an unrestricted "
        "random split can conceal domain shifts. D-02 should decide confidence filtering, deduplication and class "
        "selection; later split/feature tasks should prevent event and preprocessing leakage. The release "
        "description gives SNR > 7.5 and peak frequency 10–2048 Hz as selection criteria, but this profiler "
        "reports actual CSV values without enforcing these cuts. No waveform or auxiliary channel data "
        "is downloaded by this task.")
    sections.append("## Licence and citation\n\n"
        "The dataset's Zenodo API metadata declares `cc-by-4.0`: "
        "[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). "
        "Provide attribution and identify any later modifications. This dataset licence is separate from "
        "the project's software licence.\n\n"
        "Dataset: Glanzer, J., et al. (2021). *Gravity Spy Machine Learning Classifications of LIGO Glitches "
        "from Observing Runs O1, O2, O3a, and O3b* (v1.0.0). Zenodo. "
        "https://doi.org/10.5281/zenodo.5649212\n\n"
        "Paper: Glanzer, J., et al. (2023). *Data quality up to the third observing run of Advanced LIGO: "
        "Gravity Spy glitch classifications*. Classical and Quantum Gravity, 40, 065004. "
        "https://doi.org/10.1088/1361-6382/acb633\n\n"
        "Sources: [release and column definitions](https://zenodo.org/records/5649212), "
        "[release API (licence, sizes and MD5)](https://zenodo.org/api/records/5649212), "
        "[Omicron documentation](https://virgo.docs.ligo.org/virgoapp/Omicron/).\n\n"
        "Generated by `data/profile.py` from checksum-verified originals; "
        "exact statistics and observed column order are in `docs/data_summary.json`.")
    return "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "data_card.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs" / "data_summary.json")
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.raw_dir)
        card = render_card(summary)
        for path in (args.output, args.json_output):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(card, encoding="utf-8", newline="\n")
        args.json_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8", newline="\n")
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output} and {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
