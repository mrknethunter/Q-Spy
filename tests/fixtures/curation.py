"""Hand-sized raw-schema data, never used as a substitute for real statistics."""

import hashlib
from pathlib import Path

import pandas as pd

from data.curate import DETECTORS, FEATURE_COLUMNS, HELD_OUT, KNOWN_CLASSES, RAW_COLUMNS, RUNS


def raw_fixture() -> pd.DataFrame:
    rows = []
    for label_index, label in enumerate((*KNOWN_CLASSES, HELD_OUT)):
        for ifo, count in (("H1", 9), ("L1", 3)):
            for i in range(count):
                row = dict(
                    gravityspy_id=f"event-{label_index}-{ifo}-{i}", ifo=ifo,
                    peak_time=1_200_000_000 + label_index * 100 + i * 5,
                    peak_time_ns=123_456_789, ml_label=label,
                    ml_confidence=0.99, run=("O3a", "O3b")[i % 2] if label == HELD_OUT else RUNS[i % 4],
                    duration=0.1, peak_frequency=100., central_freq=90.,
                    bandwidth=20., amplitude=1e-21, snr=8. + i, q_value=8.,
                )
                rows.append(row)
    return pd.DataFrame(rows)


def extra_row(identifier: str, **overrides) -> dict:
    row = raw_fixture().iloc[0].to_dict()
    row.update(gravityspy_id=identifier, peak_time=1_500_000_000)
    row.update(overrides)
    return row


def write_release(directory: Path, raw: pd.DataFrame | None = None) -> dict:
    raw = raw_fixture() if raw is None else raw
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for ifo in DETECTORS:
        for run in RUNS:
            path = directory / f"{ifo}_{run}.csv"
            subset = raw.loc[(raw.ifo == ifo) & (raw.run == run), list(RAW_COLUMNS)]
            subset.to_csv(path, index=False)
            content = path.read_bytes()
            files.append(dict(name=path.name, size_bytes=len(content),
                              md5=hashlib.md5(content, usedforsecurity=False).hexdigest()))
    return {"doi": "synthetic-test-only", "version": "fixture-v1", "files": files}
