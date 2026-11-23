from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KNOWN_CLASSES = ("Blip", "Extremely_Loud", "Koi_Fish", "Low_Frequency_Burst",
                 "Low_Frequency_Lines", "Scattered_Light", "Tomte")
HELD_OUT = "Fast_Scattering"
FEATURES = ("snr", "peak_frequency", "central_freq", "bandwidth", "duration", "q_value", "amplitude")
CURATED_COLUMNS = ("id", "gps", "ifo", "label", "confidence", *FEATURES, "run", "is_held_out")


@dataclass(frozen=True)
class Split:  # mirrors data.splits.Split without importing an unmerged module
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def synthetic_curated(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = (*KNOWN_CLASSES, HELD_OUT)
    centres = rng.uniform(0.5, 3.0, size=(len(classes), len(FEATURES)))
    rows = []
    for c, label in enumerate(classes):
        runs = ("O3a", "O3b") if label == HELD_OUT else ("O1", "O2", "O3a", "O3b")
        logs = centres[c] + rng.normal(0.0, 0.15, size=(n_per_class, len(FEATURES)))
        for i in range(n_per_class):
            rows.append({"label": label, "run": runs[i % len(runs)], **dict(zip(FEATURES, 10.0 ** logs[i]))})
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"syn-{i:06d}" for i in range(len(df))])
    df.insert(1, "gps", np.sort(rng.uniform(1.126e9, 1.27e9, len(df))))
    df.insert(2, "ifo", rng.choice(["H1", "L1"], len(df)))
    df.insert(4, "confidence", rng.uniform(0.9, 1.0, len(df)))
    df["is_held_out"] = df["label"] == HELD_OUT
    return df[list(CURATED_COLUMNS)]


def synthetic_features(df: pd.DataFrame) -> np.ndarray:
    logs = np.log10(df[list(FEATURES)].to_numpy(dtype=np.float64))
    med = np.median(logs, axis=0)
    iqr = np.subtract(*np.percentile(logs, [75, 25], axis=0))
    return np.clip((logs - med) / (2.0 * iqr), -1.0, 1.0)


def synthetic_split(df: pd.DataFrame, k: int = 5, n_val: int = 5, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in KNOWN_CLASSES:
        ids = rng.permutation(df.loc[df["label"] == label, "id"].to_numpy())
        train += list(ids[:k]); val += list(ids[k:k + n_val]); test += list(ids[k + n_val:])
    return Split(tuple(train), tuple(val), tuple(test))


def synthetic_bloch(n_points: int, n_qubits: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_points, n_qubits, 3))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v * rng.uniform(0.0, 1.0, size=(n_points, n_qubits, 1))
