from __future__ import annotations

from pathlib import Path

import numpy as np


def write_bloch(run_id: str, ids, B: np.ndarray, meta: dict, root: str | Path | None = None) -> Path:
    raise NotImplementedError("I-04")


def read_bloch(run_id: str, root: str | Path | None = None) -> tuple[list[str], np.ndarray, dict]:
    raise NotImplementedError("I-04")
