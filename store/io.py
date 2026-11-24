from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parent / "runs"


def write_bloch(run_id: str, ids, B: np.ndarray, meta: dict, root: str | Path | None = None) -> Path:
    B = np.asarray(B)
    ids = [str(i) for i in ids]
    if B.ndim != 3 or B.shape[2] != 3 or len(ids) != B.shape[0]:
        raise ValueError(f"need B of shape (len(ids), n, 3), got {B.shape} for {len(ids)} ids")
    path = Path(root or DEFAULT_ROOT) / run_id
    if (path / "meta.json").exists() and json.loads((path / "meta.json").read_text()).get("tier") == "HW":
        raise FileExistsError(f"{path} holds a hardware run and is never overwritten")
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "B.npy", B)
    (path / "ids.json").write_text(json.dumps(ids))
    meta = {**meta, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit()}
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))  # default=str: never lose a run over a type
    return path


def read_bloch(run_id: str, root: str | Path | None = None) -> tuple[list[str], np.ndarray, dict]:
    path = Path(root or DEFAULT_ROOT) / run_id
    return json.loads((path / "ids.json").read_text()), np.load(path / "B.npy"), json.loads((path / "meta.json").read_text())


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=DEFAULT_ROOT.parent, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
