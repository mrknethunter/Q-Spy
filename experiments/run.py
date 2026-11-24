from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python experiments/run.py`
from store.io import _git_commit  # noqa: E402  (same provenance as the Bloch-vector store)

REQUIRED = ("name", "data", "run")


def _load(ref: str):
    module, function = ref.split(":")
    return getattr(importlib.import_module(module), function)


def run(config_path, out_dir="results"):
    config = yaml.safe_load(Path(config_path).read_text())
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise ValueError(f"{config_path}: missing keys {missing}")
    data = _load(config["data"])(**config.get("data_params", {}))  # a tuple of positional arguments for `run`
    res = _load(config["run"])(*data, **config.get("params", {}))
    out = Path(out_dir) / f"{config['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    out.with_suffix(".run.json").write_text(json.dumps(
        {"config": config, "rows": len(res), "git_commit": _git_commit(),
         "finished_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run experiments described by YAML files (I-03); see docs/contracts.md.")
    p.add_argument("configs", nargs="+")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    for c in a.configs:
        print(f"{c} -> {run(c, a.out)}")
