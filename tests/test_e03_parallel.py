"""Table aggregation tests for the process-parallel E-03 runner."""
import sys
import types
import warnings

import pandas as pd
import pytest

from experiments.e03_parallel import _concat_frames, _run_fold

def test_concat_frames_preserves_column_order():
    frames = [
        pd.DataFrame({
            "family": ["Blip"],
            "count": [2],
        }),
        pd.DataFrame({
            "family": ["Tomte"],
            "count": [4],
        }),
    ]

    result = _concat_frames(frames)

    assert result.columns.tolist() == ["family", "count"]
    assert result["family"].tolist() == ["Blip", "Tomte"]

def test_concat_frames_no_inputs_returns_empty_dataframe():
    result = _concat_frames([])

    assert result.empty
    assert result.columns.tolist() == []

def test_concat_frames_skips_empty_rotations_and_preserves_valid_schema():
    schema = ["family", "count"]
    frames = [
        pd.DataFrame(columns=schema),
        pd.DataFrame({"family": ["Fast_Scattering"], "count": pd.Series([3], dtype="int64")}),
        pd.DataFrame(columns=schema),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = _concat_frames(frames)

    assert result.columns.tolist() == schema
    assert result.to_dict("records") == [{"family": "Fast_Scattering", "count": 3}]
    assert result["count"].dtype == "int64"


def test_concat_frames_all_empty_keeps_columns_without_warning():
    schema = ["family", "count"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = _concat_frames([pd.DataFrame(columns=schema), pd.DataFrame(columns=schema)])

    assert result.empty
    assert result.columns.tolist() == schema


def test_concat_frames_no_inputs_returns_requested_empty_schema():
    result = _concat_frames([], columns=["id", "score"])

    assert result.empty
    assert result.columns.tolist() == ["id", "score"]


def test_concat_frames_rejects_inconsistent_schemas():
    with pytest.raises(ValueError, match="different column schemas"):
        _concat_frames([pd.DataFrame(columns=["id"]), pd.DataFrame(columns=["score"])])


def test_spawn_worker_loads_experiment_runner_lazily(monkeypatch):
    module = types.ModuleType("experiments.e03_novelty")
    module.run_experiment = lambda frame, **kwargs: (frame, kwargs)
    monkeypatch.setitem(sys.modules, "experiments.e03_novelty", module)

    result = _run_fold("fixture", "Blip", {"seed": 3})

    assert result == ("fixture", {"seed": 3, "held_outs": ["Blip"]})
