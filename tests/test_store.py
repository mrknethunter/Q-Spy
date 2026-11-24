import numpy as np
import pytest

from store.io import read_bloch, write_bloch
from tests.fixtures.synthetic import synthetic_bloch


def test_write_records_provenance(tmp_path):
    write_bloch("run-a", ["a", "b"], synthetic_bloch(2, 3), {"tier": "L0"}, root=tmp_path)
    meta = read_bloch("run-a", root=tmp_path)[2]
    assert meta["tier"] == "L0" and meta["created_at"] and "git_commit" in meta


def test_simulated_runs_can_be_overwritten(tmp_path):
    write_bloch("run-a", ["a"], synthetic_bloch(1, 2, seed=1), {"tier": "L2"}, root=tmp_path)
    B = synthetic_bloch(1, 2, seed=2)
    write_bloch("run-a", ["a"], B, {"tier": "L2"}, root=tmp_path)
    assert np.array_equal(read_bloch("run-a", root=tmp_path)[1], B)


def test_rejects_ids_that_do_not_match_rows(tmp_path):
    with pytest.raises(ValueError):
        write_bloch("run-a", ["a"], synthetic_bloch(2, 3), {"tier": "L0"}, root=tmp_path)


def test_meta_names_its_run(tmp_path):
    write_bloch("run-a", ["a"], synthetic_bloch(1, 2), {"tier": "L0"}, root=tmp_path)
    assert read_bloch("run-a", root=tmp_path)[2]["run_id"] == "run-a"
