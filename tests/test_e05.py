from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diagnostics.concentration import (
    effective_rank,
    eigen_spectrum,
    off_diagonal_values,
    off_diagonal_variance,
)
from experiments.e05.analysis import (
    E05Error,
    analyze_campaign,
    build_report,
    discover_h02_campaign,
    inspect_d05_approval,
    inspect_l05_approval,
    load_store_runs,
    read_frozen_config,
)
from kernels.projected import projected_kernel
from store.io import read_bloch, write_bloch


FROZEN = {
    "n_qubits": 8,
    "layers": 2,
    "s": 0.1778,
    "precision": 0.03,
    "concentration_qubits": [4, 8, 12],
    "device": "ibm_kingston",
}
FROZEN_WITH_GAMMA = {**FROZEN, "gamma": 0.7}


class StubG03:
    """Call recorder for the missing shared G-03 dependency in this checkout."""

    def __init__(self):
        self.bloch_calls = []
        self.compare_calls = []
        self.alignment_calls = []
        self.drift_calls = []

    def bloch_frobenius_error(self, ids_a, B_a, ids_b, B_b):
        self.bloch_calls.append((list(ids_a), list(ids_b)))
        return {
            "sample_ids": list(ids_a),
            "rmse_per_qubit": [0.125] * B_a.shape[1],
            "rmse_global": 0.125,
            "frobenius_global": float(np.sqrt(B_a.size) * 0.125),
        }

    def compare_runs(self, run_a, run_b, *, tier_a, tier_b, kernel_a=None, kernel_b=None):
        self.compare_calls.append((tier_a, tier_b, kernel_a is not None and kernel_b is not None))
        return {"comparison_type": "tier", "tier_a": tier_a, "tier_b": tier_b, "kernel_comparison": None}

    def kernel_alignment(self, K_a, K_b, ids_a, ids_b):
        self.alignment_calls.append((list(ids_a), list(ids_b), np.shape(K_a), np.shape(K_b)))
        return 0.91

    def drift_report(self, run_start, run_end, *, run_start_id=None, run_end_id=None, kernel_start=None, kernel_end=None):
        self.drift_calls.append((list(run_start[0]), list(run_end[0]), kernel_start is not None))
        return {
            "comparison_type": "drift",
            "reference_ids": list(run_start[0]),
            "bloch_drift": {
                "sample_ids": list(run_start[0]),
                "rmse_per_qubit": [0.025] * run_start[1].shape[1],
                "rmse_global": 0.025,
                "frobenius_global": float(np.sqrt(run_start[1].size) * 0.025),
            },
            "kernel_alignment": None,
        }


def _subsets():
    return {
        "status": "bound_to_hw_freeze",
        "selection_sha256": "fixture-selection-digest",
        "l05_freeze": {"tag": "hw-freeze"},
        "subsets": {
            "main": {"ids": ["main-1", "main-2", "main-3"]},
            "control": {"ids": ["control-1", "control-2", "control-3"]},
            "drift_references": {"ids": ["ref-1", "ref-2", "ref-3"]},
            "concentration": [
                {"ids": ["n4-1", "n4-2", "n4-3"]},
                {"ids": ["n8-1", "n8-2", "n8-3"]},
                {"ids": ["n12-1", "n12-2", "n12-3"]},
            ],
        },
    }


def _store_h02(
    store_root: Path,
    *,
    bad_experiment: str | None = None,
    bloch_dtype=np.float32,
    source_kind: str | None = "synthetic_fixture",
) -> str:
    subsets = _subsets()
    prefix = "hw-20261124T010203Z"
    experiments = {
        "main": (subsets["subsets"]["main"]["ids"], 8),
        "control": (subsets["subsets"]["control"]["ids"], 8),
        "drift-start": (subsets["subsets"]["drift_references"]["ids"], 8),
        "drift-end": (subsets["subsets"]["drift_references"]["ids"], 8),
        "concentration-n4": (subsets["subsets"]["concentration"][0]["ids"], 4),
        "concentration-n8": (subsets["subsets"]["concentration"][1]["ids"], 8),
        "concentration-n12": (subsets["subsets"]["concentration"][2]["ids"], 12),
    }
    rng = np.random.default_rng(7)
    for experiment, (ids, n_qubits) in experiments.items():
        for tier in ("L0", "HW"):
            if bad_experiment == f"{experiment}/{tier}":
                continue
            stored_ids = list(ids) if tier == "L0" else list(reversed(ids))
            B = rng.uniform(-0.7, 0.7, (len(ids), n_qubits, 3)).astype(bloch_dtype)
            if tier == "HW":
                B = B[::-1].copy()
            meta = {
                "tier": tier,
                "campaign_id": prefix,
                "experiment": experiment,
                "embed_config": {"n_qubits": n_qubits, "layers": 2, "s": 0.1778, "entangle": True},
            }
            if source_kind is not None:
                meta["source_kind"] = source_kind
            if tier == "HW":
                layout = list(range(n_qubits))
                meta.update({
                    "precision": FROZEN["precision"],
                    "job_ids": [f"fixture-{experiment}"],
                    "backend_name": FROZEN["device"],
                    "layout": layout,
                    "two_qubit_depth": 4,
                    "estimated_qpu_s": 1.0,
                    "usage_s": 1.0,
                    "mitigation": {
                        "measure_mitigation": experiment != "control",
                        "twirling": experiment != "control",
                        "dynamical_decoupling": experiment != "control",
                    },
                    "calibration": {
                        "last_update": "synthetic-fixture",
                        "qubits": {
                            str(qubit): {"t1": 0.1, "t2": 0.1, "readout_error": 0.01}
                            for qubit in layout
                        },
                        "two_qubit_errors": {"cz:0-1": 0.01},
                    },
                })
            run_id = f"{prefix}-{experiment}-{tier}"
            write_bloch(run_id, stored_ids, B, meta, root=store_root)
    return prefix


def test_i04_loading_uses_read_bloch(tmp_path):
    root = tmp_path / "store"
    B = np.zeros((2, 3, 3), dtype=np.float32)
    write_bloch("fixture-L0", ["a", "b"], B, {"tier": "L0"}, root=root)
    runs, errors = load_store_runs(root)
    assert not errors
    ids, loaded, meta = runs["fixture-L0"]
    expected = read_bloch("fixture-L0", root=root)
    assert ids == expected[0] == ["a", "b"]
    np.testing.assert_array_equal(loaded, expected[1])
    assert meta == expected[2]


@pytest.mark.parametrize(
    ("stored_ids", "reason"),
    [
        (["valid", 7], "not a string"),
        (["valid", "  "], "whitespace-only"),
        (["same", "same"], "unique"),
        (["only-one"], "Bloch leading dimension"),
    ],
)
def test_i04_loading_rejects_nonconforming_ids_and_reports_run(tmp_path, stored_ids, reason):
    root = tmp_path / "store"
    write_bloch("fixture-L0", ["placeholder-1", "placeholder-2"], np.zeros((2, 3, 3), dtype=np.float32),
                {"tier": "L0"}, root=root)
    (root / "fixture-L0" / "ids.json").write_text(json.dumps(stored_ids), encoding="utf-8")

    runs, errors = load_store_runs(root)
    assert "fixture-L0" not in runs
    assert len(errors) == 1
    assert errors[0]["run_id"] == "fixture-L0"
    assert reason in errors[0]["reason"]

    report = build_report(
        store_root=root,
        subsets=_subsets(),
        frozen_config=FROZEN,
        g03=StubG03(),
    )
    assert report["store_errors"][0]["run_id"] == "fixture-L0"
    assert reason in report["store_errors"][0]["reason"]


def test_i04_invalid_id_is_not_silently_stringified(tmp_path):
    root = tmp_path / "store"
    write_bloch("fixture-L0", ["placeholder-1", "placeholder-2"], np.zeros((2, 3, 3), dtype=np.float32),
                {"tier": "L0"}, root=root)
    id_path = root / "fixture-L0" / "ids.json"
    id_path.write_text('["sample", 7]', encoding="utf-8")
    raw_ids = json.loads(id_path.read_text(encoding="utf-8"))

    runs, errors = load_store_runs(root)
    assert raw_ids == ["sample", 7]
    assert "fixture-L0" not in runs
    assert "not a string" in errors[0]["reason"]


def test_g01_analytic_off_diagonal_values_and_population_variance():
    kernel = np.array(
        [[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(off_diagonal_values(kernel), [0.2, 0.4, 0.6], rtol=0, atol=1e-15)
    assert off_diagonal_variance(kernel) == pytest.approx(2 / 75, rel=1e-14, abs=1e-15)


def test_g01_analytic_spectrum_and_effective_rank_for_known_psd_kernel():
    # This matrix has eigenvalues 2, 1/2, 1/2 and normalized spectrum 2/3, 1/6, 1/6.
    kernel = 0.5 * np.eye(3, dtype=np.float64) + 0.5 * np.ones((3, 3), dtype=np.float64)
    expected_spectrum = np.array([2.0, 0.5, 0.5])
    expected_rank = np.exp(-((2 / 3) * np.log(2 / 3) + 2 * (1 / 6) * np.log(1 / 6)))
    np.testing.assert_allclose(eigen_spectrum(kernel), expected_spectrum, rtol=1e-13, atol=1e-14)
    assert effective_rank(kernel) == pytest.approx(expected_rank, rel=1e-13, abs=1e-14)


def test_g01_analytic_rank_edge_cases_and_psd_tolerance():
    zero = np.zeros((3, 3), dtype=np.float64)
    np.testing.assert_array_equal(eigen_spectrum(zero), [0.0, 0.0, 0.0])
    assert effective_rank(zero) == 0.0
    assert effective_rank(np.ones((3, 3), dtype=np.float64)) == pytest.approx(1.0, rel=1e-14)

    tiny_negative_eigenvalue = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])
    assert effective_rank(tiny_negative_eigenvalue) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError, match="semidefinita"):
        effective_rank(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_g01_off_diagonal_functions_require_two_samples():
    with pytest.raises(ValueError, match="almeno due"):
        off_diagonal_values(np.ones((1, 1)))
    with pytest.raises(ValueError, match="almeno due"):
        off_diagonal_variance(np.ones((1, 1)))


def test_h02_campaign_matches_d05_ids_and_frozen_embedding(tmp_path):
    root = tmp_path / "store"
    prefix = _store_h02(root)
    runs, errors = load_store_runs(root)
    assert not errors
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is not None
    assert campaign["campaign_id"] == prefix
    assert campaign["source"] == "synthetic_fixture"
    assert candidates[0]["valid"]
    assert len(campaign["runs"]) == 7


def test_campaign_accepts_explicit_hardware_provenance(tmp_path):
    _store_h02(tmp_path / "store", source_kind="hardware")
    runs, _ = load_store_runs(tmp_path / "store")
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is not None
    assert campaign["source"] == "hardware"
    assert candidates[0]["valid"]
    assert all("explicitly marked hardware" in detail for detail in campaign["source_details"])


def test_absent_source_kind_is_inferred_only_from_complete_h02_metadata(tmp_path):
    # Temporary synthetic vectors exercise metadata classification only; this is
    # not a hardware run or evidence of hardware validation.
    _store_h02(tmp_path / "store", source_kind=None)
    runs, _ = load_store_runs(tmp_path / "store")
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is not None
    assert campaign["source"] == "hardware"
    assert candidates[0]["valid"]
    assert all("inferred from complete H-02 metadata" in detail for detail in campaign["source_details"])

    hw_id = next(run_id for run_id in runs if run_id.endswith("-main-HW"))
    del runs[hw_id][2]["backend_name"]
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert candidates[0]["source"] == "unverified"
    assert any("provenance is unverified" in error for error in candidates[0]["errors"])


def test_campaign_with_mixed_hardware_and_fixture_provenance_is_rejected(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    hw_id = next(run_id for run_id in runs if run_id.endswith("-main-HW"))
    runs[hw_id][2]["source_kind"] = "hardware"
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert candidates[0]["source"] == "mixed_provenance"
    assert not candidates[0]["valid"]
    assert any("mixed hardware and synthetic_fixture" in error for error in candidates[0]["errors"])


def test_duplicate_experiment_tier_runs_reject_campaign_deterministically(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    entries = list(runs.items())
    main_hw = next(item for item in entries if item[0].endswith("-main-HW"))
    control_l0 = next(item for item in entries if item[0].endswith("-control-L0"))
    conflicting_main_hw = (
        main_hw[0],
        (main_hw[1][0], main_hw[1][1], {**main_hw[1][2], "source_kind": "hardware"}),
    )

    class DuplicateEntries:
        def __init__(self, values):
            self.values = values

        def items(self):
            return iter(self.values)

    duplicated = entries + [main_hw, conflicting_main_hw, main_hw, control_l0]
    _, forward = discover_h02_campaign(DuplicateEntries(duplicated), _subsets(), FROZEN)
    _, reverse = discover_h02_campaign(DuplicateEntries(list(reversed(duplicated))), _subsets(), FROZEN)
    assert forward[0]["valid"] is False
    assert reverse[0]["valid"] is False
    assert forward[0] == reverse[0]
    assert forward[0]["errors"] == reverse[0]["errors"]
    main_error = next(error for error in forward[0]["errors"] if "experiment=main, tier=HW" in error)
    control_error = next(error for error in forward[0]["errors"] if "experiment=control, tier=L0" in error)
    assert main_error.count(main_hw[0]) == 4
    assert control_error.count(control_l0[0]) == 2
    assert "hw-20261124T010203Z" in main_error


def test_h02_candidate_rejects_missing_tier_and_config_mismatch(tmp_path):
    root = tmp_path / "store"
    _store_h02(root, bad_experiment="main/HW")
    runs, _ = load_store_runs(root)
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert any("main/HW: run missing" in error for error in candidates[0]["errors"])

    root2 = tmp_path / "store2"
    _store_h02(root2)
    runs2, _ = load_store_runs(root2)
    run_id = next(name for name in runs2 if name.endswith("-main-HW"))
    runs2[run_id][2]["embed_config"]["s"] = 1.0
    campaign, candidates = discover_h02_campaign(runs2, _subsets(), FROZEN)
    assert campaign is None
    assert any("embed_config.s" in error for error in candidates[0]["errors"])


def test_h01_pilot_shape_is_not_discovered_as_h02(tmp_path):
    root = tmp_path / "store"
    write_bloch(
        "hw-20261124T005515Z-HW",
        ["pilot-1", "pilot-2"],
        np.zeros((2, 8, 3), dtype=np.float32),
        {"tier": "HW", "job_ids": ["pilot"], "backend_name": "ibm", "calibration": {"snapshot": "fixture"}},
        root=root,
    )
    runs, _ = load_store_runs(root)
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert candidates == []


def test_projected_kernel_and_g01_metrics_use_frozen_gamma(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    campaign, _ = discover_h02_campaign(runs, _subsets(), FROZEN_WITH_GAMMA)
    g03 = StubG03()
    result = analyze_campaign(campaign, FROZEN_WITH_GAMMA, g03=g03)
    main_l0 = campaign["runs"]["main"]["L0"]
    expected_kernel = projected_kernel(main_l0[1], main_l0[1], 0.7)
    expected_variance = off_diagonal_variance(expected_kernel)
    assert result["gamma"] == {"value": 0.7, "source": "frozen_config", "blocked_reason": None}
    assert result["groups"]["main"]["concentration"]["L0"]["off_diagonal_variance"] == pytest.approx(expected_variance)
    assert result["groups"]["main"]["concentration"]["L0"]["off_diagonal_count"] == 3
    assert result["groups"]["main"]["kernel_alignment"] == pytest.approx(0.91)
    assert len(g03.alignment_calls) == 7


def test_g03_receives_ids_in_stored_order_and_drift_reference_runs(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    campaign, _ = discover_h02_campaign(runs, _subsets(), FROZEN_WITH_GAMMA)
    g03 = StubG03()
    analyze_campaign(campaign, FROZEN_WITH_GAMMA, g03=g03)
    main_l0 = campaign["runs"]["main"]["L0"][0]
    main_hw = campaign["runs"]["main"]["HW"][0]
    assert main_l0 == list(reversed(main_hw))
    assert (main_l0, main_hw) in g03.bloch_calls
    assert g03.drift_calls and g03.drift_calls[0][2] is True


def test_missing_gamma_keeps_bloch_comparison_and_blocks_kernel_metrics(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    campaign, _ = discover_h02_campaign(runs, _subsets(), FROZEN)
    g03 = StubG03()
    result = analyze_campaign(campaign, FROZEN, g03=g03)
    assert result["gamma"]["value"] is None
    assert "official training-selection artifact" in result["gamma"]["blocked_reason"]
    assert result["groups"]["main"]["bloch_error"] is not None
    assert result["groups"]["main"]["concentration"] is None
    assert result["groups"]["main"]["kernel_alignment"] is None
    assert result["drift"] is not None
def test_build_report_distinguishes_synthetic_fixture_and_generates_artifacts(tmp_path):
    store_root = tmp_path / "store"
    _store_h02(store_root)
    g03 = StubG03()
    report = build_report(
        store_root=store_root,
        subsets=_subsets(),
        frozen_config=FROZEN_WITH_GAMMA,
        g03=g03,
    )
    assert report["provenance"]["campaign_source"] == "synthetic_fixture"
    assert report["hardware_validation"] == "not_validated"
    assert any("synthetic_fixture" in blocker for blocker in report["blockers"])

    first = tmp_path / "out-a"
    second = tmp_path / "out-b"
    from experiments.e05.analysis import write_outputs

    run_a = write_outputs(report, first, run_id="fixture-run")
    run_b = write_outputs(report, second, run_id="fixture-run")
    generated = sorted(path.name for path in (run_a / "figures").glob("*.svg"))
    assert "bloch_rmse_per_qubit.svg" in generated
    assert "kernel_alignment.svg" in generated
    assert "concentration_variance.svg" in generated
    assert "effective_rank.svg" in generated
    assert "drift_rmse_per_qubit.svg" in generated
    assert len(generated) == 12
    assert "synthetic_fixture" in (run_a / "summary.md").read_text(encoding="utf-8")
    assert "No QPU job was submitted" in (run_a / "summary.md").read_text(encoding="utf-8")
    assert (run_a / "analysis.json").read_bytes() == (run_b / "analysis.json").read_bytes()
    assert (run_a / "summary.md").read_bytes() == (run_b / "summary.md").read_bytes()
    figures_a = {path.name: path.read_bytes() for path in (run_a / "figures").glob("*.svg")}
    figures_b = {path.name: path.read_bytes() for path in (run_b / "figures").glob("*.svg")}
    assert figures_a == figures_b
    sample_svg = (run_a / "figures" / "bloch_rmse_per_qubit.svg").read_text(encoding="utf-8")
    assert "RMSE (dimensionless Bloch expectation)" in sample_svg
    assert "HW vs L0" in sample_svg
    from experiments.e05.cli import main

    assert main(["figures", str(run_a)]) == 0
    assert main(["summary", str(run_a)]) == 0
    assert json.loads((run_a / "analysis.json").read_text(encoding="utf-8"))["figures_generated"]


def test_current_frozen_configuration_has_no_gamma():
    frozen = read_frozen_config()
    assert frozen["n_qubits"] == 8
    assert frozen["concentration_qubits"] == [4, 8, 12]
    assert "gamma" not in frozen


def test_build_report_requests_live_tag_validation_for_d05(monkeypatch, tmp_path):
    import experiments.e05.analysis as e05_analysis

    seen = {}

    def fake_load_hardware_subsets(*, freeze_repo=None):
        seen["freeze_repo"] = Path(freeze_repo)
        return _subsets()

    monkeypatch.setattr(e05_analysis, "load_hardware_subsets", fake_load_hardware_subsets)
    e05_analysis.build_report(store_root=tmp_path / "empty-store")
    assert seen["freeze_repo"] == Path(__file__).resolve().parents[1]


def test_unfrozen_gamma_is_a_blocker_not_a_cli_override():
    from experiments.e05.analysis import _validate_gamma

    value, source, blocker = _validate_gamma(FROZEN)
    assert value is None and source is None
    assert "official training-selection artifact" in blocker


def test_invalid_frozen_gamma_is_rejected():
    from experiments.e05.analysis import _validate_gamma

    with pytest.raises(E05Error, match="finite and greater than zero"):
        _validate_gamma({**FROZEN, "gamma": float("nan")})


def test_campaign_rejects_bloch_dtype_shape_and_range_violations(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    main_hw_id = next(run_id for run_id in runs if run_id.endswith("-main-HW"))
    ids, B, meta = runs[main_hw_id]

    runs[main_hw_id] = (ids, B.astype(np.float16), meta)
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert any("dtype must be float32 or float64" in reason for reason in candidates[0]["errors"])

    runs[main_hw_id] = (ids, B.copy(), meta)
    runs[main_hw_id][1][0, 0, 0] = np.nan
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert any("NaN or infinite" in reason for reason in candidates[0]["errors"])


def test_campaign_checks_h02_provenance_without_requiring_job_status(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    campaign, _ = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is not None
    hw = campaign["runs"]["main"]["HW"][2]
    assert "job_status" not in hw

    hw["mitigation"]["twirling"] = False
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert any("mitigation metadata" in reason for reason in candidates[0]["errors"])


def test_campaign_rejects_unknown_source_kind(tmp_path):
    _store_h02(tmp_path / "store")
    runs, _ = load_store_runs(tmp_path / "store")
    hw_id = next(run_id for run_id in runs if run_id.endswith("-main-HW"))
    runs[hw_id][2]["source_kind"] = "simulated"
    campaign, candidates = discover_h02_campaign(runs, _subsets(), FROZEN)
    assert campaign is None
    assert any("not a recognized provenance" in reason for reason in candidates[0]["errors"])


def test_campaign_preserves_float64_bloch_vectors(tmp_path):
    _store_h02(tmp_path / "store", bloch_dtype=np.float64)
    runs, errors = load_store_runs(tmp_path / "store")
    assert not errors
    campaign, _ = discover_h02_campaign(runs, _subsets(), FROZEN_WITH_GAMMA)
    assert campaign is not None
    for tiers in campaign["runs"].values():
        assert tiers["L0"][1].dtype == np.float64
        assert tiers["HW"][1].dtype == np.float64
    result = analyze_campaign(campaign, FROZEN_WITH_GAMMA, g03=StubG03())
    assert result["groups"]["main"]["bloch_error"] is not None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_actual_g03_integration_when_dependency_is_available(tmp_path, dtype):
    g03 = pytest.importorskip("diagnostics.compare")
    _store_h02(tmp_path / "store", bloch_dtype=dtype)
    runs, errors = load_store_runs(tmp_path / "store")
    assert not errors
    campaign, _ = discover_h02_campaign(runs, _subsets(), FROZEN_WITH_GAMMA)
    result = analyze_campaign(campaign, FROZEN_WITH_GAMMA, g03=g03)

    run_l0, run_hw = campaign["runs"]["main"]["L0"], campaign["runs"]["main"]["HW"]
    expected = g03.bloch_frobenius_error(run_l0[0], run_l0[1], run_hw[0], run_hw[1])
    assert result["groups"]["main"]["bloch_error"] == expected
    assert result["groups"]["main"]["comparison"]["tier_a"] == "L0"
    assert result["groups"]["main"]["comparison"]["tier_b"] == "HW"
    assert result["drift"]["run_start_id"].endswith("-drift-start-HW")
    assert result["drift"]["run_end_id"].endswith("-drift-end-HW")


def test_current_l05_decision_is_not_approved_by_tag_alone():
    repository_root = Path(__file__).resolve().parents[1]
    state, detail = inspect_l05_approval(repository_root)
    assert state == "pending"
    assert "draft for lead approval" in detail


def test_current_d05_binding_does_not_imply_final_acceptance():
    repository_root = Path(__file__).resolve().parents[1]
    state, detail = inspect_d05_approval(repository_root)
    assert state == "pending"
    assert "review proposal" in detail


def test_g03_dependency_is_available_or_explicitly_reported():
    from experiments.e05.analysis import _load_g03

    module, error = _load_g03()
    if module is None:
        assert error and "diagnostics.compare" in error
    else:
        assert error is None
        assert all(callable(getattr(module, name, None)) for name in (
            "bloch_frobenius_error", "kernel_alignment", "compare_runs", "drift_report"
        ))


def test_approval_readers_require_explicit_approved_state(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "decisions.md").write_text("Status: **approved**\n", encoding="utf-8")
    (docs / "validation_d05.md").write_text("D-05 selection validation passed.\n", encoding="utf-8")
    (docs / "hardware_subsets.md").write_text(
        "Allowed lifecycle values: draft_pending_l05 or bound_to_hw_freeze.\n"
        "D-05 approval: approved\n",
        encoding="utf-8",
    )
    assert inspect_l05_approval(tmp_path)[0] == "documented_approved"
    assert inspect_d05_approval(tmp_path)[0] == "documented_approved"

    (docs / "decisions.md").write_text("Status: **not approved**\n", encoding="utf-8")
    assert inspect_l05_approval(tmp_path)[0] != "documented_approved"



