"""E-05 orchestration for diagnostics over stored Bloch-vector runs.

The experiment owns campaign discovery, provenance checks and reporting. It
delegates Bloch and kernel comparisons to G-03, projected kernels to M-01,
concentration metrics to G-01, store reads to I-04, and subset validation to
D-05. It never submits work to a QPU.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import diagnostics.compare as _g03_module
except ModuleNotFoundError as error:
    # Keep isolated E-05 tests usable before G-03 is integrated, but surface
    # only that exact missing module. Errors inside an available G-03 import
    # must not be mistaken for an unavailable dependency.
    if error.name != "diagnostics.compare":
        raise
    _g03_module = None
    _g03_import_error = str(error)
else:
    _g03_import_error = None

from data.hardware_subsets import load_hardware_subsets
from diagnostics.concentration import (
    effective_rank,
    eigen_spectrum,
    off_diagonal_values,
    off_diagonal_variance,
)
from kernels.projected import projected_kernel
from store.io import DEFAULT_ROOT, read_bloch

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "hardware_frozen.yaml"
EXPERIMENT_PATTERN = r"drift-start|drift-end|concentration-n(?:\d+)|control|main"
RUN_PATTERN = re.compile(
    rf"^(?P<campaign>hw-\d{{8}}T\d{{6}}Z)-(?P<experiment>{EXPERIMENT_PATTERN})-"
    r"(?P<tier>L0|L1|L2|HW)$"
)
EXPECTED_TIERS = ("L0", "HW")


class E05Error(ValueError):
    """Invalid E-05 configuration, run metadata or experiment input."""


def _parse_scalar(value: str) -> Any:
    """Parse the simple scalar/list subset used by the frozen YAML file."""
    value = value.strip()
    if not value:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar(part) for part in value[1:-1].split(",") if part.strip()]
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value[0:1] in ("'", '"') and value[-1:] == value[0]:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            result = float(value)
        except ValueError:
            return value
        if not math.isfinite(result):
            raise E05Error("Frozen configuration contains a non-finite number.")
        return result


def read_frozen_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the project's flat, scalar hardware freeze without adding PyYAML."""
    config_path = Path(path)
    values: dict[str, Any] = {}
    for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), 1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        if ":" not in content:
            raise E05Error(f"Unsupported frozen config syntax at {config_path}:{line_number}.")
        key, value = content.split(":", 1)
        if not key.strip() or key.strip() in values:
            raise E05Error(f"Invalid or duplicate frozen config key at {config_path}:{line_number}.")
        values[key.strip()] = _parse_scalar(value)
    required = ("n_qubits", "layers", "s", "precision", "concentration_qubits", "device")
    missing = [key for key in required if key not in values]
    if missing:
        raise E05Error(f"Frozen configuration is missing: {', '.join(missing)}.")
    if not isinstance(values["concentration_qubits"], list):
        raise E05Error("concentration_qubits must be a list in the frozen configuration.")
    return values


def inspect_l05_approval(repository_root: str | Path) -> tuple[str, str]:
    """Read the decision record; a Git tag alone is not group approval."""
    root = Path(repository_root) / "docs"
    path = root / "decisions.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return "unverified", f"L-05 decision record is unavailable: {error}"
    try:
        validation = (root / "validation_d05.md").read_text(encoding="utf-8").lower()
    except OSError:
        validation = ""
    status_match = re.search(r"^Status:\s*\*\*(.*?)\*\*", text, flags=re.IGNORECASE | re.MULTILINE)
    if status_match is None:
        return "unverified", "docs/decisions.md has no explicit L-05 status line"
    status = status_match.group(1).strip().lower()
    if (
        status.startswith("draft") or status in {"pending", "not approved"}
        or "l-05 is still pending" in validation
        or "does not establish completion of l-05" in validation
    ):
        return "pending", f"docs/decisions.md records L-05 as {status_match.group(1).strip()}"
    if status == "approved":
        return "documented_approved", f"docs/decisions.md records L-05 as {status_match.group(1).strip()}"
    return "unverified", f"docs/decisions.md status is not an explicit approval: {status_match.group(1).strip()}"


def inspect_d05_approval(repository_root: str | Path) -> tuple[str, str]:
    """Check the D-05 lifecycle documents; tag binding is not acceptance."""
    docs = Path(repository_root) / "docs"
    try:
        lifecycle = (docs / "hardware_subsets.md").read_text(encoding="utf-8").lower()
    except OSError as error:
        return "unverified", f"D-05 lifecycle documentation is unavailable: {error}"
    try:
        validation = (docs / "validation_d05.md").read_text(encoding="utf-8").lower()
    except OSError:
        validation = ""
    status_match = re.search(
        r"^D-05 approval:\s*(.*?)\s*$", lifecycle, flags=re.IGNORECASE | re.MULTILINE
    )
    status = status_match.group(1).strip().lower() if status_match else ""
    pending_markers = (
        "l-05 is not yet completed",
        "not an approved hardware configuration",
        "review proposal, not an approved hardware configuration",
        "does not establish completion of l-05 or final acceptance of d-05",
    )
    if status in {"draft", "pending", "not approved"} or any(
        marker in lifecycle or marker in validation for marker in pending_markers
    ):
        return "pending", "D-05 documentation still describes the subset as a draft/review proposal without final acceptance"
    if status == "approved":
        return "documented_approved", "docs/hardware_subsets.md records explicit D-05 approval"
    return "unverified", "D-05 documentation has no explicit approval status"


def _expected_experiments(subsets: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, list[str]]:
    groups = subsets.get("subsets", {})
    concentration = groups.get("concentration", [])
    qubits = frozen.get("concentration_qubits", [])
    if len(concentration) != len(qubits) or not concentration:
        raise E05Error("D-05 concentration groups do not match the frozen qubit list.")
    expected: dict[str, list[str]] = {
        "main": list(groups["main"]["ids"]),
        "control": list(groups["control"]["ids"]),
        "drift-start": list(groups["drift_references"]["ids"]),
        "drift-end": list(groups["drift_references"]["ids"]),
    }
    for qubit_count, subset in zip(qubits, concentration):
        expected[f"concentration-n{int(qubit_count)}"] = list(subset["ids"])
    return expected


def _same_ids(actual: list[str], expected: list[str]) -> bool:
    return len(actual) == len(expected) and len(set(actual)) == len(actual) and set(actual) == set(expected)


def _configuration_error(
    meta: Mapping[str, Any], experiment: str, frozen: Mapping[str, Any], tier: str
) -> str | None:
    config = meta.get("embed_config")
    if not isinstance(config, Mapping):
        return "embed_config metadata is missing or invalid"
    expected_n = int(experiment.rsplit("n", 1)[1]) if experiment.startswith("concentration-n") else int(frozen["n_qubits"])
    checks = {
        "n_qubits": expected_n,
        "layers": int(frozen["layers"]),
        "s": float(frozen["s"]),
    }
    for key, expected in checks.items():
        if key not in config:
            return f"embed_config.{key} is missing"
        actual = config[key]
        if key == "s":
            try:
                matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            return f"embed_config.{key}={actual!r} does not match expected {expected!r}"
    if not isinstance(config.get("entangle"), bool):
        return "embed_config.entangle is missing or is not boolean"
    if tier == "HW":
        try:
            precision = float(meta["precision"])
        except (KeyError, TypeError, ValueError):
            return "HW precision metadata is missing or invalid"
        if not math.isfinite(precision) or not math.isclose(
            precision, float(frozen["precision"]), rel_tol=0.0, abs_tol=1e-12
        ):
            return f"HW precision={precision!r} does not match frozen precision {frozen['precision']!r}"
    return None


def _hardware_metadata_error(
    meta: Mapping[str, Any], experiment: str, frozen: Mapping[str, Any], expected_n: int
) -> str | None:
    job_ids = meta.get("job_ids")
    if not isinstance(job_ids, list) or not job_ids or any(
        not isinstance(job_id, str) or not job_id.strip() for job_id in job_ids
    ):
        return "HW run has no recorded job_ids"
    if not meta.get("backend_name"):
        return "HW run has no backend_name"
    if meta["backend_name"] != frozen["device"]:
        return f"HW backend_name={meta['backend_name']!r} does not match frozen device {frozen['device']!r}"

    layout = meta.get("layout")
    if (
        not isinstance(layout, list)
        or len(layout) != expected_n
        or any(not isinstance(qubit, int) or isinstance(qubit, bool) for qubit in layout)
        or len(set(layout)) != len(layout)
    ):
        return f"HW layout must contain {expected_n} unique physical-qubit indices"
    depth = meta.get("two_qubit_depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        return "HW two_qubit_depth metadata is missing or invalid"

    calibration = meta.get("calibration")
    if not isinstance(calibration, Mapping) or not {
        "last_update", "qubits", "two_qubit_errors"
    }.issubset(calibration):
        return "HW run has no calibration snapshot metadata"
    if not calibration.get("last_update"):
        return "HW calibration last_update metadata is missing"
    qubits = calibration.get("qubits")
    if not isinstance(qubits, Mapping):
        return "HW calibration.qubits metadata is missing or invalid"
    for physical_qubit in layout:
        snapshot = qubits.get(str(physical_qubit), qubits.get(physical_qubit))
        if not isinstance(snapshot, Mapping) or not {
            "t1", "t2", "readout_error"
        }.issubset(snapshot):
            return f"HW calibration is missing T1/T2/readout metadata for physical qubit {physical_qubit}"
    if not isinstance(calibration.get("two_qubit_errors"), Mapping):
        return "HW calibration.two_qubit_errors metadata is missing or invalid"

    try:
        usage = float(meta["usage_s"])
        estimate = float(meta["estimated_qpu_s"])
    except (KeyError, TypeError, ValueError):
        return "HW usage_s or estimated_qpu_s metadata is missing or invalid"
    if not math.isfinite(usage) or usage < 0 or not math.isfinite(estimate) or estimate <= 0:
        return "HW usage_s must be finite and non-negative and estimated_qpu_s finite and positive"

    mitigation = meta.get("mitigation")
    expected_mitigation = experiment != "control"
    mitigation_fields = ("measure_mitigation", "twirling", "dynamical_decoupling")
    if not isinstance(mitigation, Mapping) or any(
        not isinstance(mitigation.get(field), bool)
        or mitigation[field] is not expected_mitigation
        for field in mitigation_fields
    ):
        mode = "enabled" if expected_mitigation else "disabled"
        return f"HW mitigation metadata does not record the H-02 {mode} setting for {experiment}"
    return None


def _bloch_data_error(B: np.ndarray, sample_count: int, expected_n: int) -> str | None:
    """Validate the L-02 Bloch-array boundary without converting its dtype."""
    if B.ndim != 3 or B.shape != (sample_count, expected_n, 3):
        return f"Bloch array shape {B.shape} does not match ({sample_count}, {expected_n}, 3)"
    if B.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        return "Bloch array dtype must be float32 or float64 per L-02"
    if not np.isfinite(B).all():
        return "Bloch array contains NaN or infinite values"
    if np.any(B < -1.0) or np.any(B > 1.0):
        return "Bloch values must be in [-1, 1] per L-02"
    return None


def load_store_runs(store_root: str | Path = DEFAULT_ROOT) -> tuple[dict[str, tuple[list[str], np.ndarray, dict]], list[dict[str, str]]]:
    """Load run records through I-04 and retain read/metadata failures."""
    root = Path(store_root)
    runs: dict[str, tuple[list[str], np.ndarray, dict]] = {}
    errors: list[dict[str, str]] = []
    if not root.exists():
        return runs, [{"run_id": "", "reason": f"store path does not exist: {root}"}]
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            result = read_bloch(entry.name, root=root)
            if not isinstance(result, tuple) or len(result) != 3:
                raise E05Error("I-04 did not return (ids, B, meta).")
            ids, B, meta = result
            if not isinstance(meta, dict):
                raise E05Error("I-04 metadata is not a dictionary.")
            if meta.get("run_id") not in (None, entry.name):
                raise E05Error("store directory and metadata run_id disagree")
            if meta.get("tier") not in ("L0", "L1", "L2", "HW"):
                raise E05Error("run metadata has no recognized tier")
            if not isinstance(ids, list):
                raise E05Error("I-04 ids must be a list of strings")
            if any(not isinstance(sample_id, str) for sample_id in ids):
                raise E05Error("I-04 ids contain a value that is not a string")
            if any(not sample_id.strip() for sample_id in ids):
                raise E05Error("I-04 ids must not be empty or whitespace-only")
            if len(set(ids)) != len(ids):
                raise E05Error("I-04 ids must be unique for E-05 matching")
            bloch = np.asarray(B)
            if bloch.ndim == 0 or len(ids) != bloch.shape[0]:
                raise E05Error(
                    f"I-04 returned {len(ids)} IDs for Bloch leading dimension "
                    f"{0 if bloch.ndim == 0 else bloch.shape[0]}"
                )
            runs[entry.name] = (ids.copy(), bloch, meta)
        except Exception as error:  # retain malformed runs in the report
            errors.append({"run_id": entry.name, "reason": f"{type(error).__name__}: {error}"})
    return runs, errors


def _validate_campaign(
    campaign_id: str,
    grouped_runs: Mapping[tuple[str, str], list[tuple[str, tuple[list[str], np.ndarray, dict]]]],
    expected: Mapping[str, list[str]],
    frozen: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for (experiment, tier), records in sorted(grouped_runs.items()):
        if len(records) > 1:
            duplicate_ids = sorted(run_id for run_id, _ in records)
            errors.append(
                f"{campaign_id}: duplicate runs for experiment={experiment}, tier={tier}: "
                f"{', '.join(duplicate_ids)}"
            )
    for experiment, expected_ids in expected.items():
        pair = {}
        for tier in EXPECTED_TIERS:
            found = grouped_runs.get((experiment, tier), [])
            if not found:
                errors.append(f"{experiment}/{tier}: run missing")
                continue
            if len(found) != 1:
                continue
            run_id, run = found[0]
            ids, B, meta = run
            if meta.get("run_id") != run_id:
                errors.append(f"{run_id}: metadata run_id is missing or disagrees with the store directory")
            if meta.get("tier") != tier:
                errors.append(f"{run_id}: metadata tier does not match the run ID tier {tier}")
            if not _same_ids(ids, expected_ids):
                errors.append(f"{run_id}: IDs do not match the D-05 {experiment} subset")
            expected_n = int(experiment.rsplit("n", 1)[1]) if experiment.startswith("concentration-n") else int(frozen["n_qubits"])
            bloch_error = _bloch_data_error(B, len(ids), expected_n)
            if bloch_error:
                errors.append(f"{run_id}: {bloch_error}")
            config_error = _configuration_error(meta, experiment, frozen, tier)
            if config_error:
                errors.append(f"{run_id}: {config_error}")
            if meta.get("experiment") is not None and meta.get("experiment") != experiment:
                errors.append(f"{run_id}: metadata experiment label conflicts with run_id")
            if meta.get("campaign_id") is not None and meta.get("campaign_id") != campaign_id:
                errors.append(f"{run_id}: metadata campaign_id conflicts with run_id")
            if tier == "HW":
                hw_error = _hardware_metadata_error(meta, experiment, frozen, expected_n)
                if hw_error:
                    errors.append(f"{run_id}: {hw_error}")
            pair[tier] = (run_id, run)
        if len(pair) == 2:
            ids_l0, _, meta_l0 = pair["L0"][1]
            ids_hw, _, meta_hw = pair["HW"][1]
            if not _same_ids(ids_l0, ids_hw):
                errors.append(f"{experiment}: L0 and HW IDs differ")
            cfg_l0, cfg_hw = meta_l0.get("embed_config"), meta_hw.get("embed_config")
            if isinstance(cfg_l0, Mapping) and isinstance(cfg_hw, Mapping):
                for key in ("n_qubits", "layers", "s", "entangle"):
                    if cfg_l0.get(key) != cfg_hw.get(key):
                        errors.append(f"{experiment}: L0 and HW embed_config.{key} differs")
    return errors


def _campaign_source(
    campaign_id: str,
    grouped_runs: Mapping[
        tuple[str, str], list[tuple[str, tuple[list[str], np.ndarray, dict]]]
    ],
    expected: Mapping[str, list[str]],
    frozen: Mapping[str, Any],
) -> tuple[str, list[str], list[str]]:
    """Classify HW provenance without treating an absent marker as hardware."""
    recognized: set[str] = set()
    errors: list[str] = []
    details: list[str] = []
    for experiment in sorted(expected):
        expected_n = (
            int(experiment.rsplit("n", 1)[1])
            if experiment.startswith("concentration-n")
            else int(frozen["n_qubits"])
        )
        records = sorted(
            grouped_runs.get((experiment, "HW"), []),
            key=lambda item: (
                item[0],
                json.dumps(item[1][2], sort_keys=True, default=repr, separators=(",", ":")),
            ),
        )
        for run_id, run in records:
            meta = run[2]
            if "source_kind" in meta:
                source_kind = meta["source_kind"]
                if not isinstance(source_kind, str) or source_kind not in {
                    "hardware", "synthetic_fixture"
                }:
                    errors.append(
                        f"{run_id}: HW source_kind {source_kind!r} is not a recognized provenance value"
                    )
                    details.append(f"{run_id}: invalid source_kind")
                    continue
                recognized.add(source_kind)
                details.append(f"{run_id}: explicitly marked {source_kind}")
                continue

            metadata_error = _hardware_metadata_error(meta, experiment, frozen, expected_n)
            if metadata_error is None:
                recognized.add("hardware")
                details.append(
                    f"{run_id}: source_kind absent; hardware inferred from complete H-02 metadata"
                )
            else:
                errors.append(
                    f"{run_id}: provenance is unverified because source_kind is absent and "
                    f"required H-02 metadata is incomplete ({metadata_error})"
                )
                details.append(f"{run_id}: unverified provenance")

    if {"hardware", "synthetic_fixture"}.issubset(recognized):
        errors.append(
            f"{campaign_id}: campaign has mixed hardware and synthetic_fixture provenance"
        )
        return "mixed_provenance", errors, details
    if errors and any("not a recognized provenance value" in error for error in errors):
        return "invalid_provenance", errors, details
    if errors:
        return "unverified", errors, details
    if recognized == {"hardware"}:
        return "hardware", errors, details
    if recognized == {"synthetic_fixture"}:
        return "synthetic_fixture", errors, details
    return "unverified", [f"{campaign_id}: HW campaign provenance could not be established"], details


def discover_h02_campaign(
    runs: Mapping[str, tuple[list[str], np.ndarray, dict]],
    subsets: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    campaign_id: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Find a complete H-02 campaign by run metadata plus D-05 ID manifests."""
    expected = _expected_experiments(subsets, frozen)
    grouped: dict[str, dict[tuple[str, str], list[tuple[str, tuple[list[str], np.ndarray, dict]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run_id, run in sorted(runs.items(), key=lambda item: item[0]):
        match = RUN_PATTERN.fullmatch(run_id)
        if not match:
            continue
        prefix = match.group("campaign")
        experiment, tier = match.group("experiment"), match.group("tier")
        key = (experiment, tier)
        grouped[prefix][key].append((run_id, run))

    candidates: list[dict[str, Any]] = []
    for prefix, group in sorted(grouped.items()):
        if campaign_id is not None and prefix != campaign_id:
            continue
        errors = _validate_campaign(prefix, group, expected, frozen)
        source, source_errors, source_details = _campaign_source(prefix, group, expected, frozen)
        errors.extend(source_errors)
        candidates.append({
            "campaign_id": prefix,
            "source": source,
            "source_details": source_details,
            "run_ids": sorted(run_id for records in group.values() for run_id, _ in records),
            "valid": not errors,
            "errors": errors,
            "experiments": sorted(expected),
        })
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        return None, candidates
    selected = valid[-1]
    prefix = selected["campaign_id"]
    selected["runs"] = {
        experiment: {
            tier: group[(experiment, tier)][0][1]
            for tier in EXPECTED_TIERS
        }
        for experiment in expected
    }
    selected["expected_ids"] = expected
    return selected, candidates


def _load_g03() -> tuple[Any | None, str | None]:
    """Return the statically imported G-03 module or its explicit absence."""
    if _g03_module is None:
        return None, f"G-03 diagnostics.compare is unavailable in this checkout: {_g03_import_error}"
    return _g03_module, None


def _validate_gamma(frozen: Mapping[str, Any]) -> tuple[float | None, str | None, str | None]:
    frozen_gamma = frozen.get("gamma")
    if frozen_gamma is not None:
        try:
            frozen_gamma = float(frozen_gamma)
        except (TypeError, ValueError) as error:
            raise E05Error("Frozen gamma must be a finite number greater than zero.") from error
        if not math.isfinite(frozen_gamma) or frozen_gamma <= 0:
            raise E05Error("Frozen gamma must be finite and greater than zero.")
        return frozen_gamma, "frozen_config", None
    return None, None, (
        "gamma has no approved frozen value or official training-selection artifact; "
        "E-05 will not select it from hardware evaluation data"
    )


def _concentration(K: np.ndarray) -> dict[str, Any]:
    values = off_diagonal_values(K)
    return {
        "off_diagonal_count": int(values.size),
        "off_diagonal_variance": off_diagonal_variance(K),
        "eigen_spectrum": [float(value) for value in eigen_spectrum(K)],
        "effective_rank": effective_rank(K),
    }


def _projected(run: tuple[list[str], np.ndarray, dict], gamma: float) -> tuple[list[str], np.ndarray]:
    ids, B, _ = run
    return ids, projected_kernel(B, B, gamma)


def analyze_campaign(
    campaign: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    g03: Any | None = None,
) -> dict[str, Any]:
    """Calculate E-05 metrics using G-03, G-01 and M-01 public functions."""
    chosen_gamma, gamma_source, gamma_block = _validate_gamma(frozen)
    if g03 is None:
        g03, _ = _load_g03()
    groups: dict[str, Any] = {}
    metric_errors: list[str] = []
    for experiment, tiers in campaign["runs"].items():
        run_l0, run_hw = tiers["L0"], tiers["HW"]
        ids_l0, B_l0, _ = run_l0
        ids_hw, B_hw, _ = run_hw
        group_result: dict[str, Any] = {"ids_count": len(ids_hw), "kernel_alignment": None}
        kernels: dict[str, tuple[list[str], np.ndarray]] = {}
        if chosen_gamma is not None:
            kernel_l0 = _projected(run_l0, chosen_gamma)
            kernel_hw = _projected(run_hw, chosen_gamma)
            kernels = {"L0": kernel_l0, "HW": kernel_hw}
            group_result["concentration"] = {
                tier: _concentration(matrix)
                for tier, (_, matrix) in kernels.items()
            }
        else:
            group_result["concentration"] = None

        if g03 is not None:
            try:
                group_result["bloch_error"] = g03.bloch_frobenius_error(ids_l0, B_l0, ids_hw, B_hw)
                group_result["comparison"] = g03.compare_runs(
                    run_l0,
                    run_hw,
                    tier_a="L0",
                    tier_b="HW",
                    kernel_a=kernels.get("L0"),
                    kernel_b=kernels.get("HW"),
                )
                if chosen_gamma is not None:
                    group_result["kernel_alignment"] = g03.kernel_alignment(
                        kernels["L0"][1], kernels["HW"][1], kernels["L0"][0], kernels["HW"][0]
                    )
            except (AttributeError, ValueError, TypeError) as error:
                metric_errors.append(f"{experiment}: G-03 comparison failed: {error}")
                group_result["bloch_error"] = None
                group_result["comparison"] = None
                group_result["kernel_alignment"] = None
        else:
            group_result["bloch_error"] = None
            group_result["comparison"] = None
            group_result["kernel_alignment"] = None
        groups[experiment] = group_result

    drift = None
    if g03 is not None:
        try:
            start = campaign["runs"]["drift-start"]["HW"]
            end = campaign["runs"]["drift-end"]["HW"]
            kwargs: dict[str, Any] = {}
            if chosen_gamma is not None:
                kwargs["kernel_start"] = _projected(start, chosen_gamma)
                kwargs["kernel_end"] = _projected(end, chosen_gamma)
            drift = g03.drift_report(start, end, **kwargs)
        except (KeyError, AttributeError, ValueError, TypeError) as error:
            metric_errors.append(f"drift: G-03 drift_report failed: {error}")

    return {
        "campaign_id": campaign["campaign_id"],
        "source": campaign["source"],
        "source_details": campaign.get("source_details", []),
        "gamma": {"value": chosen_gamma, "source": gamma_source, "blocked_reason": gamma_block},
        "g03_available": g03 is not None,
        "groups": groups,
        "drift": drift,
        "metric_errors": metric_errors,
    }


def build_report(
    *,
    store_root: str | Path = DEFAULT_ROOT,
    config_path: str | Path = CONFIG_PATH,
    campaign_id: str | None = None,
    subsets: Mapping[str, Any] | None = None,
    frozen_config: Mapping[str, Any] | None = None,
    g03: Any | None = None,
) -> dict[str, Any]:
    """Load prerequisites, find a valid H-02 campaign and report available work."""
    blockers: list[str] = []
    injected_inputs = subsets is not None or frozen_config is not None
    repository_root = Path(config_path).resolve().parent.parent
    if subsets is None:
        try:
            # Validate the frozen bytes against the live tag as well as the
            # strict D-05 lifecycle state; a recorded tag name alone can move.
            subsets = load_hardware_subsets(freeze_repo=repository_root)
        except Exception as error:
            subsets = None
            blockers.append(f"D-05 subsets are not approved/usable: {type(error).__name__}: {error}")
    else:
        blockers.append("D-05 subset object was injected; this report is a fixture run, not hardware validation")
    if subsets is not None and subsets.get("status") != "bound_to_hw_freeze":
        blockers.append(f"D-05 lifecycle state is {subsets.get('status')!r}, not bound_to_hw_freeze")

    try:
        frozen = dict(frozen_config) if frozen_config is not None else read_frozen_config(config_path)
    except Exception as error:
        frozen = {}
        blockers.append(f"Frozen hardware configuration could not be loaded: {type(error).__name__}: {error}")
    chosen_gamma, gamma_source, gamma_block = (None, None, "frozen config is unavailable")
    if frozen:
        chosen_gamma, gamma_source, gamma_block = _validate_gamma(frozen)
    if gamma_block:
        blockers.append(gamma_block)

    if injected_inputs:
        l05_approval, l05_approval_detail = "not_assessed_fixture", "L-05 approval is not assessed for injected test fixtures"
        d05_approval, d05_approval_detail = "not_assessed_fixture", "D-05 approval is not assessed for injected test fixtures"
    else:
        l05_approval, l05_approval_detail = inspect_l05_approval(repository_root)
        d05_approval, d05_approval_detail = inspect_d05_approval(repository_root)
        if l05_approval != "documented_approved":
            blockers.append(l05_approval_detail)
        if d05_approval != "documented_approved":
            blockers.append(d05_approval_detail)

    runs, store_errors = load_store_runs(store_root)
    campaign = None
    candidates: list[dict[str, Any]] = []
    analysis = None
    if subsets is not None and frozen:
        try:
            campaign, candidates = discover_h02_campaign(
                runs, subsets, frozen, campaign_id=campaign_id
            )
        except (KeyError, TypeError, ValueError, E05Error) as error:
            blockers.append(f"H-02 campaign discovery could not run: {error}")
    if campaign is None:
        blockers.append("No complete H-02 campaign matching D-05 IDs and frozen embedding metadata is available")
        for candidate in candidates:
            blockers.extend(
                f"{candidate['campaign_id']} candidate rejected: {reason}"
                for reason in candidate.get("errors", [])
            )
        if any("hw-" in run_id and run_id.endswith("-HW") for run_id in runs):
            blockers.append("Pilot-like HW runs were found, but do not match the H-02 campaign run manifest and were not substituted")
    else:
        approvals_ready = d05_approval == "documented_approved" and l05_approval == "documented_approved"
        if campaign["source"] != "synthetic_fixture" and (not approvals_ready or injected_inputs):
            blockers.append("D-05/L-05 approvals are not both explicitly documented; E-05 did not analyze candidate hardware runs")
        elif campaign["source"] == "synthetic_fixture":
            blockers.append("The selected campaign is explicitly marked synthetic_fixture; it is not hardware validation")
        if campaign["source"] == "synthetic_fixture" or (approvals_ready and not injected_inputs):
            g03_module = g03
            g03_error = None
            if g03_module is None:
                g03_module, g03_error = _load_g03()
            if g03_error:
                blockers.append(g03_error)
            analysis = analyze_campaign(campaign, frozen, g03=g03_module)
            if analysis["metric_errors"]:
                blockers.extend(analysis["metric_errors"])

    g03_module, g03_error = (g03, None) if g03 is not None else _load_g03()
    if g03_error and g03_error not in blockers:
        blockers.append(g03_error)
    if analysis is None:
        gamma_state = {"value": chosen_gamma, "source": gamma_source, "blocked_reason": gamma_block}
    else:
        gamma_state = analysis["gamma"]

    return {
        "schema": "q-spy-e05-report-v1",
        "task": "E-05",
        "status": "complete" if campaign and not blockers and analysis else "blocked_by_prerequisites",
        "hardware_validation": "complete" if campaign and campaign["source"] == "hardware" and not blockers and analysis else "not_validated",
        "provenance": {
            "store_root": str(Path(store_root)),
            "d05_status": None if subsets is None else subsets.get("status"),
            "d05_selection_sha256": None if subsets is None else subsets.get("selection_sha256"),
            "l05_tag": (subsets or {}).get("l05_freeze", {}).get("tag"),
            "l05_approval": l05_approval,
            "l05_approval_detail": l05_approval_detail,
            "d05_approval": d05_approval,
            "d05_approval_detail": d05_approval_detail,
            "campaign_id": None if campaign is None else campaign["campaign_id"],
            "campaign_source": None if campaign is None else campaign["source"],
            "gamma": gamma_state,
            "g03_available": g03_module is not None,
        },
        "frozen_config": {
            key: frozen.get(key)
            for key in ("n_qubits", "layers", "s", "precision", "concentration_qubits", "device", "gamma")
            if key in frozen
        },
        "campaign_candidates": [
            {key: value for key, value in candidate.items() if key != "runs"}
            for candidate in candidates
        ],
        "store_runs_seen": sorted(runs),
        "store_errors": store_errors,
        "analysis": analysis,
        "blockers": list(dict.fromkeys(blockers)),
        "figures_generated": [],
        "figures_not_generated": [],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_outputs(
    report: dict[str, Any],
    output_root: str | Path = "results/e05/runs",
    *,
    run_id: str | None = None,
) -> Path:
    """Write an isolated analysis JSON, generated figures and one-page summary."""
    from experiments.e05.figures import render_figures
    from experiments.e05.summary import render_summary

    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", identifier):
        raise E05Error("run_id must be a simple filename-safe identifier.")
    run_dir = Path(output_root) / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    safe_report = _json_safe(report)
    figures = render_figures(safe_report.get("analysis"), run_dir / "figures")
    safe_report["figures_generated"] = [f"figures/{path.name}" for path in figures]
    safe_report["figures_not_generated"] = _figure_absences(safe_report.get("analysis"), figures)
    (run_dir / "analysis.json").write_text(
        json.dumps(safe_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(render_summary(safe_report), encoding="utf-8")
    return run_dir


def _figure_absences(analysis: Mapping[str, Any] | None, figures: list[Path]) -> list[dict[str, str]]:
    names = {path.name for path in figures}
    required = {
        "bloch_rmse_per_qubit.svg": "Matched H-02 HW/L0 Bloch errors from G-03 are unavailable.",
        "kernel_alignment.svg": "Matched H-02 kernels, an approved frozen gamma and G-03 alignment are unavailable.",
        "concentration_variance.svg": "Matched H-02 runs and an approved frozen gamma are unavailable.",
        "effective_rank.svg": "Matched H-02 runs and an approved frozen gamma are unavailable.",
        "drift_rmse_per_qubit.svg": "H-02 drift-start/end references and G-03 drift_report are unavailable.",
    }
    groups = (analysis or {}).get("groups", {})
    if groups:
        for experiment in groups:
            required[f"eigen_spectrum_{experiment}.svg"] = (
                f"The {experiment} projected-kernel spectrum needs matched H-02 runs and an approved frozen gamma."
            )
    else:
        required["eigen_spectrum_<experiment>.svg"] = (
            "No H-02 group has matched runs and an approved frozen gamma for its projected-kernel spectrum."
        )
    return [
        {"file": name, "reason": reason}
        for name, reason in required.items()
        if name not in names
    ]

