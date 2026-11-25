"""One-page Markdown summary for E-05 analysis reports."""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _metric_rows(analysis: dict[str, Any] | None) -> list[str]:
    if not analysis:
        return ["No campaign metrics were calculated."]
    rows: list[str] = []
    groups = analysis.get("groups", {})
    main = groups.get("main", {})
    bloch = main.get("bloch_error")
    if bloch:
        per_qubit = ", ".join(_fmt(float(value)) for value in bloch.get("rmse_per_qubit", []))
        rows.extend([
            f"- Main Bloch RMSE per qubit: `{per_qubit}` (dimensionless).",
            f"- Main global Bloch RMSE: `{_fmt(bloch.get('rmse_global'))}`; Frobenius norm: `{_fmt(bloch.get('frobenius_global'))}`.",
        ])
    else:
        rows.append("- Bloch errors: blocked because a valid H-02 campaign and G-03 are required.")
    alignment = main.get("kernel_alignment")
    rows.append(
        f"- Main centered normalized Frobenius kernel alignment: `{_fmt(alignment)}`."
        if alignment is not None
        else "- Kernel alignment: unavailable; it requires matched H-02 runs, gamma and G-03."
    )
    concentration_groups = []
    for name, result in groups.items():
        concentration = result.get("concentration")
        if not isinstance(concentration, dict):
            continue
        l0, hw = concentration["L0"], concentration["HW"]
        concentration_groups.append(
            f"`{name}` variance L0/HW `{_fmt(l0['off_diagonal_variance'])}` / `{_fmt(hw['off_diagonal_variance'])}`, "
            f"effective rank `{_fmt(l0['effective_rank'])}` / `{_fmt(hw['effective_rank'])}`"
        )
    rows.append(
        "- Concentration: " + "; ".join(concentration_groups) + "."
        if concentration_groups
        else "- Concentration metrics: unavailable; no matched H-02 kernel group and explicit gamma are available."
    )
    drift = analysis.get("drift")
    if isinstance(drift, dict):
        bloch_drift = drift.get("bloch_drift", {})
        rmse = ", ".join(_fmt(float(value)) for value in bloch_drift.get("rmse_per_qubit", []))
        rows.append(f"- Start/end drift RMSE per qubit: `{rmse}` (dimensionless); no pass/fail threshold applied.")
    else:
        rows.append("- Drift: unavailable; H-02 drift-start/end references and G-03 are required.")
    return rows


def render_summary(report: dict[str, Any]) -> str:
    """Return a compact summary containing only findings present in the report."""
    provenance = report.get("provenance", {})
    frozen = report.get("frozen_config", {})
    analysis = report.get("analysis")
    source = provenance.get("campaign_source") or "no H-02 campaign"
    if report.get("analysis") is None and provenance.get("campaign_id"):
        source = f"candidate {source} campaign; metrics not analyzed"
    embedding = (
        f"n={frozen.get('n_qubits', '—')}, L={frozen.get('layers', '—')}, "
        f"s={frozen.get('s', '—')}"
    )
    gamma = provenance.get("gamma", {})
    gamma_value = gamma.get("value")
    gamma_text = "not approved/frozen" if gamma_value is None else f"{gamma_value} ({gamma.get('source')})"
    figures = report.get("figures_generated", [])
    figure_text = ", ".join(f"`{path}`" for path in figures) if figures else "None; see figure blockers below."
    blockers = report.get("blockers", [])
    if not blockers:
        blocker_lines = ["- No prerequisite blockers recorded."]
    else:
        blocker_lines = [f"- {item}" for item in blockers]
    missing_figures = report.get("figures_not_generated", [])
    missing_lines = [f"- `{item['file']}`: {item['reason']}" for item in missing_figures]
    if not missing_lines:
        missing_lines = ["- None."]
    frozen_tag = provenance.get("l05_tag") or "not linked"
    selection = provenance.get("d05_selection_sha256") or "not available"
    return "\n".join([
        "# E-05 hardware diagnostics summary",
        "",
        f"**Status:** `{report.get('status')}` · **Hardware validation:** `{report.get('hardware_validation')}` · **Data source:** `{source}`",
        "",
        f"**Campaign:** `{provenance.get('campaign_id') or 'not available'}` · **Embedding freeze:** {embedding} · **D-05:** `{provenance.get('d05_status')}` · **D-05 approval:** `{provenance.get('d05_approval', 'unverified')}` · **L-05 tag:** `{frozen_tag}` · **L-05 approval:** `{provenance.get('l05_approval', 'unverified')}`",
        f"**D-05 selection SHA-256:** `{selection}` · **Gamma:** `{gamma_text}`",
        "",
        "## Diagnostics",
        "",
        *_metric_rows(analysis),
        "",
        f"**Generated figures:** {figure_text}",
        "",
        "## Missing figures and limits",
        "",
        *missing_lines,
        *([""] if blockers else []),
        *(["**Blocked prerequisites:**", "", *blocker_lines] if blockers else []),
        "",
        "H-01 pilot data are not substituted for H-02. Synthetic fixture results, if used in tests, are not hardware results. No QPU job was submitted by E-05.",
        "",
    ])

