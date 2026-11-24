"""Small dependency-free SVG figures for E-05 result dictionaries."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any


WIDTH = 920
HEIGHT = 470
LEFT = 82
RIGHT = 28
TOP = 72
BOTTOM = 102


def _text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#152238">{html.escape(value)}</text>'
    )


def _base(title: str, subtitle: str, y_label: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _text(LEFT, 32, title, size=20, anchor="start", weight="bold"),
        _text(LEFT, 54, subtitle, size=12, anchor="start"),
        f'<path d="M {LEFT} {TOP} V {HEIGHT - BOTTOM} H {WIDTH - RIGHT}" fill="none" stroke="#53657d" stroke-width="1.2"/>',
        f'<text transform="translate(22 {(TOP + HEIGHT - BOTTOM) / 2:.1f}) rotate(-90)" font-family="Arial, sans-serif" font-size="13px" text-anchor="middle" fill="#152238">{html.escape(y_label)}</text>',
    ]


def _format(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.3g}"


def _bar_chart(
    title: str,
    subtitle: str,
    y_label: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    file: Path,
) -> None:
    values = [float(value) for _, data, _ in series for value in data]
    if not values or not labels or any(not math.isfinite(value) for value in values):
        return
    if any(len(data) != len(labels) for _, data, _ in series):
        return
    minimum = min(0.0, min(values))
    maximum = max(values)
    if maximum == minimum:
        maximum = minimum + 1.0
    span = maximum - minimum
    plot_top, plot_bottom = TOP, HEIGHT - BOTTOM
    plot_height = plot_bottom - plot_top
    zero_y = plot_bottom - (0.0 - minimum) / span * plot_height
    chunks = _base(title, subtitle, y_label)
    chunks.append(f'<path d="M {LEFT} {zero_y:.1f} H {WIDTH - RIGHT}" stroke="#9aa8b8" stroke-width="1"/>')
    for index in range(5):
        value = minimum + span * index / 4
        y = plot_bottom - (value - minimum) / span * plot_height
        chunks.append(f'<path d="M {LEFT - 5} {y:.1f} H {WIDTH - RIGHT}" stroke="#e5eaf0" stroke-width="1"/>')
        chunks.append(_text(LEFT - 10, y + 4, _format(value), size=10, anchor="end"))
    group_width = (WIDTH - RIGHT - LEFT) / len(labels)
    bar_group_width = group_width * 0.72
    bar_width = bar_group_width / max(1, len(series))
    palette = [color for _, _, color in series]
    for label_index, label in enumerate(labels):
        group_start = LEFT + group_width * label_index + (group_width - bar_group_width) / 2
        for series_index, (series_name, data, _) in enumerate(series):
            value = float(data[label_index])
            y_value = plot_bottom - (value - minimum) / span * plot_height
            x = group_start + series_index * bar_width + bar_width * 0.12
            width = bar_width * 0.76
            y = min(y_value, zero_y)
            height = max(1.2, abs(zero_y - y_value))
            chunks.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="2" fill="{palette[series_index]}"/>')
            chunks.append(_text(x + width / 2, y - 5 if value >= 0 else y + height + 14, _format(value), size=9))
        label_y = plot_bottom + 22
        chunks.append(_text(LEFT + group_width * (label_index + 0.5), label_y, label, size=10))
    legend_y = HEIGHT - 35
    legend_x = LEFT
    for name, _, color in series:
        chunks.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" rx="2" fill="{color}"/>')
        chunks.append(_text(legend_x + 18, legend_y, name, size=11, anchor="start"))
        legend_x += 130
    chunks.append('</svg>')
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _spectrum_chart(experiment: str, concentration: dict[str, Any], file: Path) -> None:
    spectra = [(tier, item.get("eigen_spectrum"), color) for tier, item, color in (
        ("L0", concentration.get("L0", {}), "#2b6cb0"),
        ("HW", concentration.get("HW", {}), "#c05621"),
    )]
    spectra = [(name, values, color) for name, values, color in spectra if isinstance(values, list) and values]
    if not spectra:
        return
    longest = max(len(values) for _, values, _ in spectra)
    peak = max((max(values) for _, values, _ in spectra), default=0.0)
    if peak <= 0 or not math.isfinite(peak):
        return
    plot_bottom = HEIGHT - BOTTOM
    plot_height = plot_bottom - TOP
    plot_width = WIDTH - RIGHT - LEFT
    chunks = _base(
        f"Kernel eigen-spectrum — {experiment}",
        "Projected kernel from stored Bloch vectors; HW and L0 use the same D-05 IDs.",
        "Eigenvalue (dimensionless)",
    )
    for tick in range(5):
        value = peak * tick / 4
        y = plot_bottom - value / peak * plot_height
        chunks.append(f'<path d="M {LEFT} {y:.1f} H {WIDTH - RIGHT}" stroke="#e5eaf0" stroke-width="1"/>')
        chunks.append(_text(LEFT - 10, y + 4, _format(value), size=10, anchor="end"))
    for name, values, color in spectra:
        points = []
        for index, value in enumerate(values):
            x = LEFT + (index / max(1, longest - 1)) * plot_width
            y = plot_bottom - float(value) / peak * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        chunks.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    chunks.append(_text((LEFT + WIDTH - RIGHT) / 2, HEIGHT - 56, "Eigenvalue rank", size=12))
    legend_x = LEFT
    for name, _, color in spectra:
        chunks.append(f'<path d="M {legend_x} {HEIGHT - 32} h 18" stroke="{color}" stroke-width="3"/>')
        chunks.append(_text(legend_x + 25, HEIGHT - 28, name, size=11, anchor="start"))
        legend_x += 90
    chunks.append('</svg>')
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def render_figures(analysis: dict[str, Any] | None, output_dir: str | Path) -> list[Path]:
    """Generate only figures supported by metrics present in the report."""
    if not analysis:
        return []
    root = Path(output_dir)
    files: list[Path] = []
    groups = analysis.get("groups", {})

    main = groups.get("main", {})
    bloch = main.get("bloch_error") or {}
    rmse = bloch.get("rmse_per_qubit")
    if isinstance(rmse, list) and rmse:
        target = root / "bloch_rmse_per_qubit.svg"
        _bar_chart(
            "Bloch RMSE per qubit — HW vs L0",
            f"Campaign {analysis.get('campaign_id', 'unknown')}; same samples matched by ID.",
            "RMSE (dimensionless Bloch expectation)",
            [str(index) for index in range(len(rmse))],
            [("HW vs L0", [float(value) for value in rmse], "#2b6cb0")],
            file=target,
        )
        if target.exists():
            files.append(target)

    alignment_labels, alignment_values = [], []
    variance_labels, variance_l0, variance_hw = [], [], []
    rank_labels, rank_l0, rank_hw = [], [], []
    gamma_info = analysis.get("gamma", {})
    gamma = gamma_info.get("value") if isinstance(gamma_info, dict) else None
    gamma_caption = f"gamma={gamma}" if gamma is not None else "gamma unavailable"
    for experiment, result in groups.items():
        alignment = result.get("kernel_alignment")
        if alignment is not None:
            alignment_labels.append(experiment)
            alignment_values.append(float(alignment))
        concentration = result.get("concentration")
        if isinstance(concentration, dict) and all(tier in concentration for tier in ("L0", "HW")):
            variance_labels.append(experiment)
            variance_l0.append(float(concentration["L0"]["off_diagonal_variance"]))
            variance_hw.append(float(concentration["HW"]["off_diagonal_variance"]))
            rank_labels.append(experiment)
            rank_l0.append(float(concentration["L0"]["effective_rank"]))
            rank_hw.append(float(concentration["HW"]["effective_rank"]))
            target = root / f"eigen_spectrum_{experiment}.svg"
            _spectrum_chart(experiment, concentration, target)
            if target.exists():
                files.append(target)

    if alignment_labels:
        target = root / "kernel_alignment.svg"
        _bar_chart(
            "Hardware vs ideal kernel alignment",
            f"Centered normalized Frobenius alignment; dimensionless; {gamma_caption}.",
            "Alignment (dimensionless)",
            alignment_labels,
            [("HW / L0", alignment_values, "#805ad5")],
            file=target,
        )
        if target.exists():
            files.append(target)
    if variance_labels:
        target = root / "concentration_variance.svg"
        _bar_chart(
            "Projected-kernel off-diagonal variance",
            f"Population variance (ddof=0); same D-05 sample IDs; {gamma_caption}.",
            "Variance (dimensionless)",
            variance_labels,
            [("L0", variance_l0, "#2b6cb0"), ("HW", variance_hw, "#c05621")],
            file=target,
        )
        if target.exists():
            files.append(target)
        target = root / "effective_rank.svg"
        _bar_chart(
            "Projected-kernel effective rank",
            f"Entropy effective rank of the G-01 eigen-spectrum; same D-05 sample IDs; {gamma_caption}.",
            "Effective rank (dimensionless)",
            rank_labels,
            [("L0", rank_l0, "#2b6cb0"), ("HW", rank_hw, "#c05621")],
            file=target,
        )
        if target.exists():
            files.append(target)

    drift = analysis.get("drift")
    drift_metrics = (drift or {}).get("bloch_drift", {}) if isinstance(drift, dict) else {}
    drift_rmse = drift_metrics.get("rmse_per_qubit") if isinstance(drift_metrics, dict) else None
    if isinstance(drift_rmse, list) and drift_rmse:
        target = root / "drift_rmse_per_qubit.svg"
        _bar_chart(
            "Bloch drift by qubit — campaign start vs end",
            "Same D-05 reference IDs; values are descriptive and use no acceptance threshold.",
            "RMSE (dimensionless Bloch expectation)",
            [str(index) for index in range(len(drift_rmse))],
            [("Start / end", [float(value) for value in drift_rmse], "#dd6b20")],
            file=target,
        )
        if target.exists():
            files.append(target)
    return files

