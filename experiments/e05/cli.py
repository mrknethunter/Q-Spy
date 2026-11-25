"""Command line entry points for E-05 analysis, figure and summary generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.e05.analysis import (
    CONFIG_PATH,
    DEFAULT_ROOT,
    _figure_absences,
    _json_safe,
    build_report,
    write_outputs,
)
from experiments.e05.figures import render_figures
from experiments.e05.summary import render_summary


def _read_report(run_dir: Path) -> dict:
    path = run_dir / "analysis.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "q-spy-e05-report-v1":
        raise ValueError(f"Not an E-05 report: {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze stored Q-Spy H-02 runs for E-05; this command never submits QPU jobs.")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="analyze the store and generate available figures and summary")
    analyze.add_argument("--store-root", type=Path, default=DEFAULT_ROOT)
    analyze.add_argument("--config", type=Path, default=CONFIG_PATH)
    analyze.add_argument("--campaign-id", help="select an exact H-02 campaign run prefix")
    analyze.add_argument("--output-root", type=Path, default=Path("results/e05/runs"))
    analyze.add_argument("--run-id", help="stable output directory name; must not already exist")

    figures = commands.add_parser("figures", help="regenerate script-produced SVG figures from a saved E-05 report")
    figures.add_argument("run_dir", type=Path)

    summary = commands.add_parser("summary", help="regenerate the one-page Markdown summary from a saved E-05 report")
    summary.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "analyze":
        report = build_report(
            store_root=args.store_root,
            config_path=args.config,
            campaign_id=args.campaign_id,
        )
        output = write_outputs(report, args.output_root, run_id=args.run_id)
        print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))
        return 0

    run_dir = args.run_dir
    report = _read_report(run_dir)
    if args.command == "figures":
        generated = render_figures(report.get("analysis"), run_dir / "figures")
        report["figures_generated"] = [f"figures/{path.name}" for path in generated]
        report["figures_not_generated"] = _figure_absences(report.get("analysis"), generated)
        (run_dir / "analysis.json").write_text(
            json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (run_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
        print(f"Generated {len(generated)} figure(s) in {run_dir / 'figures'}")
        return 0
    if args.command == "summary":
        target = run_dir / "summary.md"
        target.write_text(render_summary(report), encoding="utf-8")
        print(f"Generated {target}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

