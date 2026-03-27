import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "comparisons"


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_report_paths(reports_dir: Path) -> tuple[Path, Path]:
    report_paths = sorted(
        path for path in reports_dir.glob("agent_eval_*.json") if path.is_file()
    )
    if len(report_paths) < 2:
        raise ValueError("Need at least two JSON eval reports to compare.")
    return report_paths[-2], report_paths[-1]


def compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    baseline_results = {item["name"]: item for item in baseline["results"]}
    current_results = {item["name"]: item for item in current["results"]}

    shared_names = sorted(set(baseline_results) & set(current_results))
    new_cases = sorted(set(current_results) - set(baseline_results))
    removed_cases = sorted(set(baseline_results) - set(current_results))

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    unchanged: list[str] = []

    for name in shared_names:
        before = baseline_results[name]
        after = current_results[name]
        if before["passed"] and not after["passed"]:
            regressions.append(
                {
                    "name": name,
                    "baseline_passed": before["passed"],
                    "current_passed": after["passed"],
                    "baseline_status": before["status"],
                    "current_status": after["status"],
                    "baseline_selected_tool": before["selected_tool"],
                    "current_selected_tool": after["selected_tool"],
                }
            )
        elif not before["passed"] and after["passed"]:
            improvements.append(
                {
                    "name": name,
                    "baseline_passed": before["passed"],
                    "current_passed": after["passed"],
                    "baseline_status": before["status"],
                    "current_status": after["status"],
                    "baseline_selected_tool": before["selected_tool"],
                    "current_selected_tool": after["selected_tool"],
                }
            )
        else:
            unchanged.append(name)

    baseline_summary = baseline["summary"]
    current_summary = current["summary"]

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_report": baseline["metadata"]["generated_at_utc"],
            "current_report": current["metadata"]["generated_at_utc"],
            "baseline_git_commit": baseline["metadata"].get("git_commit"),
            "current_git_commit": current["metadata"].get("git_commit"),
        },
        "summary_delta": {
            "pass_rate_delta": round(current_summary["pass_rate"] - baseline_summary["pass_rate"], 2),
            "tool_selection_accuracy_delta": round(
                current_summary["tool_selection_accuracy"] - baseline_summary["tool_selection_accuracy"], 2
            ),
            "status_accuracy_delta": round(
                current_summary["status_accuracy"] - baseline_summary["status_accuracy"], 2
            ),
            "answer_match_rate_delta": round(
                current_summary["answer_match_rate"] - baseline_summary["answer_match_rate"], 2
            ),
        },
        "regressions": regressions,
        "improvements": improvements,
        "new_cases": new_cases,
        "removed_cases": removed_cases,
        "unchanged_cases": unchanged,
    }


def render_markdown_comparison(
    comparison: dict[str, Any], *, baseline_path: Path, current_path: Path
) -> str:
    summary_delta = comparison["summary_delta"]
    lines = [
        "# Agent Eval Comparison Report",
        "",
        f"- Generated at (UTC): `{comparison['metadata']['generated_at_utc']}`",
        f"- Baseline report: `{baseline_path.name}`",
        f"- Current report: `{current_path.name}`",
        f"- Baseline git commit: `{comparison['metadata']['baseline_git_commit'] or 'unknown'}`",
        f"- Current git commit: `{comparison['metadata']['current_git_commit'] or 'unknown'}`",
        "",
        "## Summary Delta",
        "",
        f"- Pass rate delta: `{summary_delta['pass_rate_delta']}%`",
        f"- Tool-selection accuracy delta: `{summary_delta['tool_selection_accuracy_delta']}%`",
        f"- Status accuracy delta: `{summary_delta['status_accuracy_delta']}%`",
        f"- Answer match rate delta: `{summary_delta['answer_match_rate_delta']}%`",
        "",
        "## Regressions",
        "",
    ]

    regressions = comparison["regressions"]
    if regressions:
        for item in regressions:
            lines.append(
                f"- `{item['name']}`: status `{item['baseline_status']}` -> `{item['current_status']}`, "
                f"tool `{item['baseline_selected_tool']}` -> `{item['current_selected_tool']}`"
            )
    else:
        lines.append("- `none`")

    lines.extend(["", "## Improvements", ""])
    improvements = comparison["improvements"]
    if improvements:
        for item in improvements:
            lines.append(
                f"- `{item['name']}`: status `{item['baseline_status']}` -> `{item['current_status']}`, "
                f"tool `{item['baseline_selected_tool']}` -> `{item['current_selected_tool']}`"
            )
    else:
        lines.append("- `none`")

    lines.extend(
        [
            "",
            f"- New cases: `{', '.join(comparison['new_cases']) if comparison['new_cases'] else 'none'}`",
            f"- Removed cases: `{', '.join(comparison['removed_cases']) if comparison['removed_cases'] else 'none'}`",
        ]
    )

    return "\n".join(lines)


def write_comparison_files(
    comparison: dict[str, Any], *, baseline_path: Path, current_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"agent_eval_compare_{timestamp}.json"
    md_path = output_dir / f"agent_eval_compare_{timestamp}.md"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown_comparison(comparison, baseline_path=baseline_path, current_path=current_path),
        encoding="utf-8",
    )
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two saved agent eval reports and detect regressions.")
    parser.add_argument("--baseline", default=None, help="Baseline JSON report path.")
    parser.add_argument("--current", default=None, help="Current JSON report path.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Directory containing saved JSON reports.")
    parser.add_argument("--save", action="store_true", help="Write JSON and Markdown comparison reports.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to store saved comparisons.")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    if args.baseline and args.current:
        baseline_path = Path(args.baseline)
        current_path = Path(args.current)
    else:
        baseline_path, current_path = find_latest_report_paths(reports_dir)

    baseline = load_report(baseline_path)
    current = load_report(current_path)
    comparison = compare_reports(baseline, current)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if args.save:
        output_dir = Path(args.output_dir)
        json_path, md_path = write_comparison_files(
            comparison, baseline_path=baseline_path, current_path=current_path, output_dir=output_dir
        )
        print(f"\nSaved comparison JSON report to: {json_path}")
        print(f"Saved comparison Markdown report to: {md_path}")


if __name__ == "__main__":
    main()
