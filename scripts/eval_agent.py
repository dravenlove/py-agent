import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import subprocess

# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "eval" / "agent_eval_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "reports"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_service import run_agent
from app.audit import agent_run_store
from app.memory import memory_store
from app.schemas import AgentRequest


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def seed_memory(case: dict[str, Any]) -> None:
    session_id = case.get("session_id")
    for item in case.get("seed_memory", []):
        memory_store.append_interaction(
            session_id=session_id,
            user_input=item["user_input"],
            planned_tools=item["planned_tools"],
            tool_input=item.get("tool_input"),
            tool_output=item.get("tool_output"),
            final_answer=item["final_answer"],
        )


def build_request(case: dict[str, Any]) -> AgentRequest:
    payload: dict[str, Any] = {"input": case["input"]}
    if "documents" in case:
        payload["documents"] = case["documents"]
    if "top_n" in case:
        payload["top_n"] = case["top_n"]
    if "session_id" in case:
        payload["session_id"] = case["session_id"]
    if "confirm" in case:
        payload["confirm"] = case["confirm"]
    return AgentRequest(**payload)


def evaluate_case(case: dict[str, Any], response: Any) -> dict[str, Any]:
    expected_tool = case["expected_selected_tool"]
    expected_status = case["expected_status"]
    answer = response.final_answer
    required_fragments = case.get("expected_final_answer_contains", [])

    tool_ok = response.selected_tool == expected_tool
    status_ok = response.status == expected_status
    answer_ok = all(fragment in answer for fragment in required_fragments)
    passed = tool_ok and status_ok and answer_ok

    return {
        "name": case["name"],
        "passed": passed,
        "selected_tool_ok": tool_ok,
        "status_ok": status_ok,
        "answer_ok": answer_ok,
        "selected_tool": response.selected_tool,
        "expected_selected_tool": expected_tool,
        "status": response.status,
        "expected_status": expected_status,
        "run_id": response.run_id,
        "final_answer": answer,
    }


async def run_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for case in cases:
        memory_store.reset()
        agent_run_store.reset()
        seed_memory(case)
        request = build_request(case)
        response = await run_agent(request)
        results.append(evaluate_case(case, response))

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    tool_passed = sum(1 for item in results if item["selected_tool_ok"])
    status_passed = sum(1 for item in results if item["status_ok"])
    answer_passed = sum(1 for item in results if item["answer_ok"])
    failed_results = [item for item in results if not item["passed"]]

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
        },
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
            "tool_selection_accuracy": round((tool_passed / total) * 100, 2) if total else 0.0,
            "status_accuracy": round((status_passed / total) * 100, 2) if total else 0.0,
            "answer_match_rate": round((answer_passed / total) * 100, 2) if total else 0.0,
        },
        "failure_summary": build_failure_summary(failed_results),
        "results": results,
    }


def build_failure_summary(failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "failed_cases": len(failures),
        "tool_selection_failures": sum(1 for item in failures if not item["selected_tool_ok"]),
        "status_failures": sum(1 for item in failures if not item["status_ok"]),
        "answer_failures": sum(1 for item in failures if not item["answer_ok"]),
        "failure_names": [item["name"] for item in failures],
    }


def render_markdown_report(report: dict[str, Any], *, cases_path: Path) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    failure_summary = report["failure_summary"]
    lines = [
        "# Agent Eval Report",
        "",
        f"- Generated at (UTC): `{metadata['generated_at_utc']}`",
        f"- Git commit: `{metadata['git_commit'] or 'unknown'}`",
        f"- Cases file: `{cases_path}`",
        "",
        "## Summary",
        "",
        f"- Total cases: `{summary['total']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Pass rate: `{summary['pass_rate']}%`",
        f"- Tool-selection accuracy: `{summary['tool_selection_accuracy']}%`",
        f"- Status accuracy: `{summary['status_accuracy']}%`",
        f"- Answer match rate: `{summary['answer_match_rate']}%`",
        "",
        "## Failures",
        "",
        f"- Failed cases: `{failure_summary['failed_cases']}`",
        f"- Tool-selection failures: `{failure_summary['tool_selection_failures']}`",
        f"- Status failures: `{failure_summary['status_failures']}`",
        f"- Answer failures: `{failure_summary['answer_failures']}`",
    ]
    if failure_summary["failure_names"]:
        lines.append(f"- Failed case names: `{', '.join(failure_summary['failure_names'])}`")
    else:
        lines.append("- Failed case names: `none`")

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case | Passed | Tool | Status | Run ID |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in report["results"]:
        lines.append(
            f"| {item['name']} | {item['passed']} | {item['selected_tool']} | {item['status']} | {item['run_id']} |"
        )

    return "\n".join(lines)


def write_report_files(report: dict[str, Any], *, output_dir: Path, cases_path: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"agent_eval_{timestamp}.json"
    md_path = output_dir / f"agent_eval_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report, cases_path=cases_path), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 8 local evals for the agent service.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to eval case JSON file.")
    parser.add_argument("--save", action="store_true", help="Write JSON and Markdown reports to eval/reports.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to store saved reports.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    report = asyncio.run(run_cases(cases))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save:
        output_dir = Path(args.output_dir)
        json_path, md_path = write_report_files(report, output_dir=output_dir, cases_path=cases_path)
        print(f"\nSaved JSON report to: {json_path}")
        print(f"Saved Markdown report to: {md_path}")


if __name__ == "__main__":
    main()
