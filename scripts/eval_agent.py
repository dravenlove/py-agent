import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "eval" / "agent_eval_cases.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_service import run_agent
from app.audit import agent_run_store
from app.memory import memory_store
from app.schemas import AgentRequest


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
            "tool_selection_accuracy": round((tool_passed / total) * 100, 2) if total else 0.0,
            "status_accuracy": round((status_passed / total) * 100, 2) if total else 0.0,
            "answer_match_rate": round((answer_passed / total) * 100, 2) if total else 0.0,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 8 local evals for the agent service.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to eval case JSON file.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    report = asyncio.run(run_cases(cases))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
