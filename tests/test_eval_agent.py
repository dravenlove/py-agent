from pathlib import Path
import json

from scripts.eval_agent import (
    DEFAULT_CASES_PATH,
    build_failure_summary,
    build_request,
    evaluate_case,
    load_cases,
    render_markdown_report,
    write_report_files,
)


def test_load_cases_reads_eval_dataset() -> None:
    cases = load_cases(Path(DEFAULT_CASES_PATH))

    assert len(cases) >= 6
    assert any(case["name"] == "calculator_basic" for case in cases)


def test_build_request_maps_confirm_and_session_id() -> None:
    request = build_request(
        {
            "input": "请清空这个会话的记忆",
            "session_id": "eval-risk",
            "confirm": True,
        }
    )

    assert request.input == "请清空这个会话的记忆"
    assert request.session_id == "eval-risk"
    assert request.confirm is True


def test_evaluate_case_marks_case_as_passed_when_all_checks_match() -> None:
    class Response:
        selected_tool = "calculator"
        status = "completed"
        final_answer = "计算完成：23 * 7 = 161"
        run_id = "run-1"

    result = evaluate_case(
        {
            "name": "calculator_basic",
            "expected_selected_tool": "calculator",
            "expected_status": "completed",
            "expected_final_answer_contains": ["161"],
        },
        Response(),
    )

    assert result["passed"] is True
    assert result["answer_ok"] is True


def test_build_failure_summary_counts_failed_dimensions() -> None:
    summary = build_failure_summary(
        [
            {
                "name": "case-1",
                "selected_tool_ok": False,
                "status_ok": True,
                "answer_ok": False,
                "passed": False,
            },
            {
                "name": "case-2",
                "selected_tool_ok": True,
                "status_ok": False,
                "answer_ok": True,
                "passed": False,
            },
        ]
    )

    assert summary["failed_cases"] == 2
    assert summary["tool_selection_failures"] == 1
    assert summary["status_failures"] == 1
    assert summary["answer_failures"] == 1


def test_render_markdown_report_includes_summary() -> None:
    report = {
        "metadata": {"generated_at_utc": "2026-03-27T00:00:00Z", "git_commit": "abc123"},
        "summary": {
            "total": 2,
            "passed": 1,
            "pass_rate": 50.0,
            "tool_selection_accuracy": 50.0,
            "status_accuracy": 100.0,
            "answer_match_rate": 50.0,
        },
        "failure_summary": {
            "failed_cases": 1,
            "tool_selection_failures": 1,
            "status_failures": 0,
            "answer_failures": 1,
            "failure_names": ["case-2"],
        },
        "results": [
            {
                "name": "case-1",
                "passed": True,
                "selected_tool": "calculator",
                "status": "completed",
                "run_id": "run-1",
            }
        ],
    }

    markdown = render_markdown_report(report, cases_path=Path("eval/agent_eval_cases.json"))

    assert "# Agent Eval Report" in markdown
    assert "Pass rate: `50.0%`" in markdown
    assert "case-1" in markdown


def test_write_report_files_creates_json_and_markdown(tmp_path: Path) -> None:
    report = {
        "metadata": {"generated_at_utc": "2026-03-27T00:00:00Z", "git_commit": "abc123"},
        "summary": {
            "total": 1,
            "passed": 1,
            "pass_rate": 100.0,
            "tool_selection_accuracy": 100.0,
            "status_accuracy": 100.0,
            "answer_match_rate": 100.0,
        },
        "failure_summary": {
            "failed_cases": 0,
            "tool_selection_failures": 0,
            "status_failures": 0,
            "answer_failures": 0,
            "failure_names": [],
        },
        "results": [
            {
                "name": "case-1",
                "passed": True,
                "selected_tool": "calculator",
                "status": "completed",
                "run_id": "run-1",
            }
        ],
    }

    json_path, md_path = write_report_files(report, output_dir=tmp_path, cases_path=Path("eval/agent_eval_cases.json"))

    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["passed"] == 1
    assert "# Agent Eval Report" in md_path.read_text(encoding="utf-8")
