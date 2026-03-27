from pathlib import Path

from scripts.eval_agent import DEFAULT_CASES_PATH, build_request, evaluate_case, load_cases


def test_load_cases_reads_eval_dataset() -> None:
    cases = load_cases(Path(DEFAULT_CASES_PATH))

    assert len(cases) >= 5
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
