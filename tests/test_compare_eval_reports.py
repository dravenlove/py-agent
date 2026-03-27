from pathlib import Path
import json

from scripts.compare_eval_reports import (
    compare_reports,
    find_latest_report_paths,
    render_markdown_comparison,
    write_comparison_files,
)


def test_find_latest_report_paths_returns_two_latest_jsons(tmp_path: Path) -> None:
    old = tmp_path / "agent_eval_20260327T080000Z.json"
    new = tmp_path / "agent_eval_20260327T090000Z.json"
    old.write_text("{}", encoding="utf-8")
    new.write_text("{}", encoding="utf-8")

    baseline, current = find_latest_report_paths(tmp_path)

    assert baseline == old
    assert current == new


def test_compare_reports_detects_regression_and_improvement() -> None:
    baseline = {
        "metadata": {"generated_at_utc": "2026-03-27T08:00:00Z", "git_commit": "abc123"},
        "summary": {
            "pass_rate": 50.0,
            "tool_selection_accuracy": 50.0,
            "status_accuracy": 50.0,
            "answer_match_rate": 50.0,
        },
        "results": [
            {
                "name": "case-1",
                "passed": True,
                "status": "completed",
                "selected_tool": "calculator",
            },
            {
                "name": "case-2",
                "passed": False,
                "status": "completed",
                "selected_tool": "unsupported",
            },
        ],
    }
    current = {
        "metadata": {"generated_at_utc": "2026-03-27T09:00:00Z", "git_commit": "def456"},
        "summary": {
            "pass_rate": 50.0,
            "tool_selection_accuracy": 75.0,
            "status_accuracy": 75.0,
            "answer_match_rate": 50.0,
        },
        "results": [
            {
                "name": "case-1",
                "passed": False,
                "status": "needs_confirmation",
                "selected_tool": "clear_session_memory",
            },
            {
                "name": "case-2",
                "passed": True,
                "status": "completed",
                "selected_tool": "calculator",
            },
            {
                "name": "case-3",
                "passed": True,
                "status": "completed",
                "selected_tool": "embed_text",
            },
        ],
    }

    comparison = compare_reports(baseline, current)

    assert comparison["summary_delta"]["tool_selection_accuracy_delta"] == 25.0
    assert comparison["regressions"][0]["name"] == "case-1"
    assert comparison["improvements"][0]["name"] == "case-2"
    assert comparison["new_cases"] == ["case-3"]


def test_render_markdown_comparison_includes_regression_section() -> None:
    comparison = {
        "metadata": {
            "generated_at_utc": "2026-03-27T09:00:00Z",
            "baseline_git_commit": "abc123",
            "current_git_commit": "def456",
        },
        "summary_delta": {
            "pass_rate_delta": -10.0,
            "tool_selection_accuracy_delta": -5.0,
            "status_accuracy_delta": 0.0,
            "answer_match_rate_delta": -10.0,
        },
        "regressions": [
            {
                "name": "case-1",
                "baseline_status": "completed",
                "current_status": "needs_confirmation",
                "baseline_selected_tool": "calculator",
                "current_selected_tool": "clear_session_memory",
            }
        ],
        "improvements": [],
        "new_cases": [],
        "removed_cases": [],
        "unchanged_cases": ["case-2"],
    }

    markdown = render_markdown_comparison(
        comparison,
        baseline_path=Path("baseline.json"),
        current_path=Path("current.json"),
    )

    assert "# Agent Eval Comparison Report" in markdown
    assert "case-1" in markdown
    assert "Pass rate delta: `-10.0%`" in markdown


def test_write_comparison_files_creates_json_and_markdown(tmp_path: Path) -> None:
    comparison = {
        "metadata": {
            "generated_at_utc": "2026-03-27T09:00:00Z",
            "baseline_git_commit": "abc123",
            "current_git_commit": "def456",
        },
        "summary_delta": {
            "pass_rate_delta": 0.0,
            "tool_selection_accuracy_delta": 0.0,
            "status_accuracy_delta": 0.0,
            "answer_match_rate_delta": 0.0,
        },
        "regressions": [],
        "improvements": [],
        "new_cases": [],
        "removed_cases": [],
        "unchanged_cases": ["case-1"],
    }

    json_path, md_path = write_comparison_files(
        comparison,
        baseline_path=Path("baseline.json"),
        current_path=Path("current.json"),
        output_dir=tmp_path,
    )

    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary_delta"]["pass_rate_delta"] == 0.0
    assert "# Agent Eval Comparison Report" in md_path.read_text(encoding="utf-8")
