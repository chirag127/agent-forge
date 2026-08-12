"""Integration tests — eval harness (mocked LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_forge.providers import LLMProvider, LLMResponse
from evals.run_eval import (
    EvalCase,
    EvalResult,
    check_assertions,
    load_dataset,
    run_evals,
)


# ── assertion checker ─────────────────────────────────────────────────────────


def test_contains_pass() -> None:
    passed, failed = check_assertions("The answer is 1024.", [{"type": "contains", "value": "1024"}])
    assert passed
    assert not failed


def test_contains_fail() -> None:
    passed, failed = check_assertions("The answer is 999.", [{"type": "contains", "value": "1024"}])
    assert not passed
    assert len(failed) == 1


def test_contains_ci() -> None:
    passed, _ = check_assertions("Capital is PARIS", [{"type": "contains_ci", "value": "paris"}])
    assert passed


def test_contains_any() -> None:
    passed, _ = check_assertions("red is primary", [{"type": "contains_any", "values": ["red", "blue"]}])
    assert passed


def test_contains_any_ci() -> None:
    passed, _ = check_assertions("RED is a color", [{"type": "contains_any_ci", "values": ["red"]}])
    assert passed


def test_not_contains_pass() -> None:
    passed, _ = check_assertions("no bad word here", [{"type": "not_contains", "value": "error"}])
    assert passed


def test_not_contains_fail() -> None:
    passed, _ = check_assertions("an error occurred", [{"type": "not_contains", "value": "error"}])
    assert not passed


def test_multiple_assertions_all_pass() -> None:
    passed, failed = check_assertions(
        "result is 1024 and correct",
        [
            {"type": "contains", "value": "1024"},
            {"type": "contains_ci", "value": "correct"},
        ],
    )
    assert passed
    assert not failed


def test_multiple_assertions_partial_fail() -> None:
    passed, failed = check_assertions(
        "result is 1024",
        [
            {"type": "contains", "value": "1024"},
            {"type": "contains", "value": "missing"},
        ],
    )
    assert not passed
    assert len(failed) == 1


# ── load_dataset ──────────────────────────────────────────────────────────────


def test_load_dataset_returns_eval_cases(tmp_path: Path) -> None:
    data = [
        {"id": "t1", "task": "Do X", "assertions": [{"type": "contains", "value": "X"}], "tags": ["test"]},
    ]
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    cases = load_dataset(p)
    assert len(cases) == 1
    assert isinstance(cases[0], EvalCase)
    assert cases[0].id == "t1"


def test_load_real_dataset() -> None:
    from evals.run_eval import GOLD_PATH  # noqa: PLC0415
    cases = load_dataset(GOLD_PATH)
    assert len(cases) > 0
    assert all(isinstance(c, EvalCase) for c in cases)


# ── run_evals (mocked LLM) ────────────────────────────────────────────────────


def test_run_evals_mock(tmp_path: Path) -> None:
    """run_evals with a mocked orchestrator — no live calls."""
    dataset = [
        {
            "id": "mock_001",
            "task": "What is 2+2?",
            "assertions": [{"type": "contains", "value": "4"}],
            "tags": [],
        },
        {
            "id": "mock_002",
            "task": "Capital of France?",
            "assertions": [{"type": "contains_ci", "value": "paris"}],
            "tags": [],
        },
    ]
    ds_path = tmp_path / "gold.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")
    report_path = tmp_path / "report.md"

    responses = ["The answer is 4.", "Paris is the capital."]
    call_idx = 0

    def _mock_run(task: str):  # noqa: ANN001
        nonlocal call_idx
        from agent_forge.agents import RunResult, Step  # noqa: PLC0415
        ans = responses[call_idx % len(responses)]
        call_idx += 1
        return RunResult(
            task=task,
            steps=[Step(index=1, description="mock step", result=ans)],
            final_answer=ans,
            critic_verdict="pass",
            passed=True,
            agent_name="mock",
        )

    with patch("evals.run_eval.Orchestrator") as MockOrch:
        MockOrch.return_value.run.side_effect = _mock_run
        results = run_evals(ds_path, report_path)

    assert len(results) == 2
    assert all(r.passed for r in results)
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "mock_001" in report_text
    assert "Pass rate:" in report_text


def test_run_evals_error_case(tmp_path: Path) -> None:
    """Error during agent run → EvalResult.error set, passed=False."""
    dataset = [
        {"id": "err_001", "task": "fail task", "assertions": [{"type": "contains", "value": "xyz"}], "tags": []},
    ]
    ds_path = tmp_path / "gold.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")
    report_path = tmp_path / "report.md"

    with patch("evals.run_eval.Orchestrator") as MockOrch:
        MockOrch.return_value.run.side_effect = RuntimeError("provider failed")
        results = run_evals(ds_path, report_path)

    assert len(results) == 1
    assert results[0].passed is False
    assert "provider failed" in results[0].error
