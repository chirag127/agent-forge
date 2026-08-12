"""Unit tests — agents (Planner, Executor, Critic, Orchestrator) with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_forge.agents import (
    AgentConfig,
    CriticAgent,
    ExecutorAgent,
    Orchestrator,
    PlannerAgent,
    RunResult,
    Step,
)
from agent_forge.providers import LLMConfig, LLMProvider, LLMResponse, Message


def _mock_provider(content: str) -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.complete.return_value = LLMResponse(
        content=content, provider="mock", model="mock"
    )
    return provider


def _planner_config() -> AgentConfig:
    return AgentConfig(name="planner", role="planner", system_prompt="")


def _executor_config() -> AgentConfig:
    return AgentConfig(name="executor", role="executor", system_prompt="")


def _critic_config() -> AgentConfig:
    return AgentConfig(name="critic", role="critic", system_prompt="")


# ── PlannerAgent ──────────────────────────────────────────────────────────────


def test_planner_parses_valid_json() -> None:
    plan_json = json.dumps({
        "steps": [
            {"index": 1, "description": "Do X", "tool": "calculator", "tool_args": {"expression": "1+1"}},
            {"index": 2, "description": "Summarize", "tool": None, "tool_args": {}},
        ]
    })
    provider = _mock_provider(plan_json)
    planner = PlannerAgent(_planner_config(), provider)
    steps = planner.plan("Do X then summarize")

    assert len(steps) == 2
    assert steps[0].tool == "calculator"
    assert steps[1].tool is None


def test_planner_handles_markdown_fences() -> None:
    plan_json = '```json\n' + json.dumps({
        "steps": [{"index": 1, "description": "Step", "tool": None, "tool_args": {}}]
    }) + '\n```'
    provider = _mock_provider(plan_json)
    planner = PlannerAgent(_planner_config(), provider)
    steps = planner.plan("task")
    assert len(steps) == 1


def test_planner_fallback_on_invalid_json() -> None:
    provider = _mock_provider("I cannot plan this, sorry.")
    planner = PlannerAgent(_planner_config(), provider)
    steps = planner.plan("task")
    assert len(steps) == 1
    assert steps[0].description == "I cannot plan this, sorry."


def test_planner_step_defaults() -> None:
    plan_json = json.dumps({
        "steps": [{"index": 1, "description": "reason step"}]
    })
    provider = _mock_provider(plan_json)
    planner = PlannerAgent(_planner_config(), provider)
    steps = planner.plan("task")
    assert steps[0].tool is None
    assert steps[0].tool_args == {}


# ── ExecutorAgent ─────────────────────────────────────────────────────────────


def test_executor_calls_tool_when_specified() -> None:
    provider = _mock_provider("The result is 2.")
    executor = ExecutorAgent(_executor_config(), provider)
    step = Step(index=1, description="Add 1+1", tool="calculator", tool_args={"expression": "1+1"})
    result = executor.execute_step(step)
    assert result == "The result is 2."
    provider.complete.assert_called_once()


def test_executor_reasoning_step_no_tool() -> None:
    provider = _mock_provider("Reasoning output.")
    executor = ExecutorAgent(_executor_config(), provider)
    step = Step(index=1, description="Think about X", tool=None)
    result = executor.execute_step(step)
    assert result == "Reasoning output."


def test_executor_handles_tool_error_gracefully() -> None:
    provider = _mock_provider("Error handled.")
    executor = ExecutorAgent(_executor_config(), provider)
    step = Step(index=1, description="Run broken tool", tool="calculator", tool_args={"expression": "__bad__"})
    # calculator will raise ValueError on unsafe chars
    result = executor.execute_step(step)
    # LLM is still called (with error context), result is from LLM
    assert result == "Error handled."
    provider.complete.assert_called_once()


# ── CriticAgent ───────────────────────────────────────────────────────────────


def test_critic_pass_verdict() -> None:
    verdict_json = json.dumps({"verdict": "pass", "reason": "looks good", "confidence": 0.9})
    provider = _mock_provider(verdict_json)
    critic = CriticAgent(_critic_config(), provider)
    steps = [Step(index=1, description="calc", result="10")]
    passed, reason = critic.verify("Calculate 5+5", steps)
    assert passed is True
    assert "looks good" in reason


def test_critic_fail_verdict() -> None:
    verdict_json = json.dumps({"verdict": "fail", "reason": "wrong answer", "confidence": 0.8})
    provider = _mock_provider(verdict_json)
    critic = CriticAgent(_critic_config(), provider)
    steps = [Step(index=1, description="calc", result="999")]
    passed, reason = critic.verify("Calculate 5+5", steps)
    assert passed is False
    assert "wrong answer" in reason


def test_critic_fallback_on_invalid_json() -> None:
    provider = _mock_provider("This execution passed all checks.")
    critic = CriticAgent(_critic_config(), provider)
    steps = [Step(index=1, description="step", result="ok")]
    passed, reason = critic.verify("task", steps)
    assert passed is True  # "pass" in text → True


def test_critic_fail_fallback_no_pass_keyword() -> None:
    provider = _mock_provider("The answer was completely wrong and failed.")
    critic = CriticAgent(_critic_config(), provider)
    steps = [Step(index=1, description="step", result="bad")]
    passed, _ = critic.verify("task", steps)
    assert passed is False


# ── Orchestrator ──────────────────────────────────────────────────────────────


def test_orchestrator_full_run() -> None:
    """Orchestrator coordinates planner→executor→critic with mocked LLM."""
    plan_json = json.dumps({
        "steps": [
            {"index": 1, "description": "Calculate 2+2", "tool": "calculator", "tool_args": {"expression": "2+2"}},
        ]
    })
    exec_response = "The result is 4."
    critic_response = json.dumps({"verdict": "pass", "reason": "correct", "confidence": 1.0})

    call_count = 0

    def side_effect(messages: list, **kwargs) -> LLMResponse:  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(content=plan_json, provider="mock", model="mock")
        if call_count == 2:
            return LLMResponse(content=exec_response, provider="mock", model="mock")
        return LLMResponse(content=critic_response, provider="mock", model="mock")

    with patch.object(LLMProvider, "complete", side_effect=side_effect):
        orch = Orchestrator(llm_config=LLMConfig())
        result = orch.run("What is 2+2?")

    assert isinstance(result, RunResult)
    assert len(result.steps) == 1
    assert result.steps[0].status == "done"
    assert result.steps[0].result == exec_response
    assert result.passed is True
    assert call_count == 3  # planner + executor + critic


def test_orchestrator_step_status_tracking() -> None:
    plan_json = json.dumps({
        "steps": [
            {"index": 1, "description": "Step one", "tool": None, "tool_args": {}},
            {"index": 2, "description": "Step two", "tool": None, "tool_args": {}},
        ]
    })

    responses = [
        plan_json,
        "Output one.",
        "Output two.",
        json.dumps({"verdict": "pass", "reason": "ok"}),
    ]
    call_idx = 0

    def side_effect(messages, **kwargs):  # noqa: ANN001
        nonlocal call_idx
        r = LLMResponse(content=responses[call_idx], provider="mock", model="mock")
        call_idx += 1
        return r

    with patch.object(LLMProvider, "complete", side_effect=side_effect):
        orch = Orchestrator(llm_config=LLMConfig())
        result = orch.run("task")

    assert all(s.status == "done" for s in result.steps)
    assert result.final_answer == "Step 2: Output two."
