"""Integration test — live LLM provider call. Skips if network unreachable."""

from __future__ import annotations

import pytest

from agent_forge.providers import LLMConfig, LLMProvider, Message


def _network_up() -> bool:
    try:
        import httpx  # noqa: PLC0415
        httpx.get("https://api.kilo.ai", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def require_network() -> None:
    if not _network_up():
        pytest.skip("No network — skipping live LLM tests")


def test_kilo_returns_completion() -> None:
    """kilo provider returns non-empty content."""
    from agent_forge.providers import _try_kilo  # noqa: PLC0415
    config = LLMConfig(timeout=30)
    messages = [Message(role="user", content="Reply with exactly the word: pong")]
    content = _try_kilo(messages, config)
    assert content.strip(), "kilo returned empty response"
    print(f"\nkilo response: {content!r}")


def test_provider_failover_chain() -> None:
    """Full provider chain returns a completion."""
    provider = LLMProvider(LLMConfig(timeout=30))
    messages = [Message(role="user", content="What is 2 + 2? Reply with just the number.")]
    resp = provider.complete(messages)
    assert resp.content.strip(), "provider returned empty content"
    assert resp.provider in ("kilo", "pollinations_post", "pollinations_get", "g4f")
    print(f"\nprovider={resp.provider!r} content={resp.content!r} latency={resp.latency_ms}ms")


def test_planner_executor_critic_live() -> None:
    """Full orchestrator run on a real completion — proves planner→executor→critic loop."""
    from agent_forge.agents import Orchestrator  # noqa: PLC0415
    from agent_forge.observability import configure  # noqa: PLC0415
    configure(level="WARNING", otel=False)

    orch = Orchestrator(llm_config=LLMConfig(timeout=45))
    result = orch.run("Calculate 3 + 7 and tell me the result.")

    print("\n--- Orchestrator live run ---")
    print(f"  Steps: {len(result.steps)}")
    for s in result.steps:
        print(f"  Step {s.index}: {s.description[:60]} -> {str(s.result)[:80]}")
    print(f"  Final answer: {result.final_answer[:120]}")
    print(f"  Critic verdict: {result.critic_verdict[:80]}")
    print(f"  Passed: {result.passed}")

    assert result.final_answer.strip(), "orchestrator returned empty final answer"
    assert len(result.steps) >= 1
