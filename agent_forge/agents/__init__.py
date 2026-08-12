"""Core agent loop: Planner → Executor → Critic (multi-agent capable)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent_forge.blackboard import Blackboard
from agent_forge.observability import get_tracer
from agent_forge.providers import LLMConfig, LLMProvider, Message
from agent_forge.tools import call_tool, get_registry

log = structlog.get_logger(__name__)
tracer = get_tracer()


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class Step:
    index: int
    description: str
    tool: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: str = "pending"  # pending | running | done | failed


@dataclass
class AgentConfig:
    name: str
    role: str  # "planner" | "executor" | "critic" | "custom"
    system_prompt: str
    llm_config: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class RunResult:
    task: str
    steps: list[Step]
    final_answer: str
    critic_verdict: str
    passed: bool
    agent_name: str


# ── Prompt templates ──────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a Planner agent. Given a task, decompose it into an ordered list of concrete steps.
Each step must be either:
  a) A tool call: specify the tool name and its JSON arguments.
  b) A reasoning step: describe the thinking needed.

Available tools:
{tools_schema}

Respond ONLY with valid JSON in this exact format (no markdown fences):
{{
  "steps": [
    {{"index": 1, "description": "...", "tool": "tool_name", "tool_args": {{...}}}},
    {{"index": 2, "description": "...", "tool": null, "tool_args": {{}}}}
  ]
}}
"""

_EXECUTOR_SYSTEM = """\
You are an Executor agent. You receive a step description and any prior context.
If the step has a tool result already, synthesize it into a clear finding.
If not, produce the reasoning output for this step.
Be concise. Return only the result text.
"""

_CRITIC_SYSTEM = """\
You are a Critic/Verifier agent. You receive the original task, all steps, and their results.
Your job: decide if the execution correctly solved the task.
Respond ONLY with valid JSON:
{{"verdict": "pass" | "fail", "reason": "...", "confidence": 0.0-1.0}}
"""


# ── Agent implementations ─────────────────────────────────────────────────────


class PlannerAgent:
    def __init__(self, config: AgentConfig, provider: LLMProvider) -> None:
        self.config = config
        self.provider = provider

    def plan(self, task: str) -> list[Step]:
        tools = get_registry()
        tools_schema = json.dumps(
            [t.to_openai_schema() for t in tools.values()], indent=2
        )
        system = (self.config.system_prompt or _PLANNER_SYSTEM).format(tools_schema=tools_schema)
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=f"Task: {task}"),
        ]
        with tracer.start_as_current_span("planner.plan"):
            response = self.provider.complete(messages)
        log.info("planner.response", agent=self.config.name, tokens=response.output_tokens)

        raw = response.content.strip()
        # strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        try:
            data = json.loads(raw)
            steps = [
                Step(
                    index=s["index"],
                    description=s["description"],
                    tool=s.get("tool"),
                    tool_args=s.get("tool_args", {}),
                )
                for s in data["steps"]
            ]
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("planner.parse_failed", error=str(e), raw=raw[:200])
            # fallback: single reasoning step
            steps = [Step(index=1, description=raw, tool=None)]
        return steps


class ExecutorAgent:
    def __init__(self, config: AgentConfig, provider: LLMProvider) -> None:
        self.config = config
        self.provider = provider

    def execute_step(self, step: Step, context: str = "") -> str:
        with tracer.start_as_current_span("executor.step", attributes={"step.index": step.index}):
            if step.tool:
                try:
                    result = call_tool(step.tool, step.tool_args)
                    tool_output = str(result)
                except Exception as e:  # noqa: BLE001
                    log.warning("executor.tool_error", tool=step.tool, error=str(e))
                    tool_output = f"ERROR: {e}"

                # synthesize tool output into natural language
                messages = [
                    Message(role="system", content=self.config.system_prompt or _EXECUTOR_SYSTEM),
                    Message(
                        role="user",
                        content=(
                            f"Step {step.index}: {step.description}\n"
                            f"Tool: {step.tool}\n"
                            f"Tool output: {tool_output}\n"
                            f"Prior context: {context[:500]}"
                        ),
                    ),
                ]
                resp = self.provider.complete(messages)
                return resp.content.strip()

            # pure reasoning step
            messages = [
                Message(role="system", content=self.config.system_prompt or _EXECUTOR_SYSTEM),
                Message(
                    role="user",
                    content=f"Step {step.index}: {step.description}\nPrior context: {context[:500]}",
                ),
            ]
            resp = self.provider.complete(messages)
            return resp.content.strip()


class CriticAgent:
    def __init__(self, config: AgentConfig, provider: LLMProvider) -> None:
        self.config = config
        self.provider = provider

    def verify(self, task: str, steps: list[Step]) -> tuple[bool, str]:
        steps_text = "\n".join(
            f"Step {s.index}: {s.description}\nResult: {s.result}" for s in steps
        )
        messages = [
            Message(role="system", content=self.config.system_prompt or _CRITIC_SYSTEM),
            Message(
                role="user",
                content=f"Task: {task}\n\nExecution log:\n{steps_text}",
            ),
        ]
        with tracer.start_as_current_span("critic.verify"):
            resp = self.provider.complete(messages)

        raw = resp.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        try:
            data = json.loads(raw)
            passed = data.get("verdict", "fail") == "pass"
            reason = data.get("reason", raw)
        except json.JSONDecodeError:
            passed = "pass" in raw.lower()
            reason = raw
        return passed, reason


# ── Orchestrator ──────────────────────────────────────────────────────────────


class Orchestrator:
    """
    Multi-agent orchestrator. Coordinates Planner → Executor → Critic.
    Supports multiple named executor agents collaborating via a shared Blackboard.
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        blackboard: Blackboard | None = None,
        planner_system: str = "",
        executor_system: str = "",
        critic_system: str = "",
    ) -> None:
        cfg = llm_config or LLMConfig()
        provider = LLMProvider(cfg)

        self.blackboard = blackboard or Blackboard()
        self.planner = PlannerAgent(
            AgentConfig(name="planner", role="planner", system_prompt=planner_system, llm_config=cfg),
            provider,
        )
        self.executor = ExecutorAgent(
            AgentConfig(name="executor", role="executor", system_prompt=executor_system, llm_config=cfg),
            provider,
        )
        self.critic = CriticAgent(
            AgentConfig(name="critic", role="critic", system_prompt=critic_system, llm_config=cfg),
            provider,
        )

    def run(self, task: str) -> RunResult:
        log.info("orchestrator.start", task=task[:100])
        with tracer.start_as_current_span("orchestrator.run"):
            steps = self.planner.plan(task)
            log.info("orchestrator.planned", step_count=len(steps))

            context_parts: list[str] = []
            for step in steps:
                step.status = "running"
                result = self.executor.execute_step(step, context="\n".join(context_parts))
                step.result = result
                step.status = "done"
                context_parts.append(f"Step {step.index}: {result}")
                log.info("orchestrator.step_done", index=step.index, result=result[:80])

            final_answer = context_parts[-1] if context_parts else ""
            passed, verdict = self.critic.verify(task, steps)
            log.info("orchestrator.critic", passed=passed, verdict=verdict[:80])

        return RunResult(
            task=task,
            steps=steps,
            final_answer=final_answer,
            critic_verdict=verdict,
            passed=passed,
            agent_name="orchestrator",
        )

    async def arun(self, task: str) -> RunResult:
        """Async wrapper — runs in thread executor (g4f is sync)."""
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(task))
