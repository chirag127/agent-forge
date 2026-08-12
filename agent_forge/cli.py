"""CLI entry point — agent-forge run / eval."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="agent-forge",
    help="Multi-agent LLM orchestration platform.",
    add_completion=False,
)
console = Console()


def _init_obs(verbose: bool, otel: bool) -> None:
    from agent_forge.observability import configure  # noqa: PLC0415
    configure(level="DEBUG" if verbose else "WARNING", otel=otel)


@app.command("run")
def run_task(
    task: str = typer.Argument(..., help="Task for the agent to solve"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs"),
    otel: bool = typer.Option(False, "--otel", help="Enable OpenTelemetry console spans"),
    timeout: float = typer.Option(60.0, "--timeout", "-t", help="LLM timeout seconds"),
) -> None:
    """Run a task through the Planner -> Executor -> Critic pipeline."""
    _init_obs(verbose, otel)
    from agent_forge.agents import Orchestrator  # noqa: PLC0415
    from agent_forge.providers import LLMConfig  # noqa: PLC0415

    console.print(Panel(f"[bold cyan]Task:[/] {task}", title="Agent Forge"))

    try:
        orch = Orchestrator(llm_config=LLMConfig(timeout=timeout))
        result = orch.run(task)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1) from e

    # Steps table
    table = Table(title="Execution Steps", show_lines=True)
    table.add_column("Step", style="dim", width=6)
    table.add_column("Description", max_width=50)
    table.add_column("Tool", style="yellow", max_width=15)
    table.add_column("Result", max_width=60)

    for s in result.steps:
        table.add_row(
            str(s.index),
            s.description[:50],
            s.tool or "-",
            str(s.result or "")[:60],
        )
    console.print(table)

    verdict_color = "green" if result.passed else "red"
    console.print(Panel(
        f"[bold]Final Answer:[/]\n{result.final_answer}\n\n"
        f"[bold]Critic:[/] [{verdict_color}]{result.critic_verdict[:120]}[/{verdict_color}]",
        title=f"Result ({'PASS' if result.passed else 'FAIL'})",
        border_style=verdict_color,
    ))


@app.command("eval")
def run_eval(
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Path to gold JSON dataset"),
    report: Optional[Path] = typer.Option(None, "--report", "-o", help="Report output path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    timeout: float = typer.Option(60.0, "--timeout", "-t"),
) -> None:
    """Run eval harness over gold dataset. Writes report.md."""
    _init_obs(verbose, False)
    from agent_forge.providers import LLMConfig  # noqa: PLC0415
    from evals.run_eval import GOLD_PATH, REPORT_PATH, run_evals  # noqa: PLC0415

    ds = dataset or GOLD_PATH
    rep = report or REPORT_PATH
    console.print(f"[cyan]Dataset:[/] {ds}")
    console.print(f"[cyan]Report:[/]  {rep}")

    try:
        results = run_evals(ds, rep, llm_config=LLMConfig(timeout=timeout))
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Eval error:[/] {e}")
        raise typer.Exit(1) from e

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = round(passed / total * 100, 1) if total else 0

    table = Table(title=f"Eval Results — {passed}/{total} passed ({pct}%)", show_lines=True)
    table.add_column("ID", style="dim")
    table.add_column("Pass", width=6)
    table.add_column("Latency")
    table.add_column("Failed assertions / Error", max_width=50)

    for r in results:
        status = "[green]PASS[/]" if r.passed else "[red]FAIL[/]"
        detail = "; ".join(r.failed_assertions) or (r.error[:50] if r.error else "")
        table.add_row(r.case.id, status, f"{r.latency_s}s", detail)

    console.print(table)
    console.print(f"\nReport written to: [cyan]{rep}[/]")

    if passed < total:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
