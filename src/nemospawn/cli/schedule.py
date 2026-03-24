"""Adaptive scheduling CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("analyze")
def schedule_analyze(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """Analyze agent performance and rank by GPU utilization + task completion."""
    from nemospawn.core.adaptive import analyze_performance
    from nemospawn.gpu.dcgm import poll_dcgm

    metrics = poll_dcgm()
    perf = analyze_performance(team_id, metrics or None)

    if not perf:
        console.print("[yellow]No running agents to analyze[/]")
        raise typer.Exit()

    table = Table(title=f"Agent Performance — {team_id}")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Agent", style="cyan")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("GPUs")
    table.add_column("GPU Util %", justify="right")
    table.add_column("Tasks (done/total)", justify="right")
    table.add_column("Hours", justify="right")
    table.add_column("Score", justify="right")

    for i, a in enumerate(reversed(perf)):
        util_str = f"{a['avg_gpu_util']:.0f}" if a["avg_gpu_util"] is not None else "n/a"
        util_color = "green" if a["avg_gpu_util"] and a["avg_gpu_util"] >= 50 else "red" if a["avg_gpu_util"] is not None else "dim"
        table.add_row(
            str(i + 1),
            a["agent_id"][:12],
            a.get("name", ""),
            a.get("role", ""),
            str(a["gpu_ids"]),
            f"[{util_color}]{util_str}[/]",
            f"{a['tasks_done']}/{a['tasks_total']}",
            f"{a['hours_running']:.1f}",
            f"{a['score']:.1f}",
        )
    console.print(table)


@app.command("suggest")
def schedule_suggest(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    threshold: float = typer.Option(30.0, "--threshold", help="GPU util threshold (%) below which an agent is underperforming"),
):
    """Suggest task reassignments from underperforming to better agents."""
    from nemospawn.core.adaptive import suggest_reassignments
    from nemospawn.gpu.dcgm import poll_dcgm

    metrics = poll_dcgm()
    suggestions = suggest_reassignments(team_id, metrics or None, util_threshold=threshold)

    if not suggestions:
        console.print("[green]No reassignments needed — all agents performing above threshold[/]")
        return

    table = Table(title=f"Suggested Reassignments — {team_id}")
    table.add_column("Task ID", style="cyan")
    table.add_column("Task Title")
    table.add_column("From Agent")
    table.add_column("From Util %", justify="right")
    table.add_column("To Agent")
    table.add_column("To Util %", justify="right")
    table.add_column("Reason")

    for s in suggestions:
        table.add_row(
            s["task_id"][:12],
            s["task_title"][:30],
            s["from_agent"][:12],
            f"[red]{s['from_util']:.0f}[/]",
            s["to_agent"][:12],
            f"[green]{s['to_util']:.0f}[/]",
            s["reason"],
        )
    console.print(table)
    console.print(f"\n[dim]Apply with: nemospawn schedule apply --team {team_id} --task <task_id> --to <agent_id>[/]")


@app.command("apply")
def schedule_apply(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    task_id: str = typer.Option(..., "--task", help="Task ID to reassign"),
    to_agent: str = typer.Option(..., "--to", help="Target agent ID"),
):
    """Apply a task reassignment to a different agent."""
    from nemospawn.core.adaptive import apply_reassignment

    if apply_reassignment(team_id, task_id, to_agent):
        console.print(f"[green]Task '{task_id}' reassigned to agent '{to_agent}'[/]")
    else:
        console.print(f"[red]Task '{task_id}' not found[/]")
        raise typer.Exit(1)


@app.command("auto")
def schedule_auto(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    threshold: float = typer.Option(30.0, "--threshold", help="GPU util threshold (%)"),
    interval: int = typer.Option(300, "--interval", help="Check interval in seconds"),
):
    """Continuously monitor and auto-reassign tasks from underperforming agents."""
    import time

    from nemospawn.core.adaptive import suggest_reassignments, apply_reassignment
    from nemospawn.gpu.dcgm import poll_dcgm

    console.print(f"[bold]Adaptive scheduler started for '{team_id}' (threshold: {threshold}%, every {interval}s)[/]")
    console.print("Press Ctrl+C to stop\n")

    try:
        while True:
            metrics = poll_dcgm()
            suggestions = suggest_reassignments(team_id, metrics or None, util_threshold=threshold)

            timestamp = _now()[:19]
            if suggestions:
                for s in suggestions:
                    apply_reassignment(team_id, s["task_id"], s["to_agent"])
                    console.print(
                        f"[dim]{timestamp}[/] [yellow]Reassigned[/] {s['task_id'][:12]} "
                        f"from {s['from_agent'][:12]} ({s['from_util']:.0f}%) "
                        f"to {s['to_agent'][:12]} ({s['to_util']:.0f}%)"
                    )
            else:
                console.print(f"[dim]{timestamp}[/] [green]All agents OK[/]")

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Adaptive scheduler stopped[/]")


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
