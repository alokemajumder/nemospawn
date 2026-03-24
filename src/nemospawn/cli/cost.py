"""Cost tracking CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("show")
def cost_show(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """Show current cost breakdown for a team."""
    from nemospawn.core.costs import update_costs

    record = update_costs(team_id)

    # Summary panel
    gpu_hours = record.total_gpu_seconds / 3600.0
    console.print(Panel(
        f"[bold]Team:[/] {record.team_id}\n"
        f"[bold]Total GPU-hours:[/] {gpu_hours:.2f}\n"
        f"[bold]Total Cost:[/] ${record.total_cost_usd:.4f}\n"
        f"[bold]Rate:[/] ${record.rate_per_gpu_hour:.2f}/GPU-hour\n"
        f"[bold]Last Updated:[/] {record.last_updated[:19]}",
        title="Cost Summary",
        border_style="green",
    ))

    # Per-agent breakdown
    if record.agent_costs:
        table = Table(title="Per-Agent Breakdown")
        table.add_column("Agent ID", style="cyan")
        table.add_column("Name")
        table.add_column("GPUs", justify="right")
        table.add_column("Hours", justify="right")
        table.add_column("Cost ($)", justify="right")
        table.add_column("Status")

        for agent_id, cost in sorted(record.agent_costs.items()):
            hours = cost.get("gpu_seconds", 0) / 3600.0
            status_color = "green" if cost.get("status") == "running" else "red"
            table.add_row(
                agent_id[:12],
                cost.get("name", ""),
                str(cost.get("gpu_count", 0)),
                f"{hours:.2f}",
                f"${cost.get('cost_usd', 0):.4f}",
                f"[{status_color}]{cost.get('status', '')}[/]",
            )
        console.print(table)


@app.command("reset")
def cost_reset(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """Reset cost tracking for a team."""
    from nemospawn.core.costs import reset_costs

    reset_costs(team_id)
    console.print(f"[green]Cost tracking reset for team '{team_id}'[/]")


@app.command("set-rate")
def cost_set_rate(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    rate: float = typer.Option(..., "--rate", help="Cost per GPU-hour in USD"),
):
    """Set the GPU-hour rate for cost calculations."""
    from nemospawn.core.costs import set_rate

    record = set_rate(team_id, rate)
    console.print(f"[green]Rate set to ${rate:.2f}/GPU-hour for team '{team_id}'[/]")
