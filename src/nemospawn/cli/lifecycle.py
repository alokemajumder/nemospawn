"""Agent lifecycle protocol CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("idle")
def report_idle(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID"),
    reason: str = typer.Option("", "--reason", help="Why the agent is idle"),
):
    """Report that an agent is idle (no more work to do)."""
    from nemospawn.core.lifecycle import report_idle as _report_idle

    event = _report_idle(team_id, agent_id, reason)
    console.print(f"[yellow]Agent '{agent_id}' reported idle[/]")
    if reason:
        console.print(f"[dim]Reason: {reason}[/]")


@app.command("shutdown-request")
def shutdown_request(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID to shut down"),
    requested_by: str = typer.Option("", "--by", help="Who is requesting shutdown"),
    reason: str = typer.Option("", "--reason", help="Reason for shutdown"),
):
    """Request graceful shutdown of an agent."""
    from nemospawn.core.lifecycle import request_shutdown

    event = request_shutdown(team_id, agent_id, requested_by, reason)
    console.print(f"[yellow]Shutdown requested for agent '{agent_id}'[/]")


@app.command("shutdown-approve")
def shutdown_approve(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID"),
    by: str = typer.Option("", "--by", help="Who is approving"),
):
    """Approve a pending shutdown request."""
    from nemospawn.core.lifecycle import approve_shutdown

    event = approve_shutdown(team_id, agent_id, responded_by=by)
    if not event:
        console.print(f"[red]No pending shutdown request for '{agent_id}'[/]")
        raise typer.Exit(1)
    console.print(f"[green]Shutdown approved for agent '{agent_id}'[/]")


@app.command("shutdown-reject")
def shutdown_reject(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID"),
    by: str = typer.Option("", "--by", help="Who is rejecting"),
    reason: str = typer.Option("", "--reason", help="Reason for rejection"),
):
    """Reject a shutdown request — agent should continue working."""
    from nemospawn.core.lifecycle import reject_shutdown

    event = reject_shutdown(team_id, agent_id, responded_by=by, reason=reason)
    if not event:
        console.print(f"[red]No pending shutdown request for '{agent_id}'[/]")
        raise typer.Exit(1)
    console.print(f"[green]Shutdown rejected for agent '{agent_id}' — agent continues[/]")


@app.command("status")
def lifecycle_status(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: Optional[str] = typer.Option(None, "--agent", help="Specific agent ID"),
):
    """Show lifecycle state of agents."""
    from nemospawn.core.config import AGENTS_SUBDIR
    from nemospawn.core.state import atomic_read, get_team_dir, list_json_files

    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    table = Table(title=f"Lifecycle — {team_id}")
    table.add_column("Agent", style="cyan")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Agent Status")
    table.add_column("Lifecycle State")
    table.add_column("Reason")

    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if not data:
            continue
        if agent_id and data.get("agent_id") != agent_id:
            continue

        lc = data.get("lifecycle", {})
        lc_state = lc.get("state", data.get("status", "unknown"))
        lc_reason = lc.get("reason", "")

        state_colors = {
            "running": "green", "idle": "yellow", "shutdown_requested": "yellow",
            "shutdown_approved": "red", "dead": "red", "stopped": "red",
        }
        color = state_colors.get(lc_state, "dim")

        table.add_row(
            data.get("agent_id", "")[:12],
            data.get("name", ""),
            data.get("role", ""),
            data.get("status", ""),
            f"[{color}]{lc_state}[/]",
            lc_reason[:40],
        )

    console.print(table)


@app.command("idle-list")
def idle_list(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """List all agents currently reporting idle."""
    from nemospawn.core.lifecycle import list_idle_agents

    idle = list_idle_agents(team_id)
    if not idle:
        console.print("[green]No idle agents[/]")
        raise typer.Exit()

    table = Table(title=f"Idle Agents — {team_id}")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Idle Since")
    table.add_column("Reason")

    for a in idle:
        table.add_row(
            a["agent_id"][:12],
            a.get("name", ""),
            a.get("role", ""),
            a.get("idle_since", "")[:19],
            a.get("reason", ""),
        )
    console.print(table)
