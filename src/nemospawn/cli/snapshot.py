"""Team snapshot CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("save")
def snapshot_save(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    label: str = typer.Option("", "--label", "-l", help="Optional label for the snapshot"),
):
    """Save a point-in-time snapshot of the team state."""
    from nemospawn.core.snapshot import save_snapshot

    snap = save_snapshot(team_id, label)
    console.print(Panel(
        f"[bold]Snapshot ID:[/] {snap.snapshot_id}\n"
        f"[bold]Label:[/] {snap.label or '(none)'}\n"
        f"[bold]Agents:[/] {len(snap.agents)}\n"
        f"[bold]Tasks:[/] {len(snap.tasks)}\n"
        f"[bold]Plans:[/] {len(snap.plans)}\n"
        f"[bold]Created:[/] {snap.created_at[:19]}",
        title="Snapshot Saved",
        border_style="green",
    ))


@app.command("restore")
def snapshot_restore(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    snapshot_id: str = typer.Option(..., "--snapshot", help="Snapshot ID to restore"),
):
    """Restore team state from a snapshot."""
    from nemospawn.core.snapshot import restore_snapshot

    snap = restore_snapshot(team_id, snapshot_id)
    if not snap:
        console.print(f"[red]Snapshot '{snapshot_id}' not found[/]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Restored from:[/] {snap.snapshot_id}\n"
        f"[bold]Label:[/] {snap.label or '(none)'}\n"
        f"[bold]Agents:[/] {len(snap.agents)}\n"
        f"[bold]Tasks:[/] {len(snap.tasks)}\n"
        f"[bold]Plans:[/] {len(snap.plans)}",
        title="Snapshot Restored",
        border_style="green",
    ))


@app.command("list")
def snapshot_list(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """List all snapshots for a team."""
    from nemospawn.core.snapshot import list_snapshots

    snaps = list_snapshots(team_id)
    if not snaps:
        console.print("[yellow]No snapshots found[/]")
        raise typer.Exit()

    table = Table(title=f"Snapshots — {team_id}")
    table.add_column("Snapshot ID", style="cyan")
    table.add_column("Label")
    table.add_column("Agents")
    table.add_column("Tasks")
    table.add_column("Plans")
    table.add_column("Created")

    for s in snaps:
        table.add_row(
            s.snapshot_id,
            s.label or "(none)",
            str(len(s.agents)),
            str(len(s.tasks)),
            str(len(s.plans)),
            s.created_at[:19],
        )
    console.print(table)


@app.command("delete")
def snapshot_delete(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    snapshot_id: str = typer.Option(..., "--snapshot", help="Snapshot ID to delete"),
):
    """Delete a snapshot."""
    from nemospawn.core.snapshot import delete_snapshot

    if delete_snapshot(team_id, snapshot_id):
        console.print(f"[green]Snapshot '{snapshot_id}' deleted[/]")
    else:
        console.print(f"[red]Snapshot '{snapshot_id}' not found[/]")
        raise typer.Exit(1)
