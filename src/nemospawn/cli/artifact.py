"""Artifact management commands — register, promote, and list NeMo artifacts."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("register")
def register(
    team_id: str = typer.Argument(..., help="Team ID"),
    path: str = typer.Argument(..., help="Path to artifact file or directory"),
    artifact_type: str = typer.Option("nemo-checkpoint", "--type", help="Artifact type"),
    val_loss: Optional[float] = typer.Option(None, "--val-loss", help="Validation loss"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Agent ID that produced this artifact"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
):
    """Register an artifact in the team's artifact store."""
    from nemospawn.nemo.artifacts import register_artifact, ARTIFACT_TYPES

    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    try:
        art = register_artifact(
            team_id=team_id,
            path=path,
            artifact_type=artifact_type,
            agent_id=agent,
            val_loss=val_loss,
            tags=tag_list,
        )
        console.print(Panel(
            f"[bold]ID:[/] {art.artifact_id}\n"
            f"[bold]Type:[/] {art.artifact_type}\n"
            f"[bold]Path:[/] {art.path}\n"
            f"[bold]Val Loss:[/] {art.val_loss or 'N/A'}\n"
            f"[bold]Hash:[/] {art.nemo_config_hash or 'N/A'}",
            title="Artifact Registered",
            border_style="green",
        ))
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)


@app.command("promote")
def promote(
    team_id: str = typer.Argument(..., help="Team ID"),
    artifact_id: str = typer.Argument(..., help="Artifact ID to promote"),
):
    """Promote an artifact as the best checkpoint for its type."""
    from nemospawn.nemo.artifacts import promote_artifact

    try:
        art = promote_artifact(team_id, artifact_id)
        console.print(f"[green]Promoted:[/] {art.artifact_id} ({art.artifact_type}) — val_loss: {art.val_loss}")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)


@app.command("list")
def list_artifacts(
    team_id: str = typer.Argument(..., help="Team ID"),
    artifact_type: Optional[str] = typer.Option(None, "--type", help="Filter by artifact type"),
    sort: str = typer.Option("val_loss", "--sort", help="Sort by: val_loss, created_at"),
    promoted: bool = typer.Option(False, "--promoted", help="Show only promoted artifacts"),
):
    """List artifacts in the team store."""
    from nemospawn.nemo.artifacts import list_artifacts as _list_arts

    artifacts = _list_arts(team_id, artifact_type=artifact_type, sort_by=sort, promoted_only=promoted)

    if not artifacts:
        console.print("[yellow]No artifacts found[/]")
        raise typer.Exit()

    table = Table(title=f"Artifacts — {team_id}")
    table.add_column("ID", style="cyan")
    table.add_column("Type")
    table.add_column("Val Loss", justify="right")
    table.add_column("Agent")
    table.add_column("Promoted")
    table.add_column("Path")
    table.add_column("Created")

    for a in artifacts:
        table.add_row(
            a.artifact_id,
            a.artifact_type,
            f"{a.val_loss:.4f}" if a.val_loss is not None else "",
            a.agent_id or "",
            "[green]yes[/]" if a.promoted else "",
            str(a.path)[:40],
            a.created_at[:19],
        )
    console.print(table)


@app.command("show")
def show_artifact(
    team_id: str = typer.Argument(..., help="Team ID"),
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
):
    """Show detailed artifact information."""
    from nemospawn.core.config import ARTIFACTS_SUBDIR
    from nemospawn.core.state import atomic_read, get_team_dir
    from nemospawn.nemo.artifacts import Artifact

    data = atomic_read(get_team_dir(team_id) / ARTIFACTS_SUBDIR / f"{artifact_id}.json")
    if not data:
        console.print(f"[red]Artifact '{artifact_id}' not found[/]")
        raise typer.Exit(1)

    art = Artifact.from_dict(data)
    text = (
        f"[bold]ID:[/] {art.artifact_id}\n"
        f"[bold]Type:[/] {art.artifact_type}\n"
        f"[bold]Path:[/] {art.path}\n"
        f"[bold]Agent:[/] {art.agent_id or 'N/A'}\n"
        f"[bold]Val Loss:[/] {art.val_loss or 'N/A'}\n"
        f"[bold]Config Hash:[/] {art.nemo_config_hash or 'N/A'}\n"
        f"[bold]Promoted:[/] {'yes' if art.promoted else 'no'}\n"
        f"[bold]Tags:[/] {', '.join(art.tags) if art.tags else 'none'}\n"
        f"[bold]Created:[/] {art.created_at}\n"
    )
    if art.metrics:
        text += "[bold]Metrics:[/]\n"
        for k, v in art.metrics.items():
            text += f"  {k}: {v}\n"

    console.print(Panel(text, title=f"Artifact: {artifact_id}"))
