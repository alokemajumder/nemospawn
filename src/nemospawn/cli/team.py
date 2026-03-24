"""Team lifecycle commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemospawn.core.config import TEAMS_DIR
from nemospawn.core.models import Team, _now, _short_id
from nemospawn.core.state import atomic_read, atomic_write, ensure_state_dir, ensure_team_dir

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("create")
def create_team(
    name: str = typer.Argument(..., help="Team name"),
    description: str = typer.Option("", "-d", "--description", help="Team description"),
    gpus: Optional[str] = typer.Option(None, "--gpus", help="Comma-separated GPU indices (e.g., 0,1,2,3)"),
):
    """Create a new agent team with GPU topology discovery."""
    ensure_state_dir()

    # Parse GPU list
    gpu_ids: list[int] = []
    if gpus:
        gpu_ids = [int(g.strip()) for g in gpus.split(",")]

    # Discover GPUs
    from nemospawn.gpu.discovery import discover_gpus
    from nemospawn.gpu.topology import get_nvlink_islands, parse_topology

    available_gpus = discover_gpus()
    gpu_dicts = []
    if available_gpus:
        if gpu_ids:
            gpu_dicts = [g.to_dict() for g in available_gpus if g.index in gpu_ids]
        else:
            gpu_dicts = [g.to_dict() for g in available_gpus]
            gpu_ids = [g.index for g in available_gpus]

    # Get topology
    topo = parse_topology()
    islands = get_nvlink_islands(topo) if topo["matrix"] else []

    # Create team
    team_id = _short_id(name)
    team = Team(
        team_id=team_id,
        name=name,
        description=description,
        gpu_ids=gpu_ids,
        gpus=gpu_dicts,
        topology=topo.get("matrix", {}),
        nvlink_islands=islands,
    )

    team_dir = ensure_team_dir(team_id)
    atomic_write(team_dir / "team.json", team.to_dict())

    # Display summary
    panel_text = f"[bold]Team ID:[/] {team_id}\n"
    panel_text += f"[bold]Name:[/] {name}\n"
    if description:
        panel_text += f"[bold]Description:[/] {description}\n"
    panel_text += f"[bold]GPUs:[/] {gpu_ids if gpu_ids else 'none detected'}\n"
    if islands:
        panel_text += f"[bold]NVLink Islands:[/] {islands}\n"
    panel_text += f"[bold]Status:[/] active"

    console.print(Panel(panel_text, title="Team Created", border_style="green"))


@app.command("status")
def team_status(
    team_id: str = typer.Argument(..., help="Team ID"),
):
    """Show team status, agents, and task summary."""
    from nemospawn.core.config import AGENTS_SUBDIR, TASKS_SUBDIR

    team_dir = TEAMS_DIR / team_id
    team_data = atomic_read(team_dir / "team.json")
    if not team_data:
        console.print(f"[red]Team '{team_id}' not found[/]")
        raise typer.Exit(1)

    team = Team.from_dict(team_data)
    console.print(Panel(
        f"[bold]Name:[/] {team.name}\n"
        f"[bold]Status:[/] {team.status}\n"
        f"[bold]GPUs:[/] {team.gpu_ids}\n"
        f"[bold]Created:[/] {team.created_at}",
        title=f"Team: {team_id}",
    ))

    # List agents
    agents_dir = team_dir / AGENTS_SUBDIR
    agent_files = sorted(agents_dir.glob("*.json")) if agents_dir.is_dir() else []
    if agent_files:
        table = Table(title="Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Role")
        table.add_column("GPUs")
        table.add_column("Status")

        for af in agent_files:
            a = atomic_read(af)
            if a:
                table.add_row(a["agent_id"], a.get("name", ""), a.get("role", ""), str(a.get("gpu_ids", [])), a.get("status", ""))
        console.print(table)

    # Task summary
    tasks_dir = team_dir / TASKS_SUBDIR
    task_files = sorted(tasks_dir.glob("*.json")) if tasks_dir.is_dir() else []
    if task_files:
        counts: dict[str, int] = {}
        for tf in task_files:
            t = atomic_read(tf)
            if t:
                s = t.get("status", "unknown")
                counts[s] = counts.get(s, 0) + 1
        summary = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        console.print(f"\n[bold]Tasks:[/] {summary}")


@app.command("list")
def list_teams():
    """List all teams."""
    ensure_state_dir()
    if not TEAMS_DIR.is_dir():
        console.print("[yellow]No teams found[/]")
        raise typer.Exit()

    team_dirs = sorted(TEAMS_DIR.iterdir())
    if not team_dirs:
        console.print("[yellow]No teams found[/]")
        raise typer.Exit()

    table = Table(title="Teams")
    table.add_column("Team ID", style="cyan")
    table.add_column("Name")
    table.add_column("GPUs")
    table.add_column("Status")
    table.add_column("Created")

    for td in team_dirs:
        data = atomic_read(td / "team.json")
        if data:
            table.add_row(
                data["team_id"],
                data.get("name", ""),
                str(data.get("gpu_ids", [])),
                data.get("status", ""),
                data.get("created_at", "")[:19],
            )
    console.print(table)


@app.command("topology")
def team_topology(
    team_id: str = typer.Argument(..., help="Team ID"),
):
    """Show NVLink topology for a team's GPUs."""
    team_dir = TEAMS_DIR / team_id
    team_data = atomic_read(team_dir / "team.json")
    if not team_data:
        console.print(f"[red]Team '{team_id}' not found[/]")
        raise typer.Exit(1)

    team = Team.from_dict(team_data)

    if team.topology:
        table = Table(title="NVLink Topology")
        gpu_ids = sorted(int(k) for k in team.topology.keys())
        table.add_column("GPU", style="cyan")
        for g in gpu_ids:
            table.add_column(f"GPU {g}")
        for g in gpu_ids:
            row = [str(g)]
            for g2 in gpu_ids:
                link = team.topology.get(str(g), {}).get(str(g2), team.topology.get(g, {}).get(g2, "?"))
                row.append(str(link))
            table.add_row(*row)
        console.print(table)

    if team.nvlink_islands:
        console.print("\n[bold]NVLink Islands:[/]")
        for i, island in enumerate(team.nvlink_islands):
            console.print(f"  Island {i}: {island}")
    else:
        console.print("[yellow]No NVLink topology data[/]")
