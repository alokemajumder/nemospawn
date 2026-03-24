"""Monitoring dashboard commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("attach")
def attach(
    team_id: str = typer.Argument(..., help="Team ID"),
):
    """Attach to a tiled tmux view of all team agents."""
    import subprocess

    from nemospawn.core.config import AGENTS_SUBDIR, TMUX_PREFIX
    from nemospawn.core.state import atomic_read, get_team_dir
    from nemospawn.runtime.tmux import create_tiled_view

    team_dir = get_team_dir(team_id)
    agents_dir = team_dir / AGENTS_SUBDIR
    if not agents_dir.is_dir():
        console.print(f"[yellow]No agents in team '{team_id}'[/]")
        raise typer.Exit()

    sessions = []
    for af in sorted(agents_dir.glob("*.json")):
        a = atomic_read(af)
        if a and a.get("status") == "running" and a.get("tmux_session"):
            sessions.append(a["tmux_session"])

    if not sessions:
        console.print("[yellow]No running agents to display[/]")
        raise typer.Exit()

    view_name = f"{TMUX_PREFIX}-board-{team_id}"
    if create_tiled_view(sessions, view_name):
        console.print(f"[green]Board created: {view_name}[/]")
        subprocess.run(["tmux", "attach-session", "-t", view_name])
    else:
        console.print("[red]Failed to create board view[/]")


@app.command("live")
def live(
    team_id: str = typer.Argument(..., help="Team ID"),
    interval: int = typer.Option(30, "--interval", help="Refresh interval in seconds"),
):
    """Live terminal kanban board for team tasks and agents."""
    from nemospawn.observability.kanban import live_kanban
    live_kanban(team_id, interval=interval)


@app.command("serve")
def serve(
    team_id: str = typer.Argument(..., help="Team ID"),
    port: int = typer.Option(8080, "--port", help="Web UI port"),
    metrics_port: int = typer.Option(9090, "--metrics-port", help="Prometheus scrape port"),
    grafana_url: Optional[str] = typer.Option(None, "--grafana-url", help="Grafana URL for auto-provisioning"),
    grafana_key: Optional[str] = typer.Option(None, "--grafana-key", help="Grafana API key"),
):
    """Start web UI dashboard with SSE real-time updates, Prometheus metrics, and optional Grafana."""
    from nemospawn.observability.prometheus import start_metrics_server
    from nemospawn.observability.grafana import provision_dashboard, write_dashboard
    from nemospawn.observability.web import start_web_board
    from pathlib import Path

    # Start web UI board
    web_server = start_web_board(team_id, port=port)
    console.print(f"[green]Web UI at http://localhost:{port}[/]")

    # Start Prometheus scrape endpoint
    metrics_server = start_metrics_server(team_id, port=metrics_port)
    console.print(f"[green]Prometheus metrics at http://localhost:{metrics_port}/metrics[/]")

    # Auto-provision Grafana dashboard if URL provided
    if grafana_url:
        if provision_dashboard(team_id, grafana_url, api_key=grafana_key or ""):
            console.print(f"[green]Grafana dashboard provisioned at {grafana_url}[/]")
        else:
            console.print("[yellow]Failed to provision Grafana dashboard — saving JSON instead[/]")
            path = write_dashboard(team_id, Path.home() / ".nemospawn" / "dashboards" / f"{team_id}.json")
            console.print(f"[dim]Dashboard saved to {path}[/]")
    else:
        # Save dashboard JSON for manual import
        path = write_dashboard(team_id, Path.home() / ".nemospawn" / "dashboards" / f"{team_id}.json")
        console.print(f"[dim]Grafana dashboard JSON saved to {path}[/]")

    console.print(f"\n[bold]Board serving for team '{team_id}'[/]")
    console.print("Press Ctrl+C to stop")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        web_server.shutdown()
        metrics_server.shutdown()
        console.print("\n[dim]Board stopped[/]")
