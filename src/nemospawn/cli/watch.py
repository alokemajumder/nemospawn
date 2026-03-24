"""Agent watcher CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("status")
def watch_status(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """Run a single health check on all agents."""
    from nemospawn.core.watcher import watch_once

    result = watch_once(team_id)

    console.print(
        f"\n[bold]Health Check — {team_id}[/]  "
        f"[green]{result['healthy']} healthy[/] | "
        f"[red]{result['unhealthy']} unhealthy[/] | "
        f"{result['total']} total\n"
    )

    if result["agents"]:
        table = Table(title="Agent Health")
        table.add_column("Agent", style="cyan")
        table.add_column("Name")
        table.add_column("Role")
        table.add_column("Status")
        table.add_column("Healthy")
        table.add_column("Issues")

        for a in result["agents"]:
            health_style = "green" if a["healthy"] else "red"
            health_text = "yes" if a["healthy"] else "NO"
            issues = "; ".join(a.get("issues", [])) or "-"
            table.add_row(
                a["agent_id"][:12],
                a.get("name", ""),
                a.get("role", ""),
                a.get("status", ""),
                f"[{health_style}]{health_text}[/]",
                issues,
            )
        console.print(table)


@app.command("start")
def watch_start(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    interval: int = typer.Option(60, "--interval", help="Check interval in seconds"),
):
    """Start continuous agent health monitoring."""
    from nemospawn.core.watcher import watch_loop

    console.print(f"[bold]Starting agent watcher for '{team_id}' (every {interval}s)[/]")
    console.print("Press Ctrl+C to stop\n")

    def _print_result(result: dict) -> None:
        healthy = result["healthy"]
        unhealthy = result["unhealthy"]
        timestamp = result["checked_at"][:19]
        status = f"[green]{healthy} ok[/]" if unhealthy == 0 else f"[green]{healthy} ok[/] [red]{unhealthy} UNHEALTHY[/]"
        console.print(f"[dim]{timestamp}[/] {status}")

        # Print details for unhealthy agents
        for a in result["agents"]:
            if not a["healthy"]:
                issues = "; ".join(a.get("issues", []))
                console.print(f"  [red]{a['agent_id'][:12]}[/] ({a.get('name', '')}): {issues}")

    try:
        watch_loop(team_id, interval=interval, callback=_print_result)
    except KeyboardInterrupt:
        console.print("\n[dim]Watcher stopped[/]")
