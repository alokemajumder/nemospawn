"""Authentication and audit log commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("create-user")
def create_user_cmd(
    username: str = typer.Argument(..., help="Username"),
    role: str = typer.Option("user", "--role", help="Role: user or admin"),
):
    """Create a new user with an API key."""
    from nemospawn.core.auth import create_user
    from nemospawn.core.audit import log_event

    user, key = create_user(username, role=role)
    log_event("user.create", {"username": username, "role": role})
    console.print(Panel(
        f"[bold]Username:[/] {user.username}\n"
        f"[bold]Role:[/] {user.role}\n"
        f"[bold]Namespace:[/] {user.namespace}\n"
        f"[bold]API Key:[/] [green]{key}[/]\n\n"
        f"[dim]Save this key — it cannot be retrieved later.[/]",
        title="User Created",
        border_style="green",
    ))


@app.command("verify")
def verify(
    api_key: str = typer.Argument(..., help="API key to verify"),
):
    """Verify an API key and show the associated user."""
    from nemospawn.core.auth import authenticate

    user = authenticate(api_key)
    if user:
        console.print(f"[green]Authenticated:[/] {user.username} (role: {user.role})")
    else:
        console.print("[red]Invalid API key[/]")
        raise typer.Exit(1)


@app.command("audit")
def audit_log(
    last_n: int = typer.Option(50, "--last", "-n", help="Number of recent entries"),
    event: Optional[str] = typer.Option(None, "--event", help="Filter by event type"),
    team: Optional[str] = typer.Option(None, "--team", help="Filter by team ID"),
):
    """View the audit log."""
    from nemospawn.core.audit import read_audit_log

    entries = read_audit_log(last_n=last_n, event_type=event, team_id=team)
    if not entries:
        console.print("[yellow]No audit log entries[/]")
        raise typer.Exit()

    table = Table(title="Audit Log")
    table.add_column("Time", style="dim")
    table.add_column("Event", style="cyan")
    table.add_column("User")
    table.add_column("Team")
    table.add_column("Details")

    for e in entries:
        details = ", ".join(f"{k}={v}" for k, v in (e.get("details") or {}).items())
        table.add_row(
            e.get("timestamp", "")[:19],
            e.get("event", ""),
            e.get("user", ""),
            e.get("team_id", ""),
            details[:60],
        )
    console.print(table)
