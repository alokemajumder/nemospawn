"""Configuration management CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("show")
def config_show():
    """Show all configuration settings with sources."""
    from nemospawn.core.settings import get_all

    settings = get_all()
    table = Table(title="NemoSpawn Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")
    table.add_column("Description")

    source_colors = {"env": "green", "config file": "yellow", "default": "dim"}
    for key, info in sorted(settings.items()):
        source = info["source"]
        color = "green" if "env" in source else ("yellow" if "config" in source else "dim")
        table.add_row(
            key,
            info["value"] or "(empty)",
            f"[{color}]{source}[/]",
            info["description"],
        )
    console.print(table)


@app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key to read"),
):
    """Get a specific config value."""
    from nemospawn.core.settings import get, DEFAULTS

    if key not in DEFAULTS:
        console.print(f"[red]Unknown config key: {key}[/]")
        console.print(f"[dim]Valid keys: {', '.join(sorted(DEFAULTS.keys()))}[/]")
        raise typer.Exit(1)

    value = get(key)
    console.print(value)


@app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key to set"),
    value: str = typer.Argument(..., help="Value to set"),
):
    """Set a config value (persisted to ~/.nemospawn/config.json)."""
    from nemospawn.core.settings import set_value, DEFAULTS

    if not set_value(key, value):
        console.print(f"[red]Unknown config key: {key}[/]")
        console.print(f"[dim]Valid keys: {', '.join(sorted(DEFAULTS.keys()))}[/]")
        raise typer.Exit(1)

    console.print(f"[green]{key} = {value}[/]")


@app.command("health")
def config_health():
    """Run configuration health checks."""
    from nemospawn.core.settings import health_check

    checks = health_check()
    console.print(Panel("Running configuration diagnostics...", title="Config Health", border_style="cyan"))

    all_ok = True
    for c in checks:
        icon = "[green]PASS[/]" if c["ok"] else "[red]FAIL[/]"
        console.print(f"  {icon}  {c['check']}: {c['detail']}")
        if not c["ok"]:
            all_ok = False

    console.print("")
    if all_ok:
        console.print("[green]All checks passed[/]")
    else:
        console.print("[yellow]Some checks failed — review above[/]")
