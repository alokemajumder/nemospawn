"""Agent skill management CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("install")
def skill_install(
    target: str = typer.Option("claude", "--target", "-t", help="Target agent: claude, codex, or all"),
):
    """Install NemoSpawn skill for agent CLIs (Claude Code, Codex)."""
    from nemospawn.core.skill import install_skill

    targets = ["claude", "codex"] if target == "all" else [target]
    installed = install_skill(targets)

    if installed:
        for path in installed:
            console.print(f"[green]Installed:[/] {path}")
        console.print(f"\n[dim]Agents will now auto-discover NemoSpawn coordination commands[/]")
    else:
        console.print(f"[yellow]No valid targets specified[/]")


@app.command("uninstall")
def skill_uninstall(
    target: str = typer.Option("all", "--target", "-t", help="Target agent: claude, codex, or all"),
):
    """Uninstall NemoSpawn skill from agent CLIs."""
    from nemospawn.core.skill import uninstall_skill

    targets = ["claude", "codex"] if target == "all" else [target]
    removed = uninstall_skill(targets)

    if removed:
        for path in removed:
            console.print(f"[green]Removed:[/] {path}")
    else:
        console.print(f"[yellow]No skill files found to remove[/]")


@app.command("status")
def skill_status():
    """Check if NemoSpawn skill is installed for each agent."""
    from nemospawn.core.skill import is_installed, SKILL_DIR_CLAUDE, SKILL_DIR_CODEX

    targets = {
        "claude": SKILL_DIR_CLAUDE,
        "codex": SKILL_DIR_CODEX,
    }

    for name, path in targets.items():
        installed = is_installed(name)
        status = "[green]installed[/]" if installed else "[dim]not installed[/]"
        console.print(f"  {name}: {status}  ({path})")
