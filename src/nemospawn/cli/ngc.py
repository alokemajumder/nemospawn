"""NGC (NVIDIA GPU Cloud) registry commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("pull")
def pull(
    org: str = typer.Argument(..., help="NGC org name"),
    model: str = typer.Argument(..., help="Model name"),
    version: str = typer.Option("latest", "--version", help="Model version"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Download a model from NGC registry."""
    from nemospawn.ngc.registry import download_model

    out_path = Path(output) if output else None
    result = download_model(org, model, version, output_dir=out_path)
    if result:
        console.print(f"[green]Downloaded to {result}[/]")
    else:
        console.print("[red]Download failed[/]")
        raise typer.Exit(1)


@app.command("push")
def push(
    path: str = typer.Argument(..., help="Local model path"),
    org: str = typer.Argument(..., help="NGC org name"),
    model: str = typer.Argument(..., help="Model name"),
    version: str = typer.Option("latest", "--version", help="Model version"),
    description: str = typer.Option("", "--desc", help="Model description"),
):
    """Upload a model to NGC registry."""
    from nemospawn.ngc.registry import upload_model

    success = upload_model(Path(path), org, model, version, description)
    if not success:
        raise typer.Exit(1)


@app.command("push-container")
def push_container(
    image: str = typer.Argument(..., help="Local Docker image name"),
    org: str = typer.Argument(..., help="NGC org name"),
    name: str = typer.Argument(..., help="Container name on NGC"),
    tag: str = typer.Option("latest", "--tag", help="Image tag"),
):
    """Push a Docker container to NGC registry (nvcr.io)."""
    from nemospawn.ngc.registry import push_container as _push

    success = _push(image, org, name, tag)
    if not success:
        raise typer.Exit(1)


@app.command("list")
def list_models(
    org: str = typer.Argument(..., help="NGC org name"),
    pattern: str = typer.Option("*", "--filter", help="Name filter pattern"),
):
    """List models in an NGC org."""
    from nemospawn.ngc.registry import list_models as _list

    models = _list(org, pattern)
    if not models:
        console.print("[yellow]No models found[/]")
        raise typer.Exit()

    table = Table(title=f"NGC Models — {org}")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    for m in models:
        table.add_row(m["name"], m.get("version", ""))
    console.print(table)


@app.command("auth")
def check_auth():
    """Check NGC CLI authentication status."""
    from nemospawn.ngc.registry import check_ngc_auth

    if check_ngc_auth():
        console.print("[green]NGC CLI authenticated[/]")
    else:
        console.print("[red]NGC CLI not authenticated. Run: ngc config set[/]")
        raise typer.Exit(1)
