"""Cross-cluster federation commands."""

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
    name: str = typer.Argument(..., help="Cluster name"),
    host: str = typer.Option(..., "--host", help="Cluster hostname or IP"),
    key: Optional[str] = typer.Option(None, "--key", help="SSH key path"),
    user: str = typer.Option("root", "--user", help="SSH user"),
    mount: Optional[str] = typer.Option(None, "--mount", help="Shared filesystem mount point"),
):
    """Register a remote cluster for federation."""
    from nemospawn.federation.cluster import register_cluster

    cluster = register_cluster(name, host, ssh_key=key or "", ssh_user=user, mount_point=mount or "")
    console.print(Panel(
        f"[bold]Name:[/] {cluster.name}\n"
        f"[bold]Host:[/] {cluster.host}\n"
        f"[bold]GPUs:[/] {cluster.gpu_count}\n"
        f"[bold]Mount:[/] {cluster.mount_point or 'none'}\n"
        f"[bold]Status:[/] {cluster.status}",
        title="Cluster Registered",
        border_style="green",
    ))


@app.command("list")
def list_clusters():
    """List registered clusters."""
    from nemospawn.federation.cluster import list_clusters as _list

    clusters = _list()
    if not clusters:
        console.print("[yellow]No clusters registered[/]")
        raise typer.Exit()

    table = Table(title="Registered Clusters")
    table.add_column("Name", style="cyan")
    table.add_column("Host")
    table.add_column("GPUs", justify="right")
    table.add_column("Mount")
    table.add_column("Status")

    for c in clusters:
        style = "green" if c.status == "online" else "dim"
        table.add_row(c.name, c.host, str(c.gpu_count), c.mount_point or "", f"[{style}]{c.status}[/]")
    console.print(table)


@app.command("status")
def cluster_status(
    name: str = typer.Argument(..., help="Cluster name"),
):
    """Check cluster connectivity status."""
    from nemospawn.federation.cluster import get_cluster, check_cluster_status

    cluster = get_cluster(name)
    if not cluster:
        console.print(f"[red]Cluster '{name}' not found[/]")
        raise typer.Exit(1)

    status = check_cluster_status(cluster)
    style = "green" if status == "online" else "red"
    console.print(f"Cluster '{name}': [{style}]{status}[/]")
