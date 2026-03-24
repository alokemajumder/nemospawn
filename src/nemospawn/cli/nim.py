"""NIM deployment pipeline commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("deploy")
def deploy(
    team_id: str = typer.Argument(..., help="Team ID"),
    artifact_id: str = typer.Argument(..., help="Artifact ID of .nemo checkpoint to deploy"),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel degree (1, 2, 4, 8)"),
    profile: str = typer.Option("default", "--profile", help="NIM profile (default, max-throughput, min-latency)"),
    port: int = typer.Option(8000, "--port", help="Endpoint port"),
    gpus: Optional[str] = typer.Option(None, "--gpus", help="GPU indices for the NIM container"),
):
    """Deploy a .nemo checkpoint as a NIM container endpoint."""
    from nemospawn.core.config import ARTIFACTS_SUBDIR
    from nemospawn.core.state import atomic_read, atomic_write, get_team_dir
    from nemospawn.nim.deployer import (
        NIMEndpoint, build_nim_container, start_nim_endpoint, check_nim_health,
    )
    from nemospawn.core.models import _short_id

    # Load artifact
    art_data = atomic_read(get_team_dir(team_id) / ARTIFACTS_SUBDIR / f"{artifact_id}.json")
    if not art_data:
        console.print(f"[red]Artifact '{artifact_id}' not found[/]")
        raise typer.Exit(1)

    checkpoint_path = art_data["path"]
    image_name = f"nemospawn-nim/{team_id}-{artifact_id}:latest"

    console.print(f"Building NIM container from {checkpoint_path} (TP{tp}, {profile})...")

    # Build container
    built = build_nim_container(checkpoint_path, image_name, tp_degree=tp, profile=profile)
    if not built:
        console.print("[red]NIM container build failed[/]")
        raise typer.Exit(1)

    # Start endpoint
    gpu_ids = [int(g) for g in gpus.split(",")] if gpus else None
    container_name = f"nim-{team_id}-{_short_id('nim')}"

    container_id = start_nim_endpoint(image_name, port=port, gpu_ids=gpu_ids, name=container_name)
    if not container_id:
        console.print("[red]Failed to start NIM endpoint[/]")
        raise typer.Exit(1)

    endpoint_url = f"http://localhost:{port}"

    # Health check
    console.print("Waiting for NIM endpoint to become healthy...")
    healthy = check_nim_health(endpoint_url)
    status = "running" if healthy else "unhealthy"

    # Save endpoint state
    endpoint_id = _short_id("nim")
    endpoint = NIMEndpoint(
        endpoint_id=endpoint_id,
        team_id=team_id,
        artifact_id=artifact_id,
        container_image=image_name,
        endpoint_url=endpoint_url,
        tp_degree=tp,
        profile=profile,
        status=status,
        container_id=container_id,
        port=port,
        gpu_ids=gpu_ids or [],
    )

    nim_dir = get_team_dir(team_id) / "nim"
    nim_dir.mkdir(exist_ok=True)
    atomic_write(nim_dir / f"{endpoint_id}.json", endpoint.to_dict())

    console.print(Panel(
        f"[bold]Endpoint ID:[/] {endpoint_id}\n"
        f"[bold]URL:[/] {endpoint_url}\n"
        f"[bold]Container:[/] {container_id}\n"
        f"[bold]TP Degree:[/] {tp}\n"
        f"[bold]Profile:[/] {profile}\n"
        f"[bold]Status:[/] {'[green]running[/]' if healthy else '[yellow]unhealthy[/]'}",
        title="NIM Endpoint Deployed",
        border_style="green" if healthy else "yellow",
    ))


@app.command("list")
def list_nim(
    team_id: str = typer.Argument(..., help="Team ID"),
):
    """List NIM endpoints for a team."""
    from nemospawn.core.state import atomic_read, get_team_dir, list_json_files

    nim_dir = get_team_dir(team_id) / "nim"
    if not nim_dir.is_dir():
        console.print("[yellow]No NIM endpoints[/]")
        raise typer.Exit()

    files = list_json_files(nim_dir)
    if not files:
        console.print("[yellow]No NIM endpoints[/]")
        raise typer.Exit()

    table = Table(title=f"NIM Endpoints — {team_id}")
    table.add_column("ID", style="cyan")
    table.add_column("URL")
    table.add_column("TP")
    table.add_column("Profile")
    table.add_column("Artifact")
    table.add_column("Status")

    for f in files:
        data = atomic_read(f)
        if data:
            s = data.get("status", "unknown")
            style = "green" if s == "running" else "red"
            table.add_row(
                data.get("endpoint_id", ""),
                data.get("endpoint_url", ""),
                str(data.get("tp_degree", 1)),
                data.get("profile", ""),
                data.get("artifact_id", ""),
                f"[{style}]{s}[/]",
            )
    console.print(table)


@app.command("benchmark")
def benchmark(
    team_id: str = typer.Argument(..., help="Team ID"),
    endpoint_url: str = typer.Argument(..., help="NIM endpoint URL (e.g., localhost:8000)"),
    model_name: str = typer.Option("model", "--model", help="Model name"),
    concurrency: str = typer.Option("1,4,16,64", "--concurrency", help="Comma-separated concurrency levels"),
):
    """Run perf_analyzer benchmark against a NIM endpoint."""
    from nemospawn.nim.triton import run_perf_analyzer, rank_endpoints

    conc_levels = [int(c.strip()) for c in concurrency.split(",")]

    console.print(f"Benchmarking {endpoint_url} at concurrency levels {conc_levels}...")
    results = run_perf_analyzer(endpoint_url, model_name=model_name, concurrency_levels=conc_levels)

    if not results:
        console.print("[red]No benchmark results[/]")
        raise typer.Exit(1)

    table = Table(title=f"Benchmark — {endpoint_url}")
    table.add_column("Concurrency", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("p99 (ms)", justify="right")
    table.add_column("Throughput (infer/s)", justify="right")

    for r in results:
        table.add_row(
            str(r.concurrency),
            f"{r.p50_latency_ms:.1f}",
            f"{r.p95_latency_ms:.1f}",
            f"{r.p99_latency_ms:.1f}",
            f"{r.throughput_infer_per_sec:.1f}",
        )
    console.print(table)

    # Show ranking
    ranked = rank_endpoints(results)
    if ranked:
        best = ranked[0]
        console.print(f"\n[green]Best config:[/] concurrency={best.concurrency}, "
                       f"throughput={best.throughput_infer_per_sec:.1f} infer/s, "
                       f"p95={best.p95_latency_ms:.1f}ms")


@app.command("profiles")
def profiles(
    num_gpus: int = typer.Argument(..., help="Number of available GPUs"),
):
    """Show available NIM tensor-parallel profiles for a GPU count."""
    from nemospawn.nim.deployer import generate_nim_profiles

    profs = generate_nim_profiles(num_gpus)

    table = Table(title=f"NIM Profiles ({num_gpus} GPUs)")
    table.add_column("Profile", style="cyan")
    table.add_column("TP Degree", justify="right")
    table.add_column("GPUs Required", justify="right")
    table.add_column("Description")

    for p in profs:
        table.add_row(p["profile_name"], str(p["tp_degree"]), str(p["gpus_required"]), p["description"])
    console.print(table)
