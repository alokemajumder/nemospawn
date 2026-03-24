"""GPU discovery and health commands."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def discover():
    """Discover available NVIDIA GPUs."""
    from nemospawn.gpu.discovery import discover_gpus

    gpus = discover_gpus()
    if not gpus:
        console.print("[yellow]No NVIDIA GPUs detected[/]")
        raise typer.Exit()

    table = Table(title="NVIDIA GPUs")
    table.add_column("Index", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("UUID", style="dim")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Used (MB)", justify="right")

    for g in gpus:
        table.add_row(
            str(g.index), g.name, g.uuid[:16] + "...",
            str(g.memory_total_mb), str(g.memory_used_mb),
        )
    console.print(table)


@app.command()
def health():
    """Show GPU health metrics via NVML."""
    from nemospawn.gpu.health import check_all_gpus

    metrics = check_all_gpus()
    if not metrics:
        console.print("[yellow]No GPU health data available[/]")
        raise typer.Exit()

    table = Table(title="GPU Health")
    table.add_column("GPU", style="cyan")
    table.add_column("Temp (C)", justify="right")
    table.add_column("GPU Util %", justify="right")
    table.add_column("Mem Util %", justify="right")
    table.add_column("Power (W)", justify="right")
    table.add_column("ECC Errors", justify="right")

    for m in metrics:
        if "error" in m:
            table.add_row(str(m["gpu_index"]), f"[red]{m['error']}[/]", "", "", "", "")
        else:
            table.add_row(
                str(m["gpu_index"]),
                str(m["temperature_c"]),
                str(m["gpu_utilization_pct"]),
                str(m["memory_utilization_pct"]),
                str(m["power_draw_w"]),
                str(m["ecc_errors"]),
            )
    console.print(table)


@app.command()
def topology():
    """Show NVLink topology and island groupings."""
    from nemospawn.gpu.topology import parse_topology, get_nvlink_islands

    topo = parse_topology()
    if not topo["matrix"]:
        console.print("[yellow]No GPU topology data available[/]")
        raise typer.Exit()

    console.print("[bold]GPU Topology Matrix[/]")
    console.print(topo["raw"])

    islands = get_nvlink_islands(topo)
    if islands:
        console.print("\n[bold]NVLink Islands[/]")
        for i, island in enumerate(islands):
            gpu_str = ", ".join(f"GPU {g}" for g in island)
            console.print(f"  Island {i}: [{gpu_str}]")


@app.command()
def status(
    team_id: str = typer.Argument(..., help="Team ID"),
):
    """Show DCGM GPU metrics for all agents in a team."""
    from nemospawn.gpu.dcgm import poll_dcgm, save_metrics_snapshot, detect_underperforming_gpus

    metrics = poll_dcgm()
    if not metrics:
        console.print("[yellow]No DCGM/GPU metrics available[/]")
        raise typer.Exit()

    # Save snapshot
    save_metrics_snapshot(team_id, metrics)

    table = Table(title=f"GPU Status — {team_id}")
    table.add_column("GPU", style="cyan")
    table.add_column("SM Util %", justify="right")
    table.add_column("Mem Util %", justify="right")
    table.add_column("Temp (C)", justify="right")
    table.add_column("Power (W)", justify="right")
    table.add_column("ECC SBE", justify="right")
    table.add_column("ECC DBE", justify="right")

    for m in metrics:
        table.add_row(
            str(m["gpu_id"]),
            f"{m['sm_util']:.0f}",
            f"{m['mem_util']:.0f}",
            f"{m['temp']:.0f}",
            f"{m['power']:.1f}",
            str(m.get("ecc_sbe", 0)),
            str(m.get("ecc_dbe", 0)),
        )
    console.print(table)

    # Detect problems
    problems = detect_underperforming_gpus(metrics)
    if problems:
        console.print("\n[yellow]Potential issues detected:[/]")
        for p in problems:
            console.print(f"  GPU {p['gpu_id']}: {'; '.join(p['reasons'])}")


@app.command()
def reallocate(
    team_id: str = typer.Argument(..., help="Team ID"),
    kill_below: float = typer.Option(50.0, "--kill-below", help="Kill agents with SM utilization below this threshold"),
):
    """Kill underperforming agents based on GPU utilization."""
    from nemospawn.gpu.dcgm import poll_dcgm, detect_underperforming_gpus
    from nemospawn.core.config import AGENTS_SUBDIR
    from nemospawn.core.state import atomic_read, get_team_dir, list_json_files

    metrics = poll_dcgm()
    if not metrics:
        console.print("[yellow]No GPU metrics available[/]")
        raise typer.Exit()

    problems = detect_underperforming_gpus(metrics, util_threshold=kill_below)
    if not problems:
        console.print("[green]All GPUs performing above threshold[/]")
        return

    problem_gpu_ids = {p["gpu_id"] for p in problems}

    # Find agents on those GPUs
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    agents_to_kill = []
    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if data and data.get("status") == "running":
            agent_gpus = set(data.get("gpu_ids", []))
            if agent_gpus & problem_gpu_ids:
                agents_to_kill.append(data)

    if not agents_to_kill:
        console.print("[yellow]No running agents on underperforming GPUs[/]")
        return

    console.print(f"[yellow]Found {len(agents_to_kill)} agent(s) on underperforming GPUs:[/]")
    for a in agents_to_kill:
        console.print(f"  {a['agent_id']} (GPUs: {a.get('gpu_ids', [])})")
    console.print("[dim]Use 'nemospawn spawn kill' to stop them[/]")
