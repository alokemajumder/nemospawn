"""SLURM/PBS job management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("generate")
def generate(
    job_name: str = typer.Argument(..., help="SLURM job name"),
    gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
    gpu_type: str = typer.Option("", "--gpu-type", help="GPU type (e.g., h100, a100)"),
    nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
    partition: str = typer.Option("", "--partition", help="SLURM partition"),
    time: str = typer.Option("24:00:00", "--time", help="Time limit (HH:MM:SS)"),
    command: str = typer.Option("", "--command", "-c", help="Command to run"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output script path"),
):
    """Generate a SLURM sbatch script for a NemoSpawn workload."""
    from nemospawn.runtime.slurm import SlurmJobConfig, generate_sbatch_script, write_sbatch_script

    config = SlurmJobConfig(
        job_name=job_name,
        gpu_count=gpus,
        gpu_type=gpu_type,
        num_nodes=nodes,
        partition=partition,
        time_limit=time,
        command=command,
    )

    if output:
        path = write_sbatch_script(config, Path(output))
        console.print(f"[green]Script written to {path}[/]")
    else:
        script = generate_sbatch_script(config)
        console.print(script)


@app.command("submit")
def submit(
    script: str = typer.Argument(..., help="Path to sbatch script"),
):
    """Submit a SLURM batch job."""
    from nemospawn.runtime.slurm import submit_sbatch

    job_id = submit_sbatch(Path(script))
    if job_id:
        console.print(f"[green]Submitted job {job_id}[/]")
    else:
        console.print("[red]Failed to submit job (is SLURM available?)[/]")
        raise typer.Exit(1)


@app.command("status")
def status(
    job_id: str = typer.Argument(..., help="SLURM job ID"),
):
    """Check SLURM job status."""
    from nemospawn.runtime.slurm import check_job_status

    info = check_job_status(job_id)
    style = "green" if info["state"] in ("RUNNING", "PENDING") else "yellow"
    console.print(f"Job {info['job_id']}: [{style}]{info['state']}[/]")
    if info.get("time"):
        console.print(f"  Time: {info['time']}")
    if info.get("node"):
        console.print(f"  Node: {info['node']}")


@app.command("cancel")
def cancel(
    job_id: str = typer.Argument(..., help="SLURM job ID"),
):
    """Cancel a SLURM job."""
    from nemospawn.runtime.slurm import cancel_job

    if cancel_job(job_id):
        console.print(f"[green]Job {job_id} cancelled[/]")
    else:
        console.print(f"[red]Failed to cancel job {job_id}[/]")
        raise typer.Exit(1)
