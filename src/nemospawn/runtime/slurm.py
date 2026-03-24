"""SLURM/PBS job script generation and submission.

Generates sbatch scripts for NemoSpawn agent workloads with proper
GPU reservation, module loading, and NemoSpawn coordination env vars.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SlurmJobConfig:
    """Configuration for a SLURM batch job."""
    job_name: str
    gpu_count: int = 1
    gpu_type: str = ""  # e.g., "a100", "h100"
    num_nodes: int = 1
    partition: str = ""
    time_limit: str = "24:00:00"
    output_log: str = "nemospawn-%j.out"
    error_log: str = "nemospawn-%j.err"
    modules: list[str] | None = None  # e.g., ["cuda/12.4", "nccl"]
    env_vars: dict[str, str] | None = None
    command: str = ""


def generate_sbatch_script(config: SlurmJobConfig) -> str:
    """Generate an sbatch script for a NemoSpawn workload."""
    lines = ["#!/bin/bash"]

    # SBATCH directives
    lines.append(f"#SBATCH --job-name={config.job_name}")
    lines.append(f"#SBATCH --nodes={config.num_nodes}")
    lines.append(f"#SBATCH --time={config.time_limit}")
    lines.append(f"#SBATCH --output={config.output_log}")
    lines.append(f"#SBATCH --error={config.error_log}")

    if config.gpu_count > 0:
        gpu_spec = f"gpu:{config.gpu_type}:{config.gpu_count}" if config.gpu_type else f"gpu:{config.gpu_count}"
        lines.append(f"#SBATCH --gres={gpu_spec}")

    if config.partition:
        lines.append(f"#SBATCH --partition={config.partition}")

    lines.append("")

    # Module loading
    if config.modules:
        for mod in config.modules:
            lines.append(f"module load {mod}")
        lines.append("")

    # Environment variables
    if config.env_vars:
        for key, value in config.env_vars.items():
            lines.append(f'export {key}="{value}"')
        lines.append("")

    # Command
    if config.command:
        lines.append(config.command)

    return "\n".join(lines) + "\n"


def write_sbatch_script(config: SlurmJobConfig, output_path: Path) -> Path:
    """Write an sbatch script to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = generate_sbatch_script(config)
    output_path.write_text(script)
    output_path.chmod(0o755)
    return output_path


def submit_sbatch(script_path: Path) -> str | None:
    """Submit an sbatch script and return the job ID."""
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            # Output: "Submitted batch job 12345"
            parts = result.stdout.strip().split()
            return parts[-1] if parts else None
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_job_status(job_id: str) -> dict:
    """Check SLURM job status."""
    try:
        result = subprocess.run(
            ["squeue", "--job", job_id, "--format=%i %T %M %N", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            return {
                "job_id": parts[0],
                "state": parts[1] if len(parts) > 1 else "UNKNOWN",
                "time": parts[2] if len(parts) > 2 else "",
                "node": parts[3] if len(parts) > 3 else "",
            }
        return {"job_id": job_id, "state": "COMPLETED_OR_NOT_FOUND"}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"job_id": job_id, "state": "SLURM_NOT_AVAILABLE"}


def cancel_job(job_id: str) -> bool:
    """Cancel a SLURM job."""
    try:
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
