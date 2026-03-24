"""GPU discovery via nvidia-smi and NVML."""

from __future__ import annotations

import subprocess
from rich.console import Console

from nemospawn.core.models import GPUInfo

console = Console(stderr=True)


def discover_gpus() -> list[GPUInfo]:
    """Enumerate GPUs via nvidia-smi CSV output."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            console.print("[yellow]nvidia-smi returned non-zero exit code[/]")
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        uuid=parts[2],
                        memory_total_mb=int(float(parts[3])),
                        memory_used_mb=int(float(parts[4])),
                    )
                )
        return gpus

    except FileNotFoundError:
        console.print("[yellow]nvidia-smi not found — no NVIDIA GPUs detected[/]")
        return []
    except subprocess.TimeoutExpired:
        console.print("[yellow]nvidia-smi timed out[/]")
        return []


def get_gpu_count() -> int:
    """Get GPU count, preferring NVML then falling back to nvidia-smi."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count
    except Exception:
        return len(discover_gpus())
