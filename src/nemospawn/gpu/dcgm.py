"""DCGM (Data Center GPU Manager) polling for GPU observability.

Queries DCGM metrics (temperature, power, SM utilization, memory, ECC errors)
and cross-correlates with val_loss trends to detect runaway training or
silent GPU failures.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nemospawn.core.config import METRICS_SUBDIR
from nemospawn.core.state import atomic_write, atomic_read, get_team_dir, list_json_files


@dataclass
class DCGMSnapshot:
    """A point-in-time DCGM metrics snapshot for all GPUs."""
    timestamp: str
    gpu_metrics: list[dict]  # one dict per GPU with all DCGM fields
    team_id: str = ""
    snapshot_id: str = ""


def poll_dcgm() -> list[dict]:
    """Poll DCGM metrics for all GPUs via dcgmi dmon.

    Returns a list of dicts, one per GPU, with keys:
    gpu_id, sm_util, mem_util, temp, power, ecc_sbe, ecc_dbe, pcie_tx, pcie_rx
    """
    try:
        # dcgmi dmon -e 203,204,150,155,310,311,409,410 -c 1
        # Fields: SM utilization, Memory utilization, GPU temp, Power usage,
        #         Single-bit ECC, Double-bit ECC, PCIe TX, PCIe RX
        result = subprocess.run(
            ["dcgmi", "dmon", "-e", "203,204,150,155,310,311", "-c", "1"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return _fallback_nvidia_smi()

        return _parse_dcgmi_output(result.stdout)

    except FileNotFoundError:
        return _fallback_nvidia_smi()
    except subprocess.TimeoutExpired:
        return _fallback_nvidia_smi()


def _parse_dcgmi_output(output: str) -> list[dict]:
    """Parse dcgmi dmon output into structured dicts."""
    metrics = []
    lines = output.strip().splitlines()

    # Find header line (starts with '#')
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("#") and "Entity" in line:
            header_idx = i
            break

    if header_idx < 0:
        return []

    # Parse data lines after header
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            metrics.append({
                "gpu_id": int(parts[1]) if parts[0] == "GPU" else int(parts[0]),
                "sm_util": _safe_float(parts[2] if len(parts) > 2 else "N/A"),
                "mem_util": _safe_float(parts[3] if len(parts) > 3 else "N/A"),
                "temp": _safe_float(parts[4] if len(parts) > 4 else "N/A"),
                "power": _safe_float(parts[5] if len(parts) > 5 else "N/A"),
                "ecc_sbe": _safe_int(parts[6] if len(parts) > 6 else "0"),
                "ecc_dbe": _safe_int(parts[7] if len(parts) > 7 else "0"),
            })
        except (ValueError, IndexError):
            continue

    return metrics


def _fallback_nvidia_smi() -> list[dict]:
    """Fallback to nvidia-smi for basic GPU metrics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        metrics = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                metrics.append({
                    "gpu_id": int(parts[0]),
                    "sm_util": _safe_float(parts[1]),
                    "mem_util": _safe_float(parts[2]),
                    "temp": _safe_float(parts[3]),
                    "power": _safe_float(parts[4]),
                    "ecc_sbe": 0,
                    "ecc_dbe": 0,
                })
        return metrics
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def save_metrics_snapshot(team_id: str, metrics: list[dict]) -> Path:
    """Save a DCGM metrics snapshot to the team's metrics directory."""
    metrics_dir = get_team_dir(team_id) / METRICS_SUBDIR
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "timestamp": ts,
        "gpu_metrics": metrics,
    }

    filename = f"dcgm_{ts.replace(':', '-')}.json"
    path = metrics_dir / filename
    atomic_write(path, snapshot)
    return path


def get_gpu_utilization_history(team_id: str, gpu_id: int, last_n: int = 20) -> list[dict]:
    """Get recent GPU utilization history for a specific GPU."""
    metrics_dir = get_team_dir(team_id) / METRICS_SUBDIR
    snapshots = list_json_files(metrics_dir)

    history = []
    for f in snapshots[-last_n:]:
        data = atomic_read(f)
        if data and "gpu_metrics" in data:
            for gm in data["gpu_metrics"]:
                if gm.get("gpu_id") == gpu_id:
                    history.append({
                        "timestamp": data["timestamp"],
                        **gm,
                    })
    return history


def detect_underperforming_gpus(
    metrics: list[dict],
    util_threshold: float = 50.0,
    temp_threshold: float = 90.0,
) -> list[dict]:
    """Detect GPUs that are underperforming or unhealthy.

    Returns list of problem GPUs with reasons.
    """
    problems = []
    for gm in metrics:
        reasons = []
        if gm.get("sm_util", 100) < util_threshold:
            reasons.append(f"low SM utilization ({gm['sm_util']}% < {util_threshold}%)")
        if gm.get("temp", 0) > temp_threshold:
            reasons.append(f"high temperature ({gm['temp']}C > {temp_threshold}C)")
        if gm.get("ecc_dbe", 0) > 0:
            reasons.append(f"double-bit ECC errors ({gm['ecc_dbe']})")
        if reasons:
            problems.append({"gpu_id": gm["gpu_id"], "reasons": reasons})
    return problems


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: str) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0
