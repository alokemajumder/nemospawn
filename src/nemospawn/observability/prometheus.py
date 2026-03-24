"""Prometheus metrics export for DCGM GPU metrics and NemoSpawn state.

Exposes a scrape endpoint on `nemospawn board serve` that Prometheus
can poll. Metrics include GPU utilization, temperature, power,
val_loss per agent, and task counts.
"""

from __future__ import annotations

import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from nemospawn.core.config import AGENTS_SUBDIR, TASKS_SUBDIR, METRICS_SUBDIR
from nemospawn.core.state import atomic_read, get_team_dir, list_json_files


def generate_metrics(team_id: str) -> str:
    """Generate Prometheus-format metrics for a team."""
    lines: list[str] = []

    # GPU metrics from latest DCGM snapshot
    metrics_dir = get_team_dir(team_id) / METRICS_SUBDIR
    snapshots = list_json_files(metrics_dir)
    if snapshots:
        latest = atomic_read(snapshots[-1])
        if latest and "gpu_metrics" in latest:
            for gm in latest["gpu_metrics"]:
                gpu_id = gm.get("gpu_id", 0)
                labels = f'team="{team_id}",gpu="{gpu_id}"'
                lines.append(f'nemospawn_gpu_sm_utilization{{{labels}}} {gm.get("sm_util", 0)}')
                lines.append(f'nemospawn_gpu_mem_utilization{{{labels}}} {gm.get("mem_util", 0)}')
                lines.append(f'nemospawn_gpu_temperature_celsius{{{labels}}} {gm.get("temp", 0)}')
                lines.append(f'nemospawn_gpu_power_watts{{{labels}}} {gm.get("power", 0)}')
                lines.append(f'nemospawn_gpu_ecc_sbe_total{{{labels}}} {gm.get("ecc_sbe", 0)}')
                lines.append(f'nemospawn_gpu_ecc_dbe_total{{{labels}}} {gm.get("ecc_dbe", 0)}')

    # Agent metrics
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    agent_count = {"running": 0, "stopped": 0, "spawning": 0}
    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if data:
            status = data.get("status", "unknown")
            agent_count[status] = agent_count.get(status, 0) + 1
    for status, count in agent_count.items():
        lines.append(f'nemospawn_agents_total{{team="{team_id}",status="{status}"}} {count}')

    # Task metrics
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    task_count = {"pending": 0, "blocked": 0, "running": 0, "done": 0, "failed": 0}
    for f in list_json_files(tasks_dir):
        data = atomic_read(f)
        if data:
            status = data.get("status", "unknown")
            task_count[status] = task_count.get(status, 0) + 1
            # Val loss per agent
            val_loss = data.get("metadata", {}).get("val_loss")
            agent_id = data.get("agent_id", "unassigned")
            if val_loss is not None:
                lines.append(f'nemospawn_val_loss{{team="{team_id}",agent="{agent_id}",task="{data.get("task_id","")}"}} {val_loss}')
    for status, count in task_count.items():
        lines.append(f'nemospawn_tasks_total{{team="{team_id}",status="{status}"}} {count}')

    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus scrape endpoint."""

    team_id: str = ""

    def do_GET(self):
        if self.path == "/metrics":
            metrics = generate_metrics(self.team_id)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logging


def start_metrics_server(team_id: str, port: int = 9090) -> HTTPServer:
    """Start a Prometheus metrics scrape endpoint in a background thread."""
    MetricsHandler.team_id = team_id
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
