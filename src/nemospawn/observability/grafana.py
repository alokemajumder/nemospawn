"""Grafana dashboard auto-provisioning for NemoSpawn teams.

Generates a Grafana dashboard JSON template with panels for:
  - GPU SM utilization per agent
  - GPU temperature and power
  - Val loss over time per experiment
  - Task status breakdown
  - Agent lifecycle timeline
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_dashboard(team_id: str, prometheus_url: str = "http://localhost:9090") -> dict:
    """Generate a Grafana dashboard JSON for a NemoSpawn team."""
    return {
        "dashboard": {
            "id": None,
            "uid": f"nemospawn-{team_id}",
            "title": f"NemoSpawn — {team_id}",
            "tags": ["nemospawn", "gpu", team_id],
            "timezone": "utc",
            "refresh": "10s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": [
                _gpu_utilization_panel(team_id),
                _gpu_temperature_panel(team_id),
                _val_loss_panel(team_id),
                _task_status_panel(team_id),
                _agent_count_panel(team_id),
                _gpu_power_panel(team_id),
            ],
        },
        "overwrite": True,
    }


def _gpu_utilization_panel(team_id: str) -> dict:
    return {
        "title": "GPU SM Utilization",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [{
            "expr": f'nemospawn_gpu_sm_utilization{{team="{team_id}"}}',
            "legendFormat": "GPU {{gpu}}",
        }],
        "fieldConfig": {
            "defaults": {
                "unit": "percent",
                "min": 0, "max": 100,
                "thresholds": {"steps": [
                    {"value": 0, "color": "red"},
                    {"value": 50, "color": "yellow"},
                    {"value": 80, "color": "green"},
                ]},
            },
        },
    }


def _gpu_temperature_panel(team_id: str) -> dict:
    return {
        "title": "GPU Temperature",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [{
            "expr": f'nemospawn_gpu_temperature_celsius{{team="{team_id}"}}',
            "legendFormat": "GPU {{gpu}}",
        }],
        "fieldConfig": {
            "defaults": {
                "unit": "celsius",
                "thresholds": {"steps": [
                    {"value": 0, "color": "green"},
                    {"value": 75, "color": "yellow"},
                    {"value": 85, "color": "red"},
                ]},
            },
        },
    }


def _val_loss_panel(team_id: str) -> dict:
    return {
        "title": "Validation Loss per Agent",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [{
            "expr": f'nemospawn_val_loss{{team="{team_id}"}}',
            "legendFormat": "{{agent}}",
        }],
        "fieldConfig": {
            "defaults": {"unit": "none", "decimals": 4},
        },
    }


def _task_status_panel(team_id: str) -> dict:
    return {
        "title": "Task Status",
        "type": "piechart",
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
        "targets": [{
            "expr": f'nemospawn_tasks_total{{team="{team_id}"}}',
            "legendFormat": "{{status}}",
        }],
    }


def _agent_count_panel(team_id: str) -> dict:
    return {
        "title": "Agents",
        "type": "stat",
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8},
        "targets": [{
            "expr": f'nemospawn_agents_total{{team="{team_id}",status="running"}}',
            "legendFormat": "Running",
        }],
    }


def _gpu_power_panel(team_id: str) -> dict:
    return {
        "title": "GPU Power Draw",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
        "targets": [{
            "expr": f'nemospawn_gpu_power_watts{{team="{team_id}"}}',
            "legendFormat": "GPU {{gpu}}",
        }],
        "fieldConfig": {"defaults": {"unit": "watt"}},
    }


def write_dashboard(team_id: str, output_path: Path) -> Path:
    """Write Grafana dashboard JSON to a file for import."""
    dashboard = generate_dashboard(team_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    return output_path


def provision_dashboard(team_id: str, grafana_url: str = "http://localhost:3000", api_key: str = "") -> bool:
    """Auto-provision a Grafana dashboard via the Grafana HTTP API."""
    import urllib.request
    import urllib.error

    dashboard = generate_dashboard(team_id)
    data = json.dumps(dashboard).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{grafana_url}/api/dashboards/db",
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False
