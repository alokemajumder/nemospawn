"""Tests for observability — Prometheus, Grafana, kanban."""

from unittest.mock import patch
from pathlib import Path

from nemospawn.observability.prometheus import generate_metrics
from nemospawn.observability.grafana import generate_dashboard, write_dashboard


def test_generate_metrics_empty(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.observability.prometheus.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        with patch("nemospawn.core.state.TEAMS_DIR", teams_dir):
            ensure_team_dir("t1")
        metrics = generate_metrics("t1")
        assert isinstance(metrics, str)
        # Should have task count lines even if all zeros
        assert "nemospawn_tasks_total" in metrics


def test_generate_metrics_with_data(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.observability.prometheus.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir, atomic_write
        with patch("nemospawn.core.state.TEAMS_DIR", teams_dir):
            team_dir = ensure_team_dir("t1")

        # Add a task with val_loss
        atomic_write(team_dir / "tasks" / "task-1.json", {
            "task_id": "task-1", "team_id": "t1", "title": "Train",
            "status": "done", "agent_id": "w0",
            "metadata": {"val_loss": 0.043},
        })
        # Add an agent
        atomic_write(team_dir / "agents" / "w0.json", {
            "agent_id": "w0", "team_id": "t1", "status": "running",
        })
        # Add a GPU metrics snapshot
        atomic_write(team_dir / "metrics" / "dcgm_snapshot.json", {
            "timestamp": "2026-03-23T00:00:00",
            "gpu_metrics": [{"gpu_id": 0, "sm_util": 95, "mem_util": 80, "temp": 72, "power": 300, "ecc_sbe": 0, "ecc_dbe": 0}],
        })

        metrics = generate_metrics("t1")
        assert "nemospawn_val_loss" in metrics
        assert "0.043" in metrics
        assert "nemospawn_gpu_sm_utilization" in metrics
        assert "nemospawn_agents_total" in metrics


def test_generate_dashboard():
    dashboard = generate_dashboard("test-team")
    assert dashboard["dashboard"]["uid"] == "nemospawn-test-team"
    assert len(dashboard["dashboard"]["panels"]) == 6
    assert dashboard["dashboard"]["refresh"] == "10s"


def test_write_dashboard(tmp_path):
    path = write_dashboard("t1", tmp_path / "dashboards" / "t1.json")
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["dashboard"]["uid"] == "nemospawn-t1"
