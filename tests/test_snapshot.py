"""Tests for team snapshots."""

from unittest.mock import patch

from nemospawn.core.snapshot import (
    save_snapshot, restore_snapshot, list_snapshots,
    get_snapshot, delete_snapshot,
)
from nemospawn.core.models import Agent, Task
from nemospawn.core.state import atomic_write, ensure_team_dir


def _setup_team(teams_dir, team_id):
    """Create a team with agents and tasks for snapshot testing."""
    ensure_team_dir(team_id)
    team_dir = teams_dir / team_id

    # Team JSON
    atomic_write(team_dir / "team.json", {
        "team_id": team_id, "name": "test-team", "gpu_ids": [0, 1],
    })

    # Agents
    agent = Agent(agent_id="a1", team_id=team_id, name="trainer", role="trainer", gpu_ids=[0])
    atomic_write(team_dir / "agents" / "a1.json", agent.to_dict())

    # Tasks
    task = Task(task_id="t1", team_id=team_id, title="Train model", status="running", agent_id="a1")
    atomic_write(team_dir / "tasks" / "t1.json", task.to_dict())


def test_save_snapshot(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.snapshot.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")
        snap = save_snapshot("t1", label="before-deploy")
        assert snap.snapshot_id.startswith("snap-")
        assert snap.label == "before-deploy"
        assert len(snap.agents) == 1
        assert len(snap.tasks) == 1
        assert snap.team_data["team_id"] == "t1"


def test_restore_snapshot(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.snapshot.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")

        # Save snapshot
        snap = save_snapshot("t1", label="checkpoint")

        # Modify state — add a new task
        task2 = Task(task_id="t2", team_id="t1", title="New task", status="pending")
        atomic_write(teams_dir / "t1" / "tasks" / "t2.json", task2.to_dict())

        # Verify modification
        task_files = list((teams_dir / "t1" / "tasks").glob("*.json"))
        assert len(task_files) == 2

        # Restore
        restored = restore_snapshot("t1", snap.snapshot_id)
        assert restored is not None

        # Verify state reverted
        task_files = list((teams_dir / "t1" / "tasks").glob("*.json"))
        assert len(task_files) == 1


def test_list_snapshots(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.snapshot.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")
        save_snapshot("t1", label="snap-1")
        save_snapshot("t1", label="snap-2")

        snaps = list_snapshots("t1")
        assert len(snaps) == 2


def test_delete_snapshot(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.snapshot.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")
        snap = save_snapshot("t1")

        assert delete_snapshot("t1", snap.snapshot_id) is True
        assert get_snapshot("t1", snap.snapshot_id) is None
        assert delete_snapshot("t1", "nonexistent") is False


def test_restore_nonexistent_snapshot(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.snapshot.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")
        result = restore_snapshot("t1", "nonexistent")
        assert result is None
