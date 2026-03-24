"""Tests for task management."""

from nemospawn.core.models import Task
from nemospawn.core.state import atomic_write, atomic_read


def test_create_and_read_task(state_dir):
    from unittest.mock import patch
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir):
        from nemospawn.core.state import ensure_team_dir
        team_dir = ensure_team_dir("test-team")
        tasks_dir = team_dir / "tasks"

        task = Task(task_id="task-001", team_id="test-team", title="Train model")
        atomic_write(tasks_dir / "task-001.json", task.to_dict())

        data = atomic_read(tasks_dir / "task-001.json")
        assert data["task_id"] == "task-001"
        assert data["title"] == "Train model"
        assert data["status"] == "pending"


def test_task_dependency_tracking(state_dir):
    task = Task(
        task_id="task-002",
        team_id="test-team",
        title="Deploy NIM",
        blocked_by=["task-001"],
        status="blocked",
    )
    path = state_dir / "task-002.json"
    atomic_write(path, task.to_dict())

    data = atomic_read(path)
    assert data["status"] == "blocked"
    assert data["blocked_by"] == ["task-001"]


def test_task_metadata(state_dir):
    task = Task(
        task_id="task-003",
        team_id="test-team",
        title="Eval",
        metadata={"val_loss": 0.043, "artifact_id": "ckpt-7"},
    )
    path = state_dir / "task-003.json"
    atomic_write(path, task.to_dict())

    data = atomic_read(path)
    assert data["metadata"]["val_loss"] == 0.043
    assert data["metadata"]["artifact_id"] == "ckpt-7"
