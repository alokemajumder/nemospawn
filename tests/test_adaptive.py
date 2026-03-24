"""Tests for adaptive scheduling."""

from unittest.mock import patch

from nemospawn.core.adaptive import (
    analyze_performance, suggest_reassignments, apply_reassignment,
)
from nemospawn.core.models import Agent, Task
from nemospawn.core.state import atomic_read, atomic_write, ensure_team_dir


def _setup_team(teams_dir, team_id):
    ensure_team_dir(team_id)
    team_dir = teams_dir / team_id

    # Create agents
    for aid, gpus in [("a1", [0]), ("a2", [1]), ("a3", [2])]:
        agent = Agent(agent_id=aid, team_id=team_id, name=aid, role="trainer", gpu_ids=gpus, status="running")
        atomic_write(team_dir / "agents" / f"{aid}.json", agent.to_dict())

    # Create tasks
    for tid, agent, status in [
        ("t1", "a1", "done"), ("t2", "a1", "running"),
        ("t3", "a2", "pending"), ("t4", "a3", "done"),
    ]:
        task = Task(task_id=tid, team_id=team_id, title=f"Task {tid}", agent_id=agent, status=status)
        atomic_write(team_dir / "tasks" / f"{tid}.json", task.to_dict())


def test_analyze_performance(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.adaptive.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")

        metrics = [
            {"gpu_id": 0, "sm_util": 85.0},
            {"gpu_id": 1, "sm_util": 15.0},
            {"gpu_id": 2, "sm_util": 92.0},
        ]
        perf = analyze_performance("t1", metrics)
        assert len(perf) == 3

        # Should be sorted by score ascending (worst first)
        scores = [p["score"] for p in perf]
        assert scores == sorted(scores)

        # a2 should have lowest score (15% util, 0 done / 1 total)
        assert perf[0]["agent_id"] == "a2"


def test_analyze_performance_no_metrics(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.adaptive.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")

        perf = analyze_performance("t1", None)
        assert len(perf) == 3
        # Without GPU metrics, score is based on task completion only
        for p in perf:
            assert p["avg_gpu_util"] is None


def test_suggest_reassignments(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.adaptive.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")

        metrics = [
            {"gpu_id": 0, "sm_util": 85.0},
            {"gpu_id": 1, "sm_util": 15.0},  # underperforming
            {"gpu_id": 2, "sm_util": 92.0},
        ]
        suggestions = suggest_reassignments("t1", metrics, util_threshold=30.0)

        # a2 is underperforming, t3 (pending, assigned to a2) should be reassigned
        assert len(suggestions) >= 1
        assert suggestions[0]["from_agent"] == "a2"
        assert suggestions[0]["task_id"] == "t3"


def test_suggest_no_reassignments_needed(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.adaptive.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")

        metrics = [
            {"gpu_id": 0, "sm_util": 85.0},
            {"gpu_id": 1, "sm_util": 75.0},
            {"gpu_id": 2, "sm_util": 92.0},
        ]
        suggestions = suggest_reassignments("t1", metrics, util_threshold=30.0)
        assert suggestions == []


def test_apply_reassignment(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.adaptive.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")

        assert apply_reassignment("t1", "t3", "a3") is True

        # Verify the task was reassigned
        data = atomic_read(teams_dir / "t1" / "tasks" / "t3.json")
        assert data["agent_id"] == "a3"
        assert data["metadata"]["reassigned_from"] == "a2"


def test_apply_reassignment_nonexistent_task(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.adaptive.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")
        assert apply_reassignment("t1", "nonexistent", "a3") is False
