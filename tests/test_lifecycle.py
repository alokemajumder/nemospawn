"""Tests for agent lifecycle protocol."""

from unittest.mock import patch

from nemospawn.core.lifecycle import (
    report_idle, request_shutdown, approve_shutdown,
    reject_shutdown, get_lifecycle_state, list_idle_agents,
)
from nemospawn.core.models import Agent
from nemospawn.core.state import atomic_write, ensure_team_dir


def _create_agent(teams_dir, team_id, agent_id, name="worker", status="running"):
    """Helper to create a test agent."""
    agent = Agent(
        agent_id=agent_id, team_id=team_id, name=name,
        role="trainer", gpu_ids=[0], status=status,
    )
    agents_dir = teams_dir / team_id / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(agents_dir / f"{agent_id}.json", agent.to_dict())
    return agent


def test_report_idle(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.lifecycle.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1")

        event = report_idle("t1", "a1", reason="All tasks done")
        assert event.event_type == "idle"
        assert event.reason == "All tasks done"

        state = get_lifecycle_state("t1", "a1")
        assert state["state"] == "idle"


def test_shutdown_request_approve(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.lifecycle.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1")

        # Request shutdown
        event = request_shutdown("t1", "a1", requested_by="leader", reason="No more work")
        assert event.event_type == "shutdown_request"

        state = get_lifecycle_state("t1", "a1")
        assert state["state"] == "shutdown_requested"

        # Approve
        event = approve_shutdown("t1", "a1", responded_by="leader")
        assert event is not None
        assert event.event_type == "shutdown_approved"

        state = get_lifecycle_state("t1", "a1")
        assert state["state"] == "shutdown_approved"


def test_shutdown_request_reject(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.lifecycle.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1")

        request_shutdown("t1", "a1", requested_by="leader")
        event = reject_shutdown("t1", "a1", responded_by="leader", reason="Keep training")
        assert event is not None
        assert event.event_type == "shutdown_rejected"

        state = get_lifecycle_state("t1", "a1")
        assert state["state"] == "running"


def test_approve_without_request_fails(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.lifecycle.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1")

        # No pending request — approve should return None
        result = approve_shutdown("t1", "a1")
        assert result is None


def test_list_idle_agents(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.lifecycle.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", name="trainer-0")
        _create_agent(teams_dir, "t1", "a2", name="trainer-1")

        report_idle("t1", "a1", reason="Done")

        idle = list_idle_agents("t1")
        assert len(idle) == 1
        assert idle[0]["agent_id"] == "a1"
