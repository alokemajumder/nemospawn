"""Tests for agent watcher."""

from unittest.mock import patch

from nemospawn.core.watcher import check_agent_health, watch_once
from nemospawn.core.models import Agent
from nemospawn.core.state import atomic_write, ensure_team_dir


def _create_agent(teams_dir, team_id, agent_id, status="running", tmux_session=""):
    agent = Agent(
        agent_id=agent_id, team_id=team_id, name=agent_id,
        role="trainer", gpu_ids=[0], status=status,
        tmux_session=tmux_session,
    )
    agents_dir = teams_dir / team_id / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(agents_dir / f"{agent_id}.json", agent.to_dict())


def test_check_healthy_agents(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.watcher.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", status="running", tmux_session="nemo-t1-a1")

        with patch("nemospawn.runtime.tmux.session_exists", return_value=True):
            results = check_agent_health("t1")

        assert len(results) == 1
        assert results[0]["healthy"] is True
        assert results[0]["issues"] == []


def test_check_dead_tmux_agent(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.watcher.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", status="running", tmux_session="nemo-t1-a1")

        with patch("nemospawn.runtime.tmux.session_exists", return_value=False):
            results = check_agent_health("t1")

        assert len(results) == 1
        assert results[0]["healthy"] is False
        assert "tmux session dead" in results[0]["issues"]


def test_check_stopped_agent_skipped(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.watcher.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", status="stopped", tmux_session="nemo-t1-a1")

        results = check_agent_health("t1")
        assert len(results) == 1
        # Stopped agents don't get tmux checks
        assert results[0]["healthy"] is True


def test_watch_once_summary(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.watcher.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", status="running", tmux_session="nemo-t1-a1")
        _create_agent(teams_dir, "t1", "a2", status="running", tmux_session="nemo-t1-a2")

        with patch("nemospawn.runtime.tmux.session_exists", side_effect=[True, False]):
            result = watch_once("t1")

        assert result["total"] == 2
        assert result["healthy"] == 1
        assert result["unhealthy"] == 1
        assert result["team_id"] == "t1"
