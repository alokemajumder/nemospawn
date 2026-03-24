"""Tests for web UI kanban dashboard."""

from unittest.mock import patch

from nemospawn.observability.web import _collect_board_data, WEB_HTML
from nemospawn.core.models import Agent, Task
from nemospawn.core.state import atomic_write, ensure_team_dir


def _setup_team(teams_dir, team_id):
    ensure_team_dir(team_id)
    team_dir = teams_dir / team_id

    agent = Agent(agent_id="a1", team_id=team_id, name="trainer", role="trainer", gpu_ids=[0], status="running")
    atomic_write(team_dir / "agents" / "a1.json", agent.to_dict())

    task = Task(task_id="t1", team_id=team_id, title="Train model", status="running", agent_id="a1")
    atomic_write(team_dir / "tasks" / "t1.json", task.to_dict())

    task2 = Task(task_id="t2", team_id=team_id, title="Eval model", status="pending")
    atomic_write(team_dir / "tasks" / "t2.json", task2.to_dict())


def test_collect_board_data(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.observability.web.get_team_dir", lambda tid: teams_dir / tid):
        _setup_team(teams_dir, "t1")

        data = _collect_board_data("t1")
        assert len(data["tasks"]) == 2
        assert len(data["agents"]) == 1
        assert isinstance(data["plans"], list)


def test_web_html_template():
    html = WEB_HTML.format(team_id="test-team")
    assert "NemoSpawn Board" in html
    assert "test-team" in html
    assert "EventSource" in html
    assert "/events" in html


def test_collect_board_data_empty_team(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.observability.web.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("empty")

        data = _collect_board_data("empty")
        assert data["tasks"] == []
        assert data["agents"] == []
        assert data["plans"] == []
