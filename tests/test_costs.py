"""Tests for cost tracking."""

from unittest.mock import patch

from nemospawn.core.costs import get_cost_record, update_costs, reset_costs, set_rate
from nemospawn.core.models import Agent, _now
from nemospawn.core.state import atomic_write, ensure_team_dir


def _create_agent(teams_dir, team_id, agent_id, gpu_ids, status="running"):
    agent = Agent(
        agent_id=agent_id, team_id=team_id, name=agent_id,
        role="trainer", gpu_ids=gpu_ids, status=status,
    )
    agents_dir = teams_dir / team_id / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(agents_dir / f"{agent_id}.json", agent.to_dict())


def test_get_cost_record_default(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.costs.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        record = get_cost_record("t1")
        assert record.team_id == "t1"
        assert record.total_gpu_seconds == 0.0
        assert record.total_cost_usd == 0.0
        assert record.rate_per_gpu_hour == 2.50


def test_update_costs_with_agents(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.costs.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", [0, 1], status="running")
        _create_agent(teams_dir, "t1", "a2", [2], status="running")

        record = update_costs("t1")
        # Both agents should be tracked
        assert "a1" in record.agent_costs
        assert "a2" in record.agent_costs
        # a1 has 2 GPUs, a2 has 1 GPU
        assert record.agent_costs["a1"]["gpu_count"] == 2
        assert record.agent_costs["a2"]["gpu_count"] == 1
        # Total GPU-seconds should be positive (agents just created)
        assert record.total_gpu_seconds >= 0


def test_update_costs_no_gpu_agent(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.costs.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", [], status="running")

        record = update_costs("t1")
        assert "a1" not in record.agent_costs


def test_reset_costs(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.costs.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        _create_agent(teams_dir, "t1", "a1", [0], status="running")
        update_costs("t1")

        record = reset_costs("t1")
        assert record.total_gpu_seconds == 0.0
        assert record.total_cost_usd == 0.0
        assert len(record.agent_costs) == 0


def test_set_rate(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.costs.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        record = set_rate("t1", 5.00)
        assert record.rate_per_gpu_hour == 5.00

        # Verify persistence
        record2 = get_cost_record("t1")
        assert record2.rate_per_gpu_hour == 5.00
