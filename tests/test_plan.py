"""Tests for plan approval workflow."""

from unittest.mock import patch

from nemospawn.core.plan import submit_plan, review_plan, list_plans, get_plan
from nemospawn.core.state import ensure_team_dir


def test_submit_plan(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.plan.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        plan = submit_plan("t1", "agent-1", "Train model", description="Fine-tune LLaMA", steps=["Prep data", "Train", "Eval"])
        assert plan.plan_id.startswith("plan-")
        assert plan.status == "pending"
        assert plan.agent_id == "agent-1"
        assert plan.title == "Train model"
        assert len(plan.steps) == 3


def test_approve_plan(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.plan.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        plan = submit_plan("t1", "agent-1", "Deploy NIM")
        result = review_plan("t1", plan.plan_id, "approved", reviewer="leader", comment="LGTM")
        assert result is not None
        assert result.status == "approved"
        assert result.reviewer == "leader"
        assert result.review_comment == "LGTM"
        assert result.reviewed_at != ""


def test_reject_plan(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.plan.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        plan = submit_plan("t1", "agent-1", "Risky plan")
        result = review_plan("t1", plan.plan_id, "rejected", comment="Too risky")
        assert result is not None
        assert result.status == "rejected"


def test_list_plans_filter(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.plan.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        submit_plan("t1", "agent-1", "Plan A")
        submit_plan("t1", "agent-2", "Plan B")
        p3 = submit_plan("t1", "agent-1", "Plan C")
        review_plan("t1", p3.plan_id, "approved")

        # All plans
        all_plans = list_plans("t1")
        assert len(all_plans) == 3

        # Filter by agent
        agent_plans = list_plans("t1", agent_id="agent-1")
        assert len(agent_plans) == 2

        # Filter by status
        approved = list_plans("t1", status="approved")
        assert len(approved) == 1
        assert approved[0].plan_id == p3.plan_id


def test_get_plan(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.plan.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        plan = submit_plan("t1", "agent-1", "My plan")
        fetched = get_plan("t1", plan.plan_id)
        assert fetched is not None
        assert fetched.title == "My plan"

        # Nonexistent plan
        assert get_plan("t1", "nonexistent") is None


def test_review_nonexistent_plan(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.core.plan.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")
        result = review_plan("t1", "nonexistent", "approved")
        assert result is None
