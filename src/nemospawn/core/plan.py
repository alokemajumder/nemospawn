"""Plan approval workflow — agents submit plans for leader review before execution."""

from __future__ import annotations

from nemospawn.core.config import PLANS_SUBDIR
from nemospawn.core.models import Plan, _now, _short_id
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files


def submit_plan(
    team_id: str,
    agent_id: str,
    title: str,
    description: str = "",
    steps: list[str] | None = None,
) -> Plan:
    """Submit a plan for leader review."""
    plan = Plan(
        plan_id=_short_id("plan"),
        team_id=team_id,
        agent_id=agent_id,
        title=title,
        description=description,
        steps=steps or [],
        status="pending",
    )
    plans_dir = get_team_dir(team_id) / PLANS_SUBDIR
    plans_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(plans_dir / f"{plan.plan_id}.json", plan.to_dict())
    return plan


def review_plan(
    team_id: str,
    plan_id: str,
    action: str,
    reviewer: str = "",
    comment: str = "",
) -> Plan | None:
    """Approve or reject a plan.

    Args:
        action: "approved" or "rejected"
    """
    plans_dir = get_team_dir(team_id) / PLANS_SUBDIR
    plan_file = plans_dir / f"{plan_id}.json"
    data = atomic_read(plan_file)
    if not data:
        return None

    data["status"] = action
    data["reviewer"] = reviewer
    data["review_comment"] = comment
    data["reviewed_at"] = _now()
    atomic_write(plan_file, data)
    return Plan.from_dict(data)


def list_plans(
    team_id: str,
    status: str | None = None,
    agent_id: str | None = None,
) -> list[Plan]:
    """List plans, optionally filtered by status or agent."""
    plans_dir = get_team_dir(team_id) / PLANS_SUBDIR
    plans = []
    for f in list_json_files(plans_dir):
        data = atomic_read(f)
        if not data:
            continue
        if status and data.get("status") != status:
            continue
        if agent_id and data.get("agent_id") != agent_id:
            continue
        plans.append(Plan.from_dict(data))
    return plans


def get_plan(team_id: str, plan_id: str) -> Plan | None:
    """Get a single plan by ID."""
    plans_dir = get_team_dir(team_id) / PLANS_SUBDIR
    data = atomic_read(plans_dir / f"{plan_id}.json")
    if data:
        return Plan.from_dict(data)
    return None
