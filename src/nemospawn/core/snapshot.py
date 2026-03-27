"""Team snapshots — save and restore full team state."""

from __future__ import annotations

from nemospawn.core.config import (
    AGENTS_SUBDIR, TASKS_SUBDIR, PLANS_SUBDIR, COSTS_SUBDIR, SNAPSHOTS_SUBDIR,
)
from nemospawn.core.models import TeamSnapshot, _short_id
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files


def save_snapshot(team_id: str, label: str = "") -> TeamSnapshot:
    """Capture the current team state into a snapshot."""
    team_dir = get_team_dir(team_id)

    # Collect team data
    team_data = atomic_read(team_dir / "team.json") or {}

    # Collect agents
    agents = []
    for f in list_json_files(team_dir / AGENTS_SUBDIR):
        data = atomic_read(f)
        if data:
            agents.append(data)

    # Collect tasks
    tasks = []
    for f in list_json_files(team_dir / TASKS_SUBDIR):
        data = atomic_read(f)
        if data:
            tasks.append(data)

    # Collect plans
    plans = []
    for f in list_json_files(team_dir / PLANS_SUBDIR):
        data = atomic_read(f)
        if data:
            plans.append(data)

    # Collect costs
    cost_file = team_dir / COSTS_SUBDIR / "cost_record.json"
    costs = atomic_read(cost_file) or {}

    snapshot = TeamSnapshot(
        snapshot_id=_short_id("snap"),
        team_id=team_id,
        label=label,
        team_data=team_data,
        agents=agents,
        tasks=tasks,
        plans=plans,
        costs=costs,
    )

    snapshots_dir = team_dir / SNAPSHOTS_SUBDIR
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(snapshots_dir / f"{snapshot.snapshot_id}.json", snapshot.to_dict())
    return snapshot


def restore_snapshot(team_id: str, snapshot_id: str) -> TeamSnapshot | None:
    """Restore team state from a snapshot.

    Overwrites current agents, tasks, plans, and costs with snapshot data.
    Does NOT restore team.json itself (preserving team identity).
    """
    team_dir = get_team_dir(team_id)
    snapshots_dir = team_dir / SNAPSHOTS_SUBDIR
    data = atomic_read(snapshots_dir / f"{snapshot_id}.json")
    if not data:
        return None

    snapshot = TeamSnapshot.from_dict(data)

    # Restore agents
    agents_dir = team_dir / AGENTS_SUBDIR
    agents_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing (handle concurrent access gracefully)
    for f in list_json_files(agents_dir):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    for agent in snapshot.agents:
        agent_id = agent.get("agent_id", "unknown")
        atomic_write(agents_dir / f"{agent_id}.json", agent)

    # Restore tasks
    tasks_dir = team_dir / TASKS_SUBDIR
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for f in list_json_files(tasks_dir):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    for task in snapshot.tasks:
        task_id = task.get("task_id", "unknown")
        atomic_write(tasks_dir / f"{task_id}.json", task)

    # Restore plans
    plans_dir = team_dir / PLANS_SUBDIR
    plans_dir.mkdir(parents=True, exist_ok=True)
    for f in list_json_files(plans_dir):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    for plan in snapshot.plans:
        plan_id = plan.get("plan_id", "unknown")
        atomic_write(plans_dir / f"{plan_id}.json", plan)

    # Restore costs
    if snapshot.costs:
        costs_dir = team_dir / COSTS_SUBDIR
        costs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(costs_dir / "cost_record.json", snapshot.costs)

    return snapshot


def list_snapshots(team_id: str) -> list[TeamSnapshot]:
    """List all snapshots for a team."""
    snapshots_dir = get_team_dir(team_id) / SNAPSHOTS_SUBDIR
    snapshots = []
    for f in list_json_files(snapshots_dir):
        data = atomic_read(f)
        if data:
            snapshots.append(TeamSnapshot.from_dict(data))
    return snapshots


def get_snapshot(team_id: str, snapshot_id: str) -> TeamSnapshot | None:
    """Get a single snapshot by ID."""
    snapshots_dir = get_team_dir(team_id) / SNAPSHOTS_SUBDIR
    data = atomic_read(snapshots_dir / f"{snapshot_id}.json")
    if data:
        return TeamSnapshot.from_dict(data)
    return None


def delete_snapshot(team_id: str, snapshot_id: str) -> bool:
    """Delete a snapshot."""
    snapshots_dir = get_team_dir(team_id) / SNAPSHOTS_SUBDIR
    snap_file = snapshots_dir / f"{snapshot_id}.json"
    if snap_file.exists():
        snap_file.unlink()
        return True
    return False
