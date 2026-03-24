"""Agent lifecycle protocol — idle reporting, graceful shutdown request/approve/reject."""

from __future__ import annotations

from nemospawn.core.config import AGENTS_SUBDIR
from nemospawn.core.models import LifecycleEvent, _now, _short_id
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files


def report_idle(team_id: str, agent_id: str, reason: str = "") -> LifecycleEvent:
    """Report that an agent is idle (no more work to do)."""
    event = LifecycleEvent(
        event_id=_short_id("lc"),
        team_id=team_id,
        agent_id=agent_id,
        event_type="idle",
        reason=reason,
    )
    _update_agent_lifecycle(team_id, agent_id, "idle", reason)
    return event


def request_shutdown(
    team_id: str, agent_id: str, requested_by: str = "", reason: str = "",
) -> LifecycleEvent:
    """Request graceful shutdown of an agent."""
    event = LifecycleEvent(
        event_id=_short_id("lc"),
        team_id=team_id,
        agent_id=agent_id,
        event_type="shutdown_request",
        reason=reason,
        requested_by=requested_by,
    )
    _update_agent_lifecycle(team_id, agent_id, "shutdown_requested", reason)
    return event


def approve_shutdown(
    team_id: str, agent_id: str, responded_by: str = "",
) -> LifecycleEvent | None:
    """Approve a pending shutdown request."""
    agent_data = _get_agent_data(team_id, agent_id)
    if not agent_data:
        return None

    lifecycle = agent_data.get("lifecycle", {})
    if lifecycle.get("state") != "shutdown_requested":
        return None

    event = LifecycleEvent(
        event_id=_short_id("lc"),
        team_id=team_id,
        agent_id=agent_id,
        event_type="shutdown_approved",
        responded_by=responded_by,
    )
    _update_agent_lifecycle(team_id, agent_id, "shutdown_approved")
    return event


def reject_shutdown(
    team_id: str, agent_id: str, responded_by: str = "", reason: str = "",
) -> LifecycleEvent | None:
    """Reject a pending shutdown request — agent should continue working."""
    agent_data = _get_agent_data(team_id, agent_id)
    if not agent_data:
        return None

    lifecycle = agent_data.get("lifecycle", {})
    if lifecycle.get("state") != "shutdown_requested":
        return None

    event = LifecycleEvent(
        event_id=_short_id("lc"),
        team_id=team_id,
        agent_id=agent_id,
        event_type="shutdown_rejected",
        reason=reason,
        responded_by=responded_by,
    )
    _update_agent_lifecycle(team_id, agent_id, "running", reason)
    return event


def get_lifecycle_state(team_id: str, agent_id: str) -> dict:
    """Get the lifecycle state for an agent."""
    agent_data = _get_agent_data(team_id, agent_id)
    if not agent_data:
        return {}
    return agent_data.get("lifecycle", {"state": agent_data.get("status", "unknown")})


def list_idle_agents(team_id: str) -> list[dict]:
    """List all agents currently reporting idle."""
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    idle = []
    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if data:
            lifecycle = data.get("lifecycle", {})
            if lifecycle.get("state") == "idle":
                idle.append({
                    "agent_id": data["agent_id"],
                    "name": data.get("name", ""),
                    "role": data.get("role", ""),
                    "idle_since": lifecycle.get("updated_at", ""),
                    "reason": lifecycle.get("reason", ""),
                })
    return idle


def _get_agent_data(team_id: str, agent_id: str) -> dict | None:
    """Load agent JSON data."""
    agent_file = get_team_dir(team_id) / AGENTS_SUBDIR / f"{agent_id}.json"
    return atomic_read(agent_file)


def _update_agent_lifecycle(
    team_id: str, agent_id: str, state: str, reason: str = "",
) -> None:
    """Update the lifecycle field in an agent's JSON."""
    agent_file = get_team_dir(team_id) / AGENTS_SUBDIR / f"{agent_id}.json"
    data = atomic_read(agent_file)
    if not data:
        return

    data["lifecycle"] = {
        "state": state,
        "reason": reason,
        "updated_at": _now(),
    }
    atomic_write(agent_file, data)
