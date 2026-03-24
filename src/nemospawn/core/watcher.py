"""Agent watcher — auto-monitor agent health, detect stuck/dead agents."""

from __future__ import annotations

from datetime import datetime, timezone

from nemospawn.core.config import AGENTS_SUBDIR
from nemospawn.core.models import _now
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files


def check_agent_health(team_id: str) -> list[dict]:
    """Check health of all agents in a team.

    Verifies:
    1. tmux session still exists for running agents
    2. Agent hasn't been stuck too long without task updates
    """
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    results = []

    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if not data:
            continue

        agent_id = data.get("agent_id", "")
        status = data.get("status", "")
        session = data.get("tmux_session", "")

        health = {
            "agent_id": agent_id,
            "name": data.get("name", ""),
            "role": data.get("role", ""),
            "status": status,
            "healthy": True,
            "issues": [],
        }

        # Check if running agent's tmux session is alive
        if status == "running" and session:
            # Only check tmux sessions (not sandbox names)
            if session.startswith("nemo-") and "-" in session:
                try:
                    from nemospawn.runtime.tmux import session_exists
                    alive = session_exists(session)
                    if not alive:
                        health["healthy"] = False
                        health["issues"].append("tmux session dead")
                        # Mark agent as stopped
                        data["status"] = "stopped"
                        data["lifecycle"] = {
                            "state": "dead",
                            "reason": "tmux session no longer exists",
                            "updated_at": _now(),
                        }
                        atomic_write(f, data)
                except Exception:
                    health["issues"].append("could not check tmux session")

        # Check for stuck agents (running > 24h without lifecycle update)
        if status == "running":
            created = data.get("created_at", "")
            lifecycle = data.get("lifecycle", {})
            last_update = lifecycle.get("updated_at", created)
            try:
                last_dt = datetime.fromisoformat(last_update)
                now = datetime.now(timezone.utc)
                hours_since = (now - last_dt).total_seconds() / 3600
                if hours_since > 24:
                    health["issues"].append(
                        f"no lifecycle update for {hours_since:.1f}h"
                    )
            except (ValueError, TypeError):
                pass

        results.append(health)

    return results


def watch_once(team_id: str) -> dict:
    """Run a single health check pass and return summary."""
    results = check_agent_health(team_id)
    healthy = sum(1 for r in results if r["healthy"])
    unhealthy = sum(1 for r in results if not r["healthy"])
    return {
        "team_id": team_id,
        "total": len(results),
        "healthy": healthy,
        "unhealthy": unhealthy,
        "agents": results,
        "checked_at": _now(),
    }


def watch_loop(team_id: str, interval: int = 60, callback=None) -> None:
    """Continuously monitor agent health.

    Args:
        interval: Seconds between health checks.
        callback: Optional callable receiving the watch result dict.
    """
    import time

    while True:
        result = watch_once(team_id)
        if callback:
            callback(result)
        time.sleep(interval)
