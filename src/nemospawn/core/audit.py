"""Audit logging — records all NemoSpawn operations for compliance.

All operations (team create, agent spawn, task updates, artifact promotions)
are logged to a structured audit log file at ~/.nemospawn/audit.jsonl.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from nemospawn.core.config import STATE_DIR

AUDIT_LOG = STATE_DIR / "audit.jsonl"


def log_event(
    event_type: str,
    details: dict,
    user: str = "",
    team_id: str = "",
    agent_id: str = "",
) -> None:
    """Append an audit event to the audit log.

    Args:
        event_type: Event category (team.create, agent.spawn, task.update, etc.)
        details: Event-specific data.
        user: Username of the operator.
        team_id: Relevant team ID.
        agent_id: Relevant agent ID.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "user": user or os.environ.get("USER", "unknown"),
        "team_id": team_id,
        "agent_id": agent_id,
        "details": details,
    }

    # Append atomically
    line = json.dumps(entry, default=str) + "\n"
    with open(AUDIT_LOG, "a") as f:
        f.write(line)


def read_audit_log(
    last_n: int = 100,
    event_type: str | None = None,
    team_id: str | None = None,
) -> list[dict]:
    """Read recent audit log entries."""
    if not AUDIT_LOG.exists():
        return []

    entries = []
    with open(AUDIT_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if event_type and entry.get("event") != event_type:
                    continue
                if team_id and entry.get("team_id") != team_id:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    return entries[-last_n:]
