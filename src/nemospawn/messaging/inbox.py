"""File-based inbox messaging with atomic writes."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from nemospawn.core.config import INBOX_SUBDIR
from nemospawn.core.models import Message, _now
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir


def _inbox_dir(team_id: str, agent_id: str) -> Path:
    """Get the inbox directory for an agent."""
    d = get_team_dir(team_id) / INBOX_SUBDIR / agent_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def send_message(team_id: str, from_agent: str, to_agent: str, body: str) -> Message:
    """Send a point-to-point message to an agent's inbox."""
    msg = Message(
        msg_id=uuid4().hex[:8],
        team_id=team_id,
        from_agent=from_agent,
        to_agent=to_agent,
        body=body,
        timestamp=_now(),
    )
    inbox_dir = _inbox_dir(team_id, to_agent)
    filename = f"{msg.timestamp.replace(':', '-')}_{msg.msg_id}.json"
    atomic_write(inbox_dir / filename, msg.to_dict())
    return msg


def broadcast_message(team_id: str, from_agent: str, body: str) -> list[Message]:
    """Broadcast a message to all agents in the team."""
    from nemospawn.core.config import AGENTS_SUBDIR

    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    if not agents_dir.is_dir():
        return []

    messages = []
    for agent_file in agents_dir.glob("*.json"):
        agent_data = atomic_read(agent_file)
        if agent_data and agent_data.get("agent_id") != from_agent:
            agent_id = agent_data["agent_id"]
            msg = send_message(team_id, from_agent, agent_id, body)
            msg.to_agent = None  # mark as broadcast in the message
            messages.append(msg)
    return messages


def receive_messages(team_id: str, agent_id: str, unread_only: bool = True) -> list[Message]:
    """Read messages from an agent's inbox."""
    inbox_dir = _inbox_dir(team_id, agent_id)
    messages = []

    for msg_file in sorted(inbox_dir.glob("*.json")):
        data = atomic_read(msg_file)
        if data is None:
            continue
        msg = Message.from_dict(data)
        if unread_only and msg.read:
            continue
        messages.append(msg)

    return messages


def mark_read(team_id: str, agent_id: str, msg_id: str) -> bool:
    """Mark a message as read."""
    inbox_dir = _inbox_dir(team_id, agent_id)
    for msg_file in inbox_dir.glob(f"*_{msg_id}.json"):
        data = atomic_read(msg_file)
        if data:
            data["read"] = True
            atomic_write(msg_file, data)
            return True
    return False
