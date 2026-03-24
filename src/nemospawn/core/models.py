"""Data models for NemoSpawn entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str = "") -> str:
    slug = uuid4().hex[:8]
    return f"{prefix}-{slug}" if prefix else slug


@dataclass
class GPUInfo:
    index: int
    name: str
    uuid: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GPUInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Team:
    team_id: str
    name: str
    description: str = ""
    gpu_ids: list[int] = field(default_factory=list)
    gpus: list[dict] = field(default_factory=list)
    topology: dict = field(default_factory=dict)
    nvlink_islands: list[list[int]] = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Team:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Agent:
    agent_id: str
    team_id: str
    name: str
    role: str = "worker"
    gpu_ids: list[int] = field(default_factory=list)
    tmux_session: str = ""
    worktree_path: str = ""
    task: str = ""
    pid: int | None = None
    status: str = "spawning"
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Agent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Task:
    task_id: str
    team_id: str
    title: str
    agent_id: str | None = None
    status: str = "pending"  # pending | blocked | running | done | failed
    blocked_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Message:
    msg_id: str
    team_id: str
    from_agent: str
    body: str
    to_agent: str | None = None  # None = broadcast
    timestamp: str = field(default_factory=_now)
    read: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Plan:
    """A plan submitted by an agent for leader approval before execution."""
    plan_id: str
    team_id: str
    agent_id: str
    title: str
    description: str = ""
    steps: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | approved | rejected
    reviewer: str = ""
    review_comment: str = ""
    created_at: str = field(default_factory=_now)
    reviewed_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Plan:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LifecycleEvent:
    """A lifecycle event for an agent (idle, shutdown request/approve/reject)."""
    event_id: str
    team_id: str
    agent_id: str
    event_type: str  # idle | shutdown_request | shutdown_approved | shutdown_rejected
    reason: str = ""
    requested_by: str = ""
    responded_by: str = ""
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LifecycleEvent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CostRecord:
    """Tracks GPU-hour cost for a team."""
    team_id: str
    total_gpu_seconds: float = 0.0
    total_cost_usd: float = 0.0
    agent_costs: dict[str, dict] = field(default_factory=dict)
    rate_per_gpu_hour: float = 2.50  # default $/GPU-hour
    last_updated: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CostRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TeamSnapshot:
    """A point-in-time snapshot of a team's full state."""
    snapshot_id: str
    team_id: str
    label: str = ""
    team_data: dict = field(default_factory=dict)
    agents: list[dict] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TeamSnapshot:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def resolve_ready_tasks(tasks: list[Task]) -> list[Task]:
    """Return tasks whose blocked_by dependencies are all done."""
    done_ids = {t.task_id for t in tasks if t.status == "done"}
    ready = []
    for t in tasks:
        if t.status in ("pending", "blocked"):
            if all(dep in done_ids for dep in t.blocked_by):
                ready.append(t)
    return ready
