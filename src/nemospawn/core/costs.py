"""Cost tracking — per-team GPU-hour cost accumulation."""

from __future__ import annotations

from datetime import datetime, timezone

from nemospawn.core.config import AGENTS_SUBDIR, COSTS_SUBDIR
from nemospawn.core.models import CostRecord, _now
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files


COST_FILE = "cost_record.json"


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO timestamp string to datetime."""
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def get_cost_record(team_id: str) -> CostRecord:
    """Load or create the cost record for a team."""
    costs_dir = get_team_dir(team_id) / COSTS_SUBDIR
    costs_dir.mkdir(parents=True, exist_ok=True)
    data = atomic_read(costs_dir / COST_FILE)
    if data:
        return CostRecord.from_dict(data)
    return CostRecord(team_id=team_id)


def save_cost_record(record: CostRecord) -> None:
    """Persist the cost record."""
    costs_dir = get_team_dir(record.team_id) / COSTS_SUBDIR
    costs_dir.mkdir(parents=True, exist_ok=True)
    record.last_updated = _now()
    atomic_write(costs_dir / COST_FILE, record.to_dict())


def update_costs(team_id: str) -> CostRecord:
    """Recalculate costs based on agent run times and GPU counts.

    Scans all agents, computes elapsed GPU-seconds for running agents
    since their creation time, and accumulates into the cost record.
    """
    record = get_cost_record(team_id)
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    now = datetime.now(timezone.utc)

    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if not data:
            continue

        agent_id = data.get("agent_id", "")
        gpu_count = len(data.get("gpu_ids", []))
        if gpu_count == 0:
            continue

        created = _parse_iso(data.get("created_at", ""))
        status = data.get("status", "")

        # For stopped agents, use last update time; for running agents, use now
        if status == "running":
            end_time = now
        elif status == "stopped":
            end_time = _parse_iso(data.get("updated_at", data.get("created_at", "")))
        else:
            continue

        elapsed_seconds = max(0.0, (end_time - created).total_seconds())
        gpu_seconds = elapsed_seconds * gpu_count
        cost_usd = (gpu_seconds / 3600.0) * record.rate_per_gpu_hour

        record.agent_costs[agent_id] = {
            "name": data.get("name", ""),
            "gpu_count": gpu_count,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "gpu_seconds": round(gpu_seconds, 1),
            "cost_usd": round(cost_usd, 4),
            "status": status,
        }

    # Aggregate totals
    record.total_gpu_seconds = sum(
        a.get("gpu_seconds", 0) for a in record.agent_costs.values()
    )
    record.total_cost_usd = sum(
        a.get("cost_usd", 0) for a in record.agent_costs.values()
    )
    save_cost_record(record)
    return record


def reset_costs(team_id: str) -> CostRecord:
    """Reset cost totals for a team, preserving the custom rate."""
    old = get_cost_record(team_id)
    record = CostRecord(team_id=team_id, rate_per_gpu_hour=old.rate_per_gpu_hour)
    save_cost_record(record)
    return record


def set_rate(team_id: str, rate_per_gpu_hour: float) -> CostRecord:
    """Set the GPU-hour rate for a team."""
    record = get_cost_record(team_id)
    record.rate_per_gpu_hour = rate_per_gpu_hour
    save_cost_record(record)
    return record
