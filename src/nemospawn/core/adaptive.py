"""Adaptive scheduling — dynamically reassign tasks based on agent GPU performance.

Monitors GPU utilization per agent, detects underperformers, and suggests
task reassignment to higher-performing GPUs/agents.
"""

from __future__ import annotations

from datetime import datetime, timezone

from nemospawn.core.config import AGENTS_SUBDIR, TASKS_SUBDIR
from nemospawn.core.models import _now
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files


def analyze_performance(team_id: str, metrics: list[dict] | None = None) -> list[dict]:
    """Analyze agent performance based on GPU utilization and task progress.

    Returns a ranked list of agents with performance scores.
    """
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR

    # Build GPU utilization map from metrics
    gpu_util: dict[int, float] = {}
    if metrics:
        for m in metrics:
            gpu_util[m.get("gpu_id", -1)] = m.get("sm_util", 0.0)

    # Load all tasks for progress analysis
    task_data = {}
    for f in list_json_files(tasks_dir):
        data = atomic_read(f)
        if data:
            task_data[data.get("task_id", "")] = data

    # Analyze each agent
    results = []
    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if not data or data.get("status") != "running":
            continue

        agent_id = data["agent_id"]
        gpu_ids = data.get("gpu_ids", [])
        created = data.get("created_at", "")

        # Calculate average GPU utilization for this agent's GPUs
        agent_utils = [gpu_util.get(g, -1) for g in gpu_ids]
        valid_utils = [u for u in agent_utils if u >= 0]
        avg_util = sum(valid_utils) / len(valid_utils) if valid_utils else -1

        # Count tasks assigned to this agent
        agent_tasks = [t for t in task_data.values() if t.get("agent_id") == agent_id]
        tasks_done = sum(1 for t in agent_tasks if t.get("status") == "done")
        tasks_running = sum(1 for t in agent_tasks if t.get("status") == "running")
        tasks_total = len(agent_tasks)

        # Calculate hours running
        try:
            created_dt = datetime.fromisoformat(created)
            hours_running = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_running = 0

        # Performance score: higher is better
        # Weight GPU utilization (0-100) and task completion rate
        completion_rate = (tasks_done / tasks_total * 100) if tasks_total > 0 else 0
        score = avg_util * 0.6 + completion_rate * 0.4 if avg_util >= 0 else completion_rate

        results.append({
            "agent_id": agent_id,
            "name": data.get("name", ""),
            "role": data.get("role", ""),
            "gpu_ids": gpu_ids,
            "avg_gpu_util": round(avg_util, 1) if avg_util >= 0 else None,
            "tasks_done": tasks_done,
            "tasks_running": tasks_running,
            "tasks_total": tasks_total,
            "hours_running": round(hours_running, 2),
            "score": round(score, 1),
        })

    # Sort by score (ascending — worst first for reassignment)
    results.sort(key=lambda x: x["score"])
    return results


def suggest_reassignments(
    team_id: str,
    metrics: list[dict] | None = None,
    util_threshold: float = 30.0,
) -> list[dict]:
    """Suggest task reassignments from underperforming to better-performing agents.

    Args:
        util_threshold: GPU utilization below which an agent is considered underperforming.

    Returns:
        List of suggested reassignments with from/to agents and affected tasks.
    """
    perf = analyze_performance(team_id, metrics)
    if len(perf) < 2:
        return []

    underperformers = [a for a in perf if a["avg_gpu_util"] is not None and a["avg_gpu_util"] < util_threshold]
    top_performers = [a for a in reversed(perf) if a["avg_gpu_util"] is not None and a["avg_gpu_util"] >= util_threshold]

    if not underperformers or not top_performers:
        return []

    suggestions = []
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR

    for under in underperformers:
        # Find running tasks on this underperforming agent
        reassignable_tasks = []
        for f in list_json_files(tasks_dir):
            data = atomic_read(f)
            if data and data.get("agent_id") == under["agent_id"] and data.get("status") in ("pending", "blocked"):
                reassignable_tasks.append(data)

        if not reassignable_tasks:
            continue

        # Find best available agent to reassign to
        for top in top_performers:
            if top["agent_id"] == under["agent_id"]:
                continue

            for task in reassignable_tasks:
                suggestions.append({
                    "task_id": task["task_id"],
                    "task_title": task.get("title", ""),
                    "from_agent": under["agent_id"],
                    "from_util": under["avg_gpu_util"],
                    "to_agent": top["agent_id"],
                    "to_util": top["avg_gpu_util"],
                    "reason": f"GPU util {under['avg_gpu_util']:.0f}% < {util_threshold:.0f}% threshold",
                })
            break  # One target per underperformer

    return suggestions


def apply_reassignment(team_id: str, task_id: str, new_agent_id: str) -> bool:
    """Apply a task reassignment — move a task to a different agent."""
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    task_file = tasks_dir / f"{task_id}.json"
    data = atomic_read(task_file)
    if not data:
        return False

    old_agent = data.get("agent_id", "")
    data["agent_id"] = new_agent_id
    data["updated_at"] = _now()
    data.setdefault("metadata", {})["reassigned_from"] = old_agent
    data["metadata"]["reassigned_at"] = _now()
    atomic_write(task_file, data)
    return True
