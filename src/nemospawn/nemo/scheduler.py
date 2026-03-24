"""NVLink island-aware task scheduling for NeMo workloads.

Ensures multi-GPU NeMo runs (tensor parallel, pipeline parallel) are
confined to GPUs within the same NVLink island for optimal bandwidth.
Single-GPU tasks can be placed on any available GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from nemospawn.core.models import Agent, Team
from nemospawn.core.state import atomic_read, get_team_dir, list_json_files
from nemospawn.core.config import AGENTS_SUBDIR


@dataclass
class GPUAllocation:
    """A GPU allocation recommendation."""
    gpu_ids: list[int]
    island_index: int | None  # which NVLink island, or None for single GPU
    reason: str


def get_allocated_gpus(team_id: str) -> set[int]:
    """Get the set of GPU IDs currently allocated to running agents."""
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    allocated: set[int] = set()
    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if data and data.get("status") == "running":
            allocated.update(data.get("gpu_ids", []))
    return allocated


def find_available_gpus(team: Team, num_gpus: int = 1, require_same_island: bool = False) -> GPUAllocation | None:
    """Find available GPUs respecting NVLink topology.

    Args:
        team: Team with GPU info and NVLink islands.
        num_gpus: Number of GPUs needed.
        require_same_island: If True, all GPUs must be in the same NVLink island.

    Returns:
        GPUAllocation with recommended GPU IDs, or None if not enough GPUs.
    """
    allocated = get_allocated_gpus(team.team_id)
    all_gpus = set(team.gpu_ids)
    free_gpus = all_gpus - allocated

    if len(free_gpus) < num_gpus:
        return None

    islands = team.nvlink_islands

    if require_same_island and islands:
        # Try to find an island with enough free GPUs
        for idx, island in enumerate(islands):
            free_in_island = [g for g in island if g in free_gpus]
            if len(free_in_island) >= num_gpus:
                return GPUAllocation(
                    gpu_ids=free_in_island[:num_gpus],
                    island_index=idx,
                    reason=f"NVLink island {idx}: all {num_gpus} GPUs share NVLink fabric",
                )

        # No single island has enough — fall back to cross-island
        return GPUAllocation(
            gpu_ids=sorted(free_gpus)[:num_gpus],
            island_index=None,
            reason=f"Warning: {num_gpus} GPUs span multiple NVLink islands (cross-island communication)",
        )

    # Single GPU or no island requirement
    if islands:
        # Prefer GPUs from the least-utilized island
        for idx, island in enumerate(islands):
            free_in_island = [g for g in island if g in free_gpus]
            if len(free_in_island) >= num_gpus:
                return GPUAllocation(
                    gpu_ids=free_in_island[:num_gpus],
                    island_index=idx,
                    reason=f"Allocated from NVLink island {idx}",
                )

    # No islands or fallback
    return GPUAllocation(
        gpu_ids=sorted(free_gpus)[:num_gpus],
        island_index=None,
        reason="Allocated from available GPUs (no NVLink topology)",
    )


def recommend_parallelism(num_gpus: int, model_size_b: float | None = None) -> dict:
    """Recommend tensor/pipeline parallel config based on GPU count and model size.

    Args:
        num_gpus: Number of GPUs allocated.
        model_size_b: Model size in billions of parameters (optional).

    Returns:
        Dict with tp_size, pp_size recommendations.
    """
    if num_gpus == 1:
        return {"tp_size": 1, "pp_size": 1, "reason": "Single GPU — no parallelism"}

    # Default: maximize tensor parallelism within NVLink island
    if model_size_b and model_size_b > 30:
        # Large models: split across TP and PP
        if num_gpus >= 8:
            return {"tp_size": 4, "pp_size": 2, "reason": f"{model_size_b}B model: TP4×PP2 across 8 GPUs"}
        elif num_gpus >= 4:
            return {"tp_size": 4, "pp_size": 1, "reason": f"{model_size_b}B model: TP4 across 4 GPUs"}
        else:
            return {"tp_size": num_gpus, "pp_size": 1, "reason": f"{model_size_b}B model: TP{num_gpus}"}
    else:
        # Smaller models: TP only
        if num_gpus <= 4:
            return {"tp_size": num_gpus, "pp_size": 1, "reason": f"TP{num_gpus} — NVLink-optimal"}
        else:
            return {"tp_size": 4, "pp_size": num_gpus // 4, "reason": f"TP4×PP{num_gpus // 4}"}
