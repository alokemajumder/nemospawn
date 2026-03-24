"""NVLink topology parsing from nvidia-smi topo -m."""

from __future__ import annotations

import subprocess
from rich.console import Console

console = Console(stderr=True)


def parse_topology_raw() -> str | None:
    """Run nvidia-smi topo -m and return raw output."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_topology() -> dict:
    """Parse nvidia-smi topo -m into a structured topology dict.

    Returns:
        {
            "matrix": {0: {1: "NV12", 2: "SYS", ...}, ...},
            "gpu_pairs": [{"gpu_a": 0, "gpu_b": 1, "link_type": "NV12"}, ...],
            "raw": "<full output>"
        }
    """
    raw = parse_topology_raw()
    if not raw:
        return {"matrix": {}, "gpu_pairs": [], "raw": ""}

    lines = raw.strip().splitlines()
    matrix: dict[int, dict[int, str]] = {}
    gpu_pairs: list[dict] = []
    gpu_indices: list[int] = []

    # Find the header line (contains GPU0, GPU1, ...)
    header_idx = -1
    for i, line in enumerate(lines):
        if "GPU0" in line or "GPU 0" in line:
            header_idx = i
            break

    if header_idx < 0:
        return {"matrix": {}, "gpu_pairs": [], "raw": raw}

    # Parse header to get GPU column indices
    header = lines[header_idx]
    cols = header.split()
    for col in cols:
        col_clean = col.replace("GPU", "").strip()
        if col_clean.isdigit():
            gpu_indices.append(int(col_clean))

    # Parse each GPU row
    for line in lines[header_idx + 1 :]:
        line = line.strip()
        if not line or line.startswith("Legend"):
            break

        parts = line.split()
        if not parts:
            continue

        # First column is the row label (e.g., "GPU0" or "GPU 0")
        row_label = parts[0].replace("GPU", "")
        if not row_label.isdigit():
            continue
        row_gpu = int(row_label)
        matrix[row_gpu] = {}

        # Remaining columns are link types
        link_values = parts[1 : 1 + len(gpu_indices)]
        for col_idx, link_type in enumerate(link_values):
            col_gpu = gpu_indices[col_idx]
            matrix[row_gpu][col_gpu] = link_type
            if row_gpu < col_gpu:
                gpu_pairs.append(
                    {"gpu_a": row_gpu, "gpu_b": col_gpu, "link_type": link_type}
                )

    return {"matrix": matrix, "gpu_pairs": gpu_pairs, "raw": raw}


def get_nvlink_islands(topology: dict) -> list[list[int]]:
    """Group GPUs connected via NVLink into islands using union-find.

    GPUs connected by NV* links (NV4, NV12, etc.) are in the same island.
    GPUs connected only by SYS, PHB, PIX, or PXB are in separate islands.
    """
    matrix = topology.get("matrix", {})
    if not matrix:
        return []

    gpu_ids = sorted(matrix.keys())
    parent: dict[int, int] = {g: g for g in gpu_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # NVLink types start with "NV"
    for gpu_a in gpu_ids:
        for gpu_b in gpu_ids:
            if gpu_a >= gpu_b:
                continue
            link = matrix.get(gpu_a, {}).get(gpu_b, "")
            if link.startswith("NV"):
                union(gpu_a, gpu_b)

    # Group by root
    islands: dict[int, list[int]] = {}
    for g in gpu_ids:
        root = find(g)
        islands.setdefault(root, []).append(g)

    return sorted(islands.values(), key=lambda x: x[0])
