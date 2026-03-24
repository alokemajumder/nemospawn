"""NeMo artifact management — typed .nemo checkpoint bundles with quality gates.

Artifacts are registered in the team's artifact store with typed metadata
(val_loss, artifact_type, nemo_config_hash, etc.) and support promotion
(crowning a best checkpoint for downstream NIM deployment).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemospawn.core.config import ARTIFACTS_SUBDIR
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files


ARTIFACT_TYPES = [
    "nemo-checkpoint",   # .nemo bundle (weights + config + tokenizer)
    "lora-adapter",      # LoRA/PEFT adapter files
    "nim-container",     # NIM container image reference
    "dataset",           # Curated dataset (.jsonl, .parquet)
    "benchmark",         # Benchmark results JSON
    "reward-signal",     # RLHF reward signal data
    "config-patch",      # NeMo YAML config override
]


@dataclass
class Artifact:
    """A typed artifact in the NemoSpawn artifact store."""
    artifact_id: str
    team_id: str
    artifact_type: str  # one of ARTIFACT_TYPES
    path: str  # absolute path to the artifact file/directory
    agent_id: str | None = None
    val_loss: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    nemo_config_hash: str = ""
    promoted: bool = False
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Artifact:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _artifacts_dir(team_id: str) -> Path:
    d = get_team_dir(team_id) / ARTIFACTS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_artifact(
    team_id: str,
    path: str,
    artifact_type: str = "nemo-checkpoint",
    agent_id: str | None = None,
    val_loss: float | None = None,
    metrics: dict | None = None,
    tags: list[str] | None = None,
) -> Artifact:
    """Register a new artifact in the team store.

    Validates the artifact path exists and computes a config hash for .nemo bundles.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {path}")

    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"Unknown artifact type '{artifact_type}'. Valid: {ARTIFACT_TYPES}")

    artifact_id = f"art-{uuid4().hex[:8]}"

    # Compute config hash for .nemo checkpoints
    config_hash = ""
    if artifact_type == "nemo-checkpoint" and artifact_path.suffix == ".nemo":
        config_hash = _compute_nemo_hash(artifact_path)

    artifact = Artifact(
        artifact_id=artifact_id,
        team_id=team_id,
        artifact_type=artifact_type,
        path=str(artifact_path.resolve()),
        agent_id=agent_id,
        val_loss=val_loss,
        metrics=metrics or {},
        nemo_config_hash=config_hash,
        tags=tags or [],
    )

    arts_dir = _artifacts_dir(team_id)
    atomic_write(arts_dir / f"{artifact_id}.json", artifact.to_dict())

    # Create symlink in artifacts dir pointing to the actual file
    symlink_name = f"{artifact_id}_{artifact_path.name}"
    symlink_path = arts_dir / symlink_name
    if not symlink_path.exists():
        try:
            symlink_path.symlink_to(artifact_path.resolve())
        except OSError:
            pass  # symlink creation not critical

    return artifact


def promote_artifact(team_id: str, artifact_id: str) -> Artifact:
    """Promote an artifact as the best checkpoint for this team.

    Demotes any previously promoted artifact of the same type.
    """
    arts_dir = _artifacts_dir(team_id)
    target_file = arts_dir / f"{artifact_id}.json"
    target_data = atomic_read(target_file)

    if not target_data:
        raise FileNotFoundError(f"Artifact '{artifact_id}' not found")

    artifact = Artifact.from_dict(target_data)

    # Demote any existing promoted artifact of the same type
    for f in list_json_files(arts_dir):
        data = atomic_read(f)
        if data and data.get("promoted") and data.get("artifact_type") == artifact.artifact_type:
            if data["artifact_id"] != artifact_id:
                data["promoted"] = False
                atomic_write(f, data)

    # Promote this one
    artifact.promoted = True
    atomic_write(target_file, artifact.to_dict())
    return artifact


def list_artifacts(
    team_id: str,
    artifact_type: str | None = None,
    sort_by: str = "val_loss",
    promoted_only: bool = False,
) -> list[Artifact]:
    """List artifacts, optionally filtered and sorted."""
    arts_dir = _artifacts_dir(team_id)
    artifacts = []

    for f in list_json_files(arts_dir):
        data = atomic_read(f)
        if not data or "artifact_id" not in data:
            continue
        art = Artifact.from_dict(data)
        if artifact_type and art.artifact_type != artifact_type:
            continue
        if promoted_only and not art.promoted:
            continue
        artifacts.append(art)

    # Sort
    if sort_by == "val_loss":
        artifacts.sort(key=lambda a: a.val_loss if a.val_loss is not None else float("inf"))
    elif sort_by == "created_at":
        artifacts.sort(key=lambda a: a.created_at, reverse=True)

    return artifacts


def get_promoted_artifact(team_id: str, artifact_type: str = "nemo-checkpoint") -> Artifact | None:
    """Get the currently promoted (best) artifact of a given type."""
    promoted = list_artifacts(team_id, artifact_type=artifact_type, promoted_only=True)
    return promoted[0] if promoted else None


def _compute_nemo_hash(path: Path) -> str:
    """Compute a hash of a .nemo bundle for dedup detection."""
    hasher = hashlib.sha256()
    if path.is_file():
        # Hash first 1MB for speed
        with open(path, "rb") as f:
            chunk = f.read(1024 * 1024)
            hasher.update(chunk)
    elif path.is_dir():
        # Hash directory listing
        for child in sorted(path.rglob("*")):
            if child.is_file():
                hasher.update(str(child.relative_to(path)).encode())
                hasher.update(str(child.stat().st_size).encode())
    return hasher.hexdigest()[:16]
