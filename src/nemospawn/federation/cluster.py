"""Cluster registration and management for cross-cluster federation.

Supports SSH-based remote agent spawning, NFS/SSHFS shared state,
and git-annex artifact transfer between clusters.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nemospawn.core.config import STATE_DIR
from nemospawn.core.state import atomic_read, atomic_write, list_json_files

CLUSTERS_DIR = STATE_DIR / "clusters"


@dataclass
class ClusterConfig:
    """Configuration for a registered remote cluster."""
    name: str
    host: str
    ssh_key: str = ""
    ssh_user: str = ""
    mount_point: str = ""  # shared FS mount path
    gpu_count: int = 0
    topology: dict = field(default_factory=dict)
    status: str = "registered"  # registered | online | offline
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ClusterConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def register_cluster(
    name: str,
    host: str,
    ssh_key: str = "",
    ssh_user: str = "",
    mount_point: str = "",
) -> ClusterConfig:
    """Register a remote cluster for federation."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Probe the cluster for GPU count
    gpu_count = _probe_remote_gpus(host, ssh_key, ssh_user)

    cluster = ClusterConfig(
        name=name,
        host=host,
        ssh_key=ssh_key,
        ssh_user=ssh_user or "root",
        mount_point=mount_point,
        gpu_count=gpu_count,
    )

    atomic_write(CLUSTERS_DIR / f"{name}.json", cluster.to_dict())
    return cluster


def list_clusters() -> list[ClusterConfig]:
    """List all registered clusters."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    clusters = []
    for f in list_json_files(CLUSTERS_DIR):
        data = atomic_read(f)
        if data:
            clusters.append(ClusterConfig.from_dict(data))
    return clusters


def get_cluster(name: str) -> ClusterConfig | None:
    """Get a cluster by name."""
    data = atomic_read(CLUSTERS_DIR / f"{name}.json")
    if data:
        return ClusterConfig.from_dict(data)
    return None


def check_cluster_status(cluster: ClusterConfig) -> str:
    """Check if a cluster is reachable via SSH."""
    ssh_args = _build_ssh_args(cluster)
    try:
        result = subprocess.run(
            [*ssh_args, "echo", "ok"],
            capture_output=True, text=True, timeout=10,
        )
        return "online" if result.returncode == 0 else "offline"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "offline"


def spawn_remote_agent(
    cluster: ClusterConfig,
    team_id: str,
    agent_id: str,
    gpu_ids: list[int],
    task: str = "",
) -> bool:
    """Spawn a NemoSpawn agent on a remote cluster via SSH + tmux."""
    ssh_args = _build_ssh_args(cluster)
    gpu_str = ",".join(str(g) for g in gpu_ids)
    tmux_session = f"nemo-{team_id}-{agent_id}"

    # Create tmux session on remote
    remote_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_str} "
        f"NEMOSPAWN_TEAM={team_id} "
        f"NEMOSPAWN_AGENT={agent_id} "
        f"tmux new-session -d -s {tmux_session}"
    )

    try:
        result = subprocess.run(
            [*ssh_args, remote_cmd],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def sync_artifacts_git_annex(
    source_cluster: ClusterConfig,
    dest_cluster: ClusterConfig,
    artifact_path: str,
) -> bool:
    """Transfer artifacts between clusters using git-annex."""
    import shlex

    ssh_args = _build_ssh_args(source_cluster)
    try:
        # Sanitize inputs to prevent command injection
        mount = shlex.quote(str(source_cluster.mount_point))
        dest = shlex.quote(dest_cluster.name)
        path = shlex.quote(artifact_path)
        remote_cmd = f"cd {mount} && git annex copy --to {dest} {path}"
        result = subprocess.run(
            [*ssh_args, remote_cmd],
            capture_output=True, text=True, timeout=3600,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_ssh_args(cluster: ClusterConfig) -> list[str]:
    """Build SSH command arguments for a cluster."""
    args = ["ssh"]
    if cluster.ssh_key:
        args.extend(["-i", cluster.ssh_key])
    args.extend(["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"])
    args.append(f"{cluster.ssh_user}@{cluster.host}" if cluster.ssh_user else cluster.host)
    return args


def _probe_remote_gpus(host: str, ssh_key: str = "", ssh_user: str = "") -> int:
    """Probe a remote host for GPU count."""
    args = ["ssh"]
    if ssh_key:
        args.extend(["-i", ssh_key])
    args.extend(["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"])
    args.append(f"{ssh_user}@{host}" if ssh_user else host)
    args.extend(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return len(result.stdout.strip().splitlines())
        return 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
