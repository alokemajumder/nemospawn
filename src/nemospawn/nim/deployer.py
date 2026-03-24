"""NIM container deployment — packages .nemo checkpoints into NIM containers.

Handles the full pipeline: .nemo checkpoint → NIM container build → endpoint
start → health check → artifact registration.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rich.console import Console

console = Console(stderr=True)


@dataclass
class NIMEndpoint:
    """A running NIM inference endpoint."""
    endpoint_id: str
    team_id: str
    artifact_id: str
    container_image: str
    endpoint_url: str
    tp_degree: int = 1
    profile: str = "default"
    status: str = "starting"  # starting | running | stopped | failed
    container_id: str = ""
    port: int = 8000
    gpu_ids: list[int] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> NIMEndpoint:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def build_nim_container(
    checkpoint_path: str,
    image_name: str,
    tp_degree: int = 1,
    profile: str = "default",
    base_image: str | None = None,
) -> str | None:
    """Build a NIM container from a .nemo checkpoint.

    Args:
        checkpoint_path: Path to .nemo checkpoint bundle.
        image_name: Docker image name for the built container.
        tp_degree: Tensor parallel degree (1, 2, 4, 8).
        profile: NIM profile (default, max-throughput, min-latency).
        base_image: Base NIM image from NGC (auto-detected if None).

    Returns:
        Container image name if successful, None otherwise.
    """
    args = ["nim", "build"]
    args.extend(["--checkpoint", checkpoint_path])
    args.extend(["--image", image_name])
    args.extend(["--tp", str(tp_degree)])
    args.extend(["--profile", profile])
    if base_image:
        args.extend(["--base-image", base_image])

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            return image_name
        console.print(f"[red]NIM build failed: {result.stderr.strip()}[/]")
        return None
    except FileNotFoundError:
        # Fallback to docker build with NIM Dockerfile
        return _docker_build_nim(checkpoint_path, image_name, tp_degree)
    except subprocess.TimeoutExpired:
        console.print("[red]NIM build timed out (30 min)[/]")
        return None


def _docker_build_nim(checkpoint_path: str, image_name: str, tp_degree: int) -> str | None:
    """Fallback: build NIM container via docker."""
    # Generate a simple Dockerfile for NIM
    dockerfile_content = f"""
FROM nvcr.io/nvidia/nim:latest
COPY {checkpoint_path} /model/
ENV TP_DEGREE={tp_degree}
EXPOSE 8000
CMD ["nim", "serve", "--model-path", "/model/", "--tp", "{tp_degree}"]
"""
    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f-", "."],
            input=dockerfile_content,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode == 0:
            return image_name
        console.print(f"[red]Docker build failed: {result.stderr.strip()[:500]}[/]")
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print("[red]Docker not available for NIM container build[/]")
        return None


def start_nim_endpoint(
    image_name: str,
    port: int = 8000,
    gpu_ids: list[int] | None = None,
    name: str | None = None,
) -> str | None:
    """Start a NIM container as a REST inference endpoint.

    Returns:
        Container ID if successful, None otherwise.
    """
    args = ["docker", "run", "-d", "--rm"]
    args.extend(["-p", f"{port}:8000"])

    if gpu_ids:
        gpu_str = ",".join(str(g) for g in gpu_ids)
        args.extend(["--gpus", f'"device={gpu_str}"'])
    else:
        args.extend(["--gpus", "all"])

    if name:
        args.extend(["--name", name])

    args.append(image_name)

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # container ID
        console.print(f"[red]Failed to start NIM: {result.stderr.strip()}[/]")
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def stop_nim_endpoint(container_id: str) -> bool:
    """Stop a running NIM container."""
    try:
        result = subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_nim_health(endpoint_url: str, timeout: int = 60) -> bool:
    """Wait for NIM endpoint to become healthy."""
    import urllib.request
    import urllib.error

    health_url = f"{endpoint_url}/v1/health/ready"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)

    return False


def list_nim_containers() -> list[dict]:
    """List running NIM containers via docker ps."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "ancestor=nvcr.io/nvidia/nim",
             "--format", "{{.ID}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}\t{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        containers = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 4:
                containers.append({
                    "container_id": parts[0],
                    "image": parts[1],
                    "ports": parts[2],
                    "status": parts[3],
                    "name": parts[4] if len(parts) > 4 else "",
                })
        return containers
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def generate_nim_profiles(num_gpus: int) -> list[dict]:
    """Generate NIM TP profile variants based on available GPU count.

    Returns configs for TP1, TP2, TP4, TP8 where applicable.
    """
    profiles = []
    for tp in [1, 2, 4, 8]:
        if tp <= num_gpus:
            profiles.append({
                "tp_degree": tp,
                "gpus_required": tp,
                "profile_name": f"tp{tp}",
                "description": f"Tensor parallel {tp} — {'single GPU' if tp == 1 else f'{tp} GPUs'}",
            })
    return profiles
