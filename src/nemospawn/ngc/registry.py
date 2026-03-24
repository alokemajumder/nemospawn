"""NGC registry operations — model download/upload, NIM container push, org sharing.

Wraps the NGC CLI (`ngc registry model download/upload`) and provides
programmatic access to NGC model and container registries.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console(stderr=True)


@dataclass
class NGCModel:
    """A model in the NGC registry."""
    org: str
    name: str
    version: str = "latest"
    path: str = ""  # local download path

    @property
    def full_name(self) -> str:
        return f"{self.org}/{self.name}:{self.version}"


def _run_ngc(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an NGC CLI command."""
    env = os.environ.copy()
    return subprocess.run(
        ["ngc", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def check_ngc_auth() -> bool:
    """Check if NGC CLI is authenticated."""
    try:
        result = _run_ngc("config", "current", timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_model(
    org: str,
    name: str,
    version: str = "latest",
    output_dir: Path | None = None,
) -> Path | None:
    """Download a model from NGC registry.

    Equivalent to: ngc registry model download <org>/<name>:<version>
    """
    model_ref = f"{org}/{name}:{version}"
    args = ["registry", "model", "download", model_ref]

    if output_dir:
        args.extend(["--dest", str(output_dir)])
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        console.print(f"Downloading {model_ref} from NGC...")
        result = _run_ngc(*args, timeout=3600)
        if result.returncode == 0:
            download_path = output_dir / name if output_dir else Path(name)
            return download_path
        console.print(f"[red]NGC download failed: {result.stderr.strip()[:500]}[/]")
        return None
    except FileNotFoundError:
        console.print("[red]NGC CLI not found. Install from https://ngc.nvidia.com/setup/installers/cli[/]")
        return None
    except subprocess.TimeoutExpired:
        console.print("[red]NGC download timed out[/]")
        return None


def upload_model(
    local_path: Path,
    org: str,
    name: str,
    version: str = "latest",
    description: str = "",
) -> bool:
    """Upload a model to NGC registry.

    Equivalent to: ngc registry model upload <org>/<name>:<version> --source <path>
    """
    model_ref = f"{org}/{name}:{version}"
    args = ["registry", "model", "upload", model_ref, "--source", str(local_path)]

    if description:
        args.extend(["--desc", description])

    try:
        console.print(f"Uploading to {model_ref} on NGC...")
        result = _run_ngc(*args, timeout=3600)
        if result.returncode == 0:
            console.print(f"[green]Uploaded to NGC: {model_ref}[/]")
            return True
        console.print(f"[red]NGC upload failed: {result.stderr.strip()[:500]}[/]")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def push_container(
    image_name: str,
    org: str,
    name: str,
    tag: str = "latest",
) -> bool:
    """Push a Docker container image to NGC registry.

    Tags and pushes via: docker tag ... nvcr.io/<org>/<name>:<tag> && docker push ...
    """
    ngc_image = f"nvcr.io/{org}/{name}:{tag}"

    try:
        # Tag
        result = subprocess.run(
            ["docker", "tag", image_name, ngc_image],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            console.print(f"[red]Docker tag failed: {result.stderr.strip()}[/]")
            return False

        # Push
        console.print(f"Pushing {ngc_image}...")
        result = subprocess.run(
            ["docker", "push", ngc_image],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            console.print(f"[green]Pushed to NGC: {ngc_image}[/]")
            return True
        console.print(f"[red]Docker push failed: {result.stderr.strip()[:500]}[/]")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_models(org: str, pattern: str = "*") -> list[dict]:
    """List models in an NGC org."""
    try:
        result = _run_ngc("registry", "model", "list", f"--org={org}", f"--filter={pattern}", timeout=30)
        if result.returncode != 0:
            return []

        models = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 2:
                models.append({"name": parts[0], "version": parts[1] if len(parts) > 1 else ""})
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
