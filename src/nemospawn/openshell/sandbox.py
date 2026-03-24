"""OpenShell sandbox management for NemoSpawn workers.

Each NemoSpawn worker agent runs inside an OpenShell sandbox — an isolated
container with Landlock filesystem controls, seccomp syscall filtering,
and network namespace isolation. GPU passthrough is enabled for training agents.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console(stderr=True)


@dataclass
class SandboxConfig:
    """Configuration for an OpenShell sandbox hosting a NemoSpawn worker."""
    name: str
    agent_command: str = "claude"  # default agent (claude, opencode, codex, copilot)
    gpu: bool = False
    policy_file: str | None = None
    from_source: str | None = None  # community catalog entry, Dockerfile, or image
    env_vars: dict[str, str] = field(default_factory=dict)
    remote: str | None = None  # user@host for remote deployment


def _run_openshell(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run an openshell CLI command."""
    return subprocess.run(
        ["openshell", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_openshell_installed() -> bool:
    """Check if the openshell CLI is installed."""
    try:
        result = _run_openshell("--version")
        return result.returncode == 0
    except FileNotFoundError:
        return False


def ensure_gateway() -> bool:
    """Ensure the OpenShell gateway is running (auto-starts on first sandbox create)."""
    try:
        result = _run_openshell("gateway", "status")
        if result.returncode == 0:
            return True
        # Gateway not running, try to start it
        result = _run_openshell("gateway", "start", timeout=120)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_sandbox(config: SandboxConfig) -> bool:
    """Create an OpenShell sandbox for a NemoSpawn worker agent.

    This creates an isolated container with:
    - Kernel-level isolation (Landlock, seccomp, netns)
    - Policy-enforced filesystem and network controls
    - GPU passthrough if requested
    - NemoSpawn coordination env vars injected
    """
    args = ["sandbox", "create"]

    # Source (community image, Dockerfile, or container image)
    if config.from_source:
        args.extend(["--from", config.from_source])

    # GPU passthrough for training/inference agents
    if config.gpu:
        args.append("--gpu")

    # Remote deployment
    if config.remote:
        args.extend(["--remote", config.remote])

    # Sandbox name
    args.extend(["--name", config.name])

    # Agent command separator
    args.append("--")
    args.append(config.agent_command)

    try:
        result = _run_openshell(*args, timeout=300)
        if result.returncode != 0:
            console.print(f"[red]Failed to create sandbox '{config.name}': {result.stderr.strip()}[/]")
            return False

        # Apply custom policy if provided
        if config.policy_file:
            apply_policy(config.name, config.policy_file)

        return True
    except FileNotFoundError:
        console.print("[red]openshell CLI not found. Install: curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh[/]")
        return False
    except subprocess.TimeoutExpired:
        console.print(f"[yellow]Sandbox creation timed out for '{config.name}'[/]")
        return False


def destroy_sandbox(name: str) -> bool:
    """Destroy an OpenShell sandbox."""
    try:
        result = _run_openshell("sandbox", "delete", name)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_sandboxes() -> list[dict]:
    """List all OpenShell sandboxes."""
    try:
        result = _run_openshell("sandbox", "list", "--json")
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            return data if isinstance(data, list) else []
        # Fallback to non-JSON output
        result = _run_openshell("sandbox", "list")
        if result.returncode == 0:
            return [{"raw": result.stdout}]
        return []
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def connect_sandbox(name: str) -> subprocess.CompletedProcess:
    """Open an interactive shell into a running sandbox."""
    return subprocess.run(["openshell", "sandbox", "connect", name])


def apply_policy(name: str, policy_file: str) -> bool:
    """Apply or update a policy on a running sandbox (hot-reloadable)."""
    try:
        result = _run_openshell("policy", "set", name, "--policy", policy_file)
        if result.returncode != 0:
            console.print(f"[yellow]Failed to apply policy: {result.stderr.strip()}[/]")
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_policy(name: str) -> str | None:
    """Get the active policy for a sandbox."""
    try:
        result = _run_openshell("policy", "get", name)
        if result.returncode == 0:
            return result.stdout
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_sandbox_logs(name: str, follow: bool = False) -> str:
    """Get sandbox logs."""
    args = ["logs", name]
    if follow:
        args.append("--tail")
    try:
        result = _run_openshell(*args, timeout=10)
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def configure_inference(provider: str, model: str) -> bool:
    """Configure inference routing for sandboxes.

    Routes agent LLM API calls through OpenShell's privacy router
    to controlled backends (NVIDIA NIM, local Ollama, etc.)
    """
    try:
        result = _run_openshell("inference", "set", "--provider", provider, "--model", model)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
