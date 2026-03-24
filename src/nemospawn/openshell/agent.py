"""OpenShell-backed agent spawning for NemoSpawn.

Integrates NVIDIA OpenShell sandbox lifecycle with NemoSpawn's
agent coordination. Each worker agent runs inside an OpenShell
sandbox with GPU passthrough, policy enforcement, and the
NemoSpawn coordination protocol injected.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from nemospawn.openshell.policy import generate_worker_policy, write_policy_file
from nemospawn.openshell.prompt import build_system_prompt
from nemospawn.openshell.sandbox import (
    SandboxConfig,
    check_openshell_installed,
    create_sandbox,
    destroy_sandbox,
    ensure_gateway,
)

console = Console()


def spawn_in_sandbox(
    team_id: str,
    agent_id: str,
    role: str = "worker",
    gpu_ids: list[int] | None = None,
    task_description: str = "",
    agent_command: str = "claude",
    remote: str | None = None,
    policy_dir: Path | None = None,
) -> bool:
    """Spawn a NemoSpawn worker agent inside an OpenShell sandbox.

    1. Verifies OpenShell is installed and gateway is running
    2. Generates a role-specific security policy
    3. Creates an isolated sandbox with GPU passthrough
    4. Injects NemoSpawn coordination env vars and system prompt

    Args:
        team_id: NemoSpawn team ID.
        agent_id: Unique agent identifier.
        role: Worker role (trainer, evaluator, deployer, etc.)
        gpu_ids: GPU indices for CUDA_VISIBLE_DEVICES.
        task_description: Natural language task for the agent.
        agent_command: Agent to run (claude, opencode, codex, copilot).
        remote: Remote host for SSH deployment (user@host).
        policy_dir: Directory to write policy files.

    Returns:
        True if the sandbox was created successfully.
    """
    # Pre-flight checks
    if not check_openshell_installed():
        console.print(
            "[red]OpenShell is not installed.[/]\n"
            "Install with: curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh\n"
            "Or: uv tool install -U openshell"
        )
        return False

    if not ensure_gateway():
        console.print("[red]Failed to start OpenShell gateway[/]")
        return False

    sandbox_name = f"nemo-{team_id}-{agent_id}"

    # Generate security policy
    if policy_dir is None:
        policy_dir = Path.home() / ".nemospawn" / "teams" / team_id / "policies"
    policy = generate_worker_policy(
        team_id=team_id,
        agent_id=agent_id,
        role=role,
        gpu_ids=gpu_ids,
    )
    policy_file = write_policy_file(policy, policy_dir / f"{agent_id}.yaml")

    # Build environment variables for NemoSpawn coordination
    env_vars = {
        "NEMOSPAWN_TEAM": team_id,
        "NEMOSPAWN_AGENT": agent_id,
    }
    if gpu_ids:
        env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # Generate system prompt for the agent
    system_prompt = build_system_prompt(
        team_id=team_id,
        agent_id=agent_id,
        gpu_ids=gpu_ids,
        role=role,
        task_description=task_description,
    )

    # Write system prompt to a file the agent can read inside sandbox
    prompt_file = policy_dir.parent / "prompts" / f"{agent_id}.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(system_prompt)

    # Create the sandbox
    config = SandboxConfig(
        name=sandbox_name,
        agent_command=agent_command,
        gpu=bool(gpu_ids),
        policy_file=str(policy_file),
        env_vars=env_vars,
        remote=remote,
    )

    success = create_sandbox(config)
    if success:
        console.print(Panel(
            f"[bold]Sandbox:[/] {sandbox_name}\n"
            f"[bold]Agent:[/] {agent_command}\n"
            f"[bold]Role:[/] {role}\n"
            f"[bold]GPUs:[/] {gpu_ids or 'none'}\n"
            f"[bold]Policy:[/] {policy_file}\n"
            f"[bold]Isolation:[/] Landlock + seccomp + netns",
            title="OpenShell Sandbox Spawned",
            border_style="green",
        ))
    return success


def kill_sandbox(team_id: str, agent_id: str) -> bool:
    """Destroy an agent's OpenShell sandbox."""
    sandbox_name = f"nemo-{team_id}-{agent_id}"
    return destroy_sandbox(sandbox_name)
