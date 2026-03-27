"""Agent spawn and management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemospawn.core.config import AGENTS_SUBDIR, TMUX_PREFIX, WORKSPACES_SUBDIR
from nemospawn.core.models import Agent, _now, _short_id
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("agent")
def spawn_agent(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    name: str = typer.Option(..., "--agent-name", help="Agent name"),
    role: str = typer.Option("worker", "--role", help="Agent role (trainer, evaluator, deployer, data-curator, rlhf-reward)"),
    gpus: Optional[str] = typer.Option(None, "--gpu", help="Comma-separated GPU indices"),
    task: str = typer.Option("", "--task", help="Task description for the agent"),
    repo: Optional[str] = typer.Option(None, "--repo", help="Git repo path for worktree creation"),
    runtime: str = typer.Option("tmux", "--runtime", help="Spawn mode: tmux (default) or sandbox (OpenShell)"),
    agent_cmd: str = typer.Option("claude", "--agent-cmd", help="Agent CLI (claude, codex, kimi, cursor, nanobot, aider, opencode, copilot, custom)"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Named profile to use (overrides --agent-cmd)"),
    remote: Optional[str] = typer.Option(None, "--remote", help="Remote host for SSH sandbox deployment (user@host)"),
):
    """Spawn a new GPU-pinned agent via tmux or OpenShell sandbox."""
    team_dir = get_team_dir(team_id)
    team_data = atomic_read(team_dir / "team.json")
    if not team_data:
        console.print(f"[red]Team '{team_id}' not found[/]")
        raise typer.Exit(1)

    # Parse GPU list
    gpu_ids: list[int] = []
    if gpus:
        gpu_ids = [int(g.strip()) for g in gpus.split(",")]

    # Resolve profile: --profile overrides --agent-cmd
    if profile:
        from nemospawn.core.profiles import load_profile
        resolved = load_profile(profile)
        if resolved:
            agent_cmd = resolved.agent
        else:
            console.print(f"[yellow]Profile '{profile}' not found, using --agent-cmd '{agent_cmd}'[/]")

    # Generate agent ID
    agent_id = _short_id(name)

    if runtime == "sandbox":
        _spawn_sandbox(team_id, agent_id, name, role, gpu_ids, task, agent_cmd, remote, team_dir)
    else:
        try:
            _spawn_tmux(team_id, agent_id, name, role, gpu_ids, task, repo, agent_cmd, team_dir)
        except (RuntimeError, FileNotFoundError) as e:
            console.print(f"[red]Failed to spawn agent: {e}[/]")
            console.print("[dim]Ensure tmux is installed: brew install tmux (macOS) or apt install tmux (Linux)[/]")
            raise typer.Exit(1)


def _spawn_tmux(
    team_id: str, agent_id: str, name: str, role: str,
    gpu_ids: list[int], task: str, repo: str | None,
    agent_cmd: str, team_dir: Path,
) -> None:
    """Spawn an agent in a tmux session (default mode)."""
    from nemospawn.core.state import ensure_team_dir
    from nemospawn.runtime.tmux import create_session, send_command

    # Ensure team directories exist
    ensure_team_dir(team_id)

    tmux_session = f"{TMUX_PREFIX}-{team_id}-{agent_id}"

    # Create git worktree if repo specified
    worktree_path = ""
    if repo:
        from nemospawn.runtime.worktree import create_worktree

        wt_path = team_dir / WORKSPACES_SUBDIR / agent_id
        branch = f"nemospawn/{team_id}/{agent_id}"
        try:
            create_worktree(Path(repo), wt_path, branch=branch)
            worktree_path = str(wt_path)
        except RuntimeError as e:
            console.print(f"[yellow]Worktree creation failed: {e}[/]")

    # Prepare environment — inherit PATH so nemospawn CLI is available inside tmux
    import os

    env = {
        "NEMOSPAWN_TEAM": team_id,
        "NEMOSPAWN_AGENT": agent_id,
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    # Inherit venv so pip-installed nemospawn is on PATH
    if os.environ.get("VIRTUAL_ENV"):
        env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # Create tmux session
    created = create_session(tmux_session, env=env)
    if not created:
        console.print(f"[red]Failed to spawn agent '{name}'[/]")
        raise typer.Exit(1)

    # Inject coordination prompt into the agent's environment
    from nemospawn.openshell.prompt import build_system_prompt

    coord_prompt = build_system_prompt(
        team_id=team_id,
        agent_id=agent_id,
        gpu_ids=gpu_ids,
        role=role,
        task_description=task,
    )
    # Write prompt file to team workspace for the agent to reference
    prompt_dir = team_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    prompt_file = prompt_dir / f"{agent_id}.md"
    prompt_file.write_text(coord_prompt)

    # Send initial commands
    if worktree_path:
        send_command(tmux_session, f"cd {worktree_path}")
    # Export prompt file path so agents can discover it
    send_command(tmux_session, f"export NEMOSPAWN_PROMPT={prompt_file}")

    # Build and send agent invocation command using adapter
    from nemospawn.core.profiles import load_profile, build_spawn_command, get_adapter

    agent_profile = load_profile(agent_cmd)
    if agent_profile:
        adapter = get_adapter(agent_profile.agent)
        spawn_cmd = build_spawn_command(agent_profile, task=task, prompt_file=str(prompt_file))
        send_command(tmux_session, " ".join(spawn_cmd))
    elif task:
        send_command(tmux_session, f"echo 'Task: {task}'")

    # Save agent state
    agent = Agent(
        agent_id=agent_id,
        team_id=team_id,
        name=name,
        role=role,
        gpu_ids=gpu_ids,
        tmux_session=tmux_session,
        worktree_path=worktree_path,
        task=task,
        status="running",
    )
    atomic_write(team_dir / AGENTS_SUBDIR / f"{agent_id}.json", agent.to_dict())

    console.print(Panel(
        f"[bold]Agent ID:[/] {agent_id}\n"
        f"[bold]Name:[/] {name}\n"
        f"[bold]Role:[/] {role}\n"
        f"[bold]GPUs:[/] {gpu_ids or 'none'}\n"
        f"[bold]Runtime:[/] tmux ({tmux_session})\n"
        f"[bold]Task:[/] {task or 'none'}",
        title="Agent Spawned",
        border_style="green",
    ))


def _spawn_sandbox(
    team_id: str, agent_id: str, name: str, role: str,
    gpu_ids: list[int], task: str, agent_cmd: str,
    remote: str | None, team_dir: Path,
) -> None:
    """Spawn an agent in an OpenShell sandbox (isolated mode)."""
    from nemospawn.openshell.agent import spawn_in_sandbox

    sandbox_name = f"nemo-{team_id}-{agent_id}"

    success = spawn_in_sandbox(
        team_id=team_id,
        agent_id=agent_id,
        role=role,
        gpu_ids=gpu_ids,
        task_description=task,
        agent_command=agent_cmd,
        remote=remote,
    )

    if not success:
        console.print(f"[red]Failed to spawn sandbox for agent '{name}'[/]")
        raise typer.Exit(1)

    # Save agent state
    agent = Agent(
        agent_id=agent_id,
        team_id=team_id,
        name=name,
        role=role,
        gpu_ids=gpu_ids,
        tmux_session=sandbox_name,  # store sandbox name in tmux_session field
        task=task,
        status="running",
    )
    atomic_write(team_dir / AGENTS_SUBDIR / f"{agent_id}.json", agent.to_dict())


@app.command("kill")
def kill_agent(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID"),
):
    """Kill a running agent (tmux session or OpenShell sandbox)."""
    from nemospawn.runtime.tmux import kill_session

    team_dir = get_team_dir(team_id)
    agent_file = team_dir / AGENTS_SUBDIR / f"{agent_id}.json"
    agent_data = atomic_read(agent_file)

    if not agent_data:
        console.print(f"[red]Agent '{agent_id}' not found in team '{team_id}'[/]")
        raise typer.Exit(1)

    session_name = agent_data.get("tmux_session", "")

    # Try OpenShell sandbox destroy first, fall back to tmux kill
    if session_name.startswith("nemo-") and not session_name.startswith(f"{TMUX_PREFIX}-"):
        from nemospawn.openshell.sandbox import destroy_sandbox
        destroy_sandbox(session_name)
    elif session_name:
        kill_session(session_name)

    # Remove worktree if exists
    worktree_path = agent_data.get("worktree_path")
    if worktree_path:
        from nemospawn.runtime.worktree import remove_worktree
        remove_worktree(team_dir, Path(worktree_path))

    # Update status
    agent_data["status"] = "stopped"
    agent_data["updated_at"] = _now()
    atomic_write(agent_file, agent_data)

    console.print(f"[green]Agent '{agent_id}' stopped[/]")


@app.command("list")
def list_agents(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """List all agents in a team."""
    team_dir = get_team_dir(team_id)
    agents_dir = team_dir / AGENTS_SUBDIR

    if not agents_dir.is_dir():
        console.print(f"[yellow]No agents in team '{team_id}'[/]")
        raise typer.Exit()

    agent_files = sorted(agents_dir.glob("*.json"))
    if not agent_files:
        console.print(f"[yellow]No agents in team '{team_id}'[/]")
        raise typer.Exit()

    table = Table(title=f"Agents — {team_id}")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("GPUs")
    table.add_column("Status")
    table.add_column("Runtime")

    for af in agent_files:
        a = atomic_read(af)
        if a:
            status_style = "green" if a.get("status") == "running" else "red"
            session = a.get("tmux_session", "")
            runtime_type = "sandbox" if session.startswith("nemo-") and not session.startswith(f"{TMUX_PREFIX}-") else "tmux"
            table.add_row(
                a["agent_id"],
                a.get("name", ""),
                a.get("role", ""),
                str(a.get("gpu_ids", [])),
                f"[{status_style}]{a.get('status', '')}[/]",
                f"{runtime_type} ({session})",
            )
    console.print(table)
