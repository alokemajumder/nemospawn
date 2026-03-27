"""Launch command — spawn a full team from a template."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("run")
def launch(
    template: str = typer.Argument(..., help="Template name or path to .toml file"),
    gpus: Optional[str] = typer.Option(None, "--gpus", help="GPU indices (e.g., 0,1,2,3)"),
    goal: str = typer.Option("", "--goal", help="Natural language goal to override worker tasks"),
    runtime: str = typer.Option("tmux", "--runtime", help="Runtime: tmux or sandbox"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without executing"),
):
    """Launch a full team from a template.

    Examples:
        nemospawn launch run autoresearch --gpus 0,1,2,3
        nemospawn launch run ./my_template.toml --goal "Fine-tune Llama-3-8B on medical QA"
    """
    from nemospawn.templates.engine import get_builtin_template, load_template

    # Load template
    template_path = Path(template)
    if template_path.exists() and template_path.suffix == ".toml":
        tmpl = load_template(template_path)
    else:
        tmpl = get_builtin_template(template)
        if not tmpl:
            console.print(f"[red]Template '{template}' not found[/]")
            console.print("[dim]Built-in templates: autoresearch, nim-deploy, rlhf-swarm, data-curation[/]")
            raise typer.Exit(1)

    # Parse GPUs
    gpu_ids: list[int] = []
    if gpus:
        gpu_ids = [int(g.strip()) for g in gpus.split(",")]
    else:
        from nemospawn.gpu.discovery import discover_gpus
        available = discover_gpus()
        gpu_ids = [g.index for g in available]

    if len(gpu_ids) < tmpl.min_gpus:
        console.print(f"[red]Template requires {tmpl.min_gpus} GPUs, only {len(gpu_ids)} available[/]")
        raise typer.Exit(1)

    # Show plan
    console.print(Panel(
        f"[bold]Template:[/] {tmpl.name}\n"
        f"[bold]Description:[/] {tmpl.description}\n"
        f"[bold]GPUs:[/] {gpu_ids}\n"
        f"[bold]Workers:[/] {len(tmpl.workers)}\n"
        f"[bold]Runtime:[/] {runtime}",
        title="Launch Plan",
        border_style="cyan",
    ))

    table = Table(title="Workers")
    table.add_column("Name", style="cyan")
    table.add_column("Role")
    table.add_column("GPUs")
    table.add_column("Task")
    table.add_column("Blocked By")

    # Assign GPUs to workers
    gpu_cursor = 0
    assignments: list[tuple] = []
    for w in tmpl.workers:
        count = min(w.gpu_count, len(gpu_ids) - gpu_cursor)
        assigned = gpu_ids[gpu_cursor:gpu_cursor + count]
        gpu_cursor = (gpu_cursor + count) % len(gpu_ids)
        task_desc = goal if goal else w.task
        assignments.append((w, assigned, task_desc))
        table.add_row(
            w.name,
            w.role,
            str(assigned),
            task_desc[:50],
            ", ".join(w.blocked_by) if w.blocked_by else "",
        )

    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no agents spawned[/]")
        return

    # Create team and spawn workers
    from nemospawn.core.models import _short_id
    from nemospawn.core.state import ensure_state_dir, ensure_team_dir, atomic_write

    ensure_state_dir()
    team_id = _short_id(tmpl.name)
    team_dir = ensure_team_dir(team_id)

    # Save team
    from nemospawn.core.models import Team
    from nemospawn.gpu.topology import parse_topology, get_nvlink_islands

    topo = parse_topology()
    islands = get_nvlink_islands(topo) if topo["matrix"] else []

    team = Team(
        team_id=team_id,
        name=tmpl.name,
        description=tmpl.description,
        gpu_ids=gpu_ids,
        topology=topo.get("matrix", {}),
        nvlink_islands=islands,
    )
    atomic_write(team_dir / "team.json", team.to_dict())

    console.print(f"\n[green]Team created:[/] {team_id}")

    # Spawn each worker with full coordination prompt injection
    import os
    from nemospawn.core.models import Agent
    from nemospawn.core.config import TMUX_PREFIX, AGENTS_SUBDIR
    from nemospawn.openshell.prompt import build_system_prompt

    for w, assigned_gpus, task_desc in assignments:
        agent_id = _short_id(w.name)

        if runtime == "sandbox":
            from nemospawn.openshell.agent import spawn_in_sandbox
            spawn_in_sandbox(
                team_id=team_id, agent_id=agent_id, role=w.role,
                gpu_ids=assigned_gpus, task_description=task_desc,
                agent_command=w.agent_cmd,
            )
        else:
            from nemospawn.runtime.tmux import create_session, send_command

            tmux_session = f"{TMUX_PREFIX}-{team_id}-{agent_id}"
            env = {
                "NEMOSPAWN_TEAM": team_id,
                "NEMOSPAWN_AGENT": agent_id,
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            }
            if os.environ.get("VIRTUAL_ENV"):
                env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
            if assigned_gpus:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in assigned_gpus)

            create_session(tmux_session, env=env)

            # Inject coordination prompt (leader role gets orchestration commands)
            coord_prompt = build_system_prompt(
                team_id=team_id, agent_id=agent_id,
                gpu_ids=assigned_gpus, role=w.role,
                task_description=task_desc,
            )
            prompt_dir = team_dir / "prompts"
            prompt_dir.mkdir(exist_ok=True)
            prompt_file = prompt_dir / f"{agent_id}.md"
            prompt_file.write_text(coord_prompt)
            send_command(tmux_session, f"export NEMOSPAWN_PROMPT={prompt_file}")

            # Launch the agent CLI via adapter
            from nemospawn.core.profiles import load_profile, build_spawn_command
            agent_profile = load_profile(w.agent_cmd if hasattr(w, 'agent_cmd') else "claude")
            if agent_profile:
                spawn_cmd = build_spawn_command(agent_profile, task=task_desc, prompt_file=str(prompt_file))
                send_command(tmux_session, " ".join(spawn_cmd))

        agent = Agent(
            agent_id=agent_id, team_id=team_id, name=w.name,
            role=w.role, gpu_ids=assigned_gpus, task=task_desc,
            tmux_session=f"{TMUX_PREFIX}-{team_id}-{agent_id}" if runtime != "sandbox" else f"nemo-{team_id}-{agent_id}",
            status="running",
        )
        atomic_write(team_dir / AGENTS_SUBDIR / f"{agent_id}.json", agent.to_dict())
        role_label = f"[bold cyan]{w.role}[/]" if w.role == "leader" else w.role
        console.print(f"  [green]Spawned:[/] {w.name} ({role_label}) → GPU {assigned_gpus or 'coordinator'}")

    console.print(f"\n[bold green]Team '{team_id}' launched with {len(tmpl.workers)} workers[/]")


@app.command("templates")
def list_templates():
    """List available built-in templates."""
    from nemospawn.templates.engine import list_builtin_templates

    templates = list_builtin_templates()
    table = Table(title="Built-in Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Min GPUs", justify="right")
    table.add_column("Workers", justify="right")

    for t in templates:
        table.add_row(t["name"], t["description"], str(t["min_gpus"]), str(t["workers"]))
    console.print(table)
