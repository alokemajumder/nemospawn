"""Workspace management CLI — checkpoint, merge, cleanup agent git worktrees."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from nemospawn.core.config import AGENTS_SUBDIR, WORKSPACES_SUBDIR
from nemospawn.core.state import atomic_read, get_team_dir, list_json_files

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def workspace_list(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
):
    """List all agent worktrees for a team."""
    team_dir = get_team_dir(team_id)
    agents_dir = team_dir / AGENTS_SUBDIR

    table = Table(title=f"Workspaces — {team_id}")
    table.add_column("Agent", style="cyan")
    table.add_column("Name")
    table.add_column("Worktree Path")
    table.add_column("Status")

    found = False
    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if data and data.get("worktree_path"):
            found = True
            from pathlib import Path
            wt = Path(data["worktree_path"])
            exists = wt.is_dir()
            table.add_row(
                data.get("agent_id", "")[:12],
                data.get("name", ""),
                data["worktree_path"],
                "[green]active[/]" if exists else "[red]missing[/]",
            )

    if found:
        console.print(table)
    else:
        console.print("[yellow]No worktrees found for this team[/]")


@app.command("checkpoint")
def workspace_checkpoint(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID"),
    message: str = typer.Option("nemospawn checkpoint", "--message", "-m", help="Commit message"),
):
    """Auto-commit current agent work in its worktree (checkpoint)."""
    import subprocess
    from pathlib import Path

    agent_data = _get_agent(team_id, agent_id)
    if not agent_data:
        raise typer.Exit(1)

    wt_path = agent_data.get("worktree_path")
    if not wt_path or not Path(wt_path).is_dir():
        console.print(f"[red]No active worktree for agent '{agent_id}'[/]")
        raise typer.Exit(1)

    # git add -A && git commit
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=wt_path, capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=wt_path, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            console.print(f"[green]Checkpoint saved for agent '{agent_id}'[/]")
            # Show short hash
            log = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=wt_path, capture_output=True, text=True, timeout=10,
            )
            if log.returncode == 0:
                console.print(f"[dim]{log.stdout.strip()}[/]")
        else:
            console.print(f"[yellow]Nothing to checkpoint (working tree clean)[/]")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        console.print(f"[red]Checkpoint failed: {e}[/]")
        raise typer.Exit(1)


@app.command("merge")
def workspace_merge(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID"),
    target: str = typer.Option("main", "--target", help="Branch to merge into (default: main)"),
):
    """Merge agent worktree branch back to main (or target branch)."""
    import subprocess
    from pathlib import Path

    agent_data = _get_agent(team_id, agent_id)
    if not agent_data:
        raise typer.Exit(1)

    wt_path = agent_data.get("worktree_path")
    if not wt_path or not Path(wt_path).is_dir():
        console.print(f"[red]No active worktree for agent '{agent_id}'[/]")
        raise typer.Exit(1)

    # Get the branch name from the worktree
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=wt_path, capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        console.print(f"[red]Failed to determine branch: {result.stderr.strip()}[/]")
        raise typer.Exit(1)

    branch = result.stdout.strip()

    # Find the main repo (parent of worktree)
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=wt_path, capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        console.print(f"[red]Failed to find main repo[/]")
        raise typer.Exit(1)

    git_common = Path(result.stdout.strip())
    repo_root = git_common.parent if git_common.name == ".git" else git_common.parent

    # Checkpoint before merge
    subprocess.run(
        ["git", "add", "-A"],
        cwd=wt_path, capture_output=True, text=True, timeout=30,
    )
    subprocess.run(
        ["git", "commit", "-m", f"nemospawn: final checkpoint before merge ({agent_id})", "--allow-empty"],
        cwd=wt_path, capture_output=True, text=True, timeout=30,
    )

    # Merge branch into target from the main repo
    result = subprocess.run(
        ["git", "merge", branch, "--no-ff", "-m", f"nemospawn: merge {agent_id} ({branch})"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=60,
    )

    if result.returncode == 0:
        console.print(f"[green]Merged '{branch}' into '{target}'[/]")
    else:
        console.print(f"[red]Merge failed:[/] {result.stderr.strip()}")
        console.print(f"[dim]Resolve conflicts in {repo_root} and commit manually[/]")
        raise typer.Exit(1)


@app.command("cleanup")
def workspace_cleanup(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID"),
):
    """Remove an agent's worktree and clean up the branch."""
    from pathlib import Path
    from nemospawn.runtime.worktree import remove_worktree

    agent_data = _get_agent(team_id, agent_id)
    if not agent_data:
        raise typer.Exit(1)

    wt_path = agent_data.get("worktree_path")
    if not wt_path:
        console.print(f"[yellow]No worktree configured for agent '{agent_id}'[/]")
        return

    team_dir = get_team_dir(team_id)
    removed = remove_worktree(team_dir, Path(wt_path))
    if removed:
        console.print(f"[green]Worktree removed for agent '{agent_id}'[/]")
    else:
        console.print(f"[yellow]Worktree already removed or not found[/]")


def _get_agent(team_id: str, agent_id: str) -> dict | None:
    """Load agent data, print error if not found."""
    team_dir = get_team_dir(team_id)
    agent_file = team_dir / AGENTS_SUBDIR / f"{agent_id}.json"
    data = atomic_read(agent_file)
    if not data:
        console.print(f"[red]Agent '{agent_id}' not found in team '{team_id}'[/]")
        return None
    return data
