"""Git worktree management for agent isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

console = Console(stderr=True)


def _run_git(*args: str, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )


def create_worktree(
    repo_path: Path,
    worktree_path: Path,
    branch: str | None = None,
) -> Path:
    """Create a git worktree for an agent.

    Args:
        repo_path: Path to the main git repository.
        worktree_path: Where to place the worktree.
        branch: Branch name to create. If None, uses detached HEAD.

    Returns:
        The worktree path.

    Raises:
        RuntimeError: If worktree creation fails.
    """
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    args = ["-C", str(repo_path), "worktree", "add"]
    if branch:
        args.extend(["-b", branch])
    args.append(str(worktree_path))

    if not branch:
        args.append("HEAD")

    result = _run_git(*args)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {result.stderr.strip()}")

    return worktree_path


def remove_worktree(repo_path: Path, worktree_path: Path) -> bool:
    """Remove a git worktree."""
    result = _run_git("-C", str(repo_path), "worktree", "remove", str(worktree_path), "--force")
    if result.returncode != 0:
        console.print(f"[yellow]Failed to remove worktree: {result.stderr.strip()}[/]")
        return False
    return True


def list_worktrees(repo_path: Path) -> list[dict]:
    """List all git worktrees for a repository."""
    result = _run_git("-C", str(repo_path), "worktree", "list", "--porcelain")
    if result.returncode != 0:
        return []

    worktrees = []
    current: dict = {}
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            if current:
                worktrees.append(current)
            current = {"path": line.split(" ", 1)[1]}
        elif line.startswith("HEAD "):
            current["head"] = line.split(" ", 1)[1]
        elif line.startswith("branch "):
            current["branch"] = line.split(" ", 1)[1]
        elif line == "bare":
            current["bare"] = True
        elif line == "detached":
            current["detached"] = True

    if current:
        worktrees.append(current)

    return worktrees
