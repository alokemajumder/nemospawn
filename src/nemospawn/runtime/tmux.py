"""tmux session management for agent isolation."""

from __future__ import annotations

import shutil
import subprocess

from rich.console import Console

console = Console(stderr=True)


def _check_tmux() -> bool:
    """Verify tmux is installed."""
    return shutil.which("tmux") is not None


def _run_tmux(*args: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a tmux command."""
    if not _check_tmux():
        raise RuntimeError("tmux is not installed. Install it with: brew install tmux (macOS) or apt install tmux (Linux)")
    return subprocess.run(
        ["tmux", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def create_session(session_name: str, env: dict[str, str] | None = None, command: str | None = None) -> bool:
    """Create a new detached tmux session with optional environment variables.

    Returns True if the session was created successfully.
    """
    result = _run_tmux("new-session", "-d", "-s", session_name)
    if result.returncode != 0:
        console.print(f"[red]Failed to create tmux session '{session_name}': {result.stderr.strip()}[/]")
        return False

    # Set environment variables
    if env:
        for key, value in env.items():
            _run_tmux("set-environment", "-t", session_name, key, value)

    # Send initial command if provided
    if command:
        send_command(session_name, command)

    return True


def send_command(session_name: str, command: str) -> bool:
    """Send a command to a tmux session."""
    result = _run_tmux("send-keys", "-t", session_name, command, "Enter")
    return result.returncode == 0


def kill_session(session_name: str) -> bool:
    """Kill a tmux session."""
    result = _run_tmux("kill-session", "-t", session_name)
    return result.returncode == 0


def list_sessions() -> list[str]:
    """List all tmux session names."""
    try:
        result = _run_tmux("list-sessions", "-F", "#{session_name}")
        if result.returncode != 0:
            return []
        return [s.strip() for s in result.stdout.strip().splitlines() if s.strip()]
    except (RuntimeError, subprocess.TimeoutExpired):
        return []


def session_exists(session_name: str) -> bool:
    """Check if a tmux session exists."""
    result = _run_tmux("has-session", "-t", session_name)
    return result.returncode == 0


def create_tiled_view(session_names: list[str], view_name: str = "nemo-board") -> bool:
    """Create a tiled tmux view showing multiple agent sessions.

    Creates a new tmux session with panes linked to each agent session.
    """
    if not session_names:
        console.print("[yellow]No sessions to display[/]")
        return False

    # Create the board session
    result = _run_tmux("new-session", "-d", "-s", view_name)
    if result.returncode != 0:
        console.print(f"[red]Failed to create board session: {result.stderr.strip()}[/]")
        return False

    # First pane watches the first session
    send_command(view_name, f"tmux switch-client -t {session_names[0]} 2>/dev/null || echo 'Session {session_names[0]} not found'")

    # Split for each additional session
    for i, sess in enumerate(session_names[1:], 1):
        split_dir = "-h" if i % 2 == 1 else "-v"
        _run_tmux("split-window", split_dir, "-t", view_name)
        send_command(f"{view_name}", f"tmux switch-client -t {sess} 2>/dev/null || echo 'Session {sess} not found'")

    # Tile all panes evenly
    _run_tmux("select-layout", "-t", view_name, "tiled")

    return True
