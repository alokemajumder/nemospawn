"""Terminal kanban board — live-updating task view with Rich."""

from __future__ import annotations

import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nemospawn.core.config import AGENTS_SUBDIR, TASKS_SUBDIR, METRICS_SUBDIR
from nemospawn.core.state import atomic_read, get_team_dir, list_json_files


def build_kanban(team_id: str) -> Layout:
    """Build a kanban layout showing tasks by status and agent metrics."""
    layout = Layout()
    layout.split_row(
        Layout(name="tasks", ratio=3),
        Layout(name="agents", ratio=2),
    )

    # Tasks panel
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    columns = {"pending": [], "running": [], "done": [], "failed": [], "blocked": []}

    for f in list_json_files(tasks_dir):
        data = atomic_read(f)
        if data:
            status = data.get("status", "pending")
            title = data.get("title", "")[:30]
            agent = data.get("agent_id", "")[:12]
            val = data.get("metadata", {}).get("val_loss")
            val_str = f" [{val:.4f}]" if val else ""
            columns.setdefault(status, []).append(f"{title}{val_str}\n  → {agent}" if agent else title)

    task_table = Table(title=f"Tasks — {team_id}", expand=True)
    status_colors = {"pending": "dim", "running": "blue", "done": "green", "failed": "red", "blocked": "yellow"}
    for status in ["pending", "blocked", "running", "done", "failed"]:
        items = columns.get(status, [])
        color = status_colors.get(status, "")
        header = f"[{color}]{status.upper()} ({len(items)})[/]"
        task_table.add_column(header, style=color)

    # Fill rows
    max_rows = max(len(v) for v in columns.values()) if columns else 0
    for i in range(max_rows):
        row = []
        for status in ["pending", "blocked", "running", "done", "failed"]:
            items = columns.get(status, [])
            row.append(items[i] if i < len(items) else "")
        task_table.add_row(*row)

    layout["tasks"].update(Panel(task_table, border_style="cyan"))

    # Agents panel
    agents_dir = get_team_dir(team_id) / AGENTS_SUBDIR
    agent_table = Table(title="Agents", expand=True)
    agent_table.add_column("Name", style="cyan")
    agent_table.add_column("Role")
    agent_table.add_column("GPU")
    agent_table.add_column("Status")

    for f in list_json_files(agents_dir):
        data = atomic_read(f)
        if data:
            s = data.get("status", "")
            style = "green" if s == "running" else "red"
            agent_table.add_row(
                data.get("name", "")[:15],
                data.get("role", ""),
                str(data.get("gpu_ids", [])),
                f"[{style}]{s}[/]",
            )

    layout["agents"].update(Panel(agent_table, border_style="cyan"))

    return layout


def live_kanban(team_id: str, interval: int = 30) -> None:
    """Display a live-updating terminal kanban board."""
    console = Console()

    with Live(build_kanban(team_id), console=console, refresh_per_second=0.5) as live:
        try:
            while True:
                time.sleep(interval)
                live.update(build_kanban(team_id))
        except KeyboardInterrupt:
            pass
