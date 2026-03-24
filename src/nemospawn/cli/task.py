"""Task DAG management commands."""

from __future__ import annotations

import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from nemospawn.core.config import TASKS_SUBDIR
from nemospawn.core.models import Task, _now, _short_id, resolve_ready_tasks
from nemospawn.core.state import atomic_read, atomic_write, get_team_dir, list_json_files

app = typer.Typer(no_args_is_help=True)
console = Console()


def _load_tasks(team_id: str) -> list[Task]:
    """Load all tasks for a team."""
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    tasks = []
    for f in list_json_files(tasks_dir):
        data = atomic_read(f)
        if data:
            tasks.append(Task.from_dict(data))
    return tasks


def _unblock_dependents(team_id: str, completed_task_id: str) -> int:
    """Check all blocked tasks and unblock those whose deps are now met."""
    tasks = _load_tasks(team_id)
    done_ids = {t.task_id for t in tasks if t.status == "done"}
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    unblocked = 0

    for t in tasks:
        if t.status == "blocked" and completed_task_id in t.blocked_by:
            if all(dep in done_ids for dep in t.blocked_by):
                t.status = "pending"
                t.updated_at = _now()
                atomic_write(tasks_dir / f"{t.task_id}.json", t.to_dict())
                unblocked += 1
    return unblocked


@app.command("create")
def create_task(
    team_id: str = typer.Argument(..., help="Team ID"),
    title: str = typer.Argument(..., help="Task title/description"),
    agent: Optional[str] = typer.Option(None, "-o", "--owner", help="Assigned agent ID"),
    blocked_by: Optional[str] = typer.Option(None, "--blocked-by", help="Comma-separated task IDs this depends on"),
    artifact: Optional[str] = typer.Option(None, "--artifact", help="Expected artifact type"),
    val_loss: Optional[float] = typer.Option(None, "--val-loss", help="Initial val_loss value"),
):
    """Create a new task in the team's task DAG."""
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    if not tasks_dir.is_dir():
        console.print(f"[red]Team '{team_id}' not found[/]")
        raise typer.Exit(1)

    deps = [d.strip() for d in blocked_by.split(",")] if blocked_by else []
    status = "blocked" if deps else "pending"

    metadata: dict = {}
    if artifact:
        metadata["artifact_type"] = artifact
    if val_loss is not None:
        metadata["val_loss"] = val_loss

    task_id = _short_id("task")
    task = Task(
        task_id=task_id,
        team_id=team_id,
        title=title,
        agent_id=agent,
        status=status,
        blocked_by=deps,
        metadata=metadata,
    )

    atomic_write(tasks_dir / f"{task_id}.json", task.to_dict())
    console.print(f"[green]Task created:[/] {task_id} — {title} [{status}]")


@app.command("update")
def update_task(
    team_id: str = typer.Argument(..., help="Team ID"),
    task_id: str = typer.Argument(..., help="Task ID"),
    status: Optional[str] = typer.Option(None, "--status", help="New status (pending|running|done|failed)"),
    val_loss: Optional[float] = typer.Option(None, "--val-loss", help="Update val_loss"),
    agent: Optional[str] = typer.Option(None, "-o", "--owner", help="Assign to agent"),
):
    """Update a task's status or metadata."""
    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    task_file = tasks_dir / f"{task_id}.json"
    data = atomic_read(task_file)

    if not data:
        console.print(f"[red]Task '{task_id}' not found[/]")
        raise typer.Exit(1)

    if status:
        data["status"] = status
    if val_loss is not None:
        data.setdefault("metadata", {})["val_loss"] = val_loss
    if agent:
        data["agent_id"] = agent
    data["updated_at"] = _now()

    atomic_write(task_file, data)
    console.print(f"[green]Task updated:[/] {task_id}")

    # Unblock dependents if task completed
    if status == "done":
        unblocked = _unblock_dependents(team_id, task_id)
        if unblocked:
            console.print(f"[cyan]{unblocked} task(s) unblocked[/]")


@app.command("list")
def list_tasks(
    team_id: str = typer.Argument(..., help="Team ID"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Filter by agent"),
):
    """List tasks in a team."""
    tasks = _load_tasks(team_id)
    if not tasks:
        console.print(f"[yellow]No tasks in team '{team_id}'[/]")
        raise typer.Exit()

    if status:
        tasks = [t for t in tasks if t.status == status]
    if agent:
        tasks = [t for t in tasks if t.agent_id == agent]

    table = Table(title=f"Tasks — {team_id}")
    table.add_column("Task ID", style="cyan")
    table.add_column("Title")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Blocked By")
    table.add_column("Val Loss", justify="right")

    for t in tasks:
        status_colors = {"done": "green", "running": "blue", "failed": "red", "blocked": "yellow", "pending": "dim"}
        style = status_colors.get(t.status, "")
        val = str(t.metadata.get("val_loss", "")) if t.metadata else ""
        table.add_row(
            t.task_id,
            t.title[:50],
            t.agent_id or "",
            f"[{style}]{t.status}[/]",
            ", ".join(t.blocked_by) if t.blocked_by else "",
            val,
        )
    console.print(table)


@app.command("show")
def show_task(
    team_id: str = typer.Argument(..., help="Team ID"),
    task_id: str = typer.Argument(..., help="Task ID"),
):
    """Show detailed task information."""
    from rich.panel import Panel

    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    data = atomic_read(tasks_dir / f"{task_id}.json")
    if not data:
        console.print(f"[red]Task '{task_id}' not found[/]")
        raise typer.Exit(1)

    task = Task.from_dict(data)
    text = (
        f"[bold]Title:[/] {task.title}\n"
        f"[bold]Status:[/] {task.status}\n"
        f"[bold]Agent:[/] {task.agent_id or 'unassigned'}\n"
        f"[bold]Blocked By:[/] {', '.join(task.blocked_by) if task.blocked_by else 'none'}\n"
        f"[bold]Created:[/] {task.created_at}\n"
        f"[bold]Updated:[/] {task.updated_at}\n"
    )
    if task.metadata:
        text += "[bold]Metadata:[/]\n"
        for k, v in task.metadata.items():
            text += f"  {k}: {v}\n"

    console.print(Panel(text, title=f"Task: {task_id}"))


@app.command("wait")
def wait_task(
    team_id: str = typer.Argument(..., help="Team ID"),
    task_id: str = typer.Argument(..., help="Task ID to wait for"),
    timeout: int = typer.Option(300, "--timeout", help="Timeout in seconds"),
):
    """Wait for a task to complete."""
    from nemospawn.core.config import TASK_POLL_INTERVAL_S

    tasks_dir = get_team_dir(team_id) / TASKS_SUBDIR
    task_file = tasks_dir / f"{task_id}.json"

    with console.status(f"Waiting for task {task_id}..."):
        start = time.time()
        while time.time() - start < timeout:
            data = atomic_read(task_file)
            if not data:
                console.print(f"[red]Task '{task_id}' not found[/]")
                raise typer.Exit(1)

            if data["status"] in ("done", "failed"):
                console.print(f"[green]Task {task_id}: {data['status']}[/]")
                return

            time.sleep(TASK_POLL_INTERVAL_S)

    console.print(f"[yellow]Timeout waiting for task {task_id}[/]")
    raise typer.Exit(1)
