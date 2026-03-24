"""Plan approval workflow CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("submit")
def plan_submit(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    agent_id: str = typer.Option(..., "--agent", help="Agent ID submitting the plan"),
    title: str = typer.Option(..., "--title", help="Plan title"),
    description: str = typer.Option("", "--description", "-d", help="Plan description"),
    steps: Optional[str] = typer.Option(None, "--steps", help="Comma-separated plan steps"),
):
    """Submit a plan for leader review before execution."""
    from nemospawn.core.plan import submit_plan

    step_list = [s.strip() for s in steps.split(",")] if steps else []
    plan = submit_plan(team_id, agent_id, title, description, step_list)

    console.print(Panel(
        f"[bold]Plan ID:[/] {plan.plan_id}\n"
        f"[bold]Title:[/] {plan.title}\n"
        f"[bold]Agent:[/] {plan.agent_id}\n"
        f"[bold]Steps:[/] {len(plan.steps)}\n"
        f"[bold]Status:[/] [yellow]pending[/]",
        title="Plan Submitted",
        border_style="yellow",
    ))


@app.command("approve")
def plan_approve(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    plan_id: str = typer.Option(..., "--plan", help="Plan ID to approve"),
    reviewer: str = typer.Option("", "--reviewer", help="Reviewer name"),
    comment: str = typer.Option("", "--comment", help="Approval comment"),
):
    """Approve a pending plan."""
    from nemospawn.core.plan import review_plan

    plan = review_plan(team_id, plan_id, "approved", reviewer=reviewer, comment=comment)
    if not plan:
        console.print(f"[red]Plan '{plan_id}' not found[/]")
        raise typer.Exit(1)
    console.print(f"[green]Plan '{plan_id}' approved[/]")


@app.command("reject")
def plan_reject(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    plan_id: str = typer.Option(..., "--plan", help="Plan ID to reject"),
    reviewer: str = typer.Option("", "--reviewer", help="Reviewer name"),
    comment: str = typer.Option("", "--comment", help="Rejection reason"),
):
    """Reject a pending plan."""
    from nemospawn.core.plan import review_plan

    plan = review_plan(team_id, plan_id, "rejected", reviewer=reviewer, comment=comment)
    if not plan:
        console.print(f"[red]Plan '{plan_id}' not found[/]")
        raise typer.Exit(1)
    console.print(f"[red]Plan '{plan_id}' rejected[/]")


@app.command("list")
def plan_list(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status (pending/approved/rejected)"),
    agent_id: Optional[str] = typer.Option(None, "--agent", help="Filter by agent ID"),
):
    """List plans for a team."""
    from nemospawn.core.plan import list_plans

    plans = list_plans(team_id, status=status, agent_id=agent_id)
    if not plans:
        console.print("[yellow]No plans found[/]")
        raise typer.Exit()

    table = Table(title=f"Plans — {team_id}")
    table.add_column("Plan ID", style="cyan")
    table.add_column("Title")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Steps")
    table.add_column("Created")

    status_colors = {"pending": "yellow", "approved": "green", "rejected": "red"}
    for p in plans:
        color = status_colors.get(p.status, "dim")
        table.add_row(
            p.plan_id,
            p.title[:40],
            p.agent_id[:12],
            f"[{color}]{p.status}[/]",
            str(len(p.steps)),
            p.created_at[:19],
        )
    console.print(table)


@app.command("show")
def plan_show(
    team_id: str = typer.Option(..., "--team", help="Team ID"),
    plan_id: str = typer.Option(..., "--plan", help="Plan ID"),
):
    """Show plan details."""
    from nemospawn.core.plan import get_plan

    plan = get_plan(team_id, plan_id)
    if not plan:
        console.print(f"[red]Plan '{plan_id}' not found[/]")
        raise typer.Exit(1)

    steps_text = ""
    if plan.steps:
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.steps))

    status_colors = {"pending": "yellow", "approved": "green", "rejected": "red"}
    color = status_colors.get(plan.status, "dim")

    text = (
        f"[bold]Plan ID:[/] {plan.plan_id}\n"
        f"[bold]Title:[/] {plan.title}\n"
        f"[bold]Agent:[/] {plan.agent_id}\n"
        f"[bold]Status:[/] [{color}]{plan.status}[/]\n"
        f"[bold]Created:[/] {plan.created_at}\n"
    )
    if plan.description:
        text += f"[bold]Description:[/] {plan.description}\n"
    if steps_text:
        text += f"[bold]Steps:[/]\n{steps_text}\n"
    if plan.reviewer:
        text += f"[bold]Reviewer:[/] {plan.reviewer}\n"
    if plan.review_comment:
        text += f"[bold]Comment:[/] {plan.review_comment}\n"
    if plan.reviewed_at:
        text += f"[bold]Reviewed:[/] {plan.reviewed_at}\n"

    console.print(Panel(text, title=f"Plan: {plan.plan_id}", border_style=color))
