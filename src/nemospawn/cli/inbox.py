"""Inter-agent messaging commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("send")
def send(
    team_id: str = typer.Argument(..., help="Team ID"),
    to: str = typer.Argument(..., help="Destination agent ID"),
    body: str = typer.Argument(..., help="Message body"),
    sender: str = typer.Option("leader", "--from", help="Sender agent ID"),
):
    """Send a message to an agent."""
    from nemospawn.messaging.inbox import send_message

    msg = send_message(team_id, sender, to, body)
    console.print(f"[green]Message sent:[/] {msg.msg_id} → {to}")


@app.command("broadcast")
def broadcast(
    team_id: str = typer.Argument(..., help="Team ID"),
    body: str = typer.Argument(..., help="Message body"),
    sender: str = typer.Option("leader", "--from", help="Sender agent ID"),
):
    """Broadcast a message to all agents in the team."""
    from nemospawn.messaging.inbox import broadcast_message

    msgs = broadcast_message(team_id, sender, body)
    console.print(f"[green]Broadcast sent to {len(msgs)} agent(s)[/]")


@app.command("receive")
def receive(
    team_id: str = typer.Argument(..., help="Team ID"),
    agent_id: str = typer.Argument(..., help="Agent ID"),
    all_msgs: bool = typer.Option(False, "--all", help="Include read messages"),
):
    """Receive messages from an agent's inbox."""
    from nemospawn.messaging.inbox import mark_read, receive_messages

    messages = receive_messages(team_id, agent_id, unread_only=not all_msgs)
    if not messages:
        console.print("[dim]No messages[/]")
        raise typer.Exit()

    table = Table(title=f"Inbox — {agent_id}")
    table.add_column("ID", style="cyan")
    table.add_column("From")
    table.add_column("Body")
    table.add_column("Time")
    table.add_column("Read")

    for m in messages:
        table.add_row(m.msg_id, m.from_agent, m.body[:60], m.timestamp[:19], "yes" if m.read else "no")
        if not m.read:
            mark_read(team_id, agent_id, m.msg_id)

    console.print(table)
