"""Profile management CLI — wizard, doctor, test, CRUD."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def profile_list():
    """List all available agent profiles."""
    from nemospawn.core.profiles import list_profiles

    profiles = list_profiles()
    table = Table(title="Agent Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Agent Type")
    table.add_column("Command")
    table.add_column("Auth Env")
    table.add_column("Description")

    for p in profiles:
        table.add_row(p.name, p.agent, p.command or p.agent, p.auth_env or "-", p.description)
    console.print(table)


@app.command("show")
def profile_show(
    name: str = typer.Argument(..., help="Profile name"),
):
    """Show detailed profile configuration."""
    from nemospawn.core.profiles import load_profile, get_adapter

    profile = load_profile(name)
    if not profile:
        console.print(f"[red]Profile '{name}' not found[/]")
        raise typer.Exit(1)

    adapter = get_adapter(profile.agent)

    text = (
        f"[bold]Name:[/] {profile.name}\n"
        f"[bold]Agent Type:[/] {profile.agent}\n"
        f"[bold]Command:[/] {profile.command or profile.agent}\n"
        f"[bold]Model:[/] {profile.model or '(default)'}\n"
        f"[bold]Base URL:[/] {profile.base_url or '(default)'}\n"
        f"[bold]Auth Env:[/] {profile.auth_env or '(none)'}\n"
        f"[bold]Extra Args:[/] {profile.args or '(none)'}\n"
        f"[bold]Extra Env:[/] {profile.env or '(none)'}\n"
        f"[bold]Description:[/] {profile.description}\n"
        f"\n[bold]Adapter:[/]\n"
        f"  Spawn args: {adapter.get('spawn_args', [])}\n"
        f"  Prompt method: {adapter.get('prompt_method', 'file')}\n"
        f"  Prompt flag: {adapter.get('prompt_flag', '(none)')}\n"
        f"  Trust prompt: {adapter.get('trust_prompt', False)}"
    )
    console.print(Panel(text, title=f"Profile: {name}", border_style="cyan"))


@app.command("create")
def profile_create(
    name: str = typer.Option(..., "--name", help="Profile name"),
    agent: str = typer.Option("claude", "--agent", help="Agent type (claude, codex, kimi, cursor, nanobot, aider, opencode, copilot, custom)"),
    command: str = typer.Option("", "--command", help="CLI command to invoke"),
    model: str = typer.Option("", "--model", help="LLM model name or endpoint"),
    base_url: str = typer.Option("", "--base-url", help="API base URL"),
    auth_env: str = typer.Option("", "--auth-env", help="Env var holding the API key"),
    description: str = typer.Option("", "--description", "-d", help="Profile description"),
):
    """Create a new agent profile."""
    from nemospawn.core.profiles import AgentProfile, save_profile

    profile = AgentProfile(
        name=name,
        agent=agent,
        command=command or agent,
        model=model,
        base_url=base_url,
        auth_env=auth_env,
        description=description,
    )
    save_profile(profile)
    console.print(f"[green]Profile '{name}' created[/]")


@app.command("delete")
def profile_delete(
    name: str = typer.Argument(..., help="Profile name to delete"),
):
    """Delete a saved profile."""
    from nemospawn.core.profiles import remove_profile

    if remove_profile(name):
        console.print(f"[green]Profile '{name}' deleted[/]")
    else:
        console.print(f"[yellow]Profile '{name}' not found on disk (may be a built-in default)[/]")


@app.command("test")
def profile_test(
    name: str = typer.Argument(..., help="Profile name to test"),
):
    """Smoke-test a profile (check command, auth, endpoint)."""
    from nemospawn.core.profiles import load_profile, check_profile

    profile = load_profile(name)
    if not profile:
        console.print(f"[red]Profile '{name}' not found[/]")
        raise typer.Exit(1)

    result = check_profile(profile)

    console.print(f"\n[bold]Profile Test: {name}[/]\n")
    for check, passed in result["checks"].items():
        icon = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        console.print(f"  {icon}  {check}")

    if result["ok"]:
        console.print(f"\n[green]Profile '{name}' is ready[/]")
    else:
        console.print(f"\n[red]Profile '{name}' has issues — see failures above[/]")


@app.command("wizard")
def profile_wizard():
    """Interactive step-by-step profile creation."""
    from nemospawn.core.profiles import AgentProfile, save_profile, DEFAULT_PROFILES

    console.print(Panel(
        "Create a new agent profile step by step.\n"
        "Supported agents: claude, codex, kimi, cursor, nanobot, aider, opencode, copilot, custom",
        title="Profile Wizard",
        border_style="cyan",
    ))

    # Step 1: Name
    name = typer.prompt("Profile name")

    # Step 2: Agent type
    agent_types = list(DEFAULT_PROFILES.keys()) + ["custom"]
    console.print(f"\n[bold]Available agent types:[/] {', '.join(agent_types)}")
    agent = typer.prompt("Agent type", default="claude")

    # Step 3: Command
    default_cmd = DEFAULT_PROFILES.get(agent, {}).get("command", agent)
    command = typer.prompt("CLI command", default=default_cmd)

    # Step 4: Model (optional)
    model = typer.prompt("Model name (leave blank for default)", default="")

    # Step 5: Base URL (optional)
    base_url = typer.prompt("API base URL (leave blank for default)", default="")

    # Step 6: Auth env var
    default_auth = DEFAULT_PROFILES.get(agent, {}).get("auth_env", "")
    auth_env = typer.prompt("Auth env var name", default=default_auth)

    # Step 7: Description
    description = typer.prompt("Description", default=f"Custom {agent} profile")

    profile = AgentProfile(
        name=name,
        agent=agent,
        command=command,
        model=model,
        base_url=base_url,
        auth_env=auth_env,
        description=description,
    )

    save_profile(profile)
    console.print(f"\n[green]Profile '{name}' created successfully![/]")
    console.print(f"[dim]Use 'nemospawn profile test {name}' to validate[/]")


@app.command("doctor")
def profile_doctor(
    name: str = typer.Argument(..., help="Profile name to diagnose"),
):
    """Diagnose and fix profile issues — first-run onboarding helper."""
    from nemospawn.core.profiles import load_profile, check_profile, get_adapter
    import os
    import shutil

    profile = load_profile(name)
    if not profile:
        console.print(f"[red]Profile '{name}' not found[/]")
        console.print(f"[dim]Run 'nemospawn profile wizard' to create one[/]")
        raise typer.Exit(1)

    console.print(Panel(
        f"Diagnosing profile '{name}'...",
        title="Profile Doctor",
        border_style="yellow",
    ))

    issues = []
    fixes = []

    # Check 1: Command exists
    cmd = profile.command or profile.agent
    found = shutil.which(cmd)
    if found:
        console.print(f"  [green]PASS[/]  Command '{cmd}' found at {found}")
    else:
        console.print(f"  [red]FAIL[/]  Command '{cmd}' not found in PATH")
        issues.append(f"Command '{cmd}' not installed")
        if profile.agent == "claude":
            fixes.append("Install: npm install -g @anthropic-ai/claude-code")
        elif profile.agent == "codex":
            fixes.append("Install: npm install -g @openai/codex")
        elif profile.agent == "aider":
            fixes.append("Install: pip install aider-chat")
        elif profile.agent == "kimi":
            fixes.append("Install: npm install -g @anthropic-ai/kimi-cli")
        else:
            fixes.append(f"Install the '{cmd}' CLI tool")

    # Check 2: Auth configured
    if profile.auth_env:
        val = os.environ.get(profile.auth_env)
        if val:
            console.print(f"  [green]PASS[/]  {profile.auth_env} is set ({len(val)} chars)")
        else:
            console.print(f"  [red]FAIL[/]  {profile.auth_env} is not set")
            issues.append(f"Missing environment variable: {profile.auth_env}")
            fixes.append(f"Set: export {profile.auth_env}=<your-api-key>")
    else:
        console.print(f"  [green]PASS[/]  No auth required")

    # Check 3: Adapter exists
    adapter = get_adapter(profile.agent)
    prompt_method = adapter.get("prompt_method", "file")
    console.print(f"  [green]INFO[/]  Prompt injection method: {prompt_method}")
    if adapter.get("trust_prompt"):
        console.print(f"  [green]INFO[/]  Auto-trust enabled for this agent")
    else:
        console.print(f"  [yellow]INFO[/]  Manual trust confirmation may be needed")

    # Check 4: Base URL if custom
    if profile.base_url:
        result = check_profile(profile)
        reachable = result["checks"].get("endpoint_reachable", True)
        if reachable:
            console.print(f"  [green]PASS[/]  Endpoint {profile.base_url} is reachable")
        else:
            console.print(f"  [red]FAIL[/]  Endpoint {profile.base_url} is not reachable")
            issues.append(f"Endpoint unreachable: {profile.base_url}")
            fixes.append("Check the URL and your network connectivity")

    # Summary
    console.print("")
    if not issues:
        console.print(f"[green]Profile '{name}' is healthy — ready to spawn agents[/]")
    else:
        console.print(f"[yellow]Found {len(issues)} issue(s):[/]")
        for i, issue in enumerate(issues):
            console.print(f"  {i+1}. {issue}")
        console.print(f"\n[bold]Suggested fixes:[/]")
        for fix in fixes:
            console.print(f"  - {fix}")
