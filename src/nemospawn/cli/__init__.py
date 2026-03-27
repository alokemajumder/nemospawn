"""NemoSpawn CLI — built with Typer + Rich."""

import typer

app = typer.Typer(
    name="nemospawn",
    help="GPU-native agent-swarm orchestration for the full NVIDIA AI stack.",
    no_args_is_help=True,
)

# Register command groups
from nemospawn.cli import team, spawn, task, inbox, board, gpu  # noqa: E402
from nemospawn.cli import artifact, nim, hpo, cluster, launch  # noqa: E402
from nemospawn.cli import ngc, slurm, auth  # noqa: E402
from nemospawn.cli import plan, lifecycle, cost, snapshot, watch  # noqa: E402
from nemospawn.cli import profile, config, schedule, skill  # noqa: E402
from nemospawn.cli import workspace  # noqa: E402

app.add_typer(team.app, name="team", help="Team lifecycle management")
app.add_typer(spawn.app, name="spawn", help="Agent spawn and management")
app.add_typer(task.app, name="task", help="Task DAG management")
app.add_typer(inbox.app, name="inbox", help="Inter-agent messaging")
app.add_typer(board.app, name="board", help="Monitoring dashboards")
app.add_typer(gpu.app, name="gpu", help="GPU discovery and health")
app.add_typer(artifact.app, name="artifact", help="Artifact management")
app.add_typer(nim.app, name="nim", help="NIM deployment pipeline")
app.add_typer(hpo.app, name="hpo", help="Hyperparameter optimization")
app.add_typer(cluster.app, name="cluster", help="Cross-cluster federation")
app.add_typer(launch.app, name="launch", help="Launch teams from templates")
app.add_typer(ngc.app, name="ngc", help="NGC registry operations")
app.add_typer(slurm.app, name="slurm", help="SLURM job management")
app.add_typer(auth.app, name="auth", help="Authentication and audit")
app.add_typer(plan.app, name="plan", help="Plan approval workflow")
app.add_typer(lifecycle.app, name="lifecycle", help="Agent lifecycle protocol")
app.add_typer(cost.app, name="cost", help="Cost tracking")
app.add_typer(snapshot.app, name="snapshot", help="Team state snapshots")
app.add_typer(watch.app, name="watch", help="Agent health monitoring")
app.add_typer(profile.app, name="profile", help="Agent profile management")
app.add_typer(config.app, name="config", help="Configuration management")
app.add_typer(schedule.app, name="schedule", help="Adaptive task scheduling")
app.add_typer(skill.app, name="skill", help="Agent skill management")
app.add_typer(workspace.app, name="workspace", help="Git worktree management")


@app.command()
def version():
    """Show NemoSpawn version."""
    from nemospawn import __version__
    from rich.console import Console

    Console().print(f"NemoSpawn v{__version__}")
