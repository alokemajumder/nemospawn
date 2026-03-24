"""Hyperparameter optimization commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


def _load_study(study_name: str):
    """Load or create an HPO study."""
    from nemospawn.hpo.study import HPOStudy, SearchSpace
    from nemospawn.core.config import HPO_DIR

    state_path = HPO_DIR / f"{study_name}_state.json"
    from nemospawn.core.state import atomic_read
    state = atomic_read(state_path)

    # Try to find search space
    hpo_toml = HPO_DIR / f"{study_name}_hpo.toml"
    if hpo_toml.exists():
        space = SearchSpace.from_toml(hpo_toml)
    else:
        # Default search space
        space = SearchSpace(
            parameters={
                "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
                "batch_size": {"type": "categorical", "choices": [256, 512, 1024, 2048]},
            },
        )

    return HPOStudy(study_name, space)


@app.command("init")
def init_study(
    study: str = typer.Option(..., "--study", help="Study name"),
    template: Optional[str] = typer.Option(None, "--template", help="Template name for search space"),
    hpo_file: Optional[str] = typer.Option(None, "--hpo-file", help="Path to hpo.toml search space"),
):
    """Initialize an HPO study with a search space."""
    from nemospawn.hpo.study import HPOStudy, SearchSpace
    from nemospawn.core.config import HPO_DIR
    import shutil

    HPO_DIR.mkdir(parents=True, exist_ok=True)

    if hpo_file:
        # Copy hpo.toml into HPO dir
        src = Path(hpo_file)
        if not src.exists():
            console.print(f"[red]File not found: {hpo_file}[/]")
            raise typer.Exit(1)
        dest = HPO_DIR / f"{study}_hpo.toml"
        shutil.copy2(src, dest)
        space = SearchSpace.from_toml(dest)
    elif template:
        from nemospawn.templates.engine import get_builtin_template
        tmpl = get_builtin_template(template)
        if tmpl and tmpl.hpo_config:
            space = SearchSpace.from_dict(tmpl.hpo_config)
        else:
            space = SearchSpace()
    else:
        space = SearchSpace()

    hpo_study = HPOStudy(study, space)
    console.print(Panel(
        f"[bold]Study:[/] {study}\n"
        f"[bold]Parameters:[/] {list(space.parameters.keys())}\n"
        f"[bold]Objective:[/] {space.objective_direction} {space.objective_metric}\n"
        f"[bold]Max Trials:[/] {space.max_trials}",
        title="HPO Study Initialized",
        border_style="green",
    ))


@app.command("suggest")
def suggest(
    study: str = typer.Option(..., "--study", help="Study name"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Sample next hyperparameter configuration."""
    hpo_study = _load_study(study)
    config = hpo_study.suggest()

    if output_json:
        console.print(json.dumps(config, indent=2))
    else:
        console.print(Panel(
            "\n".join(f"[bold]{k}:[/] {v}" for k, v in config.items()),
            title=f"Suggested Config (trial {config.get('trial_id', '?')})",
            border_style="cyan",
        ))


@app.command("report")
def report(
    study: str = typer.Option(..., "--study", help="Study name"),
    trial: int = typer.Option(..., "--trial", help="Trial ID"),
    step: str = typer.Option(..., "--step", help="Step number or 'final'"),
    val_loss: float = typer.Option(..., "--val-loss", help="Validation loss"),
):
    """Report trial result (intermediate or final)."""
    hpo_study = _load_study(study)
    step_int = -1 if step == "final" else int(step)
    should_continue = hpo_study.report(trial, step_int, val_loss)

    if should_continue:
        console.print(f"[green]Trial {trial} reported: step={step}, val_loss={val_loss}[/]")
    else:
        console.print(f"[yellow]Trial {trial} PRUNED at step={step}[/]")


@app.command("best")
def best(
    study: str = typer.Option(..., "--study", help="Study name"),
):
    """Show best trial so far."""
    hpo_study = _load_study(study)
    bt = hpo_study.best_trial()

    if not bt:
        console.print("[yellow]No completed trials yet[/]")
        raise typer.Exit()

    console.print(Panel(
        f"[bold]Trial:[/] {bt['trial_id']}\n"
        f"[bold]Value:[/] {bt['value']}\n"
        f"[bold]Params:[/]\n" + "\n".join(f"  {k}: {v}" for k, v in bt["params"].items()),
        title="Best Trial",
        border_style="green",
    ))


@app.command("trials")
def list_trials(
    study: str = typer.Option(..., "--study", help="Study name"),
):
    """List all trials in a study."""
    hpo_study = _load_study(study)
    trials = hpo_study.get_all_trials()

    if not trials:
        console.print("[yellow]No trials yet[/]")
        raise typer.Exit()

    table = Table(title=f"HPO Trials — {study}")
    table.add_column("Trial", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Status")
    table.add_column("Params")

    for t in trials:
        params_str = ", ".join(f"{k}={v}" for k, v in (t.get("params") or {}).items())
        val = f"{t['value']:.4f}" if t.get("value") is not None else ""
        table.add_row(str(t["trial_id"]), val, t.get("status", t.get("state", "")), params_str[:60])
    console.print(table)


@app.command("dashboard")
def dashboard(
    study: str = typer.Option(..., "--study", help="Study name"),
    port: int = typer.Option(8081, "--port", help="Dashboard port"),
):
    """Launch Optuna dashboard (requires optuna-dashboard)."""
    from nemospawn.core.config import HPO_DIR

    db_path = HPO_DIR / f"{study}.db"
    if not db_path.exists():
        console.print(f"[red]Study database not found: {db_path}[/]")
        raise typer.Exit(1)

    console.print(f"Launching Optuna dashboard on port {port}...")
    import subprocess
    try:
        subprocess.run(
            ["optuna-dashboard", f"sqlite:///{db_path}", "--port", str(port)],
            check=True,
        )
    except FileNotFoundError:
        console.print("[red]optuna-dashboard not installed. Install: pip install optuna-dashboard[/]")
    except KeyboardInterrupt:
        pass
