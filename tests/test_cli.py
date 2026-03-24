"""Smoke tests for CLI commands."""

from typer.testing import CliRunner
from nemospawn.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "1.0.0" in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "nemospawn" in result.output.lower() or "GPU" in result.output


def test_team_help():
    result = runner.invoke(app, ["team", "--help"])
    assert result.exit_code == 0
    assert "create" in result.output.lower()


def test_task_help():
    result = runner.invoke(app, ["task", "--help"])
    assert result.exit_code == 0


def test_inbox_help():
    result = runner.invoke(app, ["inbox", "--help"])
    assert result.exit_code == 0


def test_gpu_help():
    result = runner.invoke(app, ["gpu", "--help"])
    assert result.exit_code == 0


def test_spawn_help():
    result = runner.invoke(app, ["spawn", "--help"])
    assert result.exit_code == 0


def test_nim_list():
    result = runner.invoke(app, ["nim", "list", "fake-team"])
    assert result.exit_code == 0
    assert "No NIM endpoints" in result.output or "NIM" in result.output


def test_hpo_best():
    result = runner.invoke(app, ["hpo", "best", "--study", "test"])
    assert result.exit_code == 0
    assert "No completed trials" in result.output or "Trial" in result.output


def test_cluster_list():
    result = runner.invoke(app, ["cluster", "list"])
    assert result.exit_code == 0
    assert "No clusters" in result.output or "Cluster" in result.output
