"""Tests for template engine and launch."""

from nemospawn.templates.engine import (
    get_builtin_template, list_builtin_templates, load_template_from_string,
    TeamTemplate, WorkerSpec, BUILTIN_TEMPLATES,
)


def test_list_builtin_templates():
    templates = list_builtin_templates()
    assert len(templates) == 4
    names = [t["name"] for t in templates]
    assert "autoresearch" in names
    assert "nim-deploy" in names
    assert "rlhf-swarm" in names
    assert "data-curation" in names


def test_get_builtin_autoresearch():
    tmpl = get_builtin_template("autoresearch")
    assert tmpl is not None
    assert tmpl.name == "autoresearch"
    assert tmpl.min_gpus == 2
    assert len(tmpl.workers) == 3
    assert tmpl.workers[0].role == "trainer"
    assert tmpl.workers[2].role == "evaluator"
    assert "trainer-0" in tmpl.workers[2].blocked_by


def test_get_builtin_rlhf():
    tmpl = get_builtin_template("rlhf-swarm")
    assert tmpl is not None
    assert tmpl.min_gpus == 4
    assert len(tmpl.workers) == 4
    # Reward trainer should use 2 GPUs with NVLink
    reward = tmpl.workers[0]
    assert reward.role == "rlhf-reward"
    assert reward.gpu_count == 2
    assert reward.require_nvlink is True


def test_get_builtin_nim_deploy():
    tmpl = get_builtin_template("nim-deploy")
    assert tmpl is not None
    assert len(tmpl.workers) == 3
    deployer_tp2 = tmpl.workers[1]
    assert deployer_tp2.gpu_count == 2
    assert deployer_tp2.require_nvlink is True


def test_get_builtin_data_curation():
    tmpl = get_builtin_template("data-curation")
    assert tmpl is not None
    assert tmpl.workers[1].blocked_by == ["curator"]


def test_get_nonexistent_template():
    assert get_builtin_template("nonexistent") is None


def test_load_template_from_string():
    toml = '''
name = "test"
description = "test template"
min_gpus = 1

[[workers]]
name = "w0"
role = "trainer"
gpu_count = 1
task = "do stuff"
'''
    tmpl = load_template_from_string(toml)
    assert tmpl.name == "test"
    assert len(tmpl.workers) == 1
    assert tmpl.workers[0].name == "w0"


def test_cli_launch_templates():
    from typer.testing import CliRunner
    from nemospawn.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["launch", "templates"])
    assert result.exit_code == 0
    assert "autoresearch" in result.output
    assert "rlhf-swarm" in result.output
