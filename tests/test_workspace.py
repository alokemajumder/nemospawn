"""Tests for workspace checkpoint/merge/cleanup CLI."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from nemospawn.core.models import Agent
from nemospawn.core.state import atomic_write, ensure_team_dir


def _create_agent_with_worktree(teams_dir, team_id, agent_id, worktree_path):
    """Create a test agent with a worktree path."""
    ensure_team_dir(team_id)
    agent = Agent(
        agent_id=agent_id, team_id=team_id, name=agent_id,
        role="trainer", gpu_ids=[0], status="running",
        worktree_path=str(worktree_path),
    )
    agents_dir = teams_dir / team_id / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(agents_dir / f"{agent_id}.json", agent.to_dict())


def test_workspace_list(state_dir, tmp_path):
    teams_dir = state_dir / "teams"
    wt = tmp_path / "worktree"
    wt.mkdir()
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.cli.workspace.get_team_dir", lambda tid: teams_dir / tid):
        _create_agent_with_worktree(teams_dir, "t1", "a1", wt)

        from typer.testing import CliRunner
        from nemospawn.cli.workspace import app

        runner = CliRunner()
        result = runner.invoke(app, ["list", "--team", "t1"])
        assert result.exit_code == 0
        assert "a1" in result.output


def test_workspace_list_no_worktrees(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.cli.workspace.get_team_dir", lambda tid: teams_dir / tid):
        ensure_team_dir("t1")

        from typer.testing import CliRunner
        from nemospawn.cli.workspace import app

        runner = CliRunner()
        result = runner.invoke(app, ["list", "--team", "t1"])
        assert "No worktrees" in result.output


def test_workspace_checkpoint_success(state_dir, tmp_path):
    teams_dir = state_dir / "teams"
    wt = tmp_path / "worktree"
    wt.mkdir()

    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.cli.workspace.get_team_dir", lambda tid: teams_dir / tid):
        _create_agent_with_worktree(teams_dir, "t1", "a1", wt)

        mock_add = MagicMock(returncode=0, stdout="", stderr="")
        mock_commit = MagicMock(returncode=0, stdout="", stderr="")
        mock_log = MagicMock(returncode=0, stdout="abc1234 checkpoint", stderr="")

        with patch("subprocess.run", side_effect=[mock_add, mock_commit, mock_log]):
            from typer.testing import CliRunner
            from nemospawn.cli.workspace import app

            runner = CliRunner()
            result = runner.invoke(app, ["checkpoint", "--team", "t1", "--agent", "a1"])
            assert result.exit_code == 0
            assert "Checkpoint saved" in result.output


def test_workspace_cleanup(state_dir, tmp_path):
    teams_dir = state_dir / "teams"
    wt = tmp_path / "worktree"
    wt.mkdir()

    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.cli.workspace.get_team_dir", lambda tid: teams_dir / tid):
        _create_agent_with_worktree(teams_dir, "t1", "a1", wt)

        with patch("nemospawn.runtime.worktree.remove_worktree", return_value=True):
            from typer.testing import CliRunner
            from nemospawn.cli.workspace import app

            runner = CliRunner()
            result = runner.invoke(app, ["cleanup", "--team", "t1", "--agent", "a1"])
            assert result.exit_code == 0
            assert "removed" in result.output
