"""Tests for git worktree management."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from nemospawn.runtime.worktree import create_worktree, remove_worktree, list_worktrees


def test_create_worktree_success(tmp_path):
    result = MagicMock()
    result.returncode = 0

    repo = tmp_path / "repo"
    wt_path = tmp_path / "worktrees" / "agent-1"

    with patch("subprocess.run", return_value=result):
        path = create_worktree(repo, wt_path, branch="nemospawn/t1/a1")
        assert path == wt_path


def test_create_worktree_failure(tmp_path):
    result = MagicMock()
    result.returncode = 1
    result.stderr = "fatal: worktree already exists"

    repo = tmp_path / "repo"
    wt_path = tmp_path / "worktrees" / "agent-1"

    with patch("subprocess.run", return_value=result):
        import pytest
        with pytest.raises(RuntimeError, match="Failed to create worktree"):
            create_worktree(repo, wt_path)


def test_remove_worktree():
    result = MagicMock()
    result.returncode = 0

    with patch("subprocess.run", return_value=result):
        assert remove_worktree(Path("/repo"), Path("/wt/agent1")) is True


def test_list_worktrees():
    result = MagicMock()
    result.returncode = 0
    result.stdout = (
        "worktree /repo\n"
        "HEAD abc123\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /wt/agent1\n"
        "HEAD def456\n"
        "branch refs/heads/nemospawn/t1/a1\n"
    )

    with patch("subprocess.run", return_value=result):
        wts = list_worktrees(Path("/repo"))
        assert len(wts) == 2
        assert wts[0]["path"] == "/repo"
        assert wts[1]["branch"] == "refs/heads/nemospawn/t1/a1"
