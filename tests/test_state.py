"""Tests for core state management."""

from nemospawn.core.state import atomic_write, atomic_read, ensure_team_dir


def test_atomic_write_and_read(state_dir):
    path = state_dir / "test.json"
    data = {"key": "value", "number": 42}
    atomic_write(path, data)
    result = atomic_read(path)
    assert result == data


def test_atomic_read_missing_file(state_dir):
    path = state_dir / "nonexistent.json"
    assert atomic_read(path) is None


def test_atomic_write_creates_parent_dirs(state_dir):
    path = state_dir / "deep" / "nested" / "file.json"
    data = {"nested": True}
    atomic_write(path, data)
    assert atomic_read(path) == data


def test_ensure_team_dir(state_dir):
    from unittest.mock import patch
    with patch("nemospawn.core.state.TEAMS_DIR", state_dir / "teams"):
        team_dir = ensure_team_dir("test-team")
        assert team_dir.is_dir()
        assert (team_dir / "agents").is_dir()
        assert (team_dir / "tasks").is_dir()
        assert (team_dir / "inbox").is_dir()
        assert (team_dir / "artifacts").is_dir()
        assert (team_dir / "workspaces").is_dir()
        assert (team_dir / "metrics").is_dir()
