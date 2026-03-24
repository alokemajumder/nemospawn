"""Tests for tmux session management."""

from unittest.mock import patch, MagicMock
from nemospawn.runtime.tmux import create_session, kill_session, list_sessions


def test_create_session():
    result = MagicMock()
    result.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/tmux"), \
         patch("subprocess.run", return_value=result) as mock_run:
        assert create_session("test-session", env={"CUDA_VISIBLE_DEVICES": "0,1"})
        # Should call new-session + set-environment
        assert mock_run.call_count >= 2


def test_create_session_failure():
    result = MagicMock()
    result.returncode = 1
    result.stderr = "duplicate session"

    with patch("shutil.which", return_value="/usr/bin/tmux"), \
         patch("subprocess.run", return_value=result):
        assert not create_session("test-session")


def test_kill_session():
    result = MagicMock()
    result.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/tmux"), \
         patch("subprocess.run", return_value=result):
        assert kill_session("test-session")


def test_list_sessions():
    result = MagicMock()
    result.returncode = 0
    result.stdout = "nemo-team1-worker0\nnemo-team1-worker1\n"

    with patch("shutil.which", return_value="/usr/bin/tmux"), \
         patch("subprocess.run", return_value=result):
        sessions = list_sessions()
        assert len(sessions) == 2
        assert "nemo-team1-worker0" in sessions
