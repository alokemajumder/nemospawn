"""Tests for inbox messaging."""

from unittest.mock import patch


def test_send_and_receive(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.messaging.inbox.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("test-team")

        from nemospawn.messaging.inbox import send_message, receive_messages

        msg = send_message("test-team", "leader", "worker0", "Start training")
        assert msg.body == "Start training"
        assert msg.from_agent == "leader"

        messages = receive_messages("test-team", "worker0", unread_only=False)
        assert len(messages) == 1
        assert messages[0].body == "Start training"


def test_mark_read(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.messaging.inbox.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("test-team")

        from nemospawn.messaging.inbox import send_message, receive_messages, mark_read

        msg = send_message("test-team", "leader", "worker0", "Check results")
        mark_read("test-team", "worker0", msg.msg_id)

        unread = receive_messages("test-team", "worker0", unread_only=True)
        assert len(unread) == 0

        all_msgs = receive_messages("test-team", "worker0", unread_only=False)
        assert len(all_msgs) == 1
