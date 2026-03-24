"""Tests for auth and audit modules."""

from unittest.mock import patch
from nemospawn.core.auth import create_user, authenticate, authorize_team_access, generate_api_key
from nemospawn.core.audit import log_event, read_audit_log


def test_generate_api_key():
    key, key_hash = generate_api_key()
    assert key.startswith("ns_")
    assert len(key_hash) == 64  # sha256 hex


def test_create_and_authenticate_user(state_dir):
    auth_dir = state_dir / "auth"
    with patch("nemospawn.core.auth.AUTH_DIR", auth_dir):
        user, key = create_user("alice", role="admin")
        assert user.username == "alice"
        assert user.role == "admin"

        # Authenticate
        authed = authenticate(key)
        assert authed is not None
        assert authed.username == "alice"

        # Bad key
        assert authenticate("bad_key") is None


def test_authorize_team_access(state_dir):
    auth_dir = state_dir / "auth"
    with patch("nemospawn.core.auth.AUTH_DIR", auth_dir):
        user, _ = create_user("bob")
        user.teams = ["team-123"]

        assert authorize_team_access(user, "team-123") is True
        assert authorize_team_access(user, "bob-experiment") is True  # namespace prefix
        assert authorize_team_access(user, "other-team") is False

        admin, _ = create_user("admin", role="admin")
        assert authorize_team_access(admin, "any-team") is True


def test_audit_log(state_dir):
    audit_log = state_dir / "audit.jsonl"
    with patch("nemospawn.core.audit.STATE_DIR", state_dir), \
         patch("nemospawn.core.audit.AUDIT_LOG", audit_log):
        log_event("team.create", {"team_id": "t1"}, user="alice", team_id="t1")
        log_event("agent.spawn", {"agent_id": "w0"}, user="alice", team_id="t1", agent_id="w0")
        log_event("task.update", {"status": "done"}, user="alice", team_id="t1")

        entries = read_audit_log()
        assert len(entries) == 3
        assert entries[0]["event"] == "team.create"

        # Filter by event type
        spawns = read_audit_log(event_type="agent.spawn")
        assert len(spawns) == 1

        # Filter by team
        team_events = read_audit_log(team_id="t1")
        assert len(team_events) == 3
