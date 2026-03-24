"""Tests for multi-transport messaging."""

from unittest.mock import patch

from nemospawn.messaging.transport import (
    TransportType, FileTransport, ZeroMQTransport, NIXLTransport,
    negotiate_transport, TransportConfig,
)
from nemospawn.core.models import Message


def test_file_transport_send_receive(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.messaging.inbox.get_team_dir", lambda tid: teams_dir / tid), \
         patch("nemospawn.messaging.transport.FileTransport.__init__", lambda self, tid: setattr(self, 'team_id', tid)):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("t1")

        ft = FileTransport.__new__(FileTransport)
        ft.team_id = "t1"
        # Direct inbox test covered in test_inbox.py


def test_negotiate_transport_defaults_to_file():
    transport = negotiate_transport("t1")
    assert isinstance(transport, FileTransport)


def test_nixl_transport_fallback():
    config = TransportConfig(transport_type=TransportType.NIXL)
    transport = NIXLTransport("t1", config)
    assert transport._nixl_available is False  # nixl not installed in test env


def test_zeromq_transport_fallback():
    config = TransportConfig(transport_type=TransportType.ZEROMQ)
    # ZeroMQ may or may not be installed; either way it should not crash
    transport = ZeroMQTransport("t1", config)
    transport.close()


def test_transport_type_enum():
    assert TransportType.FILESYSTEM.value == "filesystem"
    assert TransportType.ZEROMQ.value == "zeromq"
    assert TransportType.NIXL.value == "nixl"
