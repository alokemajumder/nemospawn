"""Multi-transport messaging — NIXL, ZeroMQ, and filesystem.

Transport selection follows the federation matrix:
  Same node + NVLink:     NIXL (sub-microsecond)
  Same cluster + IB:      NIXL or ZeroMQ
  Cross-cluster + NFS:    Filesystem (JSON)
  Cross-cluster + WAN:    ZeroMQ P2P

Transport negotiation happens at agent spawn time based on topology.
File-based inbox is always the fallback.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nemospawn.core.models import Message


class TransportType(Enum):
    FILESYSTEM = "filesystem"
    ZEROMQ = "zeromq"
    NIXL = "nixl"


@dataclass
class TransportConfig:
    """Configuration for a messaging transport."""
    transport_type: TransportType = TransportType.FILESYSTEM
    # ZeroMQ settings
    zmq_bind_address: str = "tcp://*:5555"
    zmq_connect_address: str = ""  # tcp://host:5555
    # NIXL settings
    nixl_device_id: int = -1  # GPU device for NIXL
    nixl_fabric: str = "nvlink"  # nvlink, infiniband


class Transport(ABC):
    """Abstract base for message transports."""

    @abstractmethod
    def send(self, to_agent: str, message: Message) -> bool:
        ...

    @abstractmethod
    def receive(self, agent_id: str) -> list[Message]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class FileTransport(Transport):
    """Filesystem-based transport — always available, uses atomic JSON writes."""

    def __init__(self, team_id: str):
        self.team_id = team_id

    def send(self, to_agent: str, message: Message) -> bool:
        from nemospawn.messaging.inbox import send_message
        send_message(self.team_id, message.from_agent, to_agent, message.body)
        return True

    def receive(self, agent_id: str) -> list[Message]:
        from nemospawn.messaging.inbox import receive_messages
        return receive_messages(self.team_id, agent_id)

    def close(self) -> None:
        pass  # nothing to clean up


class ZeroMQTransport(Transport):
    """ZeroMQ-based transport for low-latency cross-node messaging.

    Uses DEALER/ROUTER pattern for async P2P messaging.
    Falls back to FileTransport if pyzmq is not installed.
    """

    def __init__(self, team_id: str, config: TransportConfig):
        self.team_id = team_id
        self.config = config
        self._context = None
        self._socket = None
        self._init_zmq()

    def _init_zmq(self) -> None:
        try:
            import zmq
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.DEALER)
            if self.config.zmq_connect_address:
                self._socket.connect(self.config.zmq_connect_address)
            else:
                self._socket.bind(self.config.zmq_bind_address)
            self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout
        except ImportError:
            self._context = None
            self._socket = None

    def send(self, to_agent: str, message: Message) -> bool:
        if self._socket is None:
            # Fallback to file transport
            return FileTransport(self.team_id).send(to_agent, message)
        try:
            import json
            payload = json.dumps({
                "to": to_agent,
                "msg": message.to_dict(),
            }).encode()
            self._socket.send(payload)
            return True
        except Exception:
            return FileTransport(self.team_id).send(to_agent, message)

    def receive(self, agent_id: str) -> list[Message]:
        if self._socket is None:
            return FileTransport(self.team_id).receive(agent_id)
        messages = []
        try:
            import json
            while True:
                try:
                    data = self._socket.recv(flags=1)  # NOBLOCK
                    payload = json.loads(data.decode())
                    if payload.get("to") == agent_id:
                        messages.append(Message.from_dict(payload["msg"]))
                except Exception:
                    break
        except Exception:
            pass
        # Also check file inbox as fallback
        messages.extend(FileTransport(self.team_id).receive(agent_id))
        return messages

    def close(self) -> None:
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()


class NIXLTransport(Transport):
    """NIXL transport for sub-microsecond messaging over NVLink/InfiniBand.

    NIXL (NVIDIA Interconnect Exchange Layer) provides the fastest inter-agent
    communication when agents are co-located on the same DGX/HGX node or
    connected via InfiniBand.

    Falls back to FileTransport when NIXL is not available.
    """

    def __init__(self, team_id: str, config: TransportConfig):
        self.team_id = team_id
        self.config = config
        self._nixl_available = self._check_nixl()

    def _check_nixl(self) -> bool:
        """Check if NIXL Python bindings are available."""
        try:
            import nixl  # noqa: F401
            return True
        except ImportError:
            return False

    def send(self, to_agent: str, message: Message) -> bool:
        if not self._nixl_available:
            return FileTransport(self.team_id).send(to_agent, message)
        # NIXL send implementation would use nixl.send() for GPU-direct messaging
        # For now, falls back to file transport with a marker
        return FileTransport(self.team_id).send(to_agent, message)

    def receive(self, agent_id: str) -> list[Message]:
        if not self._nixl_available:
            return FileTransport(self.team_id).receive(agent_id)
        return FileTransport(self.team_id).receive(agent_id)

    def close(self) -> None:
        pass


def negotiate_transport(
    team_id: str,
    gpu_ids: list[int] | None = None,
    remote: str | None = None,
    preferred: TransportType | None = None,
) -> Transport:
    """Negotiate the best transport based on topology and availability.

    Selection order:
    1. If preferred is specified and available, use it
    2. Same node + NVLink: try NIXL
    3. Cross-node or fallback: try ZeroMQ
    4. Always: filesystem fallback

    Returns:
        The best available Transport instance.
    """
    config = TransportConfig()

    if preferred == TransportType.NIXL or (gpu_ids and not remote):
        config.transport_type = TransportType.NIXL
        config.nixl_device_id = gpu_ids[0] if gpu_ids else 0
        transport = NIXLTransport(team_id, config)
        if transport._nixl_available:
            return transport

    if preferred == TransportType.ZEROMQ or remote:
        config.transport_type = TransportType.ZEROMQ
        if remote:
            config.zmq_connect_address = f"tcp://{remote}:5555"
        transport = ZeroMQTransport(team_id, config)
        if transport._socket is not None:
            return transport

    # Always-available fallback
    return FileTransport(team_id)
