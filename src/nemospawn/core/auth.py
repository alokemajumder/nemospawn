"""Authentication and authorization for multi-user NemoSpawn.

Provides API key-based auth for the board serve endpoint and
namespace isolation for multi-user environments.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from nemospawn.core.config import STATE_DIR
from nemospawn.core.state import atomic_read, atomic_write

AUTH_DIR = STATE_DIR / "auth"


@dataclass
class User:
    """A NemoSpawn user with namespace isolation."""
    username: str
    api_key_hash: str = ""
    namespace: str = ""  # defaults to username
    teams: list[str] = field(default_factory=list)
    role: str = "user"  # user | admin

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> User:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def generate_api_key() -> tuple[str, str]:
    """Generate an API key and its hash.

    Returns (plaintext_key, hash) — plaintext is shown once, hash is stored.
    """
    key = f"ns_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


def create_user(username: str, role: str = "user") -> tuple[User, str]:
    """Create a new user with an API key.

    Returns (User, plaintext_api_key).
    """
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    key, key_hash = generate_api_key()
    user = User(
        username=username,
        api_key_hash=key_hash,
        namespace=username,
        role=role,
    )
    atomic_write(AUTH_DIR / f"{username}.json", user.to_dict())
    return user, key


def authenticate(api_key: str) -> User | None:
    """Authenticate a user by API key."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    for f in AUTH_DIR.glob("*.json"):
        data = atomic_read(f)
        if data and data.get("api_key_hash") == key_hash:
            return User.from_dict(data)
    return None


def authorize_team_access(user: User, team_id: str) -> bool:
    """Check if a user can access a team."""
    if user.role == "admin":
        return True
    return team_id in user.teams or team_id.startswith(f"{user.namespace}-")
