"""Atomic JSON state management for ~/.nemospawn/."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from nemospawn.core.config import STATE_DIR, TEAMS_DIR, HPO_DIR


def ensure_state_dir() -> Path:
    """Create the top-level ~/.nemospawn/ directory structure if missing."""
    for d in (STATE_DIR, TEAMS_DIR, HPO_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def ensure_team_dir(team_id: str) -> Path:
    """Create team directory with all required subdirectories."""
    from nemospawn.core.config import TEAM_SUBDIRS

    team_dir = TEAMS_DIR / team_id
    team_dir.mkdir(parents=True, exist_ok=True)
    for sub in TEAM_SUBDIRS:
        (team_dir / sub).mkdir(exist_ok=True)
    return team_dir


def get_team_dir(team_id: str) -> Path:
    """Return the team directory path (does not create it)."""
    return TEAMS_DIR / team_id


def atomic_write(path: Path, data: dict) -> None:
    """Write JSON data atomically using tmp + rename.

    Creates a temp file in the same directory, writes JSON, then atomically
    replaces the target. This is crash-safe on POSIX (rename within same FS
    is atomic).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_read(path: Path) -> dict | None:
    """Read a JSON file, returning None if it doesn't exist."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def list_json_files(directory: Path) -> list[Path]:
    """List all .json files in a directory, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))
