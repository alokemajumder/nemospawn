"""Dynamic configuration system — env > file > defaults.

Config file lives at ~/.nemospawn/config.json. Environment variables
prefixed with NEMOSPAWN_ take highest priority.
"""

from __future__ import annotations

import os
from pathlib import Path

from nemospawn.core.config import STATE_DIR
from nemospawn.core.state import atomic_read, atomic_write

CONFIG_FILE = STATE_DIR / "config.json"

# Default settings with descriptions
DEFAULTS: dict[str, dict] = {
    "data_dir": {
        "value": str(STATE_DIR),
        "description": "Root directory for all NemoSpawn state",
        "env_var": "NEMOSPAWN_DATA_DIR",
    },
    "transport": {
        "value": "file",
        "description": "Default messaging transport (file, zeromq, nixl)",
        "env_var": "NEMOSPAWN_TRANSPORT",
    },
    "workspace": {
        "value": "auto",
        "description": "Git worktree mode (auto, always, never)",
        "env_var": "NEMOSPAWN_WORKSPACE",
    },
    "default_profile": {
        "value": "claude",
        "description": "Default agent profile for spawning",
        "env_var": "NEMOSPAWN_DEFAULT_PROFILE",
    },
    "default_runtime": {
        "value": "tmux",
        "description": "Default spawn runtime (tmux, sandbox)",
        "env_var": "NEMOSPAWN_DEFAULT_RUNTIME",
    },
    "cost_rate": {
        "value": "2.50",
        "description": "Default cost per GPU-hour in USD",
        "env_var": "NEMOSPAWN_COST_RATE",
    },
    "watch_interval": {
        "value": "60",
        "description": "Agent watcher check interval in seconds",
        "env_var": "NEMOSPAWN_WATCH_INTERVAL",
    },
    "web_port": {
        "value": "8080",
        "description": "Default web UI board port",
        "env_var": "NEMOSPAWN_WEB_PORT",
    },
    "metrics_port": {
        "value": "9090",
        "description": "Default Prometheus metrics port",
        "env_var": "NEMOSPAWN_METRICS_PORT",
    },
    "user": {
        "value": "",
        "description": "Current user identity for multi-user namespacing",
        "env_var": "NEMOSPAWN_USER",
    },
}


def _load_file_config() -> dict:
    """Load config from disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return atomic_read(CONFIG_FILE) or {}


def _save_file_config(data: dict) -> None:
    """Save config to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(CONFIG_FILE, data)


def get(key: str) -> str:
    """Get a config value. Priority: env var > config file > default.

    Args:
        key: Setting name (e.g., "transport", "default_profile")

    Returns:
        The resolved config value as a string.
    """
    spec = DEFAULTS.get(key)
    if not spec:
        return ""

    # Priority 1: Environment variable
    env_var = spec.get("env_var", "")
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val is not None:
            return env_val

    # Priority 2: Config file
    file_config = _load_file_config()
    if key in file_config:
        return str(file_config[key])

    # Priority 3: Default
    return spec["value"]


def set_value(key: str, value: str) -> bool:
    """Set a config value in the config file.

    Returns True if the key is valid, False otherwise.
    """
    if key not in DEFAULTS:
        return False

    file_config = _load_file_config()
    file_config[key] = value
    _save_file_config(file_config)
    return True


def get_all() -> dict[str, dict]:
    """Get all settings with their resolved values and sources."""
    file_config = _load_file_config()
    result = {}

    for key, spec in DEFAULTS.items():
        env_var = spec.get("env_var", "")
        env_val = os.environ.get(env_var) if env_var else None

        if env_val is not None:
            source = f"env ({env_var})"
            value = env_val
        elif key in file_config:
            source = "config file"
            value = str(file_config[key])
        else:
            source = "default"
            value = spec["value"]

        result[key] = {
            "value": value,
            "source": source,
            "description": spec["description"],
            "env_var": env_var,
        }

    return result


def health_check() -> list[dict]:
    """Run config health checks."""
    checks = []

    # Check state directory exists and is writable
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        test_file = STATE_DIR / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
        checks.append({"check": "state_dir_writable", "ok": True, "detail": str(STATE_DIR)})
    except OSError as e:
        checks.append({"check": "state_dir_writable", "ok": False, "detail": str(e)})

    # Check config file valid
    data = atomic_read(CONFIG_FILE)
    if CONFIG_FILE.exists():
        if data is not None:
            checks.append({"check": "config_file_valid", "ok": True, "detail": str(CONFIG_FILE)})
        else:
            checks.append({"check": "config_file_valid", "ok": False, "detail": "Corrupted JSON"})
    else:
        checks.append({"check": "config_file_valid", "ok": True, "detail": "No config file (using defaults)"})

    # Check transport setting is valid
    transport = get("transport")
    valid_transports = ("file", "zeromq", "nixl")
    checks.append({
        "check": "transport_valid",
        "ok": transport in valid_transports,
        "detail": f"transport={transport}" + ("" if transport in valid_transports else f" (valid: {', '.join(valid_transports)})"),
    })

    # Check default profile exists
    profile_name = get("default_profile")
    from nemospawn.core.profiles import load_profile
    profile = load_profile(profile_name)
    checks.append({
        "check": "default_profile_exists",
        "ok": profile is not None,
        "detail": f"profile={profile_name}",
    })

    return checks
