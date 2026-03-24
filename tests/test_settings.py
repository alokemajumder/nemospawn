"""Tests for dynamic configuration system."""

import os
from unittest.mock import patch

from nemospawn.core.settings import get, set_value, get_all, health_check, DEFAULTS


def test_get_default_value():
    """Getting a key with no env or file returns the default."""
    with patch("nemospawn.core.settings._load_file_config", return_value={}):
        assert get("transport") == "file"
        assert get("default_profile") == "claude"
        assert get("default_runtime") == "tmux"


def test_get_from_env():
    """Environment variables take highest priority."""
    with patch("nemospawn.core.settings._load_file_config", return_value={"transport": "zeromq"}), \
         patch.dict(os.environ, {"NEMOSPAWN_TRANSPORT": "nixl"}):
        assert get("transport") == "nixl"


def test_get_from_file():
    """File config overrides default."""
    with patch("nemospawn.core.settings._load_file_config", return_value={"transport": "zeromq"}):
        # Make sure no env var interferes
        env = dict(os.environ)
        env.pop("NEMOSPAWN_TRANSPORT", None)
        with patch.dict(os.environ, env, clear=True):
            assert get("transport") == "zeromq"


def test_set_value_valid_key(state_dir):
    with patch("nemospawn.core.settings.STATE_DIR", state_dir), \
         patch("nemospawn.core.settings.CONFIG_FILE", state_dir / "config.json"):
        assert set_value("transport", "zeromq") is True
        # Verify it persists
        assert get("transport") == "zeromq"


def test_set_value_invalid_key():
    assert set_value("nonexistent_key", "value") is False


def test_get_unknown_key():
    assert get("nonexistent_key") == ""


def test_get_all():
    with patch("nemospawn.core.settings._load_file_config", return_value={}):
        all_settings = get_all()
        assert "transport" in all_settings
        assert "default_profile" in all_settings
        assert all_settings["transport"]["source"] == "default"
        assert all_settings["transport"]["value"] == "file"


def test_get_all_shows_env_source():
    with patch("nemospawn.core.settings._load_file_config", return_value={}), \
         patch.dict(os.environ, {"NEMOSPAWN_TRANSPORT": "nixl"}):
        all_settings = get_all()
        assert "env" in all_settings["transport"]["source"]
        assert all_settings["transport"]["value"] == "nixl"


def test_health_check(state_dir):
    with patch("nemospawn.core.settings.STATE_DIR", state_dir), \
         patch("nemospawn.core.settings.CONFIG_FILE", state_dir / "config.json"):
        checks = health_check()
        assert len(checks) >= 3
        # State dir should be writable
        writable = [c for c in checks if c["check"] == "state_dir_writable"]
        assert writable[0]["ok"] is True


def test_defaults_have_descriptions():
    for key, spec in DEFAULTS.items():
        assert "description" in spec, f"Missing description for {key}"
        assert "value" in spec, f"Missing default value for {key}"
        assert "env_var" in spec, f"Missing env_var for {key}"
