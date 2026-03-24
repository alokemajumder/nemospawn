"""Tests for multi-agent CLI adapter registry and extended profiles."""

from unittest.mock import patch

from nemospawn.core.profiles import (
    AgentProfile, DEFAULT_PROFILES, AGENT_ADAPTERS,
    get_adapter, build_spawn_command, load_profile, save_profile,
    list_profiles, remove_profile, check_profile,
)


def test_default_profiles_expanded():
    """All expected agent types have default profiles."""
    expected = {"claude", "codex", "opencode", "copilot", "kimi", "cursor", "nanobot", "aider"}
    assert expected == set(DEFAULT_PROFILES.keys())


def test_agent_adapters_exist():
    """All default profiles have matching adapters."""
    for agent_type in DEFAULT_PROFILES:
        adapter = get_adapter(agent_type)
        assert "prompt_method" in adapter
        assert "spawn_args" in adapter


def test_get_adapter_custom_fallback():
    """Unknown agent types fall back to custom adapter."""
    adapter = get_adapter("unknown-agent")
    assert adapter["prompt_method"] == "file"
    assert adapter["spawn_args"] == []


def test_build_spawn_command_claude():
    profile = AgentProfile(name="test", agent="claude", command="claude")
    cmd = build_spawn_command(profile, task="Train model")
    assert cmd[0] == "claude"
    assert "--print" in cmd
    assert "--prompt" in cmd
    assert "Train model" in cmd


def test_build_spawn_command_codex():
    profile = AgentProfile(name="test", agent="codex", command="codex")
    cmd = build_spawn_command(profile, task="Deploy")
    assert cmd[0] == "codex"
    assert "--full-auto" in cmd


def test_build_spawn_command_with_model():
    profile = AgentProfile(name="test", agent="claude", command="claude", model="opus")
    cmd = build_spawn_command(profile, task="Train")
    assert "--model" in cmd
    assert "opus" in cmd


def test_build_spawn_command_kimi():
    profile = AgentProfile(name="test", agent="kimi", command="kimi")
    cmd = build_spawn_command(profile, task="Research")
    assert cmd[0] == "kimi"
    assert "--prompt" in cmd


def test_build_spawn_command_custom_no_prompt_flag():
    profile = AgentProfile(name="test", agent="custom", command="my-tool")
    cmd = build_spawn_command(profile, task="Do stuff")
    assert cmd[0] == "my-tool"
    # Custom agents use file-based prompt injection, no flag
    assert "--prompt" not in cmd


def test_build_spawn_command_with_extra_args():
    profile = AgentProfile(name="test", agent="claude", command="claude", args=["--verbose"])
    cmd = build_spawn_command(profile)
    assert "--verbose" in cmd


def test_save_and_load_profile(state_dir):
    with patch("nemospawn.core.profiles.PROFILES_DIR", state_dir / "profiles"):
        profile = AgentProfile(
            name="my-custom",
            agent="kimi",
            command="kimi",
            model="moonshot-v1",
            auth_env="MOONSHOT_API_KEY",
            description="Custom Kimi profile",
        )
        save_profile(profile)

        loaded = load_profile("my-custom")
        assert loaded is not None
        assert loaded.agent == "kimi"
        assert loaded.model == "moonshot-v1"


def test_list_profiles_includes_all_defaults():
    profiles = list_profiles()
    names = {p.name for p in profiles}
    assert "claude" in names
    assert "kimi" in names
    assert "nanobot" in names
    assert "aider" in names
