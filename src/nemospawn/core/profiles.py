"""Agent runtime profiles — configure CLI invocation, model, auth, and env vars.

A profile defines how a specific agent CLI is invoked: the command to run,
the LLM model/endpoint it connects to, authentication credentials, and
extra environment variables. Profiles are stored in ~/.nemospawn/profiles/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from nemospawn.core.config import STATE_DIR
from nemospawn.core.state import atomic_read, atomic_write, list_json_files

PROFILES_DIR = STATE_DIR / "profiles"


@dataclass
class AgentProfile:
    """Runtime profile for an agent CLI."""
    name: str
    agent: str = "claude"          # agent type: claude, codex, opencode, copilot, custom
    command: str = ""              # CLI command to invoke (e.g., "claude", "codex")
    model: str = ""                # LLM model name or endpoint
    base_url: str = ""             # API base URL (for custom endpoints)
    auth_env: str = ""             # env var name holding the API key (e.g., ANTHROPIC_API_KEY)
    env: dict[str, str] = field(default_factory=dict)  # extra environment variables
    args: list[str] = field(default_factory=list)       # extra CLI arguments
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AgentProfile:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Default profiles for supported agents
DEFAULT_PROFILES: dict[str, dict] = {
    "claude": {
        "agent": "claude",
        "command": "claude",
        "auth_env": "ANTHROPIC_API_KEY",
        "description": "Claude Code CLI agent",
    },
    "codex": {
        "agent": "codex",
        "command": "codex",
        "auth_env": "OPENAI_API_KEY",
        "description": "OpenAI Codex CLI agent",
    },
    "opencode": {
        "agent": "opencode",
        "command": "opencode",
        "description": "OpenCode CLI agent",
    },
    "copilot": {
        "agent": "copilot",
        "command": "github-copilot-cli",
        "auth_env": "GITHUB_TOKEN",
        "description": "GitHub Copilot CLI agent",
    },
    "kimi": {
        "agent": "kimi",
        "command": "kimi",
        "auth_env": "MOONSHOT_API_KEY",
        "description": "Kimi CLI agent (Moonshot AI)",
    },
    "cursor": {
        "agent": "cursor",
        "command": "cursor",
        "description": "Cursor CLI agent (experimental)",
    },
    "nanobot": {
        "agent": "nanobot",
        "command": "nanobot",
        "description": "nanobot CLI agent",
    },
    "aider": {
        "agent": "aider",
        "command": "aider",
        "auth_env": "OPENAI_API_KEY",
        "description": "aider CLI agent",
    },
}


# Agent adapter registry — defines how each agent CLI is invoked
# and how the coordination prompt is injected
AGENT_ADAPTERS: dict[str, dict] = {
    "claude": {
        "spawn_args": ["--print", "--dangerously-skip-permissions"],
        "prompt_flag": "--prompt",
        "prompt_method": "flag",  # pass prompt via CLI flag
        "trust_prompt": True,     # auto-confirms trust prompt
    },
    "codex": {
        "spawn_args": ["--full-auto"],
        "prompt_flag": "--prompt",
        "prompt_method": "flag",
        "trust_prompt": True,
    },
    "opencode": {
        "spawn_args": [],
        "prompt_flag": None,
        "prompt_method": "file",   # inject prompt via NEMOSPAWN_PROMPT env var
        "trust_prompt": False,
    },
    "copilot": {
        "spawn_args": [],
        "prompt_flag": None,
        "prompt_method": "file",
        "trust_prompt": False,
    },
    "kimi": {
        "spawn_args": [],
        "prompt_flag": "--prompt",
        "prompt_method": "flag",
        "trust_prompt": True,
    },
    "cursor": {
        "spawn_args": [],
        "prompt_flag": None,
        "prompt_method": "file",
        "trust_prompt": False,
    },
    "nanobot": {
        "spawn_args": [],
        "prompt_flag": "--system-prompt",
        "prompt_method": "flag",
        "trust_prompt": False,
    },
    "aider": {
        "spawn_args": ["--yes"],
        "prompt_flag": "--message",
        "prompt_method": "flag",
        "trust_prompt": True,
    },
    "custom": {
        "spawn_args": [],
        "prompt_flag": None,
        "prompt_method": "file",
        "trust_prompt": False,
    },
}


def get_adapter(agent_type: str) -> dict:
    """Get the adapter config for an agent type."""
    return AGENT_ADAPTERS.get(agent_type, AGENT_ADAPTERS["custom"])


def build_spawn_command(profile: AgentProfile, task: str = "", prompt_file: str = "") -> list[str]:
    """Build the full CLI command to spawn an agent based on its profile and adapter.

    Returns a list of command parts suitable for subprocess or tmux send-keys.
    """
    adapter = get_adapter(profile.agent)
    cmd_parts = [profile.command or profile.agent]

    # Add profile-specific extra args
    cmd_parts.extend(profile.args)

    # Add adapter default args
    cmd_parts.extend(adapter.get("spawn_args", []))

    # Add model if specified
    if profile.model:
        cmd_parts.extend(["--model", profile.model])

    # Add prompt injection
    if task and adapter.get("prompt_method") == "flag" and adapter.get("prompt_flag"):
        cmd_parts.extend([adapter["prompt_flag"], task])

    return cmd_parts


def save_profile(profile: AgentProfile) -> Path:
    """Save an agent profile."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILES_DIR / f"{profile.name}.json"
    atomic_write(path, profile.to_dict())
    return path


def load_profile(name: str) -> AgentProfile | None:
    """Load a profile by name. Falls back to defaults if not found on disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    data = atomic_read(PROFILES_DIR / f"{name}.json")
    if data:
        return AgentProfile.from_dict(data)
    # Check defaults
    if name in DEFAULT_PROFILES:
        return AgentProfile(name=name, **DEFAULT_PROFILES[name])
    return None


def list_profiles() -> list[AgentProfile]:
    """List all saved profiles plus defaults."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profiles = {}

    # Defaults first
    for name, defaults in DEFAULT_PROFILES.items():
        profiles[name] = AgentProfile(name=name, **defaults)

    # Saved profiles override defaults
    for f in list_json_files(PROFILES_DIR):
        data = atomic_read(f)
        if data and "name" in data:
            profiles[data["name"]] = AgentProfile.from_dict(data)

    return sorted(profiles.values(), key=lambda p: p.name)


def remove_profile(name: str) -> bool:
    """Remove a saved profile."""
    path = PROFILES_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def check_profile(profile: AgentProfile) -> dict:
    """Smoke-test a profile by checking if the command exists and auth is set."""
    import os
    import shutil

    result = {"profile": profile.name, "checks": {}}

    # Check command exists
    cmd = profile.command or profile.agent
    result["checks"]["command_found"] = shutil.which(cmd) is not None

    # Check auth env var
    if profile.auth_env:
        result["checks"]["auth_configured"] = bool(os.environ.get(profile.auth_env))
    else:
        result["checks"]["auth_configured"] = True  # no auth needed

    # Check base_url reachable
    if profile.base_url:
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(profile.base_url, method="HEAD")
            urllib.request.urlopen(req, timeout=5)
            result["checks"]["endpoint_reachable"] = True
        except (urllib.error.URLError, TimeoutError, OSError):
            result["checks"]["endpoint_reachable"] = False

    result["ok"] = all(result["checks"].values())
    return result
