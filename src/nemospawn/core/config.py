"""Global configuration constants for NemoSpawn."""

from pathlib import Path

STATE_DIR = Path.home() / ".nemospawn"
TEAMS_DIR = STATE_DIR / "teams"
HPO_DIR = STATE_DIR / "hpo"

# Subdirectory names within each team directory
AGENTS_SUBDIR = "agents"
TASKS_SUBDIR = "tasks"
INBOX_SUBDIR = "inbox"
ARTIFACTS_SUBDIR = "artifacts"
WORKSPACES_SUBDIR = "workspaces"
METRICS_SUBDIR = "metrics"
PLANS_SUBDIR = "plans"
SNAPSHOTS_SUBDIR = "snapshots"
COSTS_SUBDIR = "costs"

TEAM_SUBDIRS = [
    AGENTS_SUBDIR,
    TASKS_SUBDIR,
    INBOX_SUBDIR,
    ARTIFACTS_SUBDIR,
    WORKSPACES_SUBDIR,
    METRICS_SUBDIR,
    PLANS_SUBDIR,
    SNAPSHOTS_SUBDIR,
    COSTS_SUBDIR,
]

# tmux session prefix
TMUX_PREFIX = "nemo"

# Polling defaults
TASK_POLL_INTERVAL_S = 2
LEADER_POLL_INTERVAL_S = 1800  # 30 minutes
