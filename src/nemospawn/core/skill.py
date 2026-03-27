"""Reusable agent skill — package NemoSpawn as a Claude Code / Codex skill.

Installs a skill entry at ~/.claude/skills/nemospawn/ that teaches spawned
agents the full NemoSpawn coordination protocol.
"""

from __future__ import annotations

from pathlib import Path

SKILL_DIR_CLAUDE = Path.home() / ".claude" / "skills" / "nemospawn"
SKILL_DIR_CODEX = Path.home() / ".codex" / "skills" / "nemospawn"

SKILL_CONTENT = """\
# NemoSpawn Agent Skill

You are an agent managed by NemoSpawn — a GPU-native agent-swarm orchestration system
for the NVIDIA AI stack (NeMo, NIM, Triton, DCGM, NGC, NIXL).

## Environment

- `NEMOSPAWN_TEAM` — your team ID
- `NEMOSPAWN_AGENT` — your agent ID
- `CUDA_VISIBLE_DEVICES` — your assigned GPUs
- `NEMOSPAWN_PROMPT` — path to your full coordination prompt file

## All Commands

### Tasks
```bash
nemospawn task list $NEMOSPAWN_TEAM                                    # see all tasks
nemospawn task create $NEMOSPAWN_TEAM "title" --owner <agent>          # create task
nemospawn task update $NEMOSPAWN_TEAM <task_id> --status running       # start working
nemospawn task update $NEMOSPAWN_TEAM <task_id> --status done --val-loss <v>  # finish
```

### Messaging
```bash
nemospawn inbox receive $NEMOSPAWN_TEAM $NEMOSPAWN_AGENT               # check inbox
nemospawn inbox send $NEMOSPAWN_TEAM <to_agent> '<message>'            # direct message
nemospawn inbox broadcast $NEMOSPAWN_TEAM '<message>' --from $NEMOSPAWN_AGENT  # broadcast
```

### Spawning Agents (leader or when delegating work)
```bash
nemospawn spawn agent --team $NEMOSPAWN_TEAM --agent-name <name> \\
  --role <role> --gpu <gpu_ids> --task '<task description>'
nemospawn spawn list --team $NEMOSPAWN_TEAM                            # list agents
nemospawn spawn kill --team $NEMOSPAWN_TEAM --agent <agent_id>         # kill agent
```

### Plan Approval
```bash
nemospawn plan submit --team $NEMOSPAWN_TEAM --agent $NEMOSPAWN_AGENT \\
  --title '<title>' -d '<description>' --steps 'step1,step2,step3'
nemospawn plan list --team $NEMOSPAWN_TEAM --status pending            # review plans
nemospawn plan approve --team $NEMOSPAWN_TEAM --plan <id> --reviewer $NEMOSPAWN_AGENT
nemospawn plan reject --team $NEMOSPAWN_TEAM --plan <id> --reviewer $NEMOSPAWN_AGENT --comment '<reason>'
```

### Lifecycle
```bash
nemospawn lifecycle idle --team $NEMOSPAWN_TEAM --agent $NEMOSPAWN_AGENT --reason '<reason>'
nemospawn lifecycle shutdown-request --team $NEMOSPAWN_TEAM --agent $NEMOSPAWN_AGENT
nemospawn lifecycle shutdown-approve --team $NEMOSPAWN_TEAM --agent <agent_id>  # leader
nemospawn lifecycle idle-list --team $NEMOSPAWN_TEAM                    # see idle agents
```

### Monitoring
```bash
nemospawn watch status --team $NEMOSPAWN_TEAM                          # health check
nemospawn schedule analyze --team $NEMOSPAWN_TEAM                      # performance
nemospawn cost show --team $NEMOSPAWN_TEAM                             # GPU costs
```

### Artifacts
```bash
nemospawn artifact register $NEMOSPAWN_TEAM <path> --type nemo-checkpoint --val-loss <v>
nemospawn artifact list $NEMOSPAWN_TEAM --sort val_loss
```

### Workspace
```bash
nemospawn workspace checkpoint --team $NEMOSPAWN_TEAM --agent $NEMOSPAWN_AGENT  # save work
nemospawn workspace merge --team $NEMOSPAWN_TEAM --agent $NEMOSPAWN_AGENT       # merge to main
```

## Working Principles

1. Check your tasks: `nemospawn task list $NEMOSPAWN_TEAM`
2. Mark tasks as `running` before starting work
3. Use ONLY GPUs in `CUDA_VISIBLE_DEVICES`
4. Submit a plan before major experiments
5. Report `val_loss` at every checkpoint
6. Send findings via `nemospawn inbox send`
7. If you are leader: spawn workers, review plans, kill idle agents, respawn with new params
8. Report `idle` when done
9. On failure, set task status to `failed` with error details
"""


def install_skill(targets: list[str] | None = None) -> list[Path]:
    """Install the NemoSpawn skill for agent CLIs.

    Args:
        targets: List of target agents ("claude", "codex", or both).
                 Defaults to ["claude"].

    Returns:
        List of paths where the skill was installed.
    """
    if targets is None:
        targets = ["claude"]

    installed = []

    target_dirs = {
        "claude": SKILL_DIR_CLAUDE,
        "codex": SKILL_DIR_CODEX,
    }

    for target in targets:
        skill_dir = target_dirs.get(target)
        if not skill_dir:
            continue

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "skill.md"
        skill_file.write_text(SKILL_CONTENT)
        installed.append(skill_file)

    return installed


def uninstall_skill(targets: list[str] | None = None) -> list[Path]:
    """Uninstall the NemoSpawn skill.

    Args:
        targets: List of target agents to uninstall from. Defaults to ["claude"].

    Returns:
        List of paths that were removed.
    """
    if targets is None:
        targets = ["claude"]

    removed = []

    target_dirs = {
        "claude": SKILL_DIR_CLAUDE,
        "codex": SKILL_DIR_CODEX,
    }

    for target in targets:
        skill_dir = target_dirs.get(target)
        if not skill_dir or not skill_dir.exists():
            continue

        skill_file = skill_dir / "skill.md"
        if skill_file.exists():
            skill_file.unlink()
            removed.append(skill_file)

        # Remove empty directory
        try:
            skill_dir.rmdir()
        except OSError:
            pass

    return removed


def is_installed(target: str = "claude") -> bool:
    """Check if the skill is installed for a given target."""
    target_dirs = {
        "claude": SKILL_DIR_CLAUDE,
        "codex": SKILL_DIR_CODEX,
    }
    skill_dir = target_dirs.get(target)
    if not skill_dir:
        return False
    return (skill_dir / "skill.md").exists()
