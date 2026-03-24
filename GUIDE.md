# NemoSpawn — How It Works & Getting Started

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Getting Started — Your First Team](#getting-started--your-first-team)
5. [Agent Spawning](#agent-spawning)
6. [Task Management](#task-management)
7. [Inter-Agent Messaging](#inter-agent-messaging)
8. [Plan Approval Workflow](#plan-approval-workflow)
9. [Agent Lifecycle Protocol](#agent-lifecycle-protocol)
10. [GPU Management](#gpu-management)
11. [NeMo Integration](#nemo-integration)
12. [NIM Deployment Pipeline](#nim-deployment-pipeline)
13. [Observability & Dashboards](#observability--dashboards)
14. [Templates & Launch](#templates--launch)
15. [HPO (Hyperparameter Optimization)](#hpo-hyperparameter-optimization)
16. [Configuration System](#configuration-system)
17. [Agent Profiles](#agent-profiles)
18. [Adaptive Scheduling](#adaptive-scheduling)
19. [Cost Tracking](#cost-tracking)
20. [Team Snapshots](#team-snapshots)
21. [Agent Health Monitoring](#agent-health-monitoring)
22. [Cross-Cluster Federation](#cross-cluster-federation)
23. [Authentication & Security](#authentication--security)
24. [SLURM Integration](#slurm-integration)
25. [Agent Skill Installation](#agent-skill-installation)
26. [State Architecture](#state-architecture)

---

## Architecture Overview

NemoSpawn is a filesystem-native orchestration system. All state lives in `~/.nemospawn/` as atomic JSON files — no database, no server, no cloud dependency.

```
~/.nemospawn/
├── teams/
│   └── {team_id}/
│       ├── team.json           # Team config, GPU list, topology
│       ├── agents/{id}.json    # Agent state (status, GPUs, tmux session)
│       ├── tasks/{id}.json     # Task DAG (status, deps, metadata)
│       ├── plans/{id}.json     # Plan approval state
│       ├── inbox/{agent_id}/   # Per-agent message files
│       ├── artifacts/{id}.json # NeMo checkpoints, NIM containers
│       ├── prompts/{id}.md     # Auto-injected coordination prompts
│       ├── snapshots/{id}.json # Team state snapshots
│       ├── costs/              # GPU-hour cost records
│       ├── workspaces/         # Git worktrees per agent
│       └── metrics/            # DCGM snapshots
├── clusters/                   # Federated cluster configs
├── profiles/                   # Custom agent profiles
├── hpo/                        # HPO study databases
├── config.json                 # Dynamic configuration
└── audit.jsonl                 # Structured audit log
```

Every write uses the atomic `tmpfile + os.replace` pattern — crash-safe on POSIX systems.

Agents spawn in **tmux sessions** (default) or **OpenShell sandboxes** (kernel-level isolation). Each agent gets its own `CUDA_VISIBLE_DEVICES` environment, optional git worktree, and an auto-injected coordination prompt that teaches it the NemoSpawn CLI protocol.

---

## Installation

```bash
# Basic install
pip install nemospawn

# With hyperparameter optimization
pip install nemospawn[hpo]

# With ZeroMQ cross-node messaging
pip install nemospawn[transport]

# Everything
pip install nemospawn[all]

# Development
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

### Prerequisites
- Python >= 3.10
- tmux (for agent isolation)
- NVIDIA GPU drivers (for GPU features; CPU-only dev mode works without GPUs)

---

## Core Concepts

| Concept | What it is |
|---------|-----------|
| **Team** | A group of agents sharing GPUs, tasks, and a message inbox |
| **Agent** | A GPU-pinned worker running in a tmux session or OpenShell sandbox |
| **Task** | A unit of work with status, dependencies, and metadata |
| **Plan** | A proposal submitted by an agent for leader approval |
| **Profile** | Configuration for a specific CLI agent (Claude, Codex, Kimi, etc.) |
| **Template** | A TOML file defining a team structure for one-command launch |
| **Artifact** | A registered output (checkpoint, model, dataset) with metadata |

---

## Getting Started — Your First Team

### Step 1: Check GPUs

```bash
nemospawn gpu discover
nemospawn gpu topology
```

### Step 2: Create a team

```bash
nemospawn team create my-experiment --gpus 0,1,2,3 -d "LLM fine-tuning sweep"
```

This discovers your GPU topology, identifies NVLink islands, and creates the team directory with all subdirectories.

### Step 3: Spawn agents

```bash
# Spawn a trainer on GPU 0
nemospawn spawn agent --team my-experiment-abc \
  --agent-name trainer0 --role trainer --gpu 0 \
  --task "Fine-tune LLaMA with lr=2e-4" --agent-cmd claude

# Spawn an evaluator on GPU 1
nemospawn spawn agent --team my-experiment-abc \
  --agent-name evaluator --role evaluator --gpu 1 \
  --task "Benchmark best checkpoint via Triton"
```

Each agent gets:
- A tmux session (`nemo-{team_id}-{agent_id}`)
- `CUDA_VISIBLE_DEVICES` set to its assigned GPUs
- `NEMOSPAWN_TEAM` and `NEMOSPAWN_AGENT` env vars
- A coordination prompt file at `prompts/{agent_id}.md`

### Step 4: Create tasks with dependencies

```bash
nemospawn task create my-experiment-abc "Train model" --owner trainer0
nemospawn task create my-experiment-abc "Evaluate" --owner evaluator --blocked-by task-abc123
```

### Step 5: Monitor

```bash
# Terminal kanban
nemospawn board live my-experiment-abc

# Web UI (browser at http://localhost:8080)
nemospawn board serve my-experiment-abc

# Tiled tmux view
nemospawn board attach my-experiment-abc
```

### Step 6: Check status

```bash
nemospawn team status my-experiment-abc
nemospawn spawn list --team my-experiment-abc
nemospawn task list my-experiment-abc
```

---

## Agent Spawning

### Supported Agent CLIs

NemoSpawn works with any CLI agent. Built-in profiles:

| Agent | Command | Auth Env Var |
|-------|---------|-------------|
| Claude Code | `claude` | `ANTHROPIC_API_KEY` |
| Codex | `codex` | `OPENAI_API_KEY` |
| Kimi CLI | `kimi` | `MOONSHOT_API_KEY` |
| Cursor | `cursor` | — |
| nanobot | `nanobot` | — |
| aider | `aider` | `OPENAI_API_KEY` |
| OpenCode | `opencode` | — |
| GitHub Copilot | `github-copilot-cli` | `GITHUB_TOKEN` |

### Using profiles

```bash
# Spawn with a named profile
nemospawn spawn agent --team t1 --agent-name worker --profile my-kimi-profile --gpu 0

# Profile overrides --agent-cmd
nemospawn spawn agent --team t1 --agent-name worker --agent-cmd kimi --gpu 0
```

### Runtimes

```bash
# tmux (default) — interactive, attach to watch
nemospawn spawn agent --team t1 --agent-name worker --runtime tmux

# OpenShell sandbox — kernel-level isolation
nemospawn spawn agent --team t1 --agent-name worker --runtime sandbox
```

### Git worktree isolation

```bash
# Each agent gets its own branch: nemospawn/{team}/{agent}
nemospawn spawn agent --team t1 --agent-name worker --repo /path/to/repo
```

### Kill an agent

```bash
nemospawn spawn kill --team t1 --agent worker-abc123
```

---

## Task Management

```bash
# Create task with dependencies
nemospawn task create t1 "Deploy NIM" --owner deployer --blocked-by train-task-id

# Update status (auto-unblocks dependents when done)
nemospawn task update t1 task-abc --status running
nemospawn task update t1 task-abc --status done --val-loss 0.042

# List tasks
nemospawn task list t1 --status running

# Wait for completion
nemospawn task wait t1 task-abc --timeout 3600
```

Tasks flow through: `pending` -> `blocked` -> `running` -> `done` / `failed`

When a task is marked `done`, all tasks that depend on it via `blocked_by` are automatically unblocked.

---

## Inter-Agent Messaging

```bash
# Direct message
nemospawn inbox send t1 worker0 "Best lr=2e-4, val_loss=0.042" --from leader

# Broadcast to all agents
nemospawn inbox broadcast t1 "Stopping all training — deploying best checkpoint" --from leader

# Check inbox
nemospawn inbox receive t1 worker0
```

Messages are stored as atomic JSON files in `inbox/{agent_id}/` directories. Agents check their inbox using the commands above.

---

## Plan Approval Workflow

Agents can submit plans for leader review before executing major actions.

```bash
# Agent submits a plan
nemospawn plan submit --team t1 --agent worker0 \
  --title "Fine-tune with LoRA" \
  -d "Apply LoRA rank-16 on attention layers" \
  --steps "Download base model,Apply LoRA config,Train 5000 steps,Evaluate"

# Leader reviews pending plans
nemospawn plan list --team t1 --status pending

# Approve
nemospawn plan approve --team t1 --plan plan-abc --reviewer leader --comment "LGTM"

# Or reject
nemospawn plan reject --team t1 --plan plan-abc --reviewer leader --comment "Use rank-32 instead"

# Agent checks plan status
nemospawn plan show --team t1 --plan plan-abc
```

---

## Agent Lifecycle Protocol

### Idle reporting

When an agent finishes all its work:

```bash
nemospawn lifecycle idle --team t1 --agent worker0 --reason "All tasks completed"
nemospawn lifecycle idle-list --team t1
```

### Graceful shutdown

```bash
# Request shutdown
nemospawn lifecycle shutdown-request --team t1 --agent worker0 --by leader --reason "No more work"

# Approve (agent stops)
nemospawn lifecycle shutdown-approve --team t1 --agent worker0 --by leader

# Or reject (agent continues)
nemospawn lifecycle shutdown-reject --team t1 --agent worker0 --by leader --reason "New tasks incoming"

# Check lifecycle state of all agents
nemospawn lifecycle status --team t1
```

---

## GPU Management

```bash
# Discover GPUs
nemospawn gpu discover

# Show NVLink topology and islands
nemospawn gpu topology

# GPU health (temp, utilization, power, ECC errors)
nemospawn gpu health

# DCGM metrics for a team
nemospawn gpu status my-team

# Kill underperforming agents
nemospawn gpu reallocate my-team --kill-below 50
```

---

## NeMo Integration

### Artifacts

```bash
# Register a checkpoint
nemospawn artifact register t1 /path/to/model.nemo --type nemo-checkpoint --val-loss 0.042

# Promote best checkpoint
nemospawn artifact promote t1 artifact-abc

# List artifacts sorted by val_loss
nemospawn artifact list t1 --sort val_loss
```

Supported artifact types: `nemo-checkpoint`, `lora-adapter`, `nim-container`, `dataset`, `benchmark`, `reward-signal`, `config-patch`

---

## NIM Deployment Pipeline

```bash
# Deploy checkpoint as NIM container
nemospawn nim deploy t1 artifact-abc --tp 2 --profile max-throughput --port 8000 --gpus 0,1

# List running endpoints
nemospawn nim list t1

# Benchmark an endpoint
nemospawn nim benchmark t1 http://localhost:8000 --concurrency 1,4,16,64

# Show available TP profiles
nemospawn nim profiles 8
```

---

## Observability & Dashboards

### Web UI (browser-based kanban)

```bash
nemospawn board serve my-team --port 8080 --metrics-port 9090
```

Opens a dark-themed kanban board at `http://localhost:8080` with:
- Task columns (pending, blocked, running, done, failed) with live counts
- Agent cards with status dots and GPU assignments
- Plan section with approval status badges
- SSE auto-updates every 3 seconds

### Terminal kanban

```bash
nemospawn board live my-team --interval 10
```

### Prometheus + Grafana

```bash
nemospawn board serve my-team --grafana-url http://grafana:3000 --grafana-key <key>
```

Metrics exported: `nemospawn_gpu_sm_utilization`, `nemospawn_gpu_temperature_celsius`, `nemospawn_agents_total`, `nemospawn_tasks_total`, `nemospawn_val_loss`

---

## Templates & Launch

### One-command launch

```bash
# Launch a full team from a built-in template
nemospawn launch run autoresearch --gpus 0,1,2,3
nemospawn launch run nim-deploy --gpus 0,1,2 --goal "Deploy LLaMA-70B"
nemospawn launch run rlhf-swarm --gpus 0,1,2,3
nemospawn launch run data-curation --gpus 0,1

# Preview without executing
nemospawn launch run autoresearch --gpus 0,1,2,3 --dry-run

# List available templates
nemospawn launch templates
```

### Custom templates

Create a TOML file:

```toml
name = "my-pipeline"
description = "Custom training pipeline"
min_gpus = 2

[[workers]]
name = "data-prep"
role = "data-curator"
gpu_count = 1
task = "Download and preprocess dataset"

[[workers]]
name = "trainer"
role = "trainer"
gpu_count = 1
task = "Train model on preprocessed data"
blocked_by = ["data-prep"]
```

Launch it:

```bash
nemospawn launch run /path/to/my-pipeline.toml --gpus 0,1
```

---

## HPO (Hyperparameter Optimization)

```bash
# Initialize study with search space
nemospawn hpo init --study lr-sweep --template autoresearch

# Get next hyperparameter suggestion
nemospawn hpo suggest --study lr-sweep

# Report trial result
nemospawn hpo report --study lr-sweep --trial trial-abc --step 5000 --val-loss 0.042

# Show best trial
nemospawn hpo best --study lr-sweep

# List all trials
nemospawn hpo trials --study lr-sweep

# Launch Optuna dashboard
nemospawn hpo dashboard --study lr-sweep --port 8081
```

---

## Configuration System

NemoSpawn uses a three-tier config priority: **env var > config file > default**.

```bash
# Show all settings with their sources
nemospawn config show

# Get a specific setting
nemospawn config get transport

# Set a value (persisted to ~/.nemospawn/config.json)
nemospawn config set transport zeromq
nemospawn config set default_profile kimi
nemospawn config set cost_rate 3.50

# Health check
nemospawn config health
```

### Available settings

| Key | Default | Env Var | Description |
|-----|---------|---------|-------------|
| `data_dir` | `~/.nemospawn` | `NEMOSPAWN_DATA_DIR` | State directory |
| `transport` | `file` | `NEMOSPAWN_TRANSPORT` | Messaging backend (file/zeromq/nixl) |
| `workspace` | `auto` | `NEMOSPAWN_WORKSPACE` | Git worktree mode |
| `default_profile` | `claude` | `NEMOSPAWN_DEFAULT_PROFILE` | Default agent CLI |
| `default_runtime` | `tmux` | `NEMOSPAWN_DEFAULT_RUNTIME` | Default spawn mode |
| `cost_rate` | `2.50` | `NEMOSPAWN_COST_RATE` | USD per GPU-hour |
| `watch_interval` | `60` | `NEMOSPAWN_WATCH_INTERVAL` | Health check interval (s) |
| `web_port` | `8080` | `NEMOSPAWN_WEB_PORT` | Web UI port |
| `metrics_port` | `9090` | `NEMOSPAWN_METRICS_PORT` | Prometheus port |
| `user` | (empty) | `NEMOSPAWN_USER` | Multi-user identity |

---

## Agent Profiles

Profiles define how a specific CLI agent is invoked.

```bash
# List all profiles (built-in + custom)
nemospawn profile list

# Show profile details
nemospawn profile show claude

# Create a custom profile
nemospawn profile create --name my-kimi \
  --agent kimi --command kimi --model moonshot-v1 \
  --auth-env MOONSHOT_API_KEY -d "Kimi with Moonshot v1"

# Interactive wizard
nemospawn profile wizard

# Diagnose issues
nemospawn profile doctor claude

# Smoke test
nemospawn profile test my-kimi

# Delete
nemospawn profile delete my-kimi
```

---

## Adaptive Scheduling

NemoSpawn can monitor GPU utilization and automatically reassign tasks from underperforming agents.

```bash
# Analyze agent performance (ranked by GPU util + task completion)
nemospawn schedule analyze --team t1

# Suggest reassignments
nemospawn schedule suggest --team t1 --threshold 30

# Apply a specific reassignment
nemospawn schedule apply --team t1 --task task-abc --to agent-xyz

# Continuous auto-reassignment
nemospawn schedule auto --team t1 --threshold 30 --interval 300
```

---

## Cost Tracking

```bash
# Show GPU-hour costs per agent
nemospawn cost show --team t1

# Set the GPU-hour rate
nemospawn cost set-rate --team t1 --rate 3.50

# Reset tracking
nemospawn cost reset --team t1
```

Costs are calculated from agent creation time, GPU count, and elapsed time. Running agents use current time; stopped agents use their last update time.

---

## Team Snapshots

Save and restore team state at any point.

```bash
# Save a snapshot
nemospawn snapshot save --team t1 --label "before-deploy"

# List snapshots
nemospawn snapshot list --team t1

# Restore from snapshot (reverts agents, tasks, plans, costs)
nemospawn snapshot restore --team t1 --snapshot snap-abc

# Delete a snapshot
nemospawn snapshot delete --team t1 --snapshot snap-abc
```

---

## Agent Health Monitoring

```bash
# Single health check
nemospawn watch status --team t1

# Continuous monitoring (checks every 60s)
nemospawn watch start --team t1 --interval 60
```

The watcher:
- Verifies tmux sessions are still alive for running agents
- Marks dead agents as `stopped` with lifecycle state `dead`
- Detects stuck agents (no lifecycle update for 24+ hours)

---

## Cross-Cluster Federation

```bash
# Register a remote cluster
nemospawn cluster register dgx-2 --host dgx2.internal --key ~/.ssh/dgx2 --user admin

# List clusters
nemospawn cluster list

# Check connectivity
nemospawn cluster status dgx-2
```

Agents can be spawned on remote clusters via SSH:

```bash
nemospawn spawn agent --team t1 --agent-name remote-trainer --remote admin@dgx2.internal --gpu 0
```

Artifacts transfer between clusters via git-annex.

---

## Authentication & Security

```bash
# Create a user with API key
nemospawn auth create-user researcher1 --role user

# Verify an API key
nemospawn auth verify <api-key>

# View audit log
nemospawn auth audit --last 50 --event agent.spawn --team t1
```

All operations are logged to `~/.nemospawn/audit.jsonl` with timestamps, user identity, and event details.

---

## SLURM Integration

```bash
# Generate sbatch script
nemospawn slurm generate my-job --gpus 4 --gpu-type a100 --nodes 2 --time 48:00:00 --command "nemospawn launch run autoresearch"

# Submit to SLURM
nemospawn slurm submit my-job.sbatch

# Check status
nemospawn slurm status 12345

# Cancel
nemospawn slurm cancel 12345
```

---

## Agent Skill Installation

Install the NemoSpawn coordination protocol as a reusable skill for Claude Code or Codex:

```bash
# Install for Claude Code
nemospawn skill install --target claude

# Install for both Claude Code and Codex
nemospawn skill install --target all

# Check status
nemospawn skill status

# Uninstall
nemospawn skill uninstall --target all
```

Once installed, agents automatically discover NemoSpawn commands and the coordination protocol when spawned.

---

## State Architecture

### How writes work

Every state mutation uses atomic writes:

```python
# 1. Write to temp file in same directory
fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
json.dump(data, f)

# 2. Atomic rename (POSIX guarantees this is crash-safe)
os.replace(tmp, target_path)
```

If the process crashes mid-write, the original file is untouched.

### How agents coordinate

1. Agent spawns with `NEMOSPAWN_TEAM` and `NEMOSPAWN_AGENT` env vars
2. A coordination prompt file is written to `prompts/{agent_id}.md`
3. The prompt teaches the agent all NemoSpawn CLI commands
4. Agents read tasks, update status, send messages, submit plans — all via CLI
5. The leader monitors progress via `board live` or `board serve`
6. When done, agents report `idle` via `lifecycle idle`

### Transport negotiation

For messaging, NemoSpawn selects the best available transport:

1. **NIXL** — if agents are on the same node with NVLink (sub-microsecond)
2. **ZeroMQ** — if agents are on different nodes (TCP)
3. **File** — always works as fallback (atomic JSON in inbox dirs)
