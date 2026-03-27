# NemoSpawn User Guide

Complete reference for every NemoSpawn feature. For a quick overview, see [README.md](README.md).

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Agent Spawning](#agent-spawning)
- [Task Management](#task-management)
- [Messaging](#messaging)
- [Plan Approval](#plan-approval)
- [Lifecycle Protocol](#lifecycle-protocol)
- [GPU Management](#gpu-management)
- [NeMo Artifacts](#nemo-artifacts)
- [NIM Deployment](#nim-deployment)
- [Dashboards & Observability](#dashboards--observability)
- [Templates & Launch](#templates--launch)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Configuration](#configuration)
- [Agent Profiles](#agent-profiles)
- [Adaptive Scheduling](#adaptive-scheduling)
- [Cost Tracking](#cost-tracking)
- [Snapshots](#snapshots)
- [Health Monitoring](#health-monitoring)
- [Cross-Cluster Federation](#cross-cluster-federation)
- [Authentication & Security](#authentication--security)
- [SLURM Integration](#slurm-integration)
- [Agent Skill](#agent-skill)
- [State Architecture](#state-architecture)

---

## Architecture

NemoSpawn is filesystem-native. All state lives in `~/.nemospawn/` as atomic JSON files. No database, no server, no cloud dependency.

```
~/.nemospawn/
├── teams/
│   └── {team_id}/
│       ├── team.json             Team config, GPU list, NVLink topology
│       ├── agents/{id}.json      Agent state, GPUs, tmux session, lifecycle
│       ├── tasks/{id}.json       Task DAG — status, deps, val_loss
│       ├── plans/{id}.json       Plan approval state
│       ├── inbox/{agent_id}/     Per-agent message files
│       ├── artifacts/{id}.json   .nemo checkpoints, NIM containers
│       ├── prompts/{id}.md       Auto-injected coordination prompts
│       ├── snapshots/{id}.json   Team state snapshots
│       ├── costs/                GPU-hour cost records
│       ├── workspaces/           Git worktrees (one per agent)
│       └── metrics/              DCGM snapshots
├── clusters/                     Federated cluster configs
├── profiles/                     Custom agent profiles
├── hpo/                          Optuna study databases
├── config.json                   Dynamic configuration
└── audit.jsonl                   Structured audit log
```

Every write uses `tmpfile + os.replace()` — crash-safe on POSIX systems.

Agents spawn in **tmux sessions** (default) or **[OpenShell](https://github.com/NVIDIA/openshell) sandboxes** (kernel-level isolation via Landlock, seccomp, and network namespaces). Each agent gets its own `CUDA_VISIBLE_DEVICES`, an optional git worktree, and a coordination prompt that teaches it the NemoSpawn CLI.

---

## Installation

```bash
# Basic
pip install nemospawn

# With HPO (Optuna)
pip install nemospawn[hpo]

# With cross-node messaging (ZeroMQ)
pip install nemospawn[transport]

# Everything
pip install nemospawn[all]

# Development
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

**Prerequisites:**
- Python >= 3.10
- [tmux](https://github.com/tmux/tmux) for agent session isolation
- NVIDIA GPU drivers and [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface) for GPU features
- Optional: [OpenShell](https://github.com/NVIDIA/openshell) for sandbox mode, [SLURM](https://slurm.schedmd.com/) for HPC jobs

---

## Getting Started

### 1. Check your GPUs

```bash
nemospawn gpu discover           # List available GPUs
nemospawn gpu topology           # Show NVLink topology and islands
nemospawn gpu health             # Temperature, utilization, ECC errors
```

GPU discovery uses [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface) and [NVML](https://developer.nvidia.com/nvidia-management-library-nvml). NVLink island detection groups GPUs connected via [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) for optimal multi-GPU task placement.

### 2. Create a team

```bash
nemospawn team create my-experiment --gpus 0,1,2,3 -d "LLM fine-tuning sweep"
```

This discovers GPU topology, identifies NVLink islands, and creates the team directory with all subdirectories.

### 3. Spawn agents

```bash
nemospawn spawn agent --team my-experiment-abc \
  --agent-name trainer0 --role trainer --gpu 0 \
  --task "Fine-tune LLaMA with lr=2e-4" --agent-cmd claude

nemospawn spawn agent --team my-experiment-abc \
  --agent-name evaluator --role evaluator --gpu 1 \
  --task "Benchmark best checkpoint via Triton"
```

Each agent gets:
- A tmux session: `nemo-{team_id}-{agent_id}`
- `CUDA_VISIBLE_DEVICES` set to assigned GPUs
- `NEMOSPAWN_TEAM` and `NEMOSPAWN_AGENT` environment variables
- A coordination prompt file at `prompts/{agent_id}.md`

### 4. Create tasks with dependencies

```bash
nemospawn task create my-experiment-abc "Train model" --owner trainer0
nemospawn task create my-experiment-abc "Evaluate" --owner evaluator --blocked-by task-abc
```

### 5. Monitor

```bash
nemospawn board serve my-experiment-abc     # Web UI at http://localhost:8080
nemospawn board live my-experiment-abc      # Terminal kanban (Rich)
nemospawn board attach my-experiment-abc    # Tiled tmux panes
```

### 6. Check status

```bash
nemospawn team status my-experiment-abc
nemospawn spawn list --team my-experiment-abc
nemospawn task list my-experiment-abc
```

---

## Agent Spawning

### Supported agent CLIs

NemoSpawn works with any CLI agent via its adapter registry:

| Agent | Command | Auth Env Var | Prompt Method |
|-------|---------|-------------|---------------|
| Claude Code | `claude` | `ANTHROPIC_API_KEY` | CLI flag |
| Codex | `codex` | `OPENAI_API_KEY` | CLI flag |
| Kimi CLI | `kimi` | `MOONSHOT_API_KEY` | CLI flag |
| Cursor | `cursor` | — | File |
| nanobot | `nanobot` | — | CLI flag |
| aider | `aider` | `OPENAI_API_KEY` | CLI flag |
| OpenCode | `opencode` | — | File |
| GitHub Copilot | `github-copilot-cli` | `GITHUB_TOKEN` | File |
| Custom | any | configurable | File |

### Using profiles

```bash
nemospawn spawn agent --team t1 --agent-name worker --profile my-kimi --gpu 0
nemospawn spawn agent --team t1 --agent-name worker --agent-cmd kimi --gpu 0
```

### Runtimes

```bash
# tmux (default) — interactive, attach to watch
nemospawn spawn agent --team t1 --agent-name worker --runtime tmux

# OpenShell sandbox — kernel-level isolation via Landlock + seccomp
nemospawn spawn agent --team t1 --agent-name worker --runtime sandbox
```

### Git worktree isolation

```bash
nemospawn spawn agent --team t1 --agent-name worker --repo /path/to/repo
# Creates branch: nemospawn/{team}/{agent}
```

### Agent management

```bash
nemospawn spawn list --team t1              # List agents
nemospawn spawn kill --team t1 --agent a1   # Kill agent
```

---

## Task Management

Tasks flow through: **pending** → **blocked** → **running** → **done** / **failed**

```bash
nemospawn task create t1 "Deploy NIM" --owner deployer --blocked-by train-task
nemospawn task update t1 task-abc --status running
nemospawn task update t1 task-abc --status done --val-loss 0.042
nemospawn task list t1 --status running
nemospawn task show t1 task-abc
nemospawn task wait t1 task-abc --timeout 3600
```

When a task is marked `done`, all tasks depending on it via `blocked_by` are automatically unblocked.

---

## Messaging

```bash
nemospawn inbox send t1 worker0 "Best lr=2e-4, val_loss=0.042" --from leader
nemospawn inbox broadcast t1 "Deploying best checkpoint" --from leader
nemospawn inbox receive t1 worker0
```

Messages are atomic JSON files in `inbox/{agent_id}/`. Transport is auto-negotiated:

| Transport | When | Latency |
|-----------|------|---------|
| [NIXL](https://github.com/NVIDIA/nixl) | Same node, NVLink available | Sub-microsecond |
| [ZeroMQ](https://zeromq.org/) | Cross-node | Low (TCP) |
| File | Always available | Filesystem I/O |

---

## Plan Approval

Agents submit plans for leader review before major actions:

```bash
# Submit
nemospawn plan submit --team t1 --agent worker0 \
  --title "Fine-tune with LoRA" \
  -d "LoRA rank-16 on attention layers" \
  --steps "Download model,Apply LoRA,Train 5k steps,Evaluate"

# Review
nemospawn plan list --team t1 --status pending
nemospawn plan show --team t1 --plan plan-abc

# Approve or reject
nemospawn plan approve --team t1 --plan plan-abc --reviewer leader --comment "LGTM"
nemospawn plan reject --team t1 --plan plan-abc --reviewer leader --comment "Use rank-32"
```

---

## Lifecycle Protocol

### Idle reporting

```bash
nemospawn lifecycle idle --team t1 --agent worker0 --reason "All tasks done"
nemospawn lifecycle idle-list --team t1
```

### Graceful shutdown

```bash
nemospawn lifecycle shutdown-request --team t1 --agent worker0 --by leader
nemospawn lifecycle shutdown-approve --team t1 --agent worker0 --by leader
nemospawn lifecycle shutdown-reject --team t1 --agent worker0 --by leader --reason "New tasks"
nemospawn lifecycle status --team t1
```

---

## GPU Management

```bash
nemospawn gpu discover                          # List GPUs (nvidia-smi)
nemospawn gpu topology                          # NVLink topology and islands
nemospawn gpu health                            # Temp, utilization, power, ECC
nemospawn gpu status my-team                    # DCGM metrics per team
nemospawn gpu reallocate my-team --kill-below 50  # Kill underperforming agents
```

GPU health monitoring uses [DCGM](https://github.com/NVIDIA/DCGM) with fallback to nvidia-smi. Metrics include SM utilization, memory utilization, temperature, power draw, and ECC error counts.

---

## NeMo Artifacts

Integration with [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo):

```bash
nemospawn artifact register t1 /path/to/model.nemo \
  --type nemo-checkpoint --val-loss 0.042 --agent trainer0

nemospawn artifact promote t1 artifact-abc       # Crown best checkpoint
nemospawn artifact list t1 --sort val_loss       # Sort by validation loss
nemospawn artifact show t1 artifact-abc          # Full metadata
```

**Artifact types:** `nemo-checkpoint`, `lora-adapter`, `nim-container`, `dataset`, `benchmark`, `reward-signal`, `config-patch`

NeMo config injection generates YAML overrides from flat parameters with schema-aware type coercion. NVLink-aware scheduling places multi-GPU training tasks on the same NVLink island for optimal interconnect bandwidth.

---

## NIM Deployment

Integration with [NVIDIA NIM](https://docs.nvidia.com/nim/):

```bash
nemospawn nim deploy t1 artifact-abc --tp 2 --profile max-throughput --port 8000 --gpus 0,1
nemospawn nim list t1
nemospawn nim benchmark t1 http://localhost:8000 --concurrency 1,4,16,64
nemospawn nim profiles 8
```

Benchmarks use [Triton perf_analyzer](https://github.com/triton-inference-server/perf_analyzer) and report p50/p95/p99 latency and throughput (inferences/sec, tokens/sec).

---

## Dashboards & Observability

### Web UI

```bash
nemospawn board serve my-team --port 8080 --metrics-port 9090
```

Dark-themed kanban at `http://localhost:8080` with:
- Task columns (pending/blocked/running/done/failed) with live counts
- Agent cards with status dots, GPU assignments, and lifecycle state
- Plan section with approval badges
- Server-Sent Events (SSE) auto-updates every 3 seconds

### Terminal kanban

```bash
nemospawn board live my-team --interval 10
```

### Prometheus + Grafana

```bash
nemospawn board serve my-team --grafana-url http://grafana:3000 --grafana-key <key>
```

Exported metrics:
- `nemospawn_gpu_sm_utilization`, `nemospawn_gpu_mem_utilization`
- `nemospawn_gpu_temperature_celsius`, `nemospawn_gpu_power_watts`
- `nemospawn_gpu_ecc_sbe_total`
- `nemospawn_agents_total{status}`, `nemospawn_tasks_total{status}`
- `nemospawn_val_loss{agent,task}`

Auto-provisions a 6-panel [Grafana](https://grafana.com/) dashboard or saves JSON for manual import.

### Tiled tmux

```bash
nemospawn board attach my-team
```

---

## Templates & Launch

### One-command launch

```bash
nemospawn launch run autoresearch --gpus 0,1,2,3
nemospawn launch run nim-deploy --gpus 0,1,2 --goal "Deploy LLaMA-70B"
nemospawn launch run rlhf-swarm --gpus 0,1,2,3
nemospawn launch run data-curation --gpus 0,1
nemospawn launch run autoresearch --gpus 0-7 --dry-run   # Preview
nemospawn launch templates                                # List templates
```

### Built-in templates

| Template | Workers | Pipeline |
|----------|---------|----------|
| `autoresearch` | 2 trainers + evaluator | DataPrep → Train → Eval → HPO loop |
| `nim-deploy` | deployer TP1 + TP2 + benchmarker | Build → Benchmark → Rank → Serve |
| `rlhf-swarm` | reward + PPO + eval agents | Data → SFT → Reward → PPO |
| `data-curation` | curator + trainer | Ingest → Clean → Deduplicate → Validate |

### Custom templates

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

```bash
nemospawn launch run /path/to/my-pipeline.toml --gpus 0,1
```

---

## Hyperparameter Optimization

Powered by [Optuna](https://github.com/optuna/optuna) with TPE sampler + ASHA pruner:

```bash
nemospawn hpo init --study lr-sweep --template autoresearch
nemospawn hpo suggest --study lr-sweep               # Sample next config
nemospawn hpo report --study lr-sweep --trial t1 --step 5000 --val-loss 0.042
nemospawn hpo best --study lr-sweep                  # Show best trial
nemospawn hpo trials --study lr-sweep                # List all trials
nemospawn hpo dashboard --study lr-sweep --port 8081 # Optuna web dashboard
```

Search spaces are defined in TOML (`hpo.toml`) with parameter types: `loguniform`, `uniform`, `categorical`, `int`. Falls back to random sampling when Optuna is not installed.

---

## Configuration

Three-tier priority: **environment variable > config file > default**

```bash
nemospawn config show                  # All settings with sources
nemospawn config get transport         # Single value
nemospawn config set transport zeromq  # Persist to ~/.nemospawn/config.json
nemospawn config health                # Diagnostic checks
```

| Key | Default | Env Var | Description |
|-----|---------|---------|-------------|
| `data_dir` | `~/.nemospawn` | `NEMOSPAWN_DATA_DIR` | State root directory |
| `transport` | `file` | `NEMOSPAWN_TRANSPORT` | Messaging backend (file/zeromq/nixl) |
| `workspace` | `auto` | `NEMOSPAWN_WORKSPACE` | Git worktree mode (auto/always/never) |
| `default_profile` | `claude` | `NEMOSPAWN_DEFAULT_PROFILE` | Default agent CLI |
| `default_runtime` | `tmux` | `NEMOSPAWN_DEFAULT_RUNTIME` | Default spawn mode (tmux/sandbox) |
| `cost_rate` | `2.50` | `NEMOSPAWN_COST_RATE` | USD per GPU-hour |
| `watch_interval` | `60` | `NEMOSPAWN_WATCH_INTERVAL` | Health check interval (seconds) |
| `web_port` | `8080` | `NEMOSPAWN_WEB_PORT` | Web UI port |
| `metrics_port` | `9090` | `NEMOSPAWN_METRICS_PORT` | Prometheus scrape port |
| `user` | *(empty)* | `NEMOSPAWN_USER` | Multi-user identity |

---

## Agent Profiles

Profiles define how a CLI agent is invoked, including command, auth, model, and prompt injection method.

```bash
nemospawn profile list                                    # All profiles
nemospawn profile show claude                             # Profile details + adapter info
nemospawn profile create --name my-kimi --agent kimi \
  --model moonshot-v1 --auth-env MOONSHOT_API_KEY         # Custom profile
nemospawn profile wizard                                  # Interactive setup
nemospawn profile doctor claude                           # Diagnose issues
nemospawn profile test my-kimi                            # Smoke test
nemospawn profile delete my-kimi                          # Remove
```

---

## Adaptive Scheduling

Monitor GPU utilization per agent and auto-reassign tasks from underperformers:

```bash
nemospawn schedule analyze --team t1                     # Rank agents by performance
nemospawn schedule suggest --team t1 --threshold 30      # Suggest reassignments
nemospawn schedule apply --team t1 --task t1 --to a2     # Apply reassignment
nemospawn schedule auto --team t1 --threshold 30 --interval 300  # Continuous
```

Performance score combines GPU SM utilization (60% weight) and task completion rate (40% weight). Tasks on agents below the utilization threshold are candidates for reassignment.

---

## Cost Tracking

```bash
nemospawn cost show --team t1                # Per-agent GPU-hour breakdown
nemospawn cost set-rate --team t1 --rate 3.50  # Set $/GPU-hour
nemospawn cost reset --team t1               # Reset tracking
```

Costs are calculated from agent creation time, GPU count, and elapsed time. Running agents accumulate to current time; stopped agents freeze at their last update.

---

## Snapshots

Save and restore full team state:

```bash
nemospawn snapshot save --team t1 --label "before-deploy"
nemospawn snapshot list --team t1
nemospawn snapshot restore --team t1 --snapshot snap-abc
nemospawn snapshot delete --team t1 --snapshot snap-abc
```

A snapshot captures agents, tasks, plans, and cost records. Restore overwrites current state but preserves team identity (`team.json`).

---

## Health Monitoring

```bash
nemospawn watch status --team t1             # Single health check
nemospawn watch start --team t1 --interval 60  # Continuous monitoring
```

The watcher checks:
- tmux sessions alive for running agents (marks dead agents as `stopped`)
- Stuck agents with no lifecycle update for 24+ hours
- Provides healthy/unhealthy counts with issue details

---

## Cross-Cluster Federation

```bash
nemospawn cluster register dgx-2 --host dgx2.internal --key ~/.ssh/dgx2 --user admin
nemospawn cluster list
nemospawn cluster status dgx-2

# Spawn on remote cluster
nemospawn spawn agent --team t1 --agent-name remote-trainer --remote admin@dgx2.internal --gpu 0
```

Artifacts transfer between clusters via [git-annex](https://git-annex.branchable.com/). State coordination uses NFS or SSHFS shared mounts.

---

## Authentication & Security

```bash
nemospawn auth create-user researcher1 --role user    # Returns API key (shown once)
nemospawn auth verify <api-key>                       # Validate key
nemospawn auth audit --last 50 --event agent.spawn    # Query audit log
```

- API keys are stored as SHA-256 hashes (plaintext never persisted)
- All operations logged to `~/.nemospawn/audit.jsonl` with timestamps, user, team, and event details
- Multi-user namespace isolation with `user` / `admin` roles

---

## SLURM Integration

Integration with [SLURM](https://slurm.schedmd.com/) workload manager:

```bash
nemospawn slurm generate my-job --gpus 4 --gpu-type a100 --nodes 2 \
  --time 48:00:00 --command "nemospawn launch run autoresearch"
nemospawn slurm submit my-job.sbatch
nemospawn slurm status 12345
nemospawn slurm cancel 12345
```

Generates sbatch scripts with GPU resource requests, partition config, module loading, and environment setup.

---

## Agent Skill

Install the NemoSpawn coordination protocol as a discoverable skill:

```bash
nemospawn skill install --target claude     # ~/.claude/skills/nemospawn/skill.md
nemospawn skill install --target all        # Claude Code + Codex
nemospawn skill status                      # Check installation
nemospawn skill uninstall --target all      # Remove
```

The skill teaches agents the full NemoSpawn protocol: task management, messaging, plan submission, lifecycle reporting, and artifact registration. Agents automatically discover it when spawned.

---

## State Architecture

### Atomic writes

Every state mutation uses crash-safe writes:

```python
# Write to temp file in same directory
fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
json.dump(data, f)

# Atomic rename (POSIX guarantee)
os.replace(tmp, target_path)
```

If the process crashes mid-write, the original file remains untouched.

### Agent coordination flow

1. Agent spawns with `NEMOSPAWN_TEAM`, `NEMOSPAWN_AGENT`, `CUDA_VISIBLE_DEVICES`
2. Coordination prompt auto-written to `prompts/{agent_id}.md`
3. Agent reads tasks → updates status → sends messages → submits plans via CLI
4. Leader monitors via `board serve` or `board live`
5. Agent reports `idle` when finished → leader approves shutdown

### Transport selection

| Condition | Transport | Latency |
|-----------|-----------|---------|
| Same node + [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) | [NIXL](https://github.com/NVIDIA/nixl) | Sub-microsecond |
| Cross-node | [ZeroMQ](https://zeromq.org/) | TCP round-trip |
| Fallback | File | Filesystem I/O |

Selection is automatic based on GPU topology and network availability.
