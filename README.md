# NemoSpawn

GPU-native agent-swarm orchestration for the full NVIDIA AI stack.

A single researcher types one goal. NemoSpawn spawns a full team of specialized agents across DGX/HGX nodes, assigns GPU-pinned workloads, tracks every dependency, synthesizes results, and ships a trained or deployed model — with zero manual coordination.

## Stack

OpenShell | NeMo Framework | NIM Microservices | Triton | NIXL | DCGM | NGC

## Quick Start

```bash
pip install nemospawn

# Discover GPUs
nemospawn gpu discover
nemospawn gpu topology

# Launch from a built-in template (one command)
nemospawn launch run autoresearch --gpus 0,1,2,3

# Or build manually — create team, spawn agents, manage tasks
nemospawn team create my-experiment --gpus 0,1,2,3
nemospawn spawn agent --team my-experiment-abc --agent-name trainer0 --gpu 0 --task "LR sweep"
nemospawn task create my-experiment-abc "Train model" --owner trainer0
nemospawn task list my-experiment-abc
```

> See [GUIDE.md](GUIDE.md) for a detailed walkthrough of every feature.

## Features

### Core Orchestration
- **Team management** — create GPU-aware teams with NVLink topology discovery
- **Agent spawning** — tmux sessions or OpenShell sandboxes with `CUDA_VISIBLE_DEVICES` pinning
- **Multi-agent CLI support** — Claude Code, Codex, Kimi, Cursor, nanobot, aider, OpenCode, Copilot, or any custom CLI
- **Task DAG** — create/update/list/wait with dependency resolution (`blocked_by`)
- **Inter-agent messaging** — file-based inbox with send/broadcast/receive
- **Plan approval** — agents submit plans for leader review before execution
- **Coordination prompt injection** — auto-injected protocol for both tmux and sandbox agents

### Agent Lifecycle & Scheduling
- **Lifecycle protocol** — graceful shutdown request/approve/reject, idle reporting
- **Adaptive scheduling** — monitor GPU utilization, auto-reassign tasks from underperforming agents
- **Agent watcher** — continuous health monitoring, dead tmux detection, stuck agent alerts
- **Cost tracking** — per-agent GPU-hour accumulation with configurable rates
- **Team snapshots** — save and restore full team state at any point

### NVIDIA Stack Integration
- **NeMo** — .nemo artifact store, YAML config injection, NVLink-aware scheduling
- **NIM** — deploy checkpoints as NIM containers, benchmark with perf_analyzer, rank endpoints
- **Triton** — model repository generation, perf_analyzer benchmarks (p50/p95/p99)
- **DCGM** — GPU health polling, underperformance detection, Prometheus export
- **NGC** — model pull/push, container registry, org sharing
- **NIXL** — sub-microsecond transport over NVLink/InfiniBand (with file fallback)

### OpenShell Sandbox Runtime
- Agents run inside NVIDIA OpenShell sandboxes with kernel-level isolation
- Per-role security policies (Landlock filesystem, seccomp, network namespace)
- GPU passthrough and NemoSpawn coordination protocol auto-injected
- `--runtime sandbox` flag on spawn commands

### Observability
- **Web UI** — browser-based kanban dashboard with SSE real-time updates (`board serve`)
- **Prometheus** — scrape endpoint at `/metrics` with GPU and task metrics
- **Grafana** — auto-provisioned 6-panel dashboard (GPU util, temp, val_loss, tasks)
- **Terminal kanban** — `nemospawn board live` for Rich-based task view
- **tmux board** — `nemospawn board attach` for tiled agent view

### Configuration & Profiles
- **Dynamic config** — `nemospawn config show/get/set/health` with env > file > default priority
- **Agent profiles** — `nemospawn profile list/create/test/wizard/doctor` for any CLI agent
- **Reusable skill** — `nemospawn skill install` packages the coordination protocol for Claude Code / Codex

### Templates
- **Built-in templates** — autoresearch, nim-deploy, rlhf-swarm, data-curation
- **TOML format** — define workers, GPU requirements, dependency chains
- **One command launch** — `nemospawn launch run autoresearch --gpus 0-7`

### HPO (Hyperparameter Optimization)
- **Optuna TPE** sampler + **ASHA** pruner (SQLite backend, no server)
- **hpo.toml** search space declarations
- `nemospawn hpo suggest/report/best/trials/dashboard`
- Fallback random sampler when Optuna is not installed

### Cross-Cluster Federation
- Register remote clusters via SSH
- Spawn agents on remote DGX/HGX nodes
- git-annex artifact transfer between clusters
- Shared filesystem (NFS/SSHFS) state coordination

### Production Hardening
- API key authentication (SHA-256)
- Structured JSONL audit logging
- Multi-user namespace isolation
- SLURM/PBS job script generation and submission

## CLI Commands (23 groups)

```
nemospawn team       — Team lifecycle management
nemospawn spawn      — Agent spawn and management
nemospawn task       — Task DAG management
nemospawn inbox      — Inter-agent messaging
nemospawn board      — Monitoring dashboards (terminal + web UI)
nemospawn gpu        — GPU discovery and health
nemospawn artifact   — NeMo artifact management
nemospawn nim        — NIM deployment pipeline
nemospawn hpo        — Hyperparameter optimization
nemospawn cluster    — Cross-cluster federation
nemospawn launch     — Launch teams from templates
nemospawn ngc        — NGC registry operations
nemospawn slurm      — SLURM job management
nemospawn auth       — Authentication and audit
nemospawn plan       — Plan approval workflow
nemospawn lifecycle  — Agent lifecycle protocol
nemospawn cost       — GPU cost tracking
nemospawn snapshot   — Team state snapshots
nemospawn watch      — Agent health monitoring
nemospawn profile    — Agent profile management
nemospawn config     — Configuration management
nemospawn schedule   — Adaptive task scheduling
nemospawn skill      — Agent skill management
```

## Requirements

- Python >= 3.10
- tmux (for agent isolation)
- NVIDIA GPU drivers (for GPU features; CPU-only dev mode works without)
- Linux (production) / macOS (CPU-only dev mode)

### Optional
- `pip install nemospawn[hpo]` — Optuna for hyperparameter optimization
- `pip install nemospawn[transport]` — ZeroMQ for cross-node messaging
- `pip install nemospawn[all]` — everything
- OpenShell (`openshell` CLI) — for sandbox-mode agent spawning
- SLURM — for HPC job submission

## License

Apache-2.0
