# NemoSpawn — Developer Guide

## What is this?
GPU-native agent-swarm orchestration for NVIDIA's AI stack (NeMo, NIM, Triton, NIXL).
Workers run inside NVIDIA OpenShell sandboxes or tmux sessions with kernel-level isolation.
Supports 8 agent CLIs (Claude Code, Codex, Kimi, Cursor, nanobot, aider, OpenCode, Copilot).

## Build & Test
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture
- `src/nemospawn/core/` — state management (atomic JSON), models, auth, audit, profiles, settings, plan, lifecycle, costs, snapshot, watcher, adaptive scheduling, skill
- `src/nemospawn/cli/` — Typer CLI commands (23 command groups)
- `src/nemospawn/gpu/` — GPU discovery, NVLink topology, DCGM health
- `src/nemospawn/nemo/` — NeMo artifacts, config injection, NVLink-aware scheduling
- `src/nemospawn/nim/` — NIM deploy pipeline, Triton benchmarks
- `src/nemospawn/openshell/` — OpenShell sandbox integration, security policies, coordination prompts
- `src/nemospawn/messaging/` — file/ZeroMQ/NIXL transport, inbox
- `src/nemospawn/observability/` — Prometheus metrics, Grafana dashboards, kanban, web UI (SSE)
- `src/nemospawn/templates/` — TOML team templates, launch engine
- `src/nemospawn/federation/` — cross-cluster SSH spawn, git-annex
- `src/nemospawn/hpo/` — Optuna TPE/ASHA HPO, fallback sampler
- `src/nemospawn/ngc/` — NGC model registry operations
- `src/nemospawn/runtime/` — tmux, git worktree, SLURM

## Key Patterns
- All state: `~/.nemospawn/` as atomic JSON (tmp + os.replace)
- Agents spawn in tmux sessions or OpenShell sandboxes
- GPU pinning via `CUDA_VISIBLE_DEVICES`
- Config priority: env var (NEMOSPAWN_*) > config file > defaults
- Coordination prompt auto-injected for both tmux and sandbox agents
- Tests: `pytest tests/ -v` — 189 tests across 32 files

## Style
- Python >=3.10, Typer+Rich CLI, dataclasses (not Pydantic)
- `ruff` for linting, 100 char line length
- No database — filesystem-native state
