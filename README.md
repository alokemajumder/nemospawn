<p align="center">
  <h1 align="center">NemoSpawn</h1>
  <p align="center">GPU-native agent-swarm orchestration for the full NVIDIA AI stack</p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="GUIDE.md">User Guide</a> &bull;
  <a href="#cli-reference">CLI Reference</a> &bull;
  <a href="#nvidia-stack">NVIDIA Stack</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-blue">
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-189%20passing-brightgreen">
  <img alt="CLI" src="https://img.shields.io/badge/CLI%20groups-23-purple">
</p>

---

One researcher types one goal. NemoSpawn spawns a team of GPU-pinned agents across DGX/HGX nodes, assigns workloads, tracks dependencies, and delivers trained or deployed models — zero manual coordination.

```
┌──────────────────────────────────────────────────────────┐
│  nemospawn launch run autoresearch --gpus 0,1,2,3,4,5,6,7│
└──────────────┬───────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │    Team Orchestrator  │
    │  NVLink topology ─── GPU discovery ─── Task DAG      │
    └──┬───┬───┬───┬───┬──┘
       │   │   │   │   │
    ┌──▼┐┌─▼┐┌─▼┐┌─▼┐┌─▼┐
    │ T0 ││ T1││ T2││ E0││ D0│   Agents (tmux / OpenShell)
    │GPU0││GPU1││GPU2││GPU3││GPU4│
    └────┘└───┘└───┘└───┘└───┘
       │         │         │
    ┌──▼─────────▼─────────▼──┐
    │  Messaging (NIXL / ZMQ / File)  │
    │  Prometheus  │  Grafana  │  Web UI │
    └─────────────────────────────────┘
```

## Why NemoSpawn?

- **GPU-first** — NVLink topology discovery, island-aware scheduling, DCGM health monitoring
- **NVIDIA-native** — first-class integration with [NeMo](https://github.com/NVIDIA/NeMo), [NIM](https://docs.nvidia.com/nim/), [Triton](https://github.com/triton-inference-server/server), [NIXL](https://github.com/NVIDIA/nixl), [DCGM](https://github.com/NVIDIA/DCGM), [NGC](https://catalog.ngc.nvidia.com/), and [OpenShell](https://github.com/NVIDIA/openshell)
- **Any agent CLI** — works with Claude Code, Codex, Kimi, Cursor, nanobot, aider, or any custom tool
- **Zero infrastructure** — no database, no server. All state is atomic JSON in `~/.nemospawn/`
- **Production-ready** — API key auth, audit logging, SLURM integration, cross-cluster federation

## Quick Start

### Install

```bash
pip install nemospawn

# Optional extras
pip install nemospawn[hpo]          # Optuna for hyperparameter optimization
pip install nemospawn[transport]    # ZeroMQ for cross-node messaging
pip install nemospawn[all]          # Everything
```

### Prerequisites

- Python >= 3.10
- [tmux](https://github.com/tmux/tmux) (for agent isolation)
- NVIDIA GPU drivers (GPU features require nvidia-smi; CPU-only dev mode works without)

### Launch a team in one command

```bash
nemospawn launch run autoresearch --gpus 0,1,2,3
```

### Or build step by step

```bash
# 1. Discover your hardware
nemospawn gpu discover
nemospawn gpu topology

# 2. Create a team
nemospawn team create my-experiment --gpus 0,1,2,3 -d "LLM fine-tuning"

# 3. Spawn agents
nemospawn spawn agent --team my-experiment-abc \
  --agent-name trainer0 --role trainer --gpu 0 \
  --task "Fine-tune with lr=2e-4" --agent-cmd claude

nemospawn spawn agent --team my-experiment-abc \
  --agent-name evaluator --role evaluator --gpu 1 \
  --task "Benchmark via Triton"

# 4. Create tasks with dependencies
nemospawn task create my-experiment-abc "Train model" --owner trainer0
nemospawn task create my-experiment-abc "Evaluate" --owner evaluator --blocked-by task-abc

# 5. Monitor
nemospawn board serve my-experiment-abc     # Web UI at http://localhost:8080
nemospawn board live my-experiment-abc      # Terminal kanban
nemospawn board attach my-experiment-abc    # Tiled tmux
```

> **Full walkthrough**: See [GUIDE.md](GUIDE.md) for detailed docs on every feature.

## Features

### Core Orchestration

| Feature | Description |
|---------|------------|
| **Team management** | Create GPU-aware teams with [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) topology discovery and island detection |
| **Agent spawning** | tmux sessions or [OpenShell](https://github.com/NVIDIA/openshell) sandboxes with `CUDA_VISIBLE_DEVICES` pinning |
| **Multi-agent CLI** | 8 built-in profiles (Claude Code, Codex, Kimi, Cursor, nanobot, aider, OpenCode, Copilot) + custom |
| **Task DAG** | Dependencies via `blocked_by` with auto-unblocking, status tracking, wait/timeout |
| **Messaging** | Point-to-point and broadcast via file, [ZeroMQ](https://zeromq.org/), or [NIXL](https://github.com/NVIDIA/nixl) transport |
| **Plan approval** | Agents submit plans for leader review before execution |
| **Prompt injection** | Coordination protocol auto-injected into both tmux and sandbox agents |

### Agent Lifecycle & Scheduling

| Feature | Description |
|---------|------------|
| **Lifecycle protocol** | Graceful idle/shutdown request/approve/reject flow |
| **Adaptive scheduling** | GPU utilization monitoring + automatic task reassignment |
| **Agent watcher** | Continuous health checks, dead tmux detection, stuck agent alerts |
| **Cost tracking** | Per-agent GPU-hour accumulation with configurable rates |
| **Team snapshots** | Save and restore full team state at any point |

### NVIDIA Stack Integration

<a name="nvidia-stack"></a>

| Component | Integration | Resources |
|-----------|------------|-----------|
| **[NeMo Framework](https://github.com/NVIDIA/NeMo)** | .nemo artifact store, YAML config injection, NVLink-aware multi-GPU scheduling | [Docs](https://docs.nvidia.com/nemo-framework/) &bull; [GitHub](https://github.com/NVIDIA/NeMo) |
| **[NIM](https://docs.nvidia.com/nim/)** | Deploy checkpoints as NIM microservices, benchmark with perf_analyzer, rank endpoints by latency/throughput | [Docs](https://docs.nvidia.com/nim/) &bull; [Blog](https://developer.nvidia.com/nim) |
| **[Triton Inference Server](https://github.com/triton-inference-server/server)** | Model repository generation, perf_analyzer benchmarks (p50/p95/p99 latency) | [Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/) &bull; [GitHub](https://github.com/triton-inference-server/server) |
| **[DCGM](https://github.com/NVIDIA/DCGM)** | GPU health polling (SM util, memory, temp, power, ECC errors), Prometheus export | [Docs](https://docs.nvidia.com/datacenter/dcgm/) &bull; [GitHub](https://github.com/NVIDIA/DCGM) |
| **[NGC Catalog](https://catalog.ngc.nvidia.com/)** | Model pull/push, container registry push to nvcr.io, org sharing | [Catalog](https://catalog.ngc.nvidia.com/) &bull; [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) |
| **[NIXL](https://github.com/NVIDIA/nixl)** | Sub-microsecond messaging over NVLink/InfiniBand with automatic file fallback | [GitHub](https://github.com/NVIDIA/nixl) |
| **[OpenShell](https://github.com/NVIDIA/openshell)** | Kernel-level sandbox isolation (Landlock, seccomp, network namespace) with GPU passthrough | [GitHub](https://github.com/NVIDIA/openshell) |
| **[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)** | Topology parsing, island detection, NVLink-aware task placement for multi-GPU jobs | [Product](https://www.nvidia.com/en-us/data-center/nvlink/) |

### Observability

| Feature | Description |
|---------|------------|
| **Web UI** | Browser-based kanban with SSE real-time updates, dark theme (`board serve`) |
| **[Prometheus](https://prometheus.io/)** | Scrape endpoint at `/metrics` with GPU and task metrics |
| **[Grafana](https://grafana.com/)** | Auto-provisioned 6-panel dashboard (GPU util, temp, val_loss, tasks) |
| **Terminal kanban** | Rich-based live task view (`board live`) |
| **tmux board** | Tiled pane view of all running agents (`board attach`) |

### Configuration & Profiles

| Feature | Description |
|---------|------------|
| **Dynamic config** | `config show/get/set/health` — priority: env var > file > default |
| **Agent profiles** | `profile list/create/test/wizard/doctor` for any CLI agent |
| **Reusable skill** | `skill install` packages the protocol for Claude Code / Codex agents |

### Templates

Four built-in team templates for one-command launch:

| Template | Description |
|----------|------------|
| `autoresearch` | Multi-phase training: DataPrep → Training → Evaluation → HPO loop |
| `nim-deploy` | Checkpoint → NIM container → Benchmark → Rank → Serve |
| `rlhf-swarm` | Data → SFT → Reward modeling → PPO optimization |
| `data-curation` | Ingestion → Cleaning → Deduplication → Validation |

Custom templates use TOML format. See [GUIDE.md#templates--launch](GUIDE.md#templates--launch).

### HPO (Hyperparameter Optimization)

- [Optuna](https://github.com/optuna/optuna) TPE sampler + ASHA pruner (SQLite backend, no server)
- TOML search space declarations
- `hpo suggest/report/best/trials/dashboard`
- Fallback random sampler when Optuna is not installed

### Cross-Cluster Federation

- Register remote DGX/HGX clusters via SSH
- Spawn agents on remote nodes
- [git-annex](https://git-annex.branchable.com/) artifact transfer between clusters
- NFS/SSHFS shared state coordination

### Production Hardening

- API key authentication (SHA-256 hashing)
- Structured JSONL audit logging
- Multi-user namespace isolation with role-based access
- [SLURM](https://slurm.schedmd.com/) job script generation and submission

## CLI Reference

<a name="cli-reference"></a>

NemoSpawn ships 23 command groups:

```
Core
  nemospawn team         Create and manage GPU-aware teams
  nemospawn spawn        Spawn and kill GPU-pinned agents
  nemospawn task         Task DAG with dependency resolution
  nemospawn inbox        Inter-agent messaging (send/broadcast/receive)
  nemospawn plan         Plan approval workflow (submit/approve/reject)
  nemospawn lifecycle    Agent lifecycle (idle/shutdown request/approve/reject)

GPU & NVIDIA
  nemospawn gpu          GPU discovery, NVLink topology, DCGM health
  nemospawn artifact     NeMo artifact store (register/promote/list)
  nemospawn nim          NIM deployment pipeline (deploy/benchmark/rank)
  nemospawn ngc          NGC registry (pull/push models and containers)
  nemospawn hpo          Hyperparameter optimization (Optuna TPE/ASHA)

Infrastructure
  nemospawn launch       Launch teams from TOML templates
  nemospawn cluster      Cross-cluster federation (SSH remote spawn)
  nemospawn slurm        SLURM job management (generate/submit/status)
  nemospawn auth         Authentication and audit logging

Monitoring
  nemospawn board        Dashboards (web UI / terminal kanban / tmux tiled)
  nemospawn watch        Agent health monitoring (tmux alive, stuck detection)
  nemospawn cost         GPU-hour cost tracking per agent
  nemospawn schedule     Adaptive scheduling (analyze/suggest/auto-reassign)
  nemospawn snapshot     Save and restore team state

Configuration
  nemospawn config       Dynamic config (env > file > default)
  nemospawn profile      Agent CLI profiles (wizard/doctor/test)
  nemospawn skill        Install coordination skill for Claude Code / Codex
```

Run `nemospawn <command> --help` for subcommand details.

## Architecture

```
~/.nemospawn/                          # All state — no database
├── teams/{id}/
│   ├── team.json                      # GPU list, NVLink topology
│   ├── agents/{id}.json               # Status, GPUs, tmux session, lifecycle
│   ├── tasks/{id}.json                # DAG with blocked_by deps
│   ├── plans/{id}.json                # Approval workflow state
│   ├── inbox/{agent}/                 # Per-agent message files
│   ├── artifacts/{id}.json            # .nemo checkpoints, NIM containers
│   ├── prompts/{id}.md                # Auto-injected coordination prompts
│   ├── snapshots/{id}.json            # Team state snapshots
│   ├── costs/cost_record.json         # GPU-hour tracking
│   ├── workspaces/                    # Git worktrees per agent
│   └── metrics/                       # DCGM metric snapshots
├── clusters/                          # Federated cluster SSH configs
├── profiles/                          # Custom agent profiles
├── hpo/                               # Optuna study databases
├── config.json                        # Dynamic configuration
└── audit.jsonl                        # Structured audit log
```

Every write uses atomic `tmpfile + os.replace` — crash-safe on POSIX.

### Transport Negotiation

NemoSpawn automatically selects the best messaging transport:

1. **NIXL** — same node with NVLink (sub-microsecond latency)
2. **ZeroMQ** — cross-node TCP
3. **File** — always-available fallback (atomic JSON)

### How Agents Coordinate

1. Agent spawns with `NEMOSPAWN_TEAM`, `NEMOSPAWN_AGENT`, and `CUDA_VISIBLE_DEVICES` env vars
2. A coordination prompt is auto-injected (`prompts/{agent_id}.md`)
3. The prompt teaches the agent all NemoSpawn CLI commands
4. Agents check tasks, update status, send messages, submit plans — all via CLI
5. The leader monitors progress via `board serve` or `board live`
6. When done, agents report `idle` via `lifecycle idle`

## Development

```bash
# Clone and install
git clone https://github.com/alokemajumder/nemospawn.git
cd nemospawn
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

### Project Layout

```
src/nemospawn/
├── cli/            23 Typer command groups
├── core/           State, models, auth, profiles, config, plan, lifecycle,
│                   costs, snapshot, watcher, adaptive scheduling, skill
├── gpu/            Discovery, NVLink topology, DCGM health
├── nemo/           Artifacts, config injection, NVLink-aware scheduling
├── nim/            NIM deployer, Triton benchmarks
├── openshell/      Sandbox integration, security policies, prompt injection
├── messaging/      File / ZeroMQ / NIXL transport
├── observability/  Prometheus, Grafana, kanban, web UI (SSE)
├── templates/      TOML templates, launch engine
├── federation/     Cross-cluster SSH spawn, git-annex
├── hpo/            Optuna TPE/ASHA, fallback sampler
├── ngc/            NGC model registry
└── runtime/        tmux, git worktree, SLURM
```

## NVIDIA Stack Usage

How NemoSpawn uses each NVIDIA component — directly (calls its APIs/CLIs) or indirectly (leverages its infrastructure):

### Direct integrations (NemoSpawn calls these)

| Component | How NemoSpawn uses it | Link |
|-----------|----------------------|------|
| **nvidia-smi / NVML** | GPU discovery (`gpu discover`), topology parsing (`gpu topology`), health metrics (`gpu health`). Called via subprocess and [pynvml](https://pypi.org/project/pynvml/) bindings. | [NVML Docs](https://developer.nvidia.com/nvidia-management-library-nvml) |
| **NeMo Framework** | Artifact store manages `.nemo` checkpoint bundles. Config injection generates NeMo YAML overrides with schema-aware type coercion. NVLink-aware scheduler places multi-GPU NeMo training on the same NVLink island. | [GitHub](https://github.com/NVIDIA/NeMo) &bull; [Docs](https://docs.nvidia.com/nemo-framework/) |
| **NIM** | `nim deploy` builds NIM containers from `.nemo` checkpoints with tensor parallel profiles (TP1-TP8). `nim benchmark` runs perf_analyzer. `nim list` tracks running endpoints. | [Docs](https://docs.nvidia.com/nim/) &bull; [Developer](https://developer.nvidia.com/nim) |
| **Triton Inference Server** | `nim benchmark` invokes `perf_analyzer` against NIM/Triton endpoints. Model repository config (`config.pbtxt`) is auto-generated from NIM artifacts. Reports p50/p95/p99 latency and throughput. | [GitHub](https://github.com/triton-inference-server/server) &bull; [perf_analyzer](https://github.com/triton-inference-server/perf_analyzer) |
| **DCGM** | `gpu status` polls `dcgmi dmon` for real-time GPU metrics (SM util, memory util, temperature, power, ECC errors, PCIe throughput). Metrics are exported to Prometheus. Falls back to nvidia-smi when DCGM is not installed. | [GitHub](https://github.com/NVIDIA/DCGM) &bull; [Docs](https://docs.nvidia.com/datacenter/dcgm/) |
| **NGC CLI** | `ngc pull/push` wraps the NGC CLI for model download/upload. `ngc push-container` pushes Docker images to `nvcr.io`. `ngc auth` checks CLI authentication. | [Catalog](https://catalog.ngc.nvidia.com/) &bull; [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) |
| **NIXL** | Messaging transport for agents on the same node with NVLink. Provides sub-microsecond latency for inter-agent coordination. Auto-negotiated — falls back to ZeroMQ or file transport when NIXL is unavailable. | [GitHub](https://github.com/NVIDIA/nixl) |
| **OpenShell** | `--runtime sandbox` spawns agents inside OpenShell sandboxes with kernel-level isolation (Landlock filesystem, seccomp syscall filtering, network namespaces). GPU passthrough via `CUDA_VISIBLE_DEVICES`. Per-role security policies generated automatically. | [GitHub](https://github.com/NVIDIA/openshell) |

### Indirect integrations (infrastructure NemoSpawn runs on)

| Component | Role in NemoSpawn | Link |
|-----------|-------------------|------|
| **NVLink** | `gpu topology` parses the NVLink interconnect matrix to detect GPU islands. Multi-GPU tasks are scheduled on the same island to maximize interconnect bandwidth. Link types tracked: NV12, NV8, NV4, NV2, SYS, PHB, PXB. | [Product](https://www.nvidia.com/en-us/data-center/nvlink/) |
| **CUDA Toolkit** | Agents use `CUDA_VISIBLE_DEVICES` for GPU pinning. NeMo and NIM workloads run on CUDA. NemoSpawn itself does not call CUDA APIs directly. | [Toolkit](https://developer.nvidia.com/cuda-toolkit) |
| **NCCL** | Multi-GPU NeMo training uses NCCL for collective communication. NemoSpawn configures tensor parallel (TP) and pipeline parallel (PP) degrees that NCCL implements. | [GitHub](https://github.com/NVIDIA/nccl) |
| **cuDNN** | NeMo training workloads use cuDNN internally. NemoSpawn does not call cuDNN directly but configures mixed-precision modes (bf16-mixed, 16-mixed, 32) that cuDNN accelerates. | [Developer](https://developer.nvidia.com/cudnn) |
| **TensorRT** | NIM containers use TensorRT for inference optimization. `nim deploy --profile max-throughput` leverages TensorRT compilation. NemoSpawn manages the deployment lifecycle, not the compilation. | [GitHub](https://github.com/NVIDIA/TensorRT) |
| **NVIDIA Container Toolkit** | NIM containers require the NVIDIA Container Toolkit for GPU access inside Docker. `nim deploy` assumes this is installed on the host. | [GitHub](https://github.com/NVIDIA/nvidia-container-toolkit) |
| **DGX / HGX Systems** | NemoSpawn's cross-cluster federation (`cluster register/spawn`) targets DGX and HGX nodes. GPU topology features are optimized for DGX H100/A100 NVLink configurations. | [Platform](https://www.nvidia.com/en-us/data-center/dgx-platform/) |

## Contributing

Contributions welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest tests/ -v`)
4. Run lint (`ruff check src/ tests/`)
5. Submit a PR

See [GUIDE.md](GUIDE.md) for architecture details and development patterns.

## License

[Apache License 2.0](LICENSE)
