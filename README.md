<h1 align="center">
  NemoSpawn
</h1>

<p align="center">
  <strong>Orchestrate GPU-pinned AI agent swarms across NVIDIA DGX/HGX infrastructure</strong>
</p>

<p align="center">
  <a href="#30-second-demo">30s Demo</a> &bull;
  <a href="#what-it-does">What It Does</a> &bull;
  <a href="#install">Install</a> &bull;
  <a href="GUIDE.md">Full Guide</a> &bull;
  <a href="#nvidia-integrations">NVIDIA Integrations</a>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-201%20passing-brightgreen">
  <img alt="CLI Commands" src="https://img.shields.io/badge/CLI%20commands-24%20groups-8b5cf6">
  <img alt="NVIDIA Stack" src="https://img.shields.io/badge/NVIDIA%20stack-8%20integrations-76b900?logo=nvidia&logoColor=white">
</p>

---

You describe a research goal. NemoSpawn creates a team of AI agents, pins each to specific GPUs, wires up task dependencies, and lets the agents coordinate autonomously through messaging, plan approvals, and lifecycle management. You watch from a live dashboard. The result is a trained model, a deployed endpoint, or a completed experiment — built by agents working in parallel across your GPU cluster.

No database. No cloud service. Just a CLI and your GPUs.

## 30-Second Demo

```bash
pip install nemospawn

# One command: full autonomous research pipeline across 4 GPUs
nemospawn launch run autoresearch --gpus 0,1,2,3
```

That single command:
1. Discovers your GPU topology and NVLink islands
2. Creates a team with 2 trainers + 1 evaluator
3. Pins each agent to a GPU with `CUDA_VISIBLE_DEVICES`
4. Spawns agents in isolated tmux sessions
5. Sets up task dependencies (evaluate waits for training to finish)
6. Injects the coordination protocol so agents know how to report progress

```
                nemospawn launch run autoresearch --gpus 0,1,2,3
                                    |
                    ┌───────────────┼───────────────┐
                    v               v               v
              ┌──────────┐   ┌──────────┐   ┌──────────────┐
              │ trainer-0 │   │ trainer-1 │   │  evaluator   │
              │   GPU 0   │   │   GPU 1   │   │   GPU 2      │
              │  lr sweep │   │  lr sweep │   │  perf_analyzer│
              └─────┬─────┘   └─────┬─────┘   └──────┬───────┘
                    │               │       blocked_by │
                    └───────┬───────┘           ┌──────┘
                            v                   v
                    ┌──────────────────────────────┐
                    │  Task DAG auto-unblocks       │
                    │  Inbox messaging between agents│
                    │  Plan approval before actions  │
                    │  Web UI + Prometheus + Grafana  │
                    └──────────────────────────────┘
```

## What It Does

### The Problem

You have 8 H100s. You want to run a hyperparameter sweep, evaluate the best checkpoint, deploy it as a NIM endpoint, and benchmark latency — all in parallel where possible. Today you do this with bash scripts, tmux, and prayer.

### The Solution

NemoSpawn treats your GPU cluster as a programmable agent fabric:

```bash
# Step 1: See what you have
nemospawn gpu discover                    # List GPUs
nemospawn gpu topology                    # NVLink interconnect map

# Step 2: Create a team
nemospawn team create llama-sweep --gpus 0,1,2,3,4,5,6,7

# Step 3: Spawn agents (each gets its own GPU, tmux session, and worktree)
nemospawn spawn agent --team llama-sweep-abc --agent-name trainer0 \
  --role trainer --gpu 0,1 --task "LoRA sweep on LLaMA-70B" --agent-cmd claude

nemospawn spawn agent --team llama-sweep-abc --agent-name deployer \
  --role deployer --gpu 4 --task "Deploy best checkpoint as NIM TP4"

# Step 4: Wire up tasks with dependencies
nemospawn task create llama-sweep-abc "Train LoRA" --owner trainer0
nemospawn task create llama-sweep-abc "Deploy NIM" --owner deployer --blocked-by task-train

# Step 5: Watch everything happen
nemospawn board serve llama-sweep-abc     # Web dashboard at :8080
```

Each spawned agent receives an auto-injected coordination prompt that teaches it the full NemoSpawn protocol: how to check tasks, send messages, submit plans, report progress, and signal when it's done.

### What Agents Can Do

Once spawned, agents coordinate autonomously through the CLI:

```bash
# Inside the agent's tmux session, the agent runs these:
nemospawn task update $NEMOSPAWN_TEAM task-abc --status running
nemospawn inbox send $NEMOSPAWN_TEAM deployer "Best ckpt: epoch-42, val_loss=0.031"
nemospawn plan submit --team $NEMOSPAWN_TEAM --agent $NEMOSPAWN_AGENT \
  --title "Switch to cosine LR" --steps "Update config,Retrain,Eval"
nemospawn artifact register $NEMOSPAWN_TEAM ./model.nemo --type nemo-checkpoint --val-loss 0.031
nemospawn lifecycle idle --team $NEMOSPAWN_TEAM --agent $NEMOSPAWN_AGENT --reason "All tasks done"
```

## Install

```bash
pip install nemospawn                     # Core
pip install nemospawn[hpo]                # + Optuna HPO
pip install nemospawn[transport]          # + ZeroMQ cross-node messaging
pip install nemospawn[all]                # Everything
```

**Requires:** Python >= 3.10, [tmux](https://github.com/tmux/tmux)

**For GPU features:** NVIDIA drivers + nvidia-smi (CPU-only dev mode works without)

## Feature Overview

### 24 CLI Command Groups

<details>
<summary><strong>Core Orchestration</strong> (click to expand)</summary>

| Command | What it does |
|---------|-------------|
| `nemospawn team` | Create GPU-aware teams with NVLink topology discovery |
| `nemospawn spawn` | Spawn agents in tmux or [OpenShell](https://github.com/NVIDIA/openshell) sandboxes with GPU pinning |
| `nemospawn task` | Task DAG — `blocked_by` dependencies, auto-unblocking, val_loss tracking |
| `nemospawn inbox` | Agent-to-agent messaging — direct, broadcast, atomic JSON delivery |
| `nemospawn plan` | Plan approval — agents submit proposals for leader review before execution |
| `nemospawn lifecycle` | Graceful idle/shutdown request/approve/reject protocol |
</details>

<details>
<summary><strong>NVIDIA GPU & AI Stack</strong></summary>

| Command | What it does |
|---------|-------------|
| `nemospawn gpu` | GPU discovery, [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) topology, [DCGM](https://github.com/NVIDIA/DCGM) health monitoring |
| `nemospawn artifact` | [NeMo](https://github.com/NVIDIA/NeMo) artifact store — register, promote, val_loss ranking |
| `nemospawn nim` | [NIM](https://docs.nvidia.com/nim/) deployment — build containers, benchmark with [perf_analyzer](https://github.com/triton-inference-server/perf_analyzer), rank endpoints |
| `nemospawn ngc` | [NGC](https://catalog.ngc.nvidia.com/) registry — pull/push models and containers |
| `nemospawn hpo` | [Optuna](https://github.com/optuna/optuna) TPE + ASHA pruner for hyperparameter optimization |
</details>

<details>
<summary><strong>Infrastructure & Operations</strong></summary>

| Command | What it does |
|---------|-------------|
| `nemospawn launch` | One-command team launch from TOML templates (autoresearch, nim-deploy, rlhf-swarm, data-curation) |
| `nemospawn cluster` | Cross-cluster federation — SSH remote spawn on DGX/HGX nodes |
| `nemospawn slurm` | [SLURM](https://slurm.schedmd.com/) job script generation, submission, status, cancel |
| `nemospawn auth` | API key auth (SHA-256), multi-user namespaces, JSONL audit logging |
</details>

<details>
<summary><strong>Monitoring & Observability</strong></summary>

| Command | What it does |
|---------|-------------|
| `nemospawn board` | Web UI kanban (SSE real-time), terminal kanban (Rich), tiled tmux view |
| `nemospawn watch` | Agent health monitoring — dead tmux detection, stuck agent alerts |
| `nemospawn cost` | GPU-hour cost tracking per agent with configurable $/GPU-hour rates |
| `nemospawn schedule` | Adaptive scheduling — analyze GPU util, auto-reassign tasks from underperformers |
| `nemospawn snapshot` | Save and restore full team state |
| `nemospawn workspace` | Git worktree checkpoint, merge, cleanup per agent |
</details>

<details>
<summary><strong>Configuration</strong></summary>

| Command | What it does |
|---------|-------------|
| `nemospawn config` | Dynamic config — env var > config file > default (10 settings) |
| `nemospawn profile` | Agent CLI profiles — wizard, doctor, smoke test for 8 supported agents |
| `nemospawn skill` | Install coordination protocol as a discoverable skill for Claude Code / Codex |
| `nemospawn workspace` | Git worktree management — checkpoint, merge, cleanup per agent |
</details>

### 8 Supported Agent CLIs

| Agent | Auth | Prompt Injection |
|-------|------|-----------------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `ANTHROPIC_API_KEY` | CLI flag |
| [Codex](https://github.com/openai/codex) | `OPENAI_API_KEY` | CLI flag |
| [Kimi CLI](https://github.com/anthropics/kimi-cli) | `MOONSHOT_API_KEY` | CLI flag |
| [aider](https://github.com/paul-gauthier/aider) | `OPENAI_API_KEY` | CLI flag |
| [nanobot](https://github.com/nano-bot/nanobot) | — | CLI flag |
| Cursor | — | File |
| OpenCode | — | File |
| GitHub Copilot | `GITHUB_TOKEN` | File |

Plus `--agent-cmd custom` for any unlisted CLI tool. Create profiles with `nemospawn profile wizard`.

### 4 Built-in Templates

```bash
nemospawn launch run autoresearch --gpus 0-7    # HP sweep → train → eval loop
nemospawn launch run nim-deploy --gpus 0-3      # Checkpoint → NIM → benchmark → rank
nemospawn launch run rlhf-swarm --gpus 0-3      # SFT → reward model → PPO
nemospawn launch run data-curation --gpus 0,1   # Ingest → clean → deduplicate → validate
```

Write your own in TOML:

```toml
name = "my-pipeline"
min_gpus = 2

[[workers]]
name = "trainer"
role = "trainer"
gpu_count = 2
task = "Fine-tune LLaMA-70B with LoRA"
require_nvlink = true

[[workers]]
name = "evaluator"
role = "evaluator"
gpu_count = 1
task = "Run AlpacaEval"
blocked_by = ["trainer"]
```

<a name="nvidia-integrations"></a>
## NVIDIA Integrations

NemoSpawn directly calls 8 NVIDIA components and runs on 7 more:

### Direct (NemoSpawn calls these APIs/CLIs)

| Component | What NemoSpawn does with it |
|-----------|---------------------------|
| **[nvidia-smi / NVML](https://developer.nvidia.com/nvidia-management-library-nvml)** | `gpu discover` lists GPUs. `gpu topology` parses the NVLink interconnect matrix. `gpu health` reads temperature, utilization, power, and ECC errors via [pynvml](https://pypi.org/project/pynvml/) bindings. |
| **[NeMo Framework](https://github.com/NVIDIA/NeMo)** | Manages `.nemo` checkpoint bundles. Generates YAML config overrides with schema-aware type coercion. NVLink-aware scheduler places multi-GPU training on the same NVLink island for maximum interconnect bandwidth. |
| **[NIM](https://docs.nvidia.com/nim/)** | `nim deploy` builds inference containers from checkpoints with tensor parallel profiles (TP1-TP8). `nim benchmark` runs [perf_analyzer](https://github.com/triton-inference-server/perf_analyzer) and reports p50/p95/p99 latency. `nim list` tracks endpoint health. |
| **[Triton Inference Server](https://github.com/triton-inference-server/server)** | Auto-generates `config.pbtxt` model repository configs. Benchmarks endpoints via perf_analyzer with configurable concurrency levels. |
| **[DCGM](https://github.com/NVIDIA/DCGM)** | `gpu status` polls `dcgmi dmon` for SM utilization, memory, temperature, power, ECC errors. Exports to [Prometheus](https://prometheus.io/). Falls back to nvidia-smi gracefully. |
| **[NGC](https://catalog.ngc.nvidia.com/)** | `ngc pull/push` wraps the NGC CLI for model download/upload. `ngc push-container` pushes to `nvcr.io`. |
| **[NIXL](https://github.com/NVIDIA/nixl)** | Sub-microsecond inter-agent messaging over NVLink/InfiniBand. Auto-negotiated — falls back to ZeroMQ then file transport. |
| **[OpenShell](https://github.com/NVIDIA/openshell)** | `--runtime sandbox` spawns agents with kernel-level isolation (Landlock filesystem + seccomp syscalls + network namespaces). GPU passthrough via CUDA. Per-role security policies auto-generated. |

### Indirect (infrastructure NemoSpawn runs on)

| Component | How it's used |
|-----------|--------------|
| **[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)** | Topology parsed to detect GPU islands. Multi-GPU tasks placed on same island. Link types tracked: NV12, NV8, NV4, NV2. |
| **[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)** | Agents use `CUDA_VISIBLE_DEVICES` for GPU pinning. NeMo/NIM workloads run on CUDA. |
| **[NCCL](https://github.com/NVIDIA/nccl)** | Multi-GPU training uses NCCL for collective comms. NemoSpawn configures TP/PP degrees that NCCL implements. |
| **[cuDNN](https://developer.nvidia.com/cudnn)** | NeMo training uses cuDNN. NemoSpawn configures mixed-precision modes (bf16-mixed, 16-mixed, 32). |
| **[TensorRT](https://github.com/NVIDIA/TensorRT)** | NIM containers use TensorRT for inference. `--profile max-throughput` leverages TensorRT compilation. |
| **[Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** | Required for GPU access inside NIM Docker containers. |
| **[DGX / HGX](https://www.nvidia.com/en-us/data-center/dgx-platform/)** | Cross-cluster federation targets DGX/HGX nodes. Topology features optimized for DGX H100/A100. |

## Architecture

```
~/.nemospawn/                             All state — atomic JSON, no database
├── teams/{id}/
│   ├── team.json                         GPU list, NVLink topology, islands
│   ├── agents/{id}.json                  Status, GPUs, tmux session, lifecycle
│   ├── tasks/{id}.json                   DAG: blocked_by deps, val_loss, metadata
│   ├── plans/{id}.json                   Submit → pending → approved/rejected
│   ├── inbox/{agent}/                    Per-agent message files
│   ├── artifacts/{id}.json               .nemo checkpoints, NIM containers
│   ├── prompts/{id}.md                   Auto-injected coordination prompts
│   ├── snapshots/{id}.json               Point-in-time team state
│   ├── costs/cost_record.json            GPU-hour tracking
│   ├── workspaces/                       Git worktrees (one per agent)
│   └── metrics/                          DCGM snapshots
├── config.json                           Dynamic config (env > file > default)
└── audit.jsonl                           Structured audit log
```

**Transport negotiation** — messaging picks the fastest available backend automatically:

| Condition | Transport | Latency |
|-----------|-----------|---------|
| Same node + NVLink | [NIXL](https://github.com/NVIDIA/nixl) | Sub-microsecond |
| Cross-node | [ZeroMQ](https://zeromq.org/) | TCP |
| Fallback | File (atomic JSON) | Filesystem I/O |

**Coordination flow:**
1. Agent spawns with `NEMOSPAWN_TEAM`, `NEMOSPAWN_AGENT`, `CUDA_VISIBLE_DEVICES`
2. Coordination prompt auto-injected at `prompts/{agent_id}.md`
3. Agent checks tasks → works → updates status → messages peers → submits plans
4. Leader watches via `board serve` (web) or `board live` (terminal)
5. Agent reports `lifecycle idle` when done

## Development

```bash
git clone https://github.com/alokemajumder/nemospawn.git
cd nemospawn
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v           # 189 tests
ruff check src/ tests/     # Lint
```

<details>
<summary>Project layout</summary>

```
src/nemospawn/
├── cli/             23 Typer command groups
├── core/            State, models, auth, profiles, config, plan,
│                    lifecycle, costs, snapshot, watcher, adaptive, skill
├── gpu/             Discovery, NVLink topology, DCGM health
├── nemo/            Artifacts, config injection, NVLink-aware scheduling
├── nim/             NIM deployer, Triton benchmarks
├── openshell/       Sandbox integration, security policies, prompts
├── messaging/       File / ZeroMQ / NIXL transport
├── observability/   Prometheus, Grafana, kanban, web UI (SSE)
├── templates/       TOML templates, launch engine
├── federation/      Cross-cluster SSH spawn, git-annex
├── hpo/             Optuna TPE/ASHA, fallback sampler
├── ngc/             NGC model registry
└── runtime/         tmux, git worktree, SLURM
```
</details>

## Contributing

Contributions welcome:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest tests/ -v`) and lint (`ruff check src/ tests/`)
4. Submit a PR

See [GUIDE.md](GUIDE.md) for architecture details and development patterns.

## License

[Apache License 2.0](LICENSE)
