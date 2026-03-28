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

You describe a research goal. NemoSpawn spawns an intelligent leader agent that autonomously orchestrates specialized sub-agents across your GPUs — designing experiments, monitoring performance, reallocating resources, and synthesizing results. No human intervention after launch.

No database. No cloud service. Just a CLI and your GPUs.

## 30-Second Demo

```bash
pip install nemospawn

# One command: autonomous research across 8 GPUs
nemospawn launch run autoresearch --gpus 0,1,2,3,4,5,6,7
```

That single command:
1. Discovers your GPU topology and NVLink islands
2. Spawns an **AI leader agent** that orchestrates the entire team
3. Leader spawns **trainers** on available GPUs and an **evaluator**
4. Wires up task dependencies (evaluate waits for training)
5. Leader monitors GPU performance, kills underperformers, respawns with new hyperparameters
6. All agents get the full coordination protocol injected automatically

```
            nemospawn launch run autoresearch --gpus 0,1,2,3,4,5,6,7
                                    |
                          ┌─────────▼─────────┐
                          │   orchestrator     │
                          │   (AI leader)      │
                          │   spawns, monitors,│
                          │   reallocates      │
                          └──┬──┬──┬──┬──┬──┬──┘
                             │  │  │  │  │  │
               ┌─────────┐ ┌┘  │  │  │  │  └┐ ┌──────────┐
               │trainer-0 │ │   │  │  │  │   │ │evaluator │
               │  GPU 0   │ │   │  │  │  │   │ │  GPU 7   │
               └──────────┘ │   │  │  │  │   │ └──────────┘
                     ┌──────┘   │  │  │  └───┘
                     │trainer-1 │  │  │ trainer-5│
                     │  GPU 1   │  │  │  GPU 5  │
                     └──────────┘  │  └─────────┘
                          ...  GPUs 2-4  ...

         ┌─────────────────────────────────────────────┐
         │  Leader autonomously:                        │
         │  - Designs experiments with varied HP         │
         │  - Monitors GPU util via DCGM                │
         │  - Kills underperformers, respawns fresh     │
         │  - Reviews worker plans before execution     │
         │  - Merges results, synthesizes findings      │
         └─────────────────────────────────────────────┘
```

## What It Does

### The Problem

You have 8 H100s. You want to run a hyperparameter sweep, evaluate the best checkpoint, deploy it as a NIM endpoint, and benchmark latency — all in parallel where possible. Today you do this with bash scripts, tmux, and prayer.

### The Solution

NemoSpawn treats your GPU cluster as a programmable agent fabric. An AI leader agent orchestrates everything:

```bash
# Step 1: See what you have
nemospawn gpu discover                    # List GPUs
nemospawn gpu topology                    # NVLink interconnect map

# Step 2: Launch autonomous research (leader + workers)
nemospawn launch run autoresearch --gpus 0,1,2,3,4,5,6,7

# Step 3: Watch from the dashboard
nemospawn board serve my-team-abc         # Web dashboard at :8080
```

Or build manually with full control:

```bash
# Create team and spawn a leader agent
nemospawn team create llama-sweep --gpus 0,1,2,3,4,5,6,7
nemospawn spawn agent --team llama-sweep-abc --agent-name orchestrator \
  --role leader --task "Design HP sweep, spawn trainers, monitor, reallocate"

# Leader agent then autonomously spawns workers, creates tasks, monitors...
```

### How Agents Self-Organize

The leader agent receives a 10-step autonomous orchestration protocol:

1. **Discover GPUs** — `nemospawn gpu discover`
2. **Spawn worker agents** on available GPUs with specialized tasks
3. **Create task dependencies** — training before evaluation, evaluation before deployment
4. **Monitor performance** — `nemospawn schedule analyze` checks GPU utilization per agent
5. **Review worker plans** — approve or reject before major experiments
6. **Detect underperformers** — `nemospawn schedule suggest` finds low-utilization agents
7. **Kill idle agents** — `nemospawn spawn kill` frees GPUs
8. **Respawn with new parameters** — fresh agents with updated hyperparameters
9. **Merge results** — `nemospawn workspace merge` combines agent branches
10. **Synthesize findings** — report final results

Workers coordinate autonomously through the CLI:

```bash
# Workers run these from their tmux sessions:
nemospawn task update $NEMOSPAWN_TEAM task-abc --status running
nemospawn inbox send $NEMOSPAWN_TEAM leader "val_loss=0.031 at epoch 42"
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

### 4 Built-in Templates (all include AI leader)

Every template spawns an **orchestrator agent** (`role=leader`) that autonomously manages the team:

```bash
nemospawn launch run autoresearch --gpus 0-7    # Leader + trainers + evaluator
nemospawn launch run nim-deploy --gpus 0-3      # Leader + deployers + benchmarker
nemospawn launch run rlhf-swarm --gpus 0-3      # Leader + reward + PPO + eval
nemospawn launch run data-curation --gpus 0,1   # Leader + curator + trainer
```

Write your own in TOML:

```toml
name = "my-pipeline"
min_gpus = 3

[[workers]]
name = "orchestrator"
role = "leader"
gpu_count = 0
task = "Orchestrate team: spawn workers, monitor GPU perf, reallocate, synthesize results"

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

**Autonomous coordination flow:**
1. `nemospawn launch run autoresearch` spawns leader + workers
2. Each agent gets `NEMOSPAWN_TEAM`, `NEMOSPAWN_AGENT`, `CUDA_VISIBLE_DEVICES`, `PATH`
3. Coordination prompt auto-injected — leader gets 10-step orchestration protocol
4. Leader spawns additional agents, creates tasks, monitors GPU performance
5. Workers train, report val_loss, submit plans, send messages
6. Leader detects underperformers, kills idle agents, respawns with new params
7. Workers report `lifecycle idle` when done
8. Leader merges results and synthesizes findings

## Development

```bash
git clone https://github.com/alokemajumder/nemospawn.git
cd nemospawn
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v           # 201 tests
ruff check src/ tests/     # Lint
```

<details>
<summary>Project layout</summary>

```
src/nemospawn/
├── cli/             24 Typer command groups
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
