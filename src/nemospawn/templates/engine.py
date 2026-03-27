"""Template engine — loads TOML templates and launches teams from them.

Each template defines:
  - Team configuration (name, description, GPU requirements)
  - Worker definitions (role, GPU count, task description)
  - HPO search space (optional, in companion hpo.toml)
  - Dependency graph between workers
"""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkerSpec:
    """Specification for a worker agent from a template."""
    name: str
    role: str
    gpu_count: int = 1
    task: str = ""
    blocked_by: list[str] = field(default_factory=list)
    runtime: str = "tmux"
    agent_cmd: str = "claude"
    require_nvlink: bool = False


@dataclass
class TeamTemplate:
    """A parsed team template."""
    name: str
    description: str = ""
    min_gpus: int = 1
    workers: list[WorkerSpec] = field(default_factory=list)
    hpo_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def load_template(path: Path) -> TeamTemplate:
    """Load a team template from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    workers = []
    for w in data.get("workers", []):
        workers.append(WorkerSpec(
            name=w["name"],
            role=w.get("role", "worker"),
            gpu_count=w.get("gpu_count", 1),
            task=w.get("task", ""),
            blocked_by=w.get("blocked_by", []),
            runtime=w.get("runtime", "tmux"),
            agent_cmd=w.get("agent_cmd", "claude"),
            require_nvlink=w.get("require_nvlink", False),
        ))

    template = TeamTemplate(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        min_gpus=data.get("min_gpus", 1),
        workers=workers,
        metadata=data.get("metadata", {}),
    )

    # Load companion hpo.toml if exists
    hpo_path = path.parent / f"{path.stem}_hpo.toml"
    if not hpo_path.exists():
        hpo_path = path.parent / "hpo.toml"
    if hpo_path.exists():
        with open(hpo_path, "rb") as f:
            template.hpo_config = tomllib.load(f)

    return template


def load_template_from_string(toml_content: str) -> TeamTemplate:
    """Load a template from a TOML string (for built-in templates)."""
    data = tomllib.loads(toml_content)

    workers = []
    for w in data.get("workers", []):
        workers.append(WorkerSpec(
            name=w["name"],
            role=w.get("role", "worker"),
            gpu_count=w.get("gpu_count", 1),
            task=w.get("task", ""),
            blocked_by=w.get("blocked_by", []),
            runtime=w.get("runtime", "tmux"),
            require_nvlink=w.get("require_nvlink", False),
        ))

    return TeamTemplate(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        min_gpus=data.get("min_gpus", 1),
        workers=workers,
        hpo_config=data.get("hpo", {}),
        metadata=data.get("metadata", {}),
    )


# Built-in templates
BUILTIN_TEMPLATES: dict[str, str] = {
    "autoresearch": '''
name = "autoresearch"
description = "Autonomous NeMo hyperparameter research — leader orchestrates sub-agents across GPUs"
min_gpus = 3

[[workers]]
name = "orchestrator"
role = "leader"
gpu_count = 0
task = "Orchestrate the research team: spawn trainers on available GPUs, design experiments with varied hyperparameters, monitor GPU performance via nemospawn schedule analyze, detect underperformers and reallocate tasks, review worker plans, kill idle agents and respawn with new parameters, synthesize findings across all agents into final results"

[[workers]]
name = "trainer-0"
role = "trainer"
gpu_count = 1
task = "NeMo training sweep — explore hyperparameter space, report val_loss at every checkpoint"

[[workers]]
name = "trainer-1"
role = "trainer"
gpu_count = 1
task = "NeMo training sweep — explore hyperparameter space, report val_loss at every checkpoint"

[[workers]]
name = "evaluator"
role = "evaluator"
gpu_count = 1
task = "Benchmark best checkpoint via Triton perf_analyzer, register artifacts"
blocked_by = ["trainer-0", "trainer-1"]

[metadata]
category = "research"
nvidia_stack = ["nemo", "triton"]
''',

    "nim-deploy": '''
name = "nim-deploy"
description = "Deploy and benchmark NIM container variants — leader coordinates pipeline"
min_gpus = 3

[[workers]]
name = "orchestrator"
role = "leader"
gpu_count = 0
task = "Orchestrate the NIM deployment pipeline: monitor deployers, review plans, benchmark results, rank endpoints by throughput and latency, synthesize deployment report"

[[workers]]
name = "deployer-tp1"
role = "deployer"
gpu_count = 1
task = "Build and deploy NIM container with TP1"

[[workers]]
name = "deployer-tp2"
role = "deployer"
gpu_count = 2
task = "Build and deploy NIM container with TP2"
require_nvlink = true

[[workers]]
name = "benchmarker"
role = "evaluator"
gpu_count = 1
task = "Run perf_analyzer on all NIM endpoints and rank by throughput@p95"
blocked_by = ["deployer-tp1", "deployer-tp2"]

[metadata]
category = "deployment"
nvidia_stack = ["nim", "triton"]
''',

    "rlhf-swarm": '''
name = "rlhf-swarm"
description = "Full RLHF loop with leader orchestrating reward, PPO, and eval agents"
min_gpus = 4

[[workers]]
name = "orchestrator"
role = "leader"
gpu_count = 0
task = "Orchestrate the RLHF pipeline: monitor reward training completion, spawn PPO agents, track policy improvement, detect training instabilities via GPU metrics, reallocate underperforming agents, synthesize alignment results"

[[workers]]
name = "reward-trainer"
role = "rlhf-reward"
gpu_count = 2
task = "Train reward model on labeled preference dataset"
require_nvlink = true

[[workers]]
name = "ppo-agent-0"
role = "trainer"
gpu_count = 1
task = "PPO policy optimization using reward model"
blocked_by = ["reward-trainer"]

[[workers]]
name = "ppo-agent-1"
role = "trainer"
gpu_count = 1
task = "PPO policy optimization using reward model"
blocked_by = ["reward-trainer"]

[[workers]]
name = "eval-agent"
role = "evaluator"
gpu_count = 1
task = "Run AlpacaEval on each PPO checkpoint"
blocked_by = ["ppo-agent-0", "ppo-agent-1"]

[metadata]
category = "alignment"
nvidia_stack = ["nemo", "nemo-aligner"]
''',

    "data-curation": '''
name = "data-curation"
description = "Parallel data curation + training with leader coordination"
min_gpus = 2

[[workers]]
name = "orchestrator"
role = "leader"
gpu_count = 0
task = "Orchestrate data pipeline: monitor curation progress, spawn trainers when shards ready, track data quality metrics, synthesize results"

[[workers]]
name = "curator"
role = "data-curator"
gpu_count = 1
task = "Deduplicate, quality filter, and tokenize raw data with NeMo Data Curator"

[[workers]]
name = "trainer"
role = "trainer"
gpu_count = 1
task = "Start training on available data shards, unblock as new shards arrive"
blocked_by = ["curator"]

[metadata]
category = "data"
nvidia_stack = ["nemo", "cudf"]
''',
}


def get_builtin_template(name: str) -> TeamTemplate | None:
    """Get a built-in template by name."""
    toml_str = BUILTIN_TEMPLATES.get(name)
    if toml_str:
        return load_template_from_string(toml_str)
    return None


def list_builtin_templates() -> list[dict]:
    """List all built-in templates with their descriptions."""
    templates = []
    for name, toml_str in BUILTIN_TEMPLATES.items():
        template = load_template_from_string(toml_str)
        templates.append({
            "name": name,
            "description": template.description,
            "min_gpus": template.min_gpus,
            "workers": len(template.workers),
        })
    return templates
