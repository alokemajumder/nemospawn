"""OpenShell policy generation for NemoSpawn workers.

Generates declarative YAML policies that control what each NemoSpawn
worker agent can access — filesystem paths, network endpoints, syscalls,
and inference routing. Policies use OpenShell's four protection layers:
  - Filesystem: Landlock-enforced read/write paths
  - Network: Layer 7 HTTP method/path egress rules
  - Process: seccomp syscall allow/deny lists
  - Inference: LLM API routing rules
"""

from __future__ import annotations

import yaml
from pathlib import Path


def generate_worker_policy(
    team_id: str,
    agent_id: str,
    role: str = "worker",
    gpu_ids: list[int] | None = None,
    extra_network_allow: list[str] | None = None,
) -> dict:
    """Generate an OpenShell policy for a NemoSpawn worker.

    Args:
        team_id: NemoSpawn team ID.
        agent_id: Agent ID for this worker.
        role: Worker role (trainer, evaluator, deployer, etc.)
        gpu_ids: GPU indices this worker can access.
        extra_network_allow: Additional network endpoints to allow.

    Returns:
        Policy dict suitable for YAML serialization.
    """
    # Base filesystem access — all workers get sandbox + nemospawn state
    filesystem_rules = [
        {"path": "/sandbox", "access": "read-write"},
        {"path": "/tmp", "access": "read-write"},
        {"path": f"/root/.nemospawn/teams/{team_id}", "access": "read-write"},
    ]

    # Network rules — base NemoSpawn coordination + role-specific
    network_rules = [
        # Allow NemoSpawn coordination (local)
        {"destination": "localhost:*", "action": "allow", "comment": "NemoSpawn local coordination"},
    ]

    # Role-specific network access
    if role in ("trainer", "fine-tuner", "data-curator"):
        network_rules.extend([
            {"destination": "*.ngc.nvidia.com", "action": "allow", "comment": "NGC model registry"},
            {"destination": "huggingface.co", "action": "allow", "comment": "HuggingFace datasets"},
        ])
    elif role in ("deployer", "nim-deployer"):
        network_rules.extend([
            {"destination": "*.ngc.nvidia.com", "action": "allow", "comment": "NGC container registry"},
            {"destination": "nvcr.io", "action": "allow", "comment": "NVIDIA container registry"},
        ])
    elif role in ("evaluator", "triton-evaluator"):
        network_rules.extend([
            {"destination": "localhost:8000", "action": "allow", "comment": "Triton endpoint"},
            {"destination": "localhost:8001", "action": "allow", "comment": "Triton gRPC"},
        ])

    if extra_network_allow:
        for endpoint in extra_network_allow:
            network_rules.append({"destination": endpoint, "action": "allow"})

    # Block everything else
    network_rules.append({"destination": "*", "action": "deny", "comment": "deny all other egress"})

    # Inference routing — route through OpenShell privacy router
    inference_rules = {
        "default_provider": "nvidia-endpoint",
        "routes": [
            {"pattern": "https://inference.local/*", "provider": "managed", "comment": "OpenShell managed inference"},
        ],
    }

    policy = {
        "version": "1",
        "metadata": {
            "name": f"nemospawn-{team_id}-{agent_id}",
            "description": f"NemoSpawn worker policy for {role} agent {agent_id}",
        },
        "filesystem": {
            "rules": filesystem_rules,
            "mode": "locked",  # locked at sandbox creation
        },
        "network": {
            "rules": network_rules,
            "mode": "dynamic",  # hot-reloadable at runtime
        },
        "process": {
            "allow_privilege_escalation": False,
            "mode": "locked",
        },
        "inference": inference_rules,
    }

    return policy


def write_policy_file(policy: dict, output_path: Path) -> Path:
    """Write a policy dict to a YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(policy, f, default_flow_style=False, sort_keys=False)
    return output_path


# Default policy presets for common NemoSpawn roles
ROLE_PRESETS = {
    "trainer": {
        "description": "NeMo training agent — NGC access, GPU, checkpoint storage",
        "extra_network": ["*.ngc.nvidia.com", "huggingface.co"],
    },
    "fine-tuner": {
        "description": "NeMo fine-tuning agent — SFT/LoRA/PEFT with NGC access",
        "extra_network": ["*.ngc.nvidia.com"],
    },
    "deployer": {
        "description": "NIM deployment agent — container registry access",
        "extra_network": ["*.ngc.nvidia.com", "nvcr.io"],
    },
    "evaluator": {
        "description": "Triton benchmark agent — local inference endpoints only",
        "extra_network": [],
    },
    "data-curator": {
        "description": "NeMo Data Curator agent — dataset access",
        "extra_network": ["*.ngc.nvidia.com", "huggingface.co"],
    },
    "rlhf-reward": {
        "description": "RLHF reward model agent — local GPU only",
        "extra_network": [],
    },
}
