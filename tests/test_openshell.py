"""Tests for OpenShell integration modules."""

from unittest.mock import MagicMock, patch
from pathlib import Path

from nemospawn.openshell.prompt import build_system_prompt
from nemospawn.openshell.policy import generate_worker_policy, write_policy_file, ROLE_PRESETS
from nemospawn.openshell.sandbox import SandboxConfig


def test_build_system_prompt_standalone():
    prompt = build_system_prompt()
    assert "NemoSpawn agent" in prompt
    assert "NeMo Framework" in prompt
    assert "Triton" in prompt
    assert "Team: " not in prompt  # no team-specific injection without team_id


def test_build_system_prompt_with_coordination():
    prompt = build_system_prompt(
        team_id="my-team",
        agent_id="worker-0",
        gpu_ids=[0, 1],
        role="trainer",
        task_description="Fine-tune Llama-3",
    )
    assert "my-team" in prompt
    assert "worker-0" in prompt
    assert "0,1" in prompt
    assert "trainer" in prompt
    assert "Fine-tune Llama-3" in prompt


def test_generate_worker_policy_trainer():
    policy = generate_worker_policy(
        team_id="t1",
        agent_id="a1",
        role="trainer",
        gpu_ids=[0],
    )
    assert policy["version"] == "1"
    assert policy["metadata"]["name"] == "nemospawn-t1-a1"
    assert policy["filesystem"]["mode"] == "locked"
    assert policy["network"]["mode"] == "dynamic"
    assert policy["process"]["allow_privilege_escalation"] is False

    # Trainer should have NGC access
    network_dests = [r["destination"] for r in policy["network"]["rules"]]
    assert "*.ngc.nvidia.com" in network_dests


def test_generate_worker_policy_evaluator():
    policy = generate_worker_policy(
        team_id="t1",
        agent_id="eval0",
        role="evaluator",
    )
    network_dests = [r["destination"] for r in policy["network"]["rules"]]
    assert "localhost:8000" in network_dests  # Triton endpoint
    assert "*.ngc.nvidia.com" not in network_dests  # no NGC for evaluator


def test_generate_worker_policy_deployer():
    policy = generate_worker_policy(
        team_id="t1",
        agent_id="deploy0",
        role="deployer",
    )
    network_dests = [r["destination"] for r in policy["network"]["rules"]]
    assert "nvcr.io" in network_dests


def test_write_policy_file(tmp_path):
    policy = generate_worker_policy("t1", "a1", "worker")
    path = write_policy_file(policy, tmp_path / "policies" / "a1.yaml")
    assert path.exists()
    content = path.read_text()
    assert "nemospawn-t1-a1" in content


def test_role_presets_exist():
    expected_roles = ["trainer", "fine-tuner", "deployer", "evaluator", "data-curator", "rlhf-reward"]
    for role in expected_roles:
        assert role in ROLE_PRESETS
        assert "description" in ROLE_PRESETS[role]


def test_sandbox_config_defaults():
    config = SandboxConfig(name="test-sandbox")
    assert config.agent_command == "claude"
    assert config.gpu is False
    assert config.policy_file is None
    assert config.env_vars == {}


def test_sandbox_config_full():
    config = SandboxConfig(
        name="nemo-t1-w1",
        agent_command="opencode",
        gpu=True,
        policy_file="/tmp/policy.yaml",
        env_vars={"CUDA_VISIBLE_DEVICES": "0"},
        remote="user@dgx-pod",
    )
    assert config.gpu is True
    assert config.remote == "user@dgx-pod"
    assert config.agent_command == "opencode"


def test_check_openshell_not_installed():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        from nemospawn.openshell.sandbox import check_openshell_installed
        assert check_openshell_installed() is False


def test_list_sandboxes_not_installed():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        from nemospawn.openshell.sandbox import list_sandboxes
        assert list_sandboxes() == []
