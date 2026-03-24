"""Tests for coordination prompt injection (tmux + sandbox)."""

from nemospawn.openshell.prompt import build_system_prompt, COORDINATION_INJECTION


def test_build_system_prompt_includes_coordination():
    prompt = build_system_prompt(
        team_id="team-abc",
        agent_id="agent-1",
        gpu_ids=[0, 1],
        role="trainer",
        task_description="Fine-tune LLaMA",
    )
    assert "team-abc" in prompt
    assert "agent-1" in prompt
    assert "0,1" in prompt
    assert "trainer" in prompt
    assert "Fine-tune LLaMA" in prompt


def test_coordination_includes_plan_commands():
    prompt = build_system_prompt(
        team_id="t1", agent_id="a1", gpu_ids=[0], role="worker",
    )
    assert "plan submit" in prompt
    assert "plan list" in prompt


def test_coordination_includes_lifecycle_commands():
    prompt = build_system_prompt(
        team_id="t1", agent_id="a1", gpu_ids=[0], role="worker",
    )
    assert "lifecycle idle" in prompt
    assert "lifecycle shutdown-request" in prompt


def test_coordination_includes_task_commands():
    prompt = build_system_prompt(
        team_id="t1", agent_id="a1", gpu_ids=[0], role="worker",
    )
    assert "task update" in prompt
    assert "task list" in prompt


def test_coordination_includes_messaging():
    prompt = build_system_prompt(
        team_id="t1", agent_id="a1", gpu_ids=[0], role="worker",
    )
    assert "inbox send" in prompt
    assert "inbox receive" in prompt


def test_build_system_prompt_without_context():
    prompt = build_system_prompt()
    # Should still have base system prompt but no coordination injection
    assert "NemoSpawn worker agent" in prompt
    assert "Plan Approval" not in prompt
