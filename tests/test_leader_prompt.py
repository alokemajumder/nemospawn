"""Tests for AI leader pattern — leader vs worker prompt injection."""

from nemospawn.openshell.prompt import build_system_prompt, LEADER_INJECTION


def test_worker_prompt_has_spawn_commands():
    """Workers should know about spawn commands in the system prompt."""
    prompt = build_system_prompt(
        team_id="t1", agent_id="w1", gpu_ids=[0], role="worker",
        task_description="Train model",
    )
    # Workers get the system prompt which includes spawn info
    assert "nemospawn spawn agent" in prompt
    # But workers do NOT get the leader injection
    assert "Leader Commands" not in prompt
    assert "Leader Protocol" not in prompt


def test_leader_prompt_includes_leader_injection():
    """Leaders get full orchestration commands."""
    prompt = build_system_prompt(
        team_id="t1", agent_id="leader1", gpu_ids=[0], role="leader",
        task_description="Orchestrate research",
    )
    assert "Leader Commands" in prompt
    assert "Leader Protocol" in prompt
    assert "nemospawn spawn agent --team t1" in prompt
    assert "nemospawn spawn kill --team t1" in prompt
    assert "nemospawn plan approve" in prompt
    assert "nemospawn plan reject" in prompt
    assert "nemospawn schedule analyze" in prompt
    assert "nemospawn lifecycle shutdown-approve" in prompt
    assert "nemospawn workspace merge" in prompt


def test_leader_prompt_has_monitoring_commands():
    prompt = build_system_prompt(
        team_id="t1", agent_id="leader1", gpu_ids=[0], role="leader",
    )
    assert "nemospawn board live t1" in prompt
    assert "nemospawn watch status --team t1" in prompt
    assert "nemospawn cost show --team t1" in prompt


def test_leader_prompt_has_10_step_protocol():
    prompt = build_system_prompt(
        team_id="t1", agent_id="leader1", gpu_ids=[0], role="leader",
    )
    assert "Check available GPUs" in prompt
    assert "Spawn worker agents" in prompt
    assert "Detect underperformers" in prompt
    assert "Kill idle/underperforming agents" in prompt
    assert "Respawn with new parameters" in prompt
    assert "Synthesize findings" in prompt


def test_worker_prompt_has_workspace_commands():
    prompt = build_system_prompt(
        team_id="t1", agent_id="w1", gpu_ids=[0], role="worker",
    )
    assert "nemospawn workspace checkpoint" in prompt


def test_all_prompts_have_task_and_inbox():
    for role in ["worker", "trainer", "evaluator", "leader"]:
        prompt = build_system_prompt(
            team_id="t1", agent_id="a1", gpu_ids=[0], role=role,
        )
        assert "nemospawn task list t1" in prompt
        assert "nemospawn inbox send t1" in prompt
        assert "nemospawn inbox receive t1" in prompt


def test_prompt_without_context_has_no_leader_injection():
    prompt = build_system_prompt()
    assert "Leader Commands" not in prompt
    assert "NemoSpawn agent" in prompt
