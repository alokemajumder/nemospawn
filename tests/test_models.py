"""Tests for data models."""

from nemospawn.core.models import (
    GPUInfo, Team, Agent, Task, Message, resolve_ready_tasks, _short_id
)


def test_gpu_info_roundtrip():
    gpu = GPUInfo(index=0, name="H100", uuid="GPU-abc", memory_total_mb=81920, memory_used_mb=1024)
    d = gpu.to_dict()
    gpu2 = GPUInfo.from_dict(d)
    assert gpu2.index == 0
    assert gpu2.name == "H100"
    assert gpu2.memory_total_mb == 81920


def test_team_roundtrip():
    team = Team(team_id="test-abc", name="test", gpu_ids=[0, 1])
    d = team.to_dict()
    team2 = Team.from_dict(d)
    assert team2.team_id == "test-abc"
    assert team2.gpu_ids == [0, 1]


def test_agent_roundtrip():
    agent = Agent(agent_id="worker-abc", team_id="t1", name="worker0", gpu_ids=[0])
    d = agent.to_dict()
    agent2 = Agent.from_dict(d)
    assert agent2.agent_id == "worker-abc"
    assert agent2.gpu_ids == [0]


def test_task_roundtrip():
    task = Task(task_id="task-1", team_id="t1", title="Train model", blocked_by=["task-0"])
    d = task.to_dict()
    task2 = Task.from_dict(d)
    assert task2.blocked_by == ["task-0"]


def test_message_roundtrip():
    msg = Message(msg_id="m1", team_id="t1", from_agent="a1", body="hello", to_agent="a2")
    d = msg.to_dict()
    msg2 = Message.from_dict(d)
    assert msg2.body == "hello"
    assert msg2.to_agent == "a2"


def test_resolve_ready_tasks():
    tasks = [
        Task(task_id="t1", team_id="x", title="A", status="done"),
        Task(task_id="t2", team_id="x", title="B", status="blocked", blocked_by=["t1"]),
        Task(task_id="t3", team_id="x", title="C", status="blocked", blocked_by=["t1", "t4"]),
        Task(task_id="t4", team_id="x", title="D", status="running"),
    ]
    ready = resolve_ready_tasks(tasks)
    assert len(ready) == 1
    assert ready[0].task_id == "t2"


def test_resolve_ready_tasks_no_deps():
    tasks = [
        Task(task_id="t1", team_id="x", title="A", status="pending"),
    ]
    ready = resolve_ready_tasks(tasks)
    assert len(ready) == 1


def test_short_id():
    sid = _short_id("test")
    assert sid.startswith("test-")
    assert len(sid) == 13  # "test-" + 8 hex chars
