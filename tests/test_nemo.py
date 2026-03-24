"""Tests for NeMo integration — artifacts, config, scheduler."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nemospawn.nemo.artifacts import (
    Artifact, register_artifact, promote_artifact, list_artifacts,
    get_promoted_artifact, ARTIFACT_TYPES,
)
from nemospawn.nemo.config import (
    generate_nemo_config, parse_overrides, write_nemo_config,
)
from nemospawn.nemo.scheduler import (
    find_available_gpus, recommend_parallelism, GPUAllocation,
)
from nemospawn.core.models import Team


# --- Artifact tests ---

def test_register_artifact(state_dir, tmp_path):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.nemo.artifacts.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("t1")

        # Create a dummy checkpoint file
        ckpt = tmp_path / "model.nemo"
        ckpt.write_text("fake checkpoint")

        art = register_artifact("t1", str(ckpt), artifact_type="nemo-checkpoint", val_loss=0.043)
        assert art.artifact_id.startswith("art-")
        assert art.val_loss == 0.043
        assert art.artifact_type == "nemo-checkpoint"
        assert art.promoted is False


def test_register_artifact_bad_path(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.nemo.artifacts.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("t1")

        with pytest.raises(FileNotFoundError):
            register_artifact("t1", "/nonexistent/path", artifact_type="nemo-checkpoint")


def test_register_artifact_bad_type(state_dir, tmp_path):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.nemo.artifacts.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("t1")

        ckpt = tmp_path / "model.nemo"
        ckpt.write_text("fake")
        with pytest.raises(ValueError):
            register_artifact("t1", str(ckpt), artifact_type="invalid-type")


def test_promote_artifact(state_dir, tmp_path):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.nemo.artifacts.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("t1")

        ckpt1 = tmp_path / "ckpt1.nemo"
        ckpt1.write_text("ckpt1")
        ckpt2 = tmp_path / "ckpt2.nemo"
        ckpt2.write_text("ckpt2")

        art1 = register_artifact("t1", str(ckpt1), val_loss=0.5)
        art2 = register_artifact("t1", str(ckpt2), val_loss=0.3)

        # Promote art1
        promoted = promote_artifact("t1", art1.artifact_id)
        assert promoted.promoted is True

        # Promote art2 — should demote art1
        promoted = promote_artifact("t1", art2.artifact_id)
        assert promoted.promoted is True

        best = get_promoted_artifact("t1")
        assert best is not None
        assert best.artifact_id == art2.artifact_id


def test_list_artifacts_sorted(state_dir, tmp_path):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.core.state.TEAMS_DIR", teams_dir), \
         patch("nemospawn.nemo.artifacts.get_team_dir", lambda tid: teams_dir / tid):
        from nemospawn.core.state import ensure_team_dir
        ensure_team_dir("t1")

        for i, vl in enumerate([0.5, 0.1, 0.3]):
            f = tmp_path / f"ckpt{i}.nemo"
            f.write_text(f"ckpt{i}")
            register_artifact("t1", str(f), val_loss=vl)

        arts = list_artifacts("t1", sort_by="val_loss")
        assert len(arts) == 3
        assert arts[0].val_loss == 0.1  # best first
        assert arts[2].val_loss == 0.5


def test_artifact_types():
    assert "nemo-checkpoint" in ARTIFACT_TYPES
    assert "nim-container" in ARTIFACT_TYPES
    assert "dataset" in ARTIFACT_TYPES


# --- Config tests ---

def test_generate_nemo_config_basic():
    config = generate_nemo_config(lr=2e-4, batch_size=8, max_steps=1000)
    assert config["optim"]["lr"] == 2e-4
    assert config["model"]["micro_batch_size"] == 8
    assert config["trainer"]["max_steps"] == 1000
    assert config["optim"]["name"] == "adamw"
    assert config["optim"]["sched"]["name"] == "cosine"


def test_generate_nemo_config_multi_gpu():
    config = generate_nemo_config(num_gpus=4, tp_size=4, precision="bf16-mixed")
    assert config["trainer"]["devices"] == 4
    assert config["model"]["tensor_model_parallel_size"] == 4
    assert config["trainer"]["precision"] == "bf16-mixed"


def test_parse_overrides():
    overrides = parse_overrides({
        "optim.lr": "1e-4",
        "trainer.max_steps": "5000",
        "model.micro_batch_size": "16",
    })
    assert overrides["optim"]["lr"] == 1e-4
    assert overrides["trainer"]["max_steps"] == 5000
    assert overrides["model"]["micro_batch_size"] == 16


def test_write_nemo_config(tmp_path):
    config = generate_nemo_config(lr=3e-4, batch_size=4, max_steps=500)
    path = write_nemo_config(config, tmp_path / "nemo_config.yaml")
    assert path.exists()
    content = path.read_text()
    assert "lr:" in content
    assert "adamw" in content


# --- Scheduler tests ---

def test_find_available_gpus_single():
    team = Team(team_id="t1", name="test", gpu_ids=[0, 1, 2, 3], nvlink_islands=[[0, 1], [2, 3]])
    with patch("nemospawn.nemo.scheduler.get_allocated_gpus", return_value=set()):
        alloc = find_available_gpus(team, num_gpus=1)
        assert alloc is not None
        assert len(alloc.gpu_ids) == 1


def test_find_available_gpus_same_island():
    team = Team(team_id="t1", name="test", gpu_ids=[0, 1, 2, 3], nvlink_islands=[[0, 1], [2, 3]])
    with patch("nemospawn.nemo.scheduler.get_allocated_gpus", return_value={0}):
        alloc = find_available_gpus(team, num_gpus=2, require_same_island=True)
        assert alloc is not None
        assert len(alloc.gpu_ids) == 2
        # Should pick from island [2, 3] since GPU 0 is allocated
        assert alloc.gpu_ids == [2, 3]


def test_find_available_gpus_not_enough():
    team = Team(team_id="t1", name="test", gpu_ids=[0, 1], nvlink_islands=[[0, 1]])
    with patch("nemospawn.nemo.scheduler.get_allocated_gpus", return_value={0, 1}):
        alloc = find_available_gpus(team, num_gpus=1)
        assert alloc is None


def test_recommend_parallelism_single_gpu():
    rec = recommend_parallelism(1)
    assert rec["tp_size"] == 1
    assert rec["pp_size"] == 1


def test_recommend_parallelism_large_model():
    rec = recommend_parallelism(8, model_size_b=70.0)
    assert rec["tp_size"] == 4
    assert rec["pp_size"] == 2


def test_recommend_parallelism_small_model_4gpu():
    rec = recommend_parallelism(4, model_size_b=8.0)
    assert rec["tp_size"] == 4
    assert rec["pp_size"] == 1
