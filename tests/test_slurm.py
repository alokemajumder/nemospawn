"""Tests for SLURM job script generation."""

from nemospawn.runtime.slurm import (
    SlurmJobConfig, generate_sbatch_script, write_sbatch_script,
)


def test_generate_sbatch_script_basic():
    config = SlurmJobConfig(
        job_name="nemo-train",
        gpu_count=4,
        command="python train.py",
    )
    script = generate_sbatch_script(config)
    assert "#!/bin/bash" in script
    assert "--job-name=nemo-train" in script
    assert "--gres=gpu:4" in script
    assert "python train.py" in script


def test_generate_sbatch_script_full():
    config = SlurmJobConfig(
        job_name="nemo-finetune",
        gpu_count=8,
        gpu_type="h100",
        num_nodes=2,
        partition="gpu",
        time_limit="48:00:00",
        modules=["cuda/12.4", "nccl"],
        env_vars={"NEMOSPAWN_TEAM": "t1", "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        command="nemospawn launch run autoresearch",
    )
    script = generate_sbatch_script(config)
    assert "--gres=gpu:h100:8" in script
    assert "--nodes=2" in script
    assert "--partition=gpu" in script
    assert "module load cuda/12.4" in script
    assert "module load nccl" in script
    assert 'NEMOSPAWN_TEAM="t1"' in script


def test_write_sbatch_script(tmp_path):
    config = SlurmJobConfig(job_name="test", gpu_count=1, command="echo hello")
    path = write_sbatch_script(config, tmp_path / "scripts" / "job.sh")
    assert path.exists()
    content = path.read_text()
    assert "echo hello" in content
