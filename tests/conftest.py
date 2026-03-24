"""Shared test fixtures."""

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def state_dir(tmp_path):
    """Patch STATE_DIR and TEAMS_DIR to use a temp directory."""
    state = tmp_path / ".nemospawn"
    teams = state / "teams"
    hpo = state / "hpo"
    state.mkdir()
    teams.mkdir()
    hpo.mkdir()

    with patch("nemospawn.core.config.STATE_DIR", state), \
         patch("nemospawn.core.config.TEAMS_DIR", teams), \
         patch("nemospawn.core.config.HPO_DIR", hpo), \
         patch("nemospawn.core.state.STATE_DIR", state), \
         patch("nemospawn.core.state.TEAMS_DIR", teams), \
         patch("nemospawn.core.state.HPO_DIR", hpo):
        yield state


@pytest.fixture
def mock_nvidia_smi_csv():
    """Mock nvidia-smi CSV output."""
    return (
        "0, NVIDIA H100 80GB HBM3, GPU-abc123, 81920, 1024\n"
        "1, NVIDIA H100 80GB HBM3, GPU-def456, 81920, 2048\n"
        "2, NVIDIA H100 80GB HBM3, GPU-ghi789, 81920, 512\n"
        "3, NVIDIA H100 80GB HBM3, GPU-jkl012, 81920, 4096\n"
    )


@pytest.fixture
def mock_nvidia_smi_topo():
    """Mock nvidia-smi topo -m output."""
    return (
        "\tGPU0\tGPU1\tGPU2\tGPU3\n"
        "GPU0\t X \tNV12\tSYS\tSYS\n"
        "GPU1\tNV12\t X \tSYS\tSYS\n"
        "GPU2\tSYS\tSYS\t X \tNV12\n"
        "GPU3\tSYS\tSYS\tNV12\t X \n"
        "\n"
        "Legend:\n"
        "  X    = Self\n"
        "  NV12 = NVLink 12\n"
        "  SYS  = System\n"
    )
