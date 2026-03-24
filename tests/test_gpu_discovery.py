"""Tests for GPU discovery."""

from unittest.mock import patch, MagicMock
from nemospawn.gpu.discovery import discover_gpus


def test_discover_gpus_success(mock_nvidia_smi_csv):
    result = MagicMock()
    result.returncode = 0
    result.stdout = mock_nvidia_smi_csv

    with patch("subprocess.run", return_value=result):
        gpus = discover_gpus()

    assert len(gpus) == 4
    assert gpus[0].index == 0
    assert gpus[0].name == "NVIDIA H100 80GB HBM3"
    assert gpus[0].memory_total_mb == 81920
    assert gpus[1].memory_used_mb == 2048


def test_discover_gpus_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        gpus = discover_gpus()
    assert gpus == []


def test_discover_gpus_timeout():
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
        gpus = discover_gpus()
    assert gpus == []
