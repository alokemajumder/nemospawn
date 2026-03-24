"""Tests for DCGM GPU monitoring."""

from unittest.mock import patch, MagicMock

from nemospawn.gpu.dcgm import (
    poll_dcgm, detect_underperforming_gpus, save_metrics_snapshot,
    get_gpu_utilization_history, _parse_dcgmi_output,
)


def test_poll_dcgm_fallback_no_dcgmi():
    """When dcgmi is not installed, falls back to nvidia-smi."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = poll_dcgm()
        assert result == []


def test_detect_underperforming_gpus():
    metrics = [
        {"gpu_id": 0, "sm_util": 95, "temp": 72, "ecc_dbe": 0},
        {"gpu_id": 1, "sm_util": 30, "temp": 72, "ecc_dbe": 0},  # low util
        {"gpu_id": 2, "sm_util": 80, "temp": 92, "ecc_dbe": 0},  # high temp
        {"gpu_id": 3, "sm_util": 90, "temp": 70, "ecc_dbe": 2},  # ECC errors
    ]
    problems = detect_underperforming_gpus(metrics, util_threshold=50, temp_threshold=90)
    assert len(problems) == 3

    problem_ids = {p["gpu_id"] for p in problems}
    assert 1 in problem_ids  # low util
    assert 2 in problem_ids  # high temp
    assert 3 in problem_ids  # ECC


def test_detect_no_problems():
    metrics = [
        {"gpu_id": 0, "sm_util": 95, "temp": 72, "ecc_dbe": 0},
        {"gpu_id": 1, "sm_util": 88, "temp": 75, "ecc_dbe": 0},
    ]
    problems = detect_underperforming_gpus(metrics)
    assert len(problems) == 0


def test_save_metrics_snapshot(state_dir):
    teams_dir = state_dir / "teams"
    with patch("nemospawn.gpu.dcgm.get_team_dir", lambda tid: teams_dir / tid):
        team_dir = teams_dir / "t1" / "metrics"
        team_dir.mkdir(parents=True)

        metrics = [{"gpu_id": 0, "sm_util": 90, "temp": 70}]
        path = save_metrics_snapshot("t1", metrics)
        assert path.exists()


def test_parse_dcgmi_output():
    output = """#Entity   SMUTIL  MEMUTIL TEMP  POWER ECC_SBE ECC_DBE
GPU    0   95      80     72    300   0       0
GPU    1   45      60     68    250   0       0
"""
    metrics = _parse_dcgmi_output(output)
    assert len(metrics) == 2
    assert metrics[0]["gpu_id"] == 0
    assert metrics[0]["sm_util"] == 95.0
    assert metrics[1]["sm_util"] == 45.0
