"""Tests for NVLink topology parsing."""

from unittest.mock import patch, MagicMock
from nemospawn.gpu.topology import parse_topology, get_nvlink_islands


def test_parse_topology(mock_nvidia_smi_topo):
    result = MagicMock()
    result.returncode = 0
    result.stdout = mock_nvidia_smi_topo

    with patch("subprocess.run", return_value=result):
        topo = parse_topology()

    assert topo["matrix"]
    assert topo["gpu_pairs"]
    assert topo["raw"]


def test_get_nvlink_islands_two_islands():
    # GPU 0-1 connected via NV12, GPU 2-3 connected via NV12
    topo = {
        "matrix": {
            0: {0: "X", 1: "NV12", 2: "SYS", 3: "SYS"},
            1: {0: "NV12", 1: "X", 2: "SYS", 3: "SYS"},
            2: {0: "SYS", 1: "SYS", 2: "X", 3: "NV12"},
            3: {0: "SYS", 1: "SYS", 2: "NV12", 3: "X"},
        }
    }
    islands = get_nvlink_islands(topo)
    assert len(islands) == 2
    assert [0, 1] in islands
    assert [2, 3] in islands


def test_get_nvlink_islands_single():
    topo = {
        "matrix": {
            0: {0: "X", 1: "NV12"},
            1: {0: "NV12", 1: "X"},
        }
    }
    islands = get_nvlink_islands(topo)
    assert len(islands) == 1
    assert islands[0] == [0, 1]


def test_get_nvlink_islands_empty():
    islands = get_nvlink_islands({})
    assert islands == []
