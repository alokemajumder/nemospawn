"""Tests for cross-cluster federation."""

from unittest.mock import patch, MagicMock

from nemospawn.federation.cluster import (
    ClusterConfig, register_cluster, list_clusters, get_cluster,
)


def test_cluster_config_roundtrip():
    c = ClusterConfig(name="dgx-pod-a", host="10.0.1.10", ssh_user="root", gpu_count=8)
    d = c.to_dict()
    c2 = ClusterConfig.from_dict(d)
    assert c2.name == "dgx-pod-a"
    assert c2.gpu_count == 8


def test_register_and_list_cluster(state_dir):
    clusters_dir = state_dir / "clusters"
    clusters_dir.mkdir()

    with patch("nemospawn.federation.cluster.CLUSTERS_DIR", clusters_dir), \
         patch("nemospawn.federation.cluster._probe_remote_gpus", return_value=8):
        cluster = register_cluster("test-cluster", "10.0.1.1", ssh_key="/tmp/key")
        assert cluster.name == "test-cluster"
        assert cluster.gpu_count == 8

    with patch("nemospawn.federation.cluster.CLUSTERS_DIR", clusters_dir):
        clusters = list_clusters()
        assert len(clusters) == 1
        assert clusters[0].name == "test-cluster"


def test_get_cluster(state_dir):
    clusters_dir = state_dir / "clusters"
    clusters_dir.mkdir()

    with patch("nemospawn.federation.cluster.CLUSTERS_DIR", clusters_dir), \
         patch("nemospawn.federation.cluster._probe_remote_gpus", return_value=4):
        register_cluster("my-cluster", "192.168.1.1")

    with patch("nemospawn.federation.cluster.CLUSTERS_DIR", clusters_dir):
        c = get_cluster("my-cluster")
        assert c is not None
        assert c.host == "192.168.1.1"

        assert get_cluster("nonexistent") is None
