"""Tests for NIM pipeline — deployer and Triton integration."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from nemospawn.nim.deployer import (
    NIMEndpoint, generate_nim_profiles, list_nim_containers,
)
from nemospawn.nim.triton import (
    BenchmarkResult, generate_model_repository, rank_endpoints,
)


def test_nim_endpoint_roundtrip():
    ep = NIMEndpoint(
        endpoint_id="nim-abc",
        team_id="t1",
        artifact_id="art-123",
        container_image="nim/test:latest",
        endpoint_url="http://localhost:8000",
        tp_degree=4,
    )
    d = ep.to_dict()
    ep2 = NIMEndpoint.from_dict(d)
    assert ep2.endpoint_id == "nim-abc"
    assert ep2.tp_degree == 4


def test_generate_nim_profiles_8gpu():
    profiles = generate_nim_profiles(8)
    assert len(profiles) == 4  # TP1, TP2, TP4, TP8
    assert profiles[0]["tp_degree"] == 1
    assert profiles[3]["tp_degree"] == 8


def test_generate_nim_profiles_2gpu():
    profiles = generate_nim_profiles(2)
    assert len(profiles) == 2  # TP1, TP2 only
    assert profiles[-1]["tp_degree"] == 2


def test_generate_nim_profiles_1gpu():
    profiles = generate_nim_profiles(1)
    assert len(profiles) == 1
    assert profiles[0]["tp_degree"] == 1


def test_list_nim_containers_not_available():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        containers = list_nim_containers()
        assert containers == []


def test_benchmark_result_roundtrip():
    br = BenchmarkResult(
        endpoint_url="http://localhost:8000",
        model_name="llama3",
        concurrency=16,
        p50_latency_ms=25.3,
        p95_latency_ms=45.1,
        p99_latency_ms=78.2,
        throughput_infer_per_sec=312.5,
    )
    d = br.to_dict()
    br2 = BenchmarkResult.from_dict(d)
    assert br2.p50_latency_ms == 25.3
    assert br2.throughput_infer_per_sec == 312.5


def test_generate_model_repository(tmp_path):
    model_dir = generate_model_repository("test_model", tmp_path)
    assert model_dir.exists()
    assert (model_dir / "config.pbtxt").exists()
    assert (model_dir / "1").is_dir()
    config = (model_dir / "config.pbtxt").read_text()
    assert 'name: "test_model"' in config
    assert "tensorrt_llm" in config


def test_rank_endpoints_throughput_at_p95():
    results = [
        BenchmarkResult("u", "m", 1, 10, 20, 30, 100),
        BenchmarkResult("u", "m", 4, 30, 60, 80, 300),
        BenchmarkResult("u", "m", 16, 50, 90, 120, 500),
        BenchmarkResult("u", "m", 64, 80, 150, 200, 600),  # p95 > 100ms
    ]
    ranked = rank_endpoints(results, "throughput_at_p95_100ms")
    # Only first 3 meet p95 < 100ms, best throughput is concurrency=16
    assert ranked[0].concurrency == 16
    assert ranked[0].throughput_infer_per_sec == 500


def test_rank_endpoints_min_p99():
    results = [
        BenchmarkResult("u", "m", 1, 10, 20, 30, 100),
        BenchmarkResult("u", "m", 4, 30, 60, 25, 300),  # lowest p99
    ]
    ranked = rank_endpoints(results, "min_p99")
    assert ranked[0].p99_latency_ms == 25


def test_rank_endpoints_max_throughput():
    results = [
        BenchmarkResult("u", "m", 1, 10, 20, 30, 100),
        BenchmarkResult("u", "m", 64, 80, 150, 200, 900),
    ]
    ranked = rank_endpoints(results, "max_throughput")
    assert ranked[0].throughput_infer_per_sec == 900
