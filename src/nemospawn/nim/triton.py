"""Triton Inference Server integration — benchmarking and model repository management.

Uses perf_analyzer for standardized benchmarks and tritonclient for endpoint
interaction. Auto-generates Triton model repository configs from NIM artifacts.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a Triton perf_analyzer benchmark run."""
    endpoint_url: str
    model_name: str
    concurrency: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_infer_per_sec: float
    throughput_tokens_per_sec: float = 0.0
    gpu_utilization_pct: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def run_perf_analyzer(
    endpoint_url: str,
    model_name: str = "model",
    concurrency_levels: list[int] | None = None,
    measurement_interval_ms: int = 5000,
    protocol: str = "http",
) -> list[BenchmarkResult]:
    """Run perf_analyzer against a Triton or NIM endpoint.

    Args:
        endpoint_url: The inference endpoint URL (e.g., localhost:8000).
        model_name: Model name in the Triton repository.
        concurrency_levels: List of concurrency levels to test (default: [1, 4, 16, 64]).
        measurement_interval_ms: Measurement window in milliseconds.
        protocol: Protocol to use (http or grpc).

    Returns:
        List of BenchmarkResult, one per concurrency level.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 16, 64]

    results = []

    for conc in concurrency_levels:
        args = [
            "perf_analyzer",
            "-m", model_name,
            "-u", endpoint_url,
            "--concurrency-range", f"{conc}:{conc}",
            "--measurement-interval", str(measurement_interval_ms),
            "-i", protocol,
            "--latency-report-file", "/dev/stdout",
            "-f", "/tmp/perf_result.csv",
        ]

        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=120,
            )

            if result.returncode == 0:
                bench = _parse_perf_output(result.stdout, endpoint_url, model_name, conc)
                if bench:
                    results.append(bench)
            else:
                results.append(BenchmarkResult(
                    endpoint_url=endpoint_url,
                    model_name=model_name,
                    concurrency=conc,
                    p50_latency_ms=0,
                    p95_latency_ms=0,
                    p99_latency_ms=0,
                    throughput_infer_per_sec=0,
                ))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            results.append(BenchmarkResult(
                endpoint_url=endpoint_url,
                model_name=model_name,
                concurrency=conc,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_infer_per_sec=0,
            ))

    return results


def _parse_perf_output(
    output: str, endpoint_url: str, model_name: str, concurrency: int,
) -> BenchmarkResult | None:
    """Parse perf_analyzer stdout for latency and throughput metrics."""
    p50 = p95 = p99 = throughput = 0.0

    for line in output.splitlines():
        line = line.strip()
        if "p50 latency" in line.lower():
            p50 = _extract_number(line)
        elif "p95 latency" in line.lower():
            p95 = _extract_number(line)
        elif "p99 latency" in line.lower():
            p99 = _extract_number(line)
        elif "throughput" in line.lower() and "infer" in line.lower():
            throughput = _extract_number(line)

    # Also try CSV parsing if available
    try:
        csv_path = Path("/tmp/perf_result.csv")
        if csv_path.exists():
            csv_content = csv_path.read_text().strip()
            lines = csv_content.splitlines()
            if len(lines) >= 2:
                headers = lines[0].split(",")
                values = lines[-1].split(",")
                data = dict(zip(headers, values))
                p50 = float(data.get("p50 latency", p50))
                p95 = float(data.get("p95 latency", p95))
                p99 = float(data.get("p99 latency", p99))
                throughput = float(data.get("Inferences/Second", throughput))
    except (ValueError, OSError):
        pass

    return BenchmarkResult(
        endpoint_url=endpoint_url,
        model_name=model_name,
        concurrency=concurrency,
        p50_latency_ms=p50 / 1000 if p50 > 1000 else p50,  # convert μs to ms if needed
        p95_latency_ms=p95 / 1000 if p95 > 1000 else p95,
        p99_latency_ms=p99 / 1000 if p99 > 1000 else p99,
        throughput_infer_per_sec=throughput,
    )


def generate_model_repository(
    model_name: str,
    output_dir: Path,
    backend: str = "tensorrt_llm",
    max_batch_size: int = 64,
) -> Path:
    """Generate a Triton model repository structure.

    Creates:
        output_dir/
        └── model_name/
            ├── config.pbtxt
            └── 1/
                (empty — model files to be added)
    """
    model_dir = output_dir / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    config = f"""name: "{model_name}"
backend: "{backend}"
max_batch_size: {max_batch_size}

input {{
  name: "input_ids"
  data_type: TYPE_INT32
  dims: [-1]
}}

output {{
  name: "output_ids"
  data_type: TYPE_INT32
  dims: [-1]
}}

instance_group {{
  count: 1
  kind: KIND_GPU
}}
"""

    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config)

    return model_dir


def rank_endpoints(
    results: list[BenchmarkResult],
    objective: str = "throughput_at_p95_100ms",
) -> list[BenchmarkResult]:
    """Rank benchmark results by a configurable objective.

    Objectives:
        throughput_at_p95_100ms: max throughput where p95 < 100ms
        min_p99: lowest p99 latency
        max_throughput: highest throughput regardless of latency
    """
    if objective == "throughput_at_p95_100ms":
        # Filter to results meeting latency SLA, then sort by throughput
        meeting_sla = [r for r in results if r.p95_latency_ms < 100.0]
        if meeting_sla:
            return sorted(meeting_sla, key=lambda r: r.throughput_infer_per_sec, reverse=True)
        return sorted(results, key=lambda r: r.p95_latency_ms)
    elif objective == "min_p99":
        return sorted(results, key=lambda r: r.p99_latency_ms)
    elif objective == "max_throughput":
        return sorted(results, key=lambda r: r.throughput_infer_per_sec, reverse=True)
    else:
        return results


def _extract_number(line: str) -> float:
    """Extract a float from a line of text."""
    import re
    match = re.search(r"[\d.]+", line.split(":")[-1] if ":" in line else line)
    return float(match.group()) if match else 0.0
