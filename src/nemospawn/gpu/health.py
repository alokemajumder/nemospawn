"""GPU health monitoring via NVML."""

from __future__ import annotations

from rich.console import Console

console = Console(stderr=True)


def get_gpu_health(gpu_index: int) -> dict:
    """Query NVML for GPU health metrics."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
            import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            ecc_errors = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC,
            )
        except pynvml.NVMLError:
            ecc_errors = 0

        pynvml.nvmlShutdown()

        return {
            "gpu_index": gpu_index,
            "temperature_c": temp,
            "gpu_utilization_pct": util.gpu,
            "memory_utilization_pct": util.memory,
            "power_draw_w": round(power, 1),
            "memory_used_mb": mem_info.used // (1024 * 1024),
            "memory_total_mb": mem_info.total // (1024 * 1024),
            "ecc_errors": ecc_errors,
        }
    except Exception as e:
        return {"gpu_index": gpu_index, "error": str(e)}


def check_all_gpus() -> list[dict]:
    """Get health metrics for all GPUs."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
            import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except Exception:
        console.print("[yellow]NVML not available — cannot query GPU health[/]")
        return []

    return [get_gpu_health(i) for i in range(count)]
