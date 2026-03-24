"""NeMo YAML config injection — converts task descriptions to NeMo config overrides.

Takes natural-language or structured task parameters and generates NeMo-compatible
YAML config overrides that can be passed to `nemo train`, `nemo finetune`, etc.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


# Standard NeMo config structure for common overrides
NEMO_CONFIG_SCHEMA = {
    "trainer": {
        "max_steps": int,
        "val_check_interval": int,
        "precision": str,  # "bf16-mixed", "16-mixed", "32"
        "devices": int,
        "num_nodes": int,
        "accumulate_grad_batches": int,
        "gradient_clip_val": float,
        "log_every_n_steps": int,
    },
    "model": {
        "micro_batch_size": int,
        "global_batch_size": int,
        "tensor_model_parallel_size": int,
        "pipeline_model_parallel_size": int,
        "encoder_seq_length": int,
        "num_layers": int,
        "hidden_size": int,
        "num_attention_heads": int,
    },
    "optim": {
        "name": str,  # "adam", "adamw", "sgd", "muon"
        "lr": float,
        "weight_decay": float,
        "betas": list,
    },
    "optim.sched": {
        "name": str,  # "cosine", "warmup_cosine", "linear", "constant"
        "warmup_steps": int,
        "min_lr": float,
    },
    "data": {
        "data_prefix": str,
        "num_workers": int,
        "seq_length": int,
    },
}


def parse_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """Convert flat key=value params into nested NeMo config overrides.

    Handles dotted keys like 'optim.lr' -> {'optim': {'lr': value}}.
    Validates types against NEMO_CONFIG_SCHEMA when possible.
    """
    overrides: dict[str, Any] = {}

    for key, value in params.items():
        parts = key.split(".")
        current = overrides
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = _coerce_value(key, value)

    return overrides


def _coerce_value(key: str, value: Any) -> Any:
    """Coerce a value to the expected type based on schema."""
    # Find the expected type from schema
    for section, fields in NEMO_CONFIG_SCHEMA.items():
        field_name = key.split(".")[-1]
        if field_name in fields:
            expected_type = fields[field_name]
            try:
                if expected_type == float:
                    return float(value)
                elif expected_type == int:
                    return int(float(value))
                elif expected_type == str:
                    return str(value)
                elif expected_type == list:
                    if isinstance(value, str):
                        return [float(x.strip()) for x in value.strip("[]").split(",")]
                    return value
            except (ValueError, TypeError):
                pass
    return value


def generate_nemo_config(
    base_config: dict | None = None,
    overrides: dict[str, Any] | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    max_steps: int | None = None,
    precision: str = "bf16-mixed",
    tp_size: int = 1,
    pp_size: int = 1,
    num_gpus: int = 1,
    warmup_steps: int | None = None,
    optimizer: str = "adamw",
    scheduler: str = "cosine",
) -> dict:
    """Generate a NeMo training config from structured parameters.

    Args:
        base_config: Optional base config dict to merge into.
        overrides: Additional dotted key-value overrides.
        lr: Learning rate.
        batch_size: Micro batch size.
        max_steps: Maximum training steps.
        precision: Training precision (bf16-mixed, 16-mixed, 32).
        tp_size: Tensor model parallel size.
        pp_size: Pipeline model parallel size.
        num_gpus: Number of GPUs.
        warmup_steps: Warmup steps for scheduler.
        optimizer: Optimizer name.
        scheduler: Learning rate scheduler name.

    Returns:
        Complete NeMo config dict.
    """
    config = dict(base_config) if base_config else {}

    # Trainer section
    trainer = config.setdefault("trainer", {})
    if max_steps is not None:
        trainer["max_steps"] = max_steps
    trainer["precision"] = precision
    trainer["devices"] = num_gpus
    trainer.setdefault("num_nodes", 1)
    trainer.setdefault("val_check_interval", max(1, (max_steps or 1000) // 10))
    trainer.setdefault("log_every_n_steps", 10)

    # Model section
    model = config.setdefault("model", {})
    if batch_size is not None:
        model["micro_batch_size"] = batch_size
    if tp_size > 1:
        model["tensor_model_parallel_size"] = tp_size
    if pp_size > 1:
        model["pipeline_model_parallel_size"] = pp_size

    # Optimizer section
    optim = config.setdefault("optim", {})
    optim["name"] = optimizer
    if lr is not None:
        optim["lr"] = lr
    optim.setdefault("weight_decay", 0.01)
    if optimizer in ("adam", "adamw"):
        optim.setdefault("betas", [0.9, 0.999])

    # Scheduler section
    sched = optim.setdefault("sched", {})
    sched["name"] = scheduler
    if warmup_steps is not None:
        sched["warmup_steps"] = warmup_steps
    sched.setdefault("min_lr", 0.0)

    # Merge additional overrides
    if overrides:
        parsed = parse_overrides(overrides) if any("." in k for k in overrides) else overrides
        _deep_merge(config, parsed)

    return config


def write_nemo_config(config: dict, output_path: Path) -> Path:
    """Write a NeMo config to a YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return output_path


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
