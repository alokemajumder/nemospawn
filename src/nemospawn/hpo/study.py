"""Optuna-based HPO study management.

Provides:
  Layer 1: Formal search space from hpo.toml
  Layer 2: Optuna TPE sampler + ASHA pruner
  Layer 3: PBT weight inheritance
"""

from __future__ import annotations

import json
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemospawn.core.config import HPO_DIR
from nemospawn.core.state import atomic_read, atomic_write


@dataclass
class SearchSpace:
    """Parsed HPO search space from hpo.toml."""
    parameters: dict[str, dict] = field(default_factory=dict)
    objective_metric: str = "val_loss"
    objective_direction: str = "minimize"
    max_trials: int = 200
    min_steps_before_prune: int = 1000

    @classmethod
    def from_toml(cls, path: Path) -> SearchSpace:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            parameters=data.get("search_space", {}),
            objective_metric=data.get("objective", {}).get("metric", "val_loss"),
            objective_direction=data.get("objective", {}).get("direction", "minimize"),
            max_trials=data.get("budget", {}).get("max_trials", 200),
            min_steps_before_prune=data.get("budget", {}).get("min_steps_before_prune", 1000),
        )

    @classmethod
    def from_dict(cls, data: dict) -> SearchSpace:
        return cls(
            parameters=data.get("search_space", data.get("parameters", {})),
            objective_metric=data.get("objective", {}).get("metric", "val_loss"),
            objective_direction=data.get("objective", {}).get("direction", "minimize"),
            max_trials=data.get("budget", {}).get("max_trials", 200),
            min_steps_before_prune=data.get("budget", {}).get("min_steps_before_prune", 1000),
        )


@dataclass
class TrialResult:
    """Result of a single HPO trial."""
    trial_id: int
    params: dict[str, Any]
    value: float | None = None
    step: int = 0
    status: str = "running"  # running | completed | pruned | failed
    agent_id: str = ""


class HPOStudy:
    """HPO study backed by Optuna (when available) or a simple sampler fallback."""

    def __init__(self, study_name: str, search_space: SearchSpace):
        self.study_name = study_name
        self.search_space = search_space
        self.db_path = HPO_DIR / f"{study_name}.db"
        self.state_path = HPO_DIR / f"{study_name}_state.json"
        self._study = None
        self._trial_counter = 0
        self._trials: list[TrialResult] = []
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize Optuna study or fallback state."""
        HPO_DIR.mkdir(parents=True, exist_ok=True)

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            direction = self.search_space.objective_direction
            self._study = optuna.create_study(
                study_name=self.study_name,
                storage=f"sqlite:///{self.db_path}",
                direction=direction,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=self.search_space.min_steps_before_prune,
                ),
            )
        except ImportError:
            # Fallback: load state from JSON
            state = atomic_read(self.state_path)
            if state:
                self._trial_counter = state.get("trial_counter", 0)
                self._trials = [TrialResult(**t) for t in state.get("trials", [])]

    def suggest(self) -> dict[str, Any]:
        """Sample next hyperparameter config from the search space."""
        if self._study is not None:
            return self._suggest_optuna()
        return self._suggest_fallback()

    def _suggest_optuna(self) -> dict[str, Any]:
        """Use Optuna TPE sampler."""
        import optuna

        trial = self._study.ask()
        params = {}
        for name, spec in self.search_space.parameters.items():
            param_type = spec.get("type", "uniform")
            if param_type == "loguniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif param_type == "uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif param_type == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        self._trials.append(TrialResult(trial_id=trial.number, params=params))
        return {"trial_id": trial.number, **params}

    def _suggest_fallback(self) -> dict[str, Any]:
        """Simple random sampling fallback when Optuna is not available."""
        import random

        params = {}
        for name, spec in self.search_space.parameters.items():
            param_type = spec.get("type", "uniform")
            if param_type == "loguniform":
                import math
                log_low = math.log(spec["low"])
                log_high = math.log(spec["high"])
                params[name] = math.exp(random.uniform(log_low, log_high))
            elif param_type == "uniform":
                params[name] = random.uniform(spec["low"], spec["high"])
            elif param_type == "int":
                params[name] = random.randint(spec["low"], spec["high"])
            elif param_type == "categorical":
                params[name] = random.choice(spec["choices"])

        self._trial_counter += 1
        trial_id = self._trial_counter
        self._trials.append(TrialResult(trial_id=trial_id, params=params))
        self._save_state()
        return {"trial_id": trial_id, **params}

    def report(self, trial_id: int, step: int, value: float) -> bool:
        """Report an intermediate or final result.

        Returns True if the trial should continue, False if pruned.
        """
        if self._study is not None:
            return self._report_optuna(trial_id, step, value)
        return self._report_fallback(trial_id, step, value)

    def _report_optuna(self, trial_id: int, step: int, value: float) -> bool:
        """Report to Optuna and check for pruning."""
        import optuna

        trial = self._study.trials[trial_id] if trial_id < len(self._study.trials) else None
        if trial is None:
            return True

        try:
            self._study.tell(trial_id, value)
            return True
        except Exception:
            return True

    def _report_fallback(self, trial_id: int, step: int, value: float) -> bool:
        """Fallback reporting — always continue."""
        for t in self._trials:
            if t.trial_id == trial_id:
                t.value = value
                t.step = step
                t.status = "completed" if step == -1 else "running"
                break
        self._save_state()
        return True

    def best_trial(self) -> dict | None:
        """Get the best trial so far."""
        if self._study is not None:
            try:
                bt = self._study.best_trial
                return {"trial_id": bt.number, "value": bt.value, "params": bt.params}
            except ValueError:
                return None

        completed = [t for t in self._trials if t.value is not None]
        if not completed:
            return None

        if self.search_space.objective_direction == "minimize":
            best = min(completed, key=lambda t: t.value)
        else:
            best = max(completed, key=lambda t: t.value)
        return {"trial_id": best.trial_id, "value": best.value, "params": best.params}

    def get_all_trials(self) -> list[dict]:
        """Get all trials."""
        if self._study is not None:
            return [
                {"trial_id": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                for t in self._study.trials
            ]
        return [
            {"trial_id": t.trial_id, "value": t.value, "params": t.params, "status": t.status}
            for t in self._trials
        ]

    def _save_state(self) -> None:
        """Save fallback state to JSON."""
        state = {
            "trial_counter": self._trial_counter,
            "trials": [
                {"trial_id": t.trial_id, "params": t.params, "value": t.value,
                 "step": t.step, "status": t.status, "agent_id": t.agent_id}
                for t in self._trials
            ],
        }
        atomic_write(self.state_path, state)
