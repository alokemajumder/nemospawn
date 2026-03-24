"""Tests for HPO study management."""

from unittest.mock import patch

from nemospawn.hpo.study import HPOStudy, SearchSpace, TrialResult


def test_search_space_defaults():
    space = SearchSpace()
    assert space.objective_metric == "val_loss"
    assert space.objective_direction == "minimize"
    assert space.max_trials == 200


def test_search_space_from_dict():
    data = {
        "search_space": {
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
            "batch_size": {"type": "categorical", "choices": [256, 512]},
        },
        "objective": {"metric": "val_loss", "direction": "minimize"},
        "budget": {"max_trials": 50, "min_steps_before_prune": 500},
    }
    space = SearchSpace.from_dict(data)
    assert "lr" in space.parameters
    assert space.max_trials == 50


def test_hpo_study_fallback_suggest(state_dir):
    """Test HPO study with fallback sampler (no Optuna)."""
    hpo_dir = state_dir / "hpo"
    hpo_dir.mkdir(exist_ok=True)

    space = SearchSpace(parameters={
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
        "batch_size": {"type": "categorical", "choices": [256, 512, 1024]},
        "depth": {"type": "int", "low": 10, "high": 16},
        "warmup": {"type": "uniform", "low": 0.01, "high": 0.1},
    })

    with patch("nemospawn.hpo.study.HPO_DIR", hpo_dir):
        # Force fallback by making optuna import fail
        with patch.dict("sys.modules", {"optuna": None}):
            study = HPOStudy.__new__(HPOStudy)
            study.study_name = "test"
            study.search_space = space
            study.db_path = hpo_dir / "test.db"
            study.state_path = hpo_dir / "test_state.json"
            study._study = None
            study._trial_counter = 0
            study._trials = []

            config = study.suggest()
            assert "trial_id" in config
            assert "lr" in config
            assert 1e-5 <= config["lr"] <= 1e-3
            assert config["batch_size"] in [256, 512, 1024]
            assert 10 <= config["depth"] <= 16

            # Report
            study.report(config["trial_id"], 1000, 0.5)
            study.report(config["trial_id"], -1, 0.3)

            # Best
            best = study.best_trial()
            assert best is not None
            assert best["value"] == 0.3


def test_hpo_study_multiple_trials(state_dir):
    hpo_dir = state_dir / "hpo"
    hpo_dir.mkdir(exist_ok=True)

    space = SearchSpace(parameters={
        "lr": {"type": "uniform", "low": 0.001, "high": 0.01},
    })

    with patch("nemospawn.hpo.study.HPO_DIR", hpo_dir):
        study = HPOStudy.__new__(HPOStudy)
        study.study_name = "multi"
        study.search_space = space
        study.db_path = hpo_dir / "multi.db"
        study.state_path = hpo_dir / "multi_state.json"
        study._study = None
        study._trial_counter = 0
        study._trials = []

        # Run 3 trials
        for val in [0.5, 0.2, 0.8]:
            config = study.suggest()
            study.report(config["trial_id"], -1, val)

        best = study.best_trial()
        assert best["value"] == 0.2

        trials = study.get_all_trials()
        assert len(trials) == 3


def test_trial_result_dataclass():
    t = TrialResult(trial_id=1, params={"lr": 0.001}, value=0.5, status="completed")
    assert t.trial_id == 1
    assert t.value == 0.5
