"""Tests for model training wired into the scheduler pipeline.

All tests use synthetic data and require no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.services.model_service import ModelService


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------

@pytest.fixture
def model_config(tmp_path: Path) -> dict:
    """Config for model training tests using temp directory."""
    return {
        "general": {
            "base_currency": "GBP",
            "timezone": "Europe/London",
            "log_level": "WARNING",
            "data_dir": str(tmp_path / "data"),
            "random_seed": 42,
        },
        "universe": {
            "name": "test",
            "tickers": ["TEST_A", "TEST_B", "TEST_C"],
            "benchmark": "^TEST",
        },
        "data": {
            "source": "synthetic",
            "start_date": "2020-01-01",
            "end_date": None,
            "interval": "1d",
            "fields": ["Open", "High", "Low", "Close", "Volume"],
            "adjust_prices": True,
            "output_format": "parquet",
        },
        "features": {
            "technical": {
                "sma_windows": [5, 20],
                "rsi_window": 14,
            },
            "returns": {
                "windows": [1, 5],
                "log_returns": True,
            },
        },
        "models": {
            "output_dir": str(tmp_path / "models"),
            "target_column": "forward_return_5d",
            "cv_splits": 3,
            "default_model": {
                "class": "GradientBoostingClassifier",
                "params": {
                    "n_estimators": 10,
                    "learning_rate": 0.1,
                    "max_depth": 2,
                },
            },
        },
        "scheduler": {
            "daily_run_time": "17:30",
            "timezone": "Europe/London",
            "retrain_day": "sunday",
            "retrain_time": "08:00",
            "enabled": True,
        },
        "alerts": {
            "enabled": True,
            "methods": ["log"],
        },
    }


@pytest.fixture
def model_service(model_config: dict) -> ModelService:
    """A ModelService initialised with test config."""
    return ModelService(model_config)


@pytest.fixture
def training_features() -> pd.DataFrame:
    """Synthetic feature DataFrame with target column (~500 rows)."""
    n = 500
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2022-01-03", periods=n, freq="B")

    df = pd.DataFrame(
        {
            "feat_1": rng.standard_normal(n),
            "feat_2": rng.standard_normal(n),
            "feat_3": rng.standard_normal(n),
            "forward_return_5d": rng.choice([0.0, 1.0], size=n),
        },
        index=dates,
    )
    return df


@pytest.fixture
def small_features() -> pd.DataFrame:
    """Feature DataFrame with too few rows for training."""
    n = 100  # below _MIN_TRAIN_ROWS (252)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2024-01-02", periods=n, freq="B")

    return pd.DataFrame(
        {
            "feat_1": rng.standard_normal(n),
            "forward_return_5d": rng.choice([0.0, 1.0], size=n),
        },
        index=dates,
    )


@pytest.fixture
def no_feature_cols() -> pd.DataFrame:
    """Feature DataFrame with target but zero feature columns."""
    n = 300
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2022-01-03", periods=n, freq="B")

    return pd.DataFrame(
        {"forward_return_5d": rng.choice([0.0, 1.0], size=n)},
        index=dates,
    )


# -----------------------------------------------------------------
# ModelService.train_and_save
# -----------------------------------------------------------------

class TestTrainAndSave:
    """Tests for ModelService.train_and_save()."""

    def test_returns_expected_keys(
        self, model_service: ModelService, training_features: pd.DataFrame
    ) -> None:
        """train_and_save should return a dict with all documented keys."""
        result = model_service.train_and_save("TEST_A", training_features)

        assert "ticker" in result
        assert "cv_mean" in result
        assert "cv_std" in result
        assert "model_path" in result
        assert "feature_count" in result

    def test_ticker_matches(
        self, model_service: ModelService, training_features: pd.DataFrame
    ) -> None:
        """Returned ticker should match input."""
        result = model_service.train_and_save("MY_TICK", training_features)
        assert result["ticker"] == "MY_TICK"

    def test_cv_mean_is_positive(
        self, model_service: ModelService, training_features: pd.DataFrame
    ) -> None:
        """CV mean accuracy should be > 0."""
        result = model_service.train_and_save("TEST_A", training_features)
        assert result["cv_mean"] > 0

    def test_feature_count_correct(
        self, model_service: ModelService, training_features: pd.DataFrame
    ) -> None:
        """Feature count should match number of non-target columns."""
        result = model_service.train_and_save("TEST_A", training_features)
        assert result["feature_count"] == 3

    def test_model_file_written(
        self,
        model_service: ModelService,
        training_features: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Model .pkl file should be created on disk."""
        model_service.train_and_save("TEST_A", training_features)
        model_path = tmp_path / "models" / "TEST_A_model.pkl"
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_metrics_json_written(
        self,
        model_service: ModelService,
        training_features: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Metrics JSON should be created with expected keys."""
        model_service.train_and_save("TEST_A", training_features)
        metrics_path = tmp_path / "models" / "TEST_A_metrics.json"
        assert metrics_path.exists()

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert metrics["ticker"] == "TEST_A"
        assert "trained_at" in metrics
        assert "model_type" in metrics
        assert "cv_mean" in metrics
        assert "cv_std" in metrics
        assert "feature_count" in metrics
        assert "train_rows" in metrics

    def test_metrics_json_values(
        self,
        model_service: ModelService,
        training_features: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Metrics JSON values should be sensible."""
        result = model_service.train_and_save("TEST_A", training_features)
        metrics_path = tmp_path / "models" / "TEST_A_metrics.json"

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert metrics["cv_mean"] == result["cv_mean"]
        assert metrics["feature_count"] == 3
        assert metrics["train_rows"] == len(training_features)


# -----------------------------------------------------------------
# Skip conditions
# -----------------------------------------------------------------

class TestSkipConditions:
    """Tests for conditions that skip model training."""

    def test_skip_on_too_few_rows(
        self, model_service: ModelService, small_features: pd.DataFrame
    ) -> None:
        """Should skip training when rows < 252."""
        result = model_service.train_and_save("TEST_A", small_features)
        assert result.get("skipped") is True
        assert result["model_path"] is None
        assert "too few" in result.get("reason", "").lower()

    def test_skip_on_zero_features(
        self, model_service: ModelService, no_feature_cols: pd.DataFrame
    ) -> None:
        """Should skip training when there are zero feature columns."""
        result = model_service.train_and_save("TEST_A", no_feature_cols)
        assert result.get("skipped") is True
        assert result["model_path"] is None
        assert "zero" in result.get("reason", "").lower()

    def test_skip_on_missing_target(
        self, model_service: ModelService
    ) -> None:
        """Should skip when target column is not present."""
        n = 300
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start="2022-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {"feat_1": rng.standard_normal(n), "wrong_target": rng.standard_normal(n)},
            index=dates,
        )
        result = model_service.train_and_save("TEST_A", df)
        assert result.get("skipped") is True


# -----------------------------------------------------------------
# TimeSeriesSplit usage
# -----------------------------------------------------------------

class TestTimeSeriesSplitUsed:
    """Verify that sklearn TimeSeriesSplit is used (no random splits)."""

    def test_time_series_split_called(
        self, model_service: ModelService, training_features: pd.DataFrame
    ) -> None:
        """cross_val_score should receive a TimeSeriesSplit instance."""
        with patch(
            "src.services.model_service.cross_val_score", wraps=__import__(
                "sklearn.model_selection", fromlist=["cross_val_score"]
            ).cross_val_score,
        ) as mock_cvs:
            model_service.train_and_save("TEST_A", training_features)

            assert mock_cvs.called
            _, kwargs = mock_cvs.call_args
            # cv argument can be positional (3rd arg) or keyword
            args = mock_cvs.call_args.args
            cv_arg = kwargs.get("cv", args[2] if len(args) > 2 else None)

            from sklearn.model_selection import TimeSeriesSplit
            assert isinstance(cv_arg, TimeSeriesSplit)


# -----------------------------------------------------------------
# ModelService.load_model
# -----------------------------------------------------------------

class TestLoadModel:
    """Tests for ModelService.load_model()."""

    def test_load_after_save(
        self, model_service: ModelService, training_features: pd.DataFrame
    ) -> None:
        """load_model should return the trained model after train_and_save."""
        model_service.train_and_save("TEST_A", training_features)
        model = model_service.load_model("TEST_A")
        assert model is not None

    def test_load_nonexistent_returns_none(
        self, model_service: ModelService
    ) -> None:
        """load_model should return None for a ticker without a saved model."""
        assert model_service.load_model("NONEXISTENT") is None


# -----------------------------------------------------------------
# ModelService.evaluate
# -----------------------------------------------------------------

class TestEvaluate:
    """Tests for ModelService.evaluate()."""

    def test_evaluate_returns_scores(
        self, model_service: ModelService, training_features: pd.DataFrame
    ) -> None:
        """evaluate should return cv_mean and cv_std."""
        result = model_service.evaluate("TEST_A", training_features)
        assert "cv_mean" in result
        assert "cv_std" in result
        assert result["cv_mean"] > 0

    def test_evaluate_does_not_save(
        self,
        model_service: ModelService,
        training_features: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """evaluate should NOT write a model file."""
        model_service.evaluate("TEST_A", training_features)
        model_path = tmp_path / "models" / "TEST_A_model.pkl"
        assert not model_path.exists()


# -----------------------------------------------------------------
# PipelineRunner integration
# -----------------------------------------------------------------

class TestPipelineRunnerContinuesOnFailure:
    """PipelineRunner should continue when a single ticker fails."""

    def test_continues_after_single_ticker_failure(
        self, model_config: dict
    ) -> None:
        """Pipeline _run_model_training should not crash on per-ticker error."""
        from src.scheduler.pipeline import PipelineRunner

        runner = PipelineRunner(model_config)

        # Build features dict: one valid, one that will trigger an error
        n = 500
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start="2022-01-03", periods=n, freq="B")

        good_features = pd.DataFrame(
            {
                "feat_1": rng.standard_normal(n),
                "feat_2": rng.standard_normal(n),
                "forward_return_5d": rng.choice([0.0, 1.0], size=n),
            },
            index=dates,
        )

        # Empty DataFrame triggers skip, not crash
        bad_features = pd.DataFrame(
            {"forward_return_5d": rng.choice([0.0, 1.0], size=50)},
            index=pd.bdate_range(start="2024-01-02", periods=50, freq="B"),
        )

        features_dict = {"GOOD": good_features, "BAD": bad_features}
        results = runner._run_model_training(features_dict)

        assert len(results) == 2
        good_result = next(r for r in results if r["ticker"] == "GOOD")
        bad_result = next(r for r in results if r["ticker"] == "BAD")

        assert good_result["model_path"] is not None
        assert bad_result.get("skipped") is True
