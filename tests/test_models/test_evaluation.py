"""Tests for time-series aware cross-validation (src.models.evaluation).

Covers:
- time_series_split produces correct number of folds
- No overlap between any train and test indices
- Gap is respected: verify_no_leakage passes for all folds
- CRITICAL: For each fold, max(train_dates) + gap_days < min(test_dates)
- walk_forward_cv returns correct structure with all expected keys
- compare_models returns DataFrame with one row per model
- Test with synthetic features from FeaturePipeline + synthetic OHLCV data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.classical import RandomForestModel, GradientBoostingModel
from src.models.evaluation import (
    DataValidationError,
    time_series_split,
    verify_no_leakage,
    walk_forward_cv,
    compare_models,
    plot_cv_results,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def cv_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """1000-row synthetic classification dataset for CV tests."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2016-01-01", periods=1000, freq="B")
    X = pd.DataFrame(
        {
            "feat_a": rng.standard_normal(1000),
            "feat_b": rng.standard_normal(1000),
            "feat_c": rng.uniform(-1, 1, 1000),
        },
        index=dates,
    )
    y = pd.Series(
        (rng.standard_normal(1000) > 0).astype(int),
        index=dates,
        name="target",
    )
    return X, y


@pytest.fixture
def small_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Tiny dataset too small for default CV parameters."""
    dates = pd.bdate_range("2020-01-01", periods=50, freq="B")
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"f1": rng.standard_normal(50)}, index=dates)
    y = pd.Series((rng.standard_normal(50) > 0).astype(int), index=dates)
    return X, y


@pytest.fixture
def test_config() -> dict:
    """Minimal config for evaluation tests."""
    return {
        "general": {"random_seed": 42},
        "models": {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
            },
            "gradient_boosting": {
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.1,
            },
            "evaluation": {
                "n_splits": 5,
                "gap": 5,
                "min_train_size": 252,
            },
        },
    }


# ------------------------------------------------------------------
# time_series_split
# ------------------------------------------------------------------

class TestTimeSeriesSplit:

    def test_correct_number_of_folds(self, cv_dataset):
        X, y = cv_dataset
        folds = list(time_series_split(X, y, n_splits=4, gap=5, min_train_size=252))
        assert len(folds) == 4

    def test_no_overlap_between_train_and_test(self, cv_dataset):
        X, y = cv_dataset
        for train_idx, test_idx in time_series_split(
            X, y, n_splits=5, gap=5, min_train_size=252,
        ):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_gap_is_respected(self, cv_dataset):
        X, y = cv_dataset
        gap = 10
        for train_idx, test_idx in time_series_split(
            X, y, n_splits=3, gap=gap, min_train_size=252,
        ):
            assert np.max(train_idx) + gap < np.min(test_idx), (
                f"Gap violated: max(train)={np.max(train_idx)}, "
                f"min(test)={np.min(test_idx)}, gap={gap}"
            )

    def test_critical_temporal_ordering(self, cv_dataset):
        """CRITICAL: For each fold, max(train_dates) + gap_days < min(test_dates)."""
        X, y = cv_dataset
        gap = 5
        for train_idx, test_idx in time_series_split(
            X, y, n_splits=5, gap=gap, min_train_size=252,
        ):
            train_dates = X.index[train_idx]
            test_dates = X.index[test_idx]

            assert train_dates.max() < test_dates.min(), (
                f"Train end {train_dates.max()} not before test start {test_dates.min()}"
            )

            # Count business days between train end and test start
            gap_days = len(pd.bdate_range(train_dates.max(), test_dates.min(), freq="B")) - 1
            assert gap_days >= gap, (
                f"Gap is only {gap_days} days, expected >= {gap}"
            )

    def test_expanding_window(self, cv_dataset):
        """Each successive fold must train on more data."""
        X, y = cv_dataset
        folds = list(time_series_split(X, y, n_splits=5, gap=5, min_train_size=252))
        train_sizes = [len(t) for t, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_last_fold_includes_end(self, cv_dataset):
        """Last fold's test set must extend to the end of data."""
        X, y = cv_dataset
        folds = list(time_series_split(X, y, n_splits=3, gap=5, min_train_size=252))
        _, last_test = folds[-1]
        assert np.max(last_test) == len(X) - 1

    def test_too_little_data_raises(self, small_dataset):
        X, y = small_dataset
        with pytest.raises(ValueError, match="Not enough data"):
            list(time_series_split(X, y, n_splits=5, gap=5, min_train_size=252))


# ------------------------------------------------------------------
# verify_no_leakage
# ------------------------------------------------------------------

class TestVerifyNoLeakage:

    def test_passes_valid_split(self):
        train_idx = np.arange(100)
        test_idx = np.arange(110, 200)
        verify_no_leakage(train_idx, test_idx, gap=5)  # Should not raise

    def test_fails_with_leakage(self):
        train_idx = np.arange(100)
        test_idx = np.arange(102, 200)
        with pytest.raises(DataValidationError, match="Data leakage"):
            verify_no_leakage(train_idx, test_idx, gap=5)

    def test_fails_at_boundary(self):
        train_idx = np.arange(100)
        test_idx = np.arange(105, 200)  # gap=5: 99+5=104 >= 105? No, 104 < 105
        # Actually max(train_idx) = 99, 99 + 5 = 104, 104 < 105. This passes.
        verify_no_leakage(train_idx, test_idx, gap=5)

    def test_fails_exact_boundary(self):
        train_idx = np.arange(100)  # max = 99
        test_idx = np.arange(104, 200)  # min = 104; 99 + 5 = 104 >= 104
        with pytest.raises(DataValidationError, match="Data leakage"):
            verify_no_leakage(train_idx, test_idx, gap=5)


# ------------------------------------------------------------------
# walk_forward_cv
# ------------------------------------------------------------------

class TestWalkForwardCV:

    def test_returns_correct_structure(self, cv_dataset, test_config):
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=3, gap=5, min_train_size=252,
        )
        assert "per_fold" in result
        assert "aggregate" in result
        assert "model_name" in result
        assert "n_splits" in result
        assert "gap" in result
        assert result["model_name"] == "rf"
        assert result["n_splits"] == 3

    def test_correct_number_of_folds(self, cv_dataset, test_config):
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=4, gap=5, min_train_size=252,
        )
        assert len(result["per_fold"]) == 4

    def test_per_fold_has_expected_keys(self, cv_dataset, test_config):
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=3, gap=5, min_train_size=252,
        )
        for fold in result["per_fold"]:
            assert "fold" in fold
            assert "train_size" in fold
            assert "test_size" in fold
            assert "metrics" in fold
            assert "accuracy" in fold["metrics"]
            assert "information_coefficient" in fold["metrics"]
            assert "hit_rate" in fold["metrics"]
            assert "rmse" in fold["metrics"]
            assert "mae" in fold["metrics"]

    def test_aggregate_has_mean_and_std(self, cv_dataset, test_config):
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=3, gap=5, min_train_size=252,
        )
        agg = result["aggregate"]
        assert "mean_information_coefficient" in agg
        assert "std_information_coefficient" in agg
        assert "mean_hit_rate" in agg
        assert "mean_accuracy" in agg

    def test_train_end_before_test_start(self, cv_dataset, test_config):
        """CRITICAL: train end must be before test start in every fold."""
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=5, gap=5, min_train_size=252,
        )
        for fold in result["per_fold"]:
            train_end = pd.Timestamp(fold["train_end"])
            test_start = pd.Timestamp(fold["test_start"])
            assert train_end < test_start, (
                f"Fold {fold['fold']}: train_end={train_end} >= test_start={test_start}"
            )

    def test_gap_respected_in_cv(self, cv_dataset, test_config):
        X, y = cv_dataset
        gap = 10
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=3, gap=gap, min_train_size=252,
        )
        for fold in result["per_fold"]:
            train_end = pd.Timestamp(fold["train_end"])
            test_start = pd.Timestamp(fold["test_start"])
            gap_days = len(pd.bdate_range(train_end, test_start, freq="B")) - 1
            assert gap_days >= gap

    def test_unsorted_index_raises(self, cv_dataset, test_config):
        X, y = cv_dataset
        X_shuffled = X.sample(frac=1.0, random_state=0)
        y_shuffled = y.loc[X_shuffled.index]
        model = RandomForestModel("rf", config=test_config)
        with pytest.raises(ValueError, match="sorted chronologically"):
            walk_forward_cv(model, X_shuffled, y_shuffled)

    def test_gap_zero_allowed(self, cv_dataset, test_config):
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=3, gap=0, min_train_size=252,
        )
        assert len(result["per_fold"]) == 3

    def test_single_fold(self, cv_dataset, test_config):
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=1, gap=5, min_train_size=252,
        )
        assert len(result["per_fold"]) == 1


# ------------------------------------------------------------------
# compare_models
# ------------------------------------------------------------------

class TestCompareModels:

    def test_returns_dataframe(self, cv_dataset, test_config):
        X, y = cv_dataset
        models = [
            RandomForestModel("rf", config=test_config),
            GradientBoostingModel("gb", config=test_config),
        ]
        result = compare_models(models, X, y, n_splits=3, gap=5, min_train_size=252)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_model(self, cv_dataset, test_config):
        X, y = cv_dataset
        models = [
            RandomForestModel("rf", config=test_config),
            GradientBoostingModel("gb", config=test_config),
        ]
        result = compare_models(models, X, y, n_splits=3, gap=5, min_train_size=252)
        assert len(result) == 2
        assert set(result.index) == {"rf", "gb"}

    def test_has_aggregate_metrics(self, cv_dataset, test_config):
        X, y = cv_dataset
        models = [RandomForestModel("rf", config=test_config)]
        result = compare_models(models, X, y, n_splits=3, gap=5, min_train_size=252)
        assert "mean_information_coefficient" in result.columns
        assert "mean_hit_rate" in result.columns


# ------------------------------------------------------------------
# plot_cv_results
# ------------------------------------------------------------------

class TestPlotCVResults:

    def test_returns_figure(self, cv_dataset, test_config):
        import matplotlib.pyplot as plt
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=3, gap=5, min_train_size=252,
        )
        fig = plot_cv_results(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, cv_dataset, test_config, tmp_path):
        X, y = cv_dataset
        model = RandomForestModel("rf", config=test_config)
        result = walk_forward_cv(
            model, X, y, n_splits=3, gap=5, min_train_size=252,
        )
        save_path = tmp_path / "cv_results.png"
        plot_cv_results(result, save_path=save_path)
        assert save_path.exists()


# ------------------------------------------------------------------
# Integration: synthetic pipeline features
# ------------------------------------------------------------------

class TestWithSyntheticPipelineFeatures:
    """Test walk_forward_cv using features from the FeaturePipeline."""

    def test_cv_with_pipeline_features(self, sample_ohlcv, sample_config):
        from src.features.pipeline import FeaturePipeline
        from src.models.targets import create_direction_target, align_features_and_target

        pipeline = FeaturePipeline(config=sample_config)
        features = pipeline.generate(sample_ohlcv)

        target = create_direction_target(sample_ohlcv["Close"], horizon=5)
        X, y = align_features_and_target(features, target)

        assert len(X) > 300

        model_config = {
            "general": {"random_seed": 42},
            "models": {
                "random_forest": {"n_estimators": 10, "max_depth": 3},
            },
        }
        model = RandomForestModel("rf", config=model_config)
        result = walk_forward_cv(
            model, X, y.astype(int),
            n_splits=3, gap=5, min_train_size=252,
        )

        assert len(result["per_fold"]) == 3
        for fold in result["per_fold"]:
            assert "accuracy" in fold["metrics"]
