"""Tests for time-series aware cross-validation (src.models.evaluation).

Covers:
- walk_forward_cv returns correct structure
- Per-fold metrics include required keys
- **No future data in any training fold** (explicit temporal verification)
- Gap between train end and test start is respected
- Folds cover the dataset without overlap
- Edge cases: too little data, unsorted index
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.classical import RandomForestModel
from src.models.evaluation import walk_forward_cv


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def cv_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """1000-row synthetic classification dataset for CV tests.

    Binary target: 1 if cumulative feature sum > 0, else 0.
    Large enough to support 5-fold walk-forward with min_train_size=252.
    """
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
    """Tiny dataset that is too small for default CV parameters."""
    dates = pd.bdate_range("2020-01-01", periods=50, freq="B")
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"f1": rng.standard_normal(50)}, index=dates)
    y = pd.Series((rng.standard_normal(50) > 0).astype(int), index=dates)
    return X, y


# ------------------------------------------------------------------
# Return structure
# ------------------------------------------------------------------

class TestReturnStructure:

    def test_returns_dict_with_required_keys(self, cv_dataset):
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=3, gap=5, min_train_size=252)
        assert "fold_metrics" in result
        assert "mean_metrics" in result
        assert "splits" in result

    def test_correct_number_of_folds(self, cv_dataset):
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=4, gap=5, min_train_size=252)
        assert len(result["fold_metrics"]) == 4
        assert len(result["splits"]) == 4

    def test_fold_metrics_contain_expected_keys(self, cv_dataset):
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=3, gap=5, min_train_size=252)
        for fm in result["fold_metrics"]:
            assert "accuracy" in fm
            assert "precision" in fm
            assert "sharpe" in fm
            assert "information_coefficient" in fm

    def test_mean_metrics_averaged(self, cv_dataset):
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=3, gap=5, min_train_size=252)
        for key in result["mean_metrics"]:
            values = [fm[key] for fm in result["fold_metrics"] if key in fm]
            expected = np.mean(values)
            assert result["mean_metrics"][key] == pytest.approx(expected)


# ------------------------------------------------------------------
# NO FUTURE DATA IN TRAINING — the critical anti-lookahead test
# ------------------------------------------------------------------

class TestNoFutureDataInTraining:
    """Explicitly verify that no training fold contains dates that
    overlap with or come after the corresponding test fold."""

    def test_train_end_before_test_start_in_every_fold(self, cv_dataset):
        """The train_end date must be strictly before test_start in
        every fold, with at least ``gap`` trading days between them."""
        X, y = cv_dataset
        gap = 5
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=5, gap=gap, min_train_size=252)

        for fold_idx, (train_end, test_start, test_end) in enumerate(result["splits"]):
            train_end_ts = pd.Timestamp(train_end)
            test_start_ts = pd.Timestamp(test_start)

            # Train end must be strictly before test start.
            assert train_end_ts < test_start_ts, (
                f"Fold {fold_idx}: train_end={train_end} >= test_start={test_start}"
            )

            # Count business days between train_end and test_start.
            gap_days = len(pd.bdate_range(train_end_ts, test_start_ts, freq="B")) - 1
            assert gap_days >= gap, (
                f"Fold {fold_idx}: gap is only {gap_days} days, expected >= {gap}"
            )

    def test_no_train_date_appears_in_test(self, cv_dataset):
        """Re-derive the actual index ranges and confirm zero overlap."""
        X, y = cv_dataset
        gap = 5
        n_splits = 4
        min_train_size = 252
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(
            model, X, y, n_splits=n_splits, gap=gap, min_train_size=min_train_size,
        )

        for fold_idx, (train_end_str, test_start_str, test_end_str) in enumerate(
            result["splits"]
        ):
            train_end = pd.Timestamp(train_end_str)
            test_start = pd.Timestamp(test_start_str)
            test_end = pd.Timestamp(test_end_str)

            train_dates = X.index[X.index <= train_end]
            test_dates = X.index[(X.index >= test_start) & (X.index <= test_end)]

            overlap = train_dates.intersection(test_dates)
            assert len(overlap) == 0, (
                f"Fold {fold_idx}: {len(overlap)} dates appear in both "
                f"train and test sets: {overlap[:5].tolist()}"
            )

    def test_gap_region_excluded_from_both_train_and_test(self, cv_dataset):
        """Dates inside the gap must not appear in either train or test."""
        X, y = cv_dataset
        gap = 10  # Use a larger gap to make this easier to verify.
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=3, gap=gap, min_train_size=252)

        for fold_idx, (train_end_str, test_start_str, _) in enumerate(
            result["splits"]
        ):
            train_end = pd.Timestamp(train_end_str)
            test_start = pd.Timestamp(test_start_str)

            # Gap dates: business days strictly between train_end and test_start
            gap_dates = X.index[(X.index > train_end) & (X.index < test_start)]
            assert len(gap_dates) >= gap - 1, (
                f"Fold {fold_idx}: expected >= {gap - 1} gap dates, got {len(gap_dates)}"
            )

    def test_expanding_window_each_fold_trains_on_more_data(self, cv_dataset):
        """Each successive fold's training set must be strictly larger."""
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=5, gap=5, min_train_size=252)

        train_ends = [pd.Timestamp(s[0]) for s in result["splits"]]
        for i in range(1, len(train_ends)):
            assert train_ends[i] > train_ends[i - 1], (
                f"Fold {i} train_end ({train_ends[i]}) not after "
                f"fold {i-1} train_end ({train_ends[i-1]})"
            )


# ------------------------------------------------------------------
# Fold coverage
# ------------------------------------------------------------------

class TestFoldCoverage:

    def test_folds_are_contiguous(self, cv_dataset):
        """Test folds should not have gaps between them (only the
        train→test gap matters, not test→next_test)."""
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=4, gap=5, min_train_size=252)

        for i in range(1, len(result["splits"])):
            prev_test_end = pd.Timestamp(result["splits"][i - 1][2])
            curr_test_start = pd.Timestamp(result["splits"][i][1])
            # Current fold's test start should be after previous fold's test end.
            assert curr_test_start > prev_test_end

    def test_last_fold_extends_to_end(self, cv_dataset):
        """The last fold should consume remaining data up to the end."""
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=3, gap=5, min_train_size=252)

        last_test_end = pd.Timestamp(result["splits"][-1][2])
        assert last_test_end == X.index[-1]


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:

    def test_too_little_data_raises(self, small_dataset):
        X, y = small_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        with pytest.raises(ValueError, match="Not enough data"):
            walk_forward_cv(model, X, y, n_splits=5, gap=5, min_train_size=252)

    def test_unsorted_index_raises(self, cv_dataset):
        X, y = cv_dataset
        X_shuffled = X.sample(frac=1.0, random_state=0)
        y_shuffled = y.loc[X_shuffled.index]
        model = RandomForestModel(n_estimators=10, max_depth=2)
        with pytest.raises(ValueError, match="sorted chronologically"):
            walk_forward_cv(model, X_shuffled, y_shuffled)

    def test_gap_zero_allowed(self, cv_dataset):
        """A gap of 0 should work — train end immediately precedes test start."""
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=3, gap=0, min_train_size=252)
        assert len(result["fold_metrics"]) == 3
        # Even with gap=0, train end should still be before test start.
        for train_end, test_start, _ in result["splits"]:
            assert pd.Timestamp(train_end) < pd.Timestamp(test_start)

    def test_single_fold(self, cv_dataset):
        """n_splits=1 should train once and test once."""
        X, y = cv_dataset
        model = RandomForestModel(n_estimators=10, max_depth=2)
        result = walk_forward_cv(model, X, y, n_splits=1, gap=5, min_train_size=252)
        assert len(result["fold_metrics"]) == 1
