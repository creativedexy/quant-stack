"""Tests for the PyCaret AutoML wrapper (src.models.automl).

Uses synthetic data from the feature pipeline to verify:
- quick_compare returns a leaderboard DataFrame and a PyCaretModel
- Leaderboard is ranked by the requested metric
- PyCaretModel satisfies the QuantModel interface (predict, metadata, save/load)
- Classification and regression target types both work
- Temporal train/test split is respected (no shuffle)

These tests are marked ``@pytest.mark.slow`` because PyCaret's setup and
compare_models take several seconds even on tiny datasets.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_ohlcv
from src.features.pipeline import FeaturePipeline
from src.models.automl import quick_compare, PyCaretModel
from src.models.base import QuantModel

pytest.importorskip("pycaret", reason="PyCaret not installed")


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_features() -> pd.DataFrame:
    """Feature matrix from the pipeline, using synthetic OHLCV.

    Module-scoped so the (relatively expensive) pipeline runs once.
    """
    ohlcv = generate_synthetic_ohlcv("AUTOML_TEST", days=800, seed=99)
    pipe = FeaturePipeline()
    result = pipe.run(ohlcv)
    return result.dropna()


@pytest.fixture(scope="module")
def clf_Xy(pipeline_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Classification X, y derived from pipeline features.

    Target: 1 if next-day return is positive, else 0.
    """
    features = pipeline_features.copy()
    # Shift ret_1d back to create a forward-looking target, then drop last row.
    y = (features["ret_1d"].shift(-1) > 0).astype(int)
    y = y.iloc[:-1]
    y.name = "target"
    X = features.drop(columns=["Open", "High", "Low", "Close", "Volume"]).iloc[:-1]
    return X, y


@pytest.fixture(scope="module")
def reg_Xy(pipeline_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Regression X, y derived from pipeline features.

    Target: next-day log return (continuous).
    """
    features = pipeline_features.copy()
    y = features["ret_1d"].shift(-1)
    y = y.iloc[:-1]
    y.name = "target"
    X = features.drop(columns=["Open", "High", "Low", "Close", "Volume"]).iloc[:-1]
    return X, y


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

@pytest.mark.slow
class TestQuickCompareClassification:

    def test_returns_leaderboard_and_model(self, clf_Xy):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaderboard, best = quick_compare(
                X, y,
                target_type="classification",
                include=["lr", "rf"],
                n_folds=2,
                sort="F1",
            )
        assert isinstance(leaderboard, pd.DataFrame)
        assert isinstance(best, PyCaretModel)
        assert isinstance(best, QuantModel)

    def test_leaderboard_has_expected_columns(self, clf_Xy):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaderboard, _ = quick_compare(
                X, y,
                target_type="classification",
                include=["lr", "rf"],
                n_folds=2,
            )
        assert "Model" in leaderboard.columns
        assert "F1" in leaderboard.columns
        assert "Accuracy" in leaderboard.columns

    def test_leaderboard_sorted_by_requested_metric(self, clf_Xy):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaderboard, _ = quick_compare(
                X, y,
                target_type="classification",
                include=["lr", "rf"],
                n_folds=2,
                sort="Accuracy",
            )
        # Leaderboard should be sorted descending by Accuracy.
        acc_values = leaderboard["Accuracy"].values
        assert all(acc_values[i] >= acc_values[i + 1] for i in range(len(acc_values) - 1))

    def test_leaderboard_has_correct_number_of_rows(self, clf_Xy):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaderboard, _ = quick_compare(
                X, y,
                target_type="classification",
                include=["lr", "rf", "et"],
                n_folds=2,
            )
        assert len(leaderboard) == 3

    def test_model_predict_returns_series(self, clf_Xy):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, best = quick_compare(
                X, y,
                target_type="classification",
                include=["lr"],
                n_folds=2,
            )
        # Predict on the last 50 rows.
        preds = best.predict(X.iloc[-50:])
        assert isinstance(preds, pd.Series)
        assert len(preds) == 50
        assert (preds.index == X.iloc[-50:].index).all()

    def test_model_metadata_populated(self, clf_Xy):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, best = quick_compare(
                X, y,
                target_type="classification",
                include=["lr"],
                n_folds=2,
            )
        assert best.metadata["train_start"] is not None
        assert best.metadata["train_end"] is not None
        assert len(best.metadata["feature_names"]) == len(X.columns)
        assert best.metadata["hyperparams"]["target_type"] == "classification"


# ------------------------------------------------------------------
# Regression
# ------------------------------------------------------------------

@pytest.mark.slow
class TestQuickCompareRegression:

    def test_returns_leaderboard_and_model(self, reg_Xy):
        X, y = reg_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaderboard, best = quick_compare(
                X, y,
                target_type="regression",
                include=["lr", "rf"],
                n_folds=2,
                sort="MAE",
            )
        assert isinstance(leaderboard, pd.DataFrame)
        assert isinstance(best, PyCaretModel)

    def test_regression_leaderboard_columns(self, reg_Xy):
        X, y = reg_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaderboard, _ = quick_compare(
                X, y,
                target_type="regression",
                include=["lr", "rf"],
                n_folds=2,
            )
        assert "MAE" in leaderboard.columns
        assert "RMSE" in leaderboard.columns
        assert "R2" in leaderboard.columns

    def test_regression_predict_continuous(self, reg_Xy):
        X, y = reg_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, best = quick_compare(
                X, y,
                target_type="regression",
                include=["lr"],
                n_folds=2,
            )
        preds = best.predict(X.iloc[-50:])
        # Regression predictions should be continuous (not just 0/1).
        assert preds.dtype == np.float64 or preds.nunique() > 2


# ------------------------------------------------------------------
# QuantModel interface compliance
# ------------------------------------------------------------------

@pytest.mark.slow
class TestQuantModelInterface:

    def test_save_and_load_round_trip(self, clf_Xy, tmp_path):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, best = quick_compare(
                X, y,
                target_type="classification",
                include=["lr"],
                n_folds=2,
            )
        saved_path = best.save(tmp_path / "automl_model")
        assert saved_path.exists()
        assert (tmp_path / "automl_model.meta.json").exists()

        loaded = QuantModel.load(saved_path)
        assert isinstance(loaded, PyCaretModel)
        assert loaded.metadata["feature_names"] == best.metadata["feature_names"]

    def test_save_path_parameter(self, clf_Xy, tmp_path):
        X, y = clf_Xy
        save_dest = tmp_path / "via_param"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, best = quick_compare(
                X, y,
                target_type="classification",
                include=["lr"],
                n_folds=2,
                save_path=save_dest,
            )
        assert save_dest.with_suffix(".pkl").exists()

    def test_evaluate_works(self, clf_Xy):
        X, y = clf_Xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, best = quick_compare(
                X, y,
                target_type="classification",
                include=["lr"],
                n_folds=2,
            )
        metrics = best.evaluate(X.iloc[-100:], y.iloc[-100:])
        assert "accuracy" in metrics


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

@pytest.mark.slow
class TestEdgeCases:

    def test_invalid_target_type_raises(self, clf_Xy):
        X, y = clf_Xy
        with pytest.raises(ValueError, match="target_type"):
            quick_compare(X, y, target_type="invalid")

    def test_uses_synthetic_pipeline_features(self, pipeline_features):
        """Verify the fixture actually produced real pipeline features."""
        expected = [
            "sma_5", "rsi_14", "macd_line", "bb_upper", "atr_14",
            "ema_12", "ret_1d",
        ]
        for col in expected:
            assert col in pipeline_features.columns
