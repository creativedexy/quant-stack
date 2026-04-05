"""Tests for scikit-learn model wrappers (src.models.classical).

Covers:
- RandomForestModel fit/predict/evaluate round-trip
- GradientBoostingModel fit/predict/evaluate round-trip
- Config-driven hyperparameters and caller overrides
- Metadata populated correctly after fit
- Predictions are a pd.Series with the correct index
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.classical import RandomForestModel, GradientBoostingModel


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture
def feature_target_pair() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic classification dataset with DatetimeIndex.

    500 rows, 4 features, binary target (1 if next-day return positive).
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=500, freq="B")
    X = pd.DataFrame(
        {
            "sma_5": rng.standard_normal(500),
            "rsi_14": rng.uniform(20, 80, 500),
            "ret_1d": rng.standard_normal(500) * 0.01,
            "bb_width": rng.uniform(0.5, 3.0, 500),
        },
        index=dates,
    )
    y = pd.Series(
        (rng.standard_normal(500) > 0).astype(int),
        index=dates,
        name="target",
    )
    return X, y


# ------------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------------

class TestRandomForestModel:

    def test_fit_predict_evaluate(self, feature_target_pair):
        X, y = feature_target_pair
        X_train, y_train = X.iloc[:400], y.iloc[:400]
        X_test, y_test = X.iloc[400:], y.iloc[400:]

        model = RandomForestModel("rf", n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X_test)
        assert (preds.index == X_test.index).all()

        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "hit_rate" in metrics

    def test_metadata_after_fit(self, feature_target_pair):
        X, y = feature_target_pair
        model = RandomForestModel("rf", n_estimators=10)
        model.fit(X, y)

        assert model.metadata["name"] == "rf"
        assert model.metadata["trained"] is True
        assert model.metadata["train_date_range"] == (
            str(X.index.min().date()),
            str(X.index.max().date()),
        )
        assert model.metadata["feature_names"] == list(X.columns)
        assert model.metadata["n_train_samples"] == len(X)

    def test_config_defaults_loaded(self):
        model = RandomForestModel()
        params = model.metadata["hyperparameters"]
        # From config/settings.yaml models.random_forest
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 10

    def test_caller_overrides_config(self):
        model = RandomForestModel("rf", n_estimators=50, max_depth=2)
        params = model.metadata["hyperparameters"]
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 2

    def test_predictions_are_binary(self, feature_target_pair):
        X, y = feature_target_pair
        model = RandomForestModel("rf", n_estimators=10, max_depth=3)
        model.fit(X.iloc[:400], y.iloc[:400])
        preds = model.predict(X.iloc[400:])
        assert set(preds.unique()).issubset({0, 1})

    def test_save_load_round_trip(self, feature_target_pair, tmp_path):
        X, y = feature_target_pair
        model = RandomForestModel("rf_model", n_estimators=10)
        model.fit(X.iloc[:400], y.iloc[:400])

        path = model.save(tmp_path)
        from src.models.base import QuantModel
        loaded = QuantModel.load(path)

        assert isinstance(loaded, RandomForestModel)
        preds_original = model.predict(X.iloc[400:])
        preds_loaded = loaded.predict(X.iloc[400:])
        pd.testing.assert_series_equal(preds_original, preds_loaded)


# ------------------------------------------------------------------
# Gradient Boosting
# ------------------------------------------------------------------

class TestGradientBoostingModel:

    def test_fit_predict_evaluate(self, feature_target_pair):
        X, y = feature_target_pair
        X_train, y_train = X.iloc[:400], y.iloc[:400]
        X_test, y_test = X.iloc[400:], y.iloc[400:]

        model = GradientBoostingModel("gb", n_estimators=20, max_depth=2)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X_test)

        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics

    def test_metadata_after_fit(self, feature_target_pair):
        X, y = feature_target_pair
        model = GradientBoostingModel("gb", n_estimators=20)
        model.fit(X, y)

        assert model.metadata["name"] == "gb"
        assert model.metadata["n_train_samples"] == len(X)

    def test_config_defaults_loaded(self):
        model = GradientBoostingModel()
        params = model.metadata["hyperparameters"]
        assert params["n_estimators"] == 300
        assert params["learning_rate"] == 0.05
        assert params["subsample"] == 0.8

    def test_caller_overrides_config(self):
        model = GradientBoostingModel("gb", n_estimators=30, learning_rate=0.1)
        params = model.metadata["hyperparameters"]
        assert params["n_estimators"] == 30
        assert params["learning_rate"] == 0.1
