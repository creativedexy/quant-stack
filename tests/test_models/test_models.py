"""Tests for model wrappers and ModelRegistry (src.models).

Covers:
- RandomForestModel and GradientBoostingModel instantiation
- fit() runs without error on synthetic feature data
- predict() returns Series of correct length
- predict() before fit() raises clear error
- feature_importances() returns Series with correct feature names
- save() and load() round-trip correctly
- ModelRegistry.create() returns correct model types
- ModelRegistry.list_models() returns all registered names
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import QuantModel, ModelRegistry, model_registry
from src.models.classical import RandomForestModel, GradientBoostingModel


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def feature_target_clf() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic classification dataset: 500 rows, 4 features, binary target."""
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


@pytest.fixture
def feature_target_reg() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic regression dataset: 500 rows, 4 features, continuous target."""
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
        rng.standard_normal(500) * 0.01,
        index=dates,
        name="fwd_ret_5d",
    )
    return X, y


@pytest.fixture
def test_config() -> dict:
    """Minimal config for model tests."""
    return {
        "general": {"random_seed": 42},
        "models": {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_leaf": 5,
            },
            "gradient_boosting": {
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.1,
            },
        },
    }


# ------------------------------------------------------------------
# RandomForestModel
# ------------------------------------------------------------------

class TestRandomForestModel:

    def test_instantiation_clf(self, test_config):
        model = RandomForestModel("rf_clf", config=test_config)
        assert model.name == "rf_clf"
        assert model.metadata["trained"] is False

    def test_instantiation_reg(self, test_config):
        model = RandomForestModel("rf_reg", config=test_config, target_type="regression")
        assert model.metadata["target_type"] == "regression"

    def test_invalid_target_type_raises(self, test_config):
        with pytest.raises(ValueError, match="target_type"):
            RandomForestModel("bad", config=test_config, target_type="invalid")

    def test_fit_clf(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        model = RandomForestModel("rf", config=test_config)
        result = model.fit(X, y)
        assert result is model  # fit returns self
        assert model.metadata["trained"] is True
        assert model.metadata["feature_names"] == list(X.columns)
        assert model.metadata["train_date_range"] is not None

    def test_fit_reg(self, feature_target_reg, test_config):
        X, y = feature_target_reg
        model = RandomForestModel("rf", config=test_config, target_type="regression")
        model.fit(X, y)
        assert model.metadata["trained"] is True

    def test_predict_returns_series(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        model = RandomForestModel("rf", config=test_config)
        model.fit(X_train, y.iloc[:400])
        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X_test)
        assert (preds.index == X_test.index).all()

    def test_predict_before_fit_raises(self, feature_target_clf, test_config):
        X, _ = feature_target_clf
        model = RandomForestModel("rf", config=test_config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_predict_wrong_columns_raises(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        model = RandomForestModel("rf", config=test_config)
        model.fit(X, y)
        X_bad = X.rename(columns={"sma_5": "wrong"})
        with pytest.raises(ValueError, match="Feature mismatch"):
            model.predict(X_bad)

    def test_feature_importances(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        model = RandomForestModel("rf", config=test_config)
        model.fit(X, y)
        imp = model.feature_importances()
        assert isinstance(imp, pd.Series)
        assert set(imp.index) == set(X.columns)
        assert (imp >= 0).all()
        assert imp.sum() == pytest.approx(1.0)

    def test_feature_importances_before_fit_raises(self, test_config):
        model = RandomForestModel("rf", config=test_config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.feature_importances()

    def test_save_load_round_trip(self, feature_target_clf, test_config, tmp_path):
        X, y = feature_target_clf
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        model = RandomForestModel("rf_test", config=test_config)
        model.fit(X_train, y.iloc[:400])
        preds_original = model.predict(X_test)

        saved_path = model.save(tmp_path)
        loaded = QuantModel.load(saved_path)

        assert isinstance(loaded, RandomForestModel)
        assert loaded.name == "rf_test"
        preds_loaded = loaded.predict(X_test)
        pd.testing.assert_series_equal(preds_original, preds_loaded)

    def test_config_defaults(self, test_config):
        model = RandomForestModel("rf", config=test_config)
        params = model.metadata["hyperparameters"]
        assert params["n_estimators"] == 10
        assert params["max_depth"] == 3

    def test_kwargs_override_config(self, test_config):
        model = RandomForestModel("rf", config=test_config, n_estimators=50)
        assert model.metadata["hyperparameters"]["n_estimators"] == 50

    def test_classification_predictions_binary(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        model = RandomForestModel("rf", config=test_config)
        model.fit(X.iloc[:400], y.iloc[:400])
        preds = model.predict(X.iloc[400:])
        assert set(preds.unique()).issubset({0, 1})

    def test_regression_predictions_continuous(self, feature_target_reg, test_config):
        X, y = feature_target_reg
        model = RandomForestModel("rf", config=test_config, target_type="regression")
        model.fit(X.iloc[:400], y.iloc[:400])
        preds = model.predict(X.iloc[400:])
        # Regression predictions are continuous — more than 2 unique values
        assert preds.nunique() > 2


# ------------------------------------------------------------------
# GradientBoostingModel
# ------------------------------------------------------------------

class TestGradientBoostingModel:

    def test_instantiation(self, test_config):
        model = GradientBoostingModel("gb", config=test_config)
        assert model.name == "gb"
        assert model.metadata["trained"] is False

    def test_instantiation_regression(self, test_config):
        model = GradientBoostingModel("gb", config=test_config, target_type="regression")
        assert model.metadata["target_type"] == "regression"

    def test_fit_predict(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        model = GradientBoostingModel("gb", config=test_config)
        model.fit(X_train, y.iloc[:400])
        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X_test)

    def test_predict_before_fit_raises(self, feature_target_clf, test_config):
        X, _ = feature_target_clf
        model = GradientBoostingModel("gb", config=test_config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_feature_importances(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        model = GradientBoostingModel("gb", config=test_config)
        model.fit(X, y)
        imp = model.feature_importances()
        assert isinstance(imp, pd.Series)
        assert set(imp.index) == set(X.columns)

    def test_evaluate(self, feature_target_clf, test_config):
        X, y = feature_target_clf
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        model = GradientBoostingModel("gb", config=test_config)
        model.fit(X_train, y.iloc[:400])
        metrics = model.evaluate(X_test, y.iloc[400:])
        assert "accuracy" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "hit_rate" in metrics
        assert "information_coefficient" in metrics

    def test_config_defaults(self, test_config):
        model = GradientBoostingModel("gb", config=test_config)
        params = model.metadata["hyperparameters"]
        assert params["n_estimators"] == 20
        assert params["max_depth"] == 2

    def test_save_load_round_trip(self, feature_target_clf, test_config, tmp_path):
        X, y = feature_target_clf
        model = GradientBoostingModel("gb_test", config=test_config)
        model.fit(X.iloc[:400], y.iloc[:400])
        preds_original = model.predict(X.iloc[400:])

        saved_path = model.save(tmp_path)
        loaded = QuantModel.load(saved_path)

        assert isinstance(loaded, GradientBoostingModel)
        preds_loaded = loaded.predict(X.iloc[400:])
        pd.testing.assert_series_equal(preds_original, preds_loaded)


# ------------------------------------------------------------------
# QuantModel abstract class
# ------------------------------------------------------------------

class TestQuantModelAbstract:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            QuantModel()

    def test_evaluate_regression(self, feature_target_reg, test_config):
        """evaluate() works for regression targets (no accuracy)."""
        X, y = feature_target_reg
        model = RandomForestModel("rf", config=test_config, target_type="regression")
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "information_coefficient" in metrics
        assert "hit_rate" in metrics
        # accuracy should NOT be present for regression
        assert "accuracy" not in metrics

    def test_evaluate_classification(self, feature_target_clf, test_config):
        """evaluate() includes accuracy for classification targets."""
        X, y = feature_target_clf
        model = RandomForestModel("rf", config=test_config)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "hit_rate" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ------------------------------------------------------------------
# ModelRegistry
# ------------------------------------------------------------------

class TestModelRegistry:

    def test_create_random_forest(self, test_config):
        model = model_registry.create("random_forest", config=test_config)
        assert isinstance(model, RandomForestModel)

    def test_create_gradient_boosting(self, test_config):
        model = model_registry.create("gradient_boosting", config=test_config)
        assert isinstance(model, GradientBoostingModel)

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            model_registry.create("nonexistent")

    def test_list_models(self):
        names = model_registry.list_models()
        assert "random_forest" in names
        assert "gradient_boosting" in names
        assert names == sorted(names)

    def test_register_custom_model(self, test_config):
        """Custom models can be added to the registry."""
        class _DummyModel(QuantModel):
            def fit(self, X_train, y_train):
                return self
            def predict(self, X):
                return pd.Series(0, index=X.index)

        reg = ModelRegistry()
        reg.register("dummy", _DummyModel)
        model = reg.create("dummy", config=test_config)
        assert isinstance(model, _DummyModel)

    def test_register_non_quant_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(TypeError):
            reg.register("bad", dict)  # type: ignore[arg-type]

    def test_create_with_kwargs(self, test_config):
        model = model_registry.create(
            "random_forest", config=test_config, target_type="regression",
        )
        assert isinstance(model, RandomForestModel)
        assert model.metadata["target_type"] == "regression"
