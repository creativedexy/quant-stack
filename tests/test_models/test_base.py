"""Tests for the abstract QuantModel base class (src.models.base).

Covers:
- Metadata initialisation
- evaluate() with classification and regression targets
- save/load round-trip
- Cannot instantiate QuantModel directly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import QuantModel


# ------------------------------------------------------------------
# Concrete stub for testing the abstract base
# ------------------------------------------------------------------

class _StubModel(QuantModel):
    """Minimal concrete subclass used only in tests."""

    def __init__(self, **kwargs) -> None:
        super().__init__(name="stub_model")
        self.metadata["hyperparameters"] = kwargs

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "_StubModel":
        self.metadata["trained"] = True
        self.metadata["train_date_range"] = (
            str(X_train.index.min().date()),
            str(X_train.index.max().date()),
        )
        self.metadata["feature_names"] = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Predict the majority class (1) for all rows.
        return pd.Series(np.ones(len(X), dtype=int), index=X.index, name="prediction")


# ------------------------------------------------------------------
# Fixtures local to this module
# ------------------------------------------------------------------

@pytest.fixture
def xy_classification() -> tuple[pd.DataFrame, pd.Series]:
    """Small classification dataset with DatetimeIndex."""
    dates = pd.bdate_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {"feat_a": rng.standard_normal(100), "feat_b": rng.standard_normal(100)},
        index=dates,
    )
    y = pd.Series((rng.standard_normal(100) > 0).astype(int), index=dates, name="target")
    return X, y


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestQuantModelCannotInstantiate:
    def test_raises_type_error(self):
        with pytest.raises(TypeError):
            QuantModel()


class TestMetadata:
    def test_initial_metadata(self):
        model = _StubModel()
        assert model.metadata["name"] == "stub_model"
        assert model.metadata["trained"] is False
        assert model.metadata["feature_names"] == []

    def test_metadata_after_fit(self, xy_classification):
        X, y = xy_classification
        model = _StubModel()
        model.fit(X, y)
        assert model.metadata["trained"] is True
        assert model.metadata["train_date_range"] == (
            str(X.index.min().date()),
            str(X.index.max().date()),
        )
        assert model.metadata["feature_names"] == ["feat_a", "feat_b"]


class TestEvaluate:
    def test_classification_metrics(self, xy_classification):
        X, y = xy_classification
        model = _StubModel()
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "hit_rate" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_information_coefficient_present(self, xy_classification):
        X, y = xy_classification
        model = _StubModel()
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "information_coefficient" in metrics


class TestSaveLoad:
    def test_round_trip(self, xy_classification, tmp_path):
        X, y = xy_classification
        model = _StubModel(alpha=0.1)
        model.fit(X, y)

        saved_path = model.save(tmp_path)
        assert saved_path.exists()
        assert (tmp_path / "stub_model" / "metadata.json").exists()

        loaded = QuantModel.load(saved_path)
        assert isinstance(loaded, _StubModel)
        assert loaded.metadata["train_date_range"] == model.metadata["train_date_range"]
        assert loaded.metadata["hyperparameters"] == {"alpha": 0.1}

    def test_load_from_directory(self, xy_classification, tmp_path):
        X, y = xy_classification
        model = _StubModel()
        model.fit(X, y)
        model.save(tmp_path)
        # Load by pointing at the model directory
        loaded = QuantModel.load(tmp_path / "stub_model")
        assert isinstance(loaded, _StubModel)
