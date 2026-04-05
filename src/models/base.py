"""Abstract base class and model registry for all quantitative models.

Every model in the pipeline inherits from :class:`QuantModel`, which enforces
a consistent interface for training, prediction, evaluation, and
serialisation.  A ``metadata`` dictionary stores provenance information
(training date range, feature names, hyperparameters) so that any saved
model can be audited later.

The :class:`ModelRegistry` provides a central catalogue of available model
types and is pre-populated with the concrete models at import time.

Usage:
    class MyModel(QuantModel):
        def fit(self, X_train, y_train): ...
        def predict(self, X): ...

    from src.models.base import model_registry
    model = model_registry.create("random_forest", config=cfg)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


class QuantModel(ABC):
    """Base class for all quant models.

    Enforces a consistent interface so models are interchangeable in the
    pipeline.

    Subclasses **must** implement :meth:`fit` and :meth:`predict`.
    :meth:`evaluate` has a sensible default but can be overridden.

    Args:
        name: Human-readable model name used for serialisation and logging.
        config: Full project config dict.  If ``None``, a minimal empty
            dict is used.
    """

    def __init__(self, name: str | None = None, config: dict | None = None) -> None:
        self.name = name or type(self).__name__
        self.config = config or {}
        self.metadata: dict[str, Any] = {
            "name": self.name,
            "trained": False,
            "train_date_range": None,
            "feature_names": [],
            "hyperparameters": {},
            "metrics": {},
        }

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "QuantModel":
        """Train the model.

        Implementations **must** update ``self.metadata`` with training info
        (at least ``trained``, ``train_date_range``, ``feature_names``).

        Args:
            X_train: Feature matrix with DatetimeIndex.
            y_train: Target series aligned to *X_train*.

        Returns:
            Self for chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions.  Must NOT modify *X*.

        Args:
            X: Feature matrix with DatetimeIndex.

        Returns:
            Series of predictions indexed to match *X*.
        """

    # ------------------------------------------------------------------
    # Default evaluate
    # ------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Run predictions and compute standard metrics.

        Returns:
            Dictionary with: ``accuracy`` (if classification), ``rmse``,
            ``mae``, ``information_coefficient`` (rank correlation between
            predictions and actuals), ``hit_rate`` (% of correct direction
            predictions).
        """
        preds = self.predict(X_test)
        metrics: dict[str, float] = {}

        is_clf = _is_classification_target(y_test)

        if is_clf:
            metrics["accuracy"] = float(accuracy_score(y_test, preds))

        # RMSE and MAE
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["mae"] = float(mean_absolute_error(y_test, preds))

        # Information coefficient: Spearman rank correlation
        if len(preds) > 2:
            corr, _ = spearmanr(preds, y_test)
            metrics["information_coefficient"] = float(corr) if np.isfinite(corr) else 0.0

        # Hit rate: % of correct direction predictions
        if is_clf:
            metrics["hit_rate"] = float(accuracy_score(y_test, preds))
        else:
            if len(preds) > 0:
                same_sign = (np.sign(preds) == np.sign(y_test)).astype(float)
                metrics["hit_rate"] = float(same_sign.mean())

        return metrics

    # ------------------------------------------------------------------
    # Serialisation — joblib + JSON metadata
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise model + metadata to disk using joblib.

        Creates a directory named after ``self.name`` under *path*
        containing ``model.joblib`` and ``metadata.json``.

        Args:
            path: Parent directory for the model directory.

        Returns:
            Path to the saved ``model.joblib`` file.
        """
        path = Path(path)
        model_dir = path / self.name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"

        joblib.dump(self, model_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info("Model saved to %s", model_path)
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> "QuantModel":
        """Deserialise model + metadata from disk.

        Args:
            path: Path to either the ``model.joblib`` file or the model
                directory containing it.

        Returns:
            The deserialised :class:`QuantModel` instance.
        """
        path = Path(path)
        if path.suffix == ".joblib":
            model_path = path
        elif path.is_dir():
            model_path = path / "model.joblib"
        else:
            model_path = path.with_suffix(".joblib")

        model = joblib.load(model_path)

        if not isinstance(model, QuantModel):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected QuantModel subclass"
            )

        logger.info("Model loaded from %s", model_path)
        return model


# ------------------------------------------------------------------
# Model Registry
# ------------------------------------------------------------------


class ModelRegistry:
    """Central catalogue of available model types.

    Usage:
        registry = ModelRegistry()
        registry.register("random_forest", RandomForestModel)
        model = registry.create("random_forest", config=cfg)
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[QuantModel]] = {}

    def register(self, name: str, model_class: type[QuantModel]) -> None:
        """Add a model class to the registry.

        Args:
            name: Short name used for lookup (e.g. ``"random_forest"``).
            model_class: The model class (must be a QuantModel subclass).
        """
        if not (isinstance(model_class, type) and issubclass(model_class, QuantModel)):
            raise TypeError(f"{model_class} is not a QuantModel subclass")
        self._registry[name] = model_class

    def create(self, name: str, config: dict | None = None, **kwargs: Any) -> QuantModel:
        """Instantiate a registered model.

        Args:
            name: Registered model name.
            config: Project config dict passed to the model constructor.
            **kwargs: Additional keyword arguments forwarded to the model
                constructor.

        Returns:
            A fresh, unfitted model instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._registry:
            raise KeyError(
                f"Unknown model '{name}'. Available: {self.list_models()}"
            )
        return self._registry[name](name=name, config=config, **kwargs)

    def list_models(self) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(self._registry.keys())


# Module-level registry instance (populated in __init__.py after models are defined).
model_registry = ModelRegistry()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_classification_target(y: pd.Series) -> bool:
    """Heuristic: treat integer targets with <= 10 unique values as classification."""
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y):
        return y.nunique() <= 10
    if pd.api.types.is_float_dtype(y):
        return y.nunique() <= 10 and (y == y.round()).all()
    return False
