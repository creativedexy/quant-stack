"""Abstract base class for all quantitative models.

Every model in the pipeline inherits from :class:`QuantModel`, which enforces
a consistent interface for training, prediction, evaluation, and
serialisation.  A ``metadata`` dictionary stores provenance information
(training date range, feature names, hyperparameters) so that any saved
model can be audited later.

Usage:
    class MyModel(QuantModel):
        def fit(self, X_train, y_train): ...
        def predict(self, X): ...
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    mean_squared_error,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


class QuantModel(ABC):
    """Abstract interface shared by every model in the quant stack.

    Subclasses **must** implement :meth:`fit` and :meth:`predict`.
    :meth:`evaluate` has a sensible default but can be overridden.

    Attributes:
        metadata: Dictionary of provenance information populated during
            :meth:`fit`.  Keys include ``train_start``, ``train_end``,
            ``feature_names``, ``hyperparams``, and ``model_class``.
    """

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {
            "model_class": type(self).__name__,
            "train_start": None,
            "train_end": None,
            "feature_names": [],
            "hyperparams": {},
        }

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model on labelled data.

        Implementations **must** populate ``self.metadata`` with at least
        ``train_start``, ``train_end``, and ``feature_names``.

        Args:
            X_train: Feature matrix with DatetimeIndex.
            y_train: Target series aligned to *X_train*.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data.

        Args:
            X: Feature matrix with DatetimeIndex.

        Returns:
            Series of predictions indexed to match *X*.
        """

    # ------------------------------------------------------------------
    # Default evaluate — works for classification and regression
    # ------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Score the model on held-out data.

        For classification targets (binary or categorical) the default
        metrics are accuracy and precision.  For continuous targets the
        default is RMSE and information coefficient (rank correlation).

        Args:
            X_test: Feature matrix with DatetimeIndex.
            y_test: True labels / values.

        Returns:
            Dictionary mapping metric name → value.
        """
        preds = self.predict(X_test)
        metrics: dict[str, float] = {}

        is_classification = _is_classification_target(y_test)

        if is_classification:
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            metrics["precision"] = float(
                precision_score(y_test, preds, average="binary", zero_division=0.0)
            )
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))

        # Information coefficient: Spearman rank correlation between
        # predictions and actuals.  Useful for both classification scores
        # and continuous targets.
        if len(preds) > 2:
            ic = float(preds.corr(y_test, method="spearman"))
            metrics["information_coefficient"] = ic

        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the model and its metadata to disk.

        Creates two files at *path*:
        - ``<path>.pkl`` — pickled model object
        - ``<path>.meta.json`` — human-readable metadata

        Args:
            path: Base file path (without extension).

        Returns:
            Path to the saved pickle file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pkl_path = path.with_suffix(".pkl")
        meta_path = path.with_suffix(".meta.json")

        with open(pkl_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {pkl_path}")
        return pkl_path

    @classmethod
    def load(cls, path: str | Path) -> "QuantModel":
        """Load a previously saved model.

        Args:
            path: Path to the ``.pkl`` file (or base path without extension).

        Returns:
            The deserialised :class:`QuantModel` instance.
        """
        path = Path(path)
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")

        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301

        if not isinstance(model, QuantModel):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected QuantModel subclass"
            )

        logger.info(f"Model loaded from {path}")
        return model


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_classification_target(y: pd.Series) -> bool:
    """Heuristic: treat integer targets with ≤ 10 unique values as classification."""
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y):
        return y.nunique() <= 10
    if pd.api.types.is_float_dtype(y):
        return y.nunique() <= 10 and (y == y.round()).all()
    return False
