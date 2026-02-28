"""Scikit-learn model wrappers for the quant stack.

Each class inherits from :class:`~src.models.base.QuantModel` and wraps a
scikit-learn estimator.  Hyperparameters default to values in
``config/settings.yaml`` under the ``models`` section but can be overridden
at construction time.

Usage:
    from src.models.classical import RandomForestModel
    model = RandomForestModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from src.models.base import QuantModel
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _model_cfg(name: str) -> dict[str, Any]:
    """Return the config section for a named model."""
    cfg = load_config()
    return dict(cfg.get("models", {}).get(name, {}))


# ------------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------------


class RandomForestModel(QuantModel):
    """Random-forest classifier driven by config defaults.

    Args:
        **kwargs: Override any scikit-learn ``RandomForestClassifier``
            parameter.  Values not supplied fall back to
            ``config.models.random_forest``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        params = _model_cfg("random_forest")
        params.update(kwargs)

        # Use the project-wide random seed unless the caller overrides it.
        if "random_state" not in params:
            general_cfg = load_config().get("general", {})
            params["random_state"] = general_cfg.get("random_seed", 42)

        self._estimator = RandomForestClassifier(**params)
        self.metadata["hyperparams"] = params

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the random-forest on labelled features.

        Args:
            X_train: Feature matrix with DatetimeIndex.
            y_train: Binary target series aligned to *X_train*.
        """
        self._estimator.fit(X_train.values, y_train.values)

        self.metadata["train_start"] = str(X_train.index.min().date())
        self.metadata["train_end"] = str(X_train.index.max().date())
        self.metadata["feature_names"] = list(X_train.columns)
        self.metadata["n_train_samples"] = len(X_train)

        logger.info(
            f"RandomForestModel fitted on {len(X_train)} samples "
            f"({self.metadata['train_start']} → {self.metadata['train_end']})"
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate class predictions.

        Args:
            X: Feature matrix with DatetimeIndex.

        Returns:
            Series of predicted classes indexed to match *X*.
        """
        preds = self._estimator.predict(X.values)
        return pd.Series(preds, index=X.index, name="prediction")


# ------------------------------------------------------------------
# Gradient Boosting
# ------------------------------------------------------------------


class GradientBoostingModel(QuantModel):
    """Gradient-boosting classifier driven by config defaults.

    Args:
        **kwargs: Override any scikit-learn ``GradientBoostingClassifier``
            parameter.  Values not supplied fall back to
            ``config.models.gradient_boosting``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        params = _model_cfg("gradient_boosting")
        params.update(kwargs)

        if "random_state" not in params:
            general_cfg = load_config().get("general", {})
            params["random_state"] = general_cfg.get("random_seed", 42)

        self._estimator = GradientBoostingClassifier(**params)
        self.metadata["hyperparams"] = params

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the gradient-boosting model on labelled features.

        Args:
            X_train: Feature matrix with DatetimeIndex.
            y_train: Binary target series aligned to *X_train*.
        """
        self._estimator.fit(X_train.values, y_train.values)

        self.metadata["train_start"] = str(X_train.index.min().date())
        self.metadata["train_end"] = str(X_train.index.max().date())
        self.metadata["feature_names"] = list(X_train.columns)
        self.metadata["n_train_samples"] = len(X_train)

        logger.info(
            f"GradientBoostingModel fitted on {len(X_train)} samples "
            f"({self.metadata['train_start']} → {self.metadata['train_end']})"
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate class predictions.

        Args:
            X: Feature matrix with DatetimeIndex.

        Returns:
            Series of predicted classes indexed to match *X*.
        """
        preds = self._estimator.predict(X.values)
        return pd.Series(preds, index=X.index, name="prediction")
