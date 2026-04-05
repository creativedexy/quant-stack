"""Scikit-learn model wrappers for the quant stack.

Each class inherits from :class:`~src.models.base.QuantModel` and wraps a
scikit-learn estimator.  Hyperparameters default to values in
``config/settings.yaml`` under the ``models`` section but can be overridden
at construction time.

Both models support ``target_type='classification'`` (default) for direction
prediction and ``target_type='regression'`` for return prediction, switching
between Classifier and Regressor variants automatically.

Usage:
    from src.models.classical import RandomForestModel
    model = RandomForestModel("rf_direction")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    importances = model.feature_importances()
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from src.models.base import QuantModel
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _model_cfg(name: str, config: dict | None = None) -> dict[str, Any]:
    """Return the config section for a named model."""
    if config is not None:
        return dict(config.get("models", {}).get(name, {}))
    cfg = load_config()
    return dict(cfg.get("models", {}).get(name, {}))


def _random_state(config: dict | None = None) -> int:
    """Return the project-wide random seed."""
    if config is not None:
        return config.get("general", {}).get("random_seed", 42)
    return load_config().get("general", {}).get("random_seed", 42)


# ------------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------------


class RandomForestModel(QuantModel):
    """Random-forest model driven by config defaults.

    Wraps :class:`~sklearn.ensemble.RandomForestClassifier` for direction
    prediction or :class:`~sklearn.ensemble.RandomForestRegressor` for
    return prediction.

    Args:
        name: Model name for serialisation.
        config: Full project config dict.  If ``None``, loads from file.
        target_type: ``"classification"`` or ``"regression"``.
        **kwargs: Override any scikit-learn parameter.  Values not supplied
            fall back to ``config.models.random_forest``.
    """

    def __init__(
        self,
        name: str | None = None,
        config: dict | None = None,
        target_type: str = "classification",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name or "random_forest", config=config)
        self._target_type = target_type
        params = _model_cfg("random_forest", config)
        params.update(kwargs)

        if "random_state" not in params:
            params["random_state"] = _random_state(config)

        self._estimator_params = dict(params)

        if target_type == "classification":
            self._estimator = RandomForestClassifier(**params)
        elif target_type == "regression":
            self._estimator = RandomForestRegressor(**params)
        else:
            raise ValueError(
                f"target_type must be 'classification' or 'regression', "
                f"got '{target_type}'"
            )

        self.metadata["hyperparameters"] = params
        self.metadata["target_type"] = target_type

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestModel":
        """Train the random-forest on labelled features.

        Args:
            X_train: Feature matrix with DatetimeIndex.
            y_train: Target series aligned to *X_train*.

        Returns:
            Self for chaining.
        """
        self._estimator.fit(X_train.values, y_train.values)

        self.metadata["trained"] = True
        self.metadata["train_date_range"] = (
            str(X_train.index.min().date()),
            str(X_train.index.max().date()),
        )
        self.metadata["feature_names"] = list(X_train.columns)
        self.metadata["n_train_samples"] = len(X_train)

        logger.info(
            "RandomForestModel fitted on %d samples (%s -> %s)",
            len(X_train),
            self.metadata["train_date_range"][0],
            self.metadata["train_date_range"][1],
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions.

        Args:
            X: Feature matrix with DatetimeIndex.

        Returns:
            Series of predictions indexed to match *X*.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.metadata["trained"]:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )
        expected = self.metadata["feature_names"]
        if list(X.columns) != expected:
            raise ValueError(
                f"Feature mismatch: expected {expected}, got {list(X.columns)}"
            )
        preds = self._estimator.predict(X.values)
        return pd.Series(preds, index=X.index, name="prediction")

    def feature_importances(self) -> pd.Series:
        """Return feature importances as a named Series.

        Returns:
            Series mapping feature name to importance score,
            sorted descending.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.metadata["trained"]:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before feature_importances()."
            )
        importances = self._estimator.feature_importances_
        return pd.Series(
            importances,
            index=self.metadata["feature_names"],
            name="importance",
        ).sort_values(ascending=False)


# ------------------------------------------------------------------
# Gradient Boosting
# ------------------------------------------------------------------


class GradientBoostingModel(QuantModel):
    """Gradient-boosting model driven by config defaults.

    Wraps :class:`~sklearn.ensemble.GradientBoostingClassifier` for
    direction prediction or :class:`~sklearn.ensemble.GradientBoostingRegressor`
    for return prediction.

    Args:
        name: Model name for serialisation.
        config: Full project config dict.  If ``None``, loads from file.
        target_type: ``"classification"`` or ``"regression"``.
        **kwargs: Override any scikit-learn parameter.
    """

    def __init__(
        self,
        name: str | None = None,
        config: dict | None = None,
        target_type: str = "classification",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name or "gradient_boosting", config=config)
        self._target_type = target_type
        params = _model_cfg("gradient_boosting", config)
        params.update(kwargs)

        if "random_state" not in params:
            params["random_state"] = _random_state(config)

        self._estimator_params = dict(params)

        if target_type == "classification":
            self._estimator = GradientBoostingClassifier(**params)
        elif target_type == "regression":
            self._estimator = GradientBoostingRegressor(**params)
        else:
            raise ValueError(
                f"target_type must be 'classification' or 'regression', "
                f"got '{target_type}'"
            )

        self.metadata["hyperparameters"] = params
        self.metadata["target_type"] = target_type

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "GradientBoostingModel":
        """Train the gradient-boosting model on labelled features.

        Args:
            X_train: Feature matrix with DatetimeIndex.
            y_train: Target series aligned to *X_train*.

        Returns:
            Self for chaining.
        """
        self._estimator.fit(X_train.values, y_train.values)

        self.metadata["trained"] = True
        self.metadata["train_date_range"] = (
            str(X_train.index.min().date()),
            str(X_train.index.max().date()),
        )
        self.metadata["feature_names"] = list(X_train.columns)
        self.metadata["n_train_samples"] = len(X_train)

        logger.info(
            "GradientBoostingModel fitted on %d samples (%s -> %s)",
            len(X_train),
            self.metadata["train_date_range"][0],
            self.metadata["train_date_range"][1],
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions.

        Args:
            X: Feature matrix with DatetimeIndex.

        Returns:
            Series of predictions indexed to match *X*.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.metadata["trained"]:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )
        expected = self.metadata["feature_names"]
        if list(X.columns) != expected:
            raise ValueError(
                f"Feature mismatch: expected {expected}, got {list(X.columns)}"
            )
        preds = self._estimator.predict(X.values)
        return pd.Series(preds, index=X.index, name="prediction")

    def feature_importances(self) -> pd.Series:
        """Return feature importances as a named Series.

        Returns:
            Series mapping feature name to importance score,
            sorted descending.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.metadata["trained"]:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before feature_importances()."
            )
        importances = self._estimator.feature_importances_
        return pd.Series(
            importances,
            index=self.metadata["feature_names"],
            name="importance",
        ).sort_values(ascending=False)
