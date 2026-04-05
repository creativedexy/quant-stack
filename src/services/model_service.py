"""Model service -- train, evaluate, and persist ML models.

Thin service layer wrapping model training and evaluation for use by
the scheduler pipeline and dashboard.  The service never imports from
src/data/ directly -- it receives feature DataFrames and delegates to
sklearn and src/models/ for the heavy lifting.

Usage:
    from src.services.model_service import ModelService
    service = ModelService(config)
    result = service.train_and_save("SHEL.L", features_df)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_TRAIN_ROWS = 252  # One year of trading days


class ModelService:
    """Train, evaluate, and persist ML models for the pipeline.

    Args:
        config: Project configuration dict.  If ``None``, uses empty
            defaults.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._config = config

        models_cfg = config.get("models", {})
        self._output_dir = Path(
            models_cfg.get("output_dir", "data/processed/models")
        )
        self._target_column = models_cfg.get("target_column", "forward_return_5d")
        self._cv_splits = models_cfg.get("cv_splits", 5)
        self._default_model_cfg = models_cfg.get("default_model", {})
        self._seed = config.get("general", {}).get("random_seed", 42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_and_save(
        self,
        ticker: str,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """Train a model with walk-forward CV and persist to disk.

        Uses :class:`sklearn.model_selection.TimeSeriesSplit` for
        cross-validation to avoid lookahead bias.

        Args:
            ticker: Ticker symbol (used for file naming).
            features: DataFrame with DatetimeIndex containing feature
                columns and the target column (``target_column`` from
                config).

        Returns:
            Dictionary with:
            - ticker: str
            - cv_mean: float -- mean CV score
            - cv_std: float -- std of CV scores
            - model_path: str -- path to saved model
            - feature_count: int
        """
        target_col = self._target_column

        if target_col not in features.columns:
            logger.error(
                "[%s] Target column '%s' not found in features", ticker, target_col
            )
            return self._skip_result(ticker, f"Target column '{target_col}' missing")

        X = features.drop(columns=[target_col])
        y = features[target_col]

        # Drop rows with NaN in target or features
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < _MIN_TRAIN_ROWS:
            logger.warning(
                "[%s] Only %d rows (need %d) -- skipping training",
                ticker, len(X), _MIN_TRAIN_ROWS,
            )
            return self._skip_result(ticker, f"Too few rows ({len(X)})")

        if X.shape[1] == 0:
            logger.warning("[%s] Zero feature columns -- skipping training", ticker)
            return self._skip_result(ticker, "Zero feature columns")

        model = self._create_model()
        tscv = TimeSeriesSplit(n_splits=self._cv_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")

        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))

        logger.info(
            "[%s] CV accuracy: %.4f +/- %.4f (%d splits, %d features, %d rows)",
            ticker, cv_mean, cv_std, self._cv_splits, X.shape[1], len(X),
        )

        # Final fit on all data
        model.fit(X, y)

        # Persist model and metrics
        model_path = self._save_model(ticker, model)
        self._save_metrics(ticker, cv_mean, cv_std, X.shape[1], len(X))

        return {
            "ticker": ticker,
            "cv_mean": round(cv_mean, 6),
            "cv_std": round(cv_std, 6),
            "model_path": str(model_path),
            "feature_count": X.shape[1],
        }

    def load_model(self, ticker: str) -> Any | None:
        """Load a previously saved model for a ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            The deserialised model, or ``None`` if no saved model exists.
        """
        safe = self._safe_name(ticker)
        path = self._output_dir / f"{safe}_model.pkl"
        if not path.exists():
            return None
        return joblib.load(path)

    def evaluate(
        self,
        ticker: str,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """Re-run walk-forward CV without retraining (evaluation only).

        Args:
            ticker: Ticker symbol.
            features: DataFrame with target column included.

        Returns:
            Dictionary with CV scores and metadata.
        """
        target_col = self._target_column

        if target_col not in features.columns:
            return self._skip_result(ticker, f"Target column '{target_col}' missing")

        X = features.drop(columns=[target_col])
        y = features[target_col]

        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < _MIN_TRAIN_ROWS:
            return self._skip_result(ticker, f"Too few rows ({len(X)})")

        if X.shape[1] == 0:
            return self._skip_result(ticker, "Zero feature columns")

        model = self._create_model()
        tscv = TimeSeriesSplit(n_splits=self._cv_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")

        return {
            "ticker": ticker,
            "cv_mean": round(float(np.mean(scores)), 6),
            "cv_std": round(float(np.std(scores)), 6),
            "model_path": None,
            "feature_count": X.shape[1],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_model(self) -> Any:
        """Instantiate a fresh sklearn model from config."""
        class_name = self._default_model_cfg.get("class", "GradientBoostingClassifier")
        params = dict(self._default_model_cfg.get("params", {}))
        params.setdefault("random_state", self._seed)

        # Resolve class from sklearn
        from sklearn import ensemble

        cls = getattr(ensemble, class_name, None)
        if cls is None:
            raise ValueError(
                f"Unknown sklearn ensemble class: {class_name}"
            )
        return cls(**params)

    def _save_model(self, ticker: str, model: Any) -> Path:
        """Persist a trained model to disk via joblib."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe = self._safe_name(ticker)
        path = self._output_dir / f"{safe}_model.pkl"
        joblib.dump(model, path)
        logger.info("[%s] Model saved to %s", ticker, path)
        return path

    def _save_metrics(
        self,
        ticker: str,
        cv_mean: float,
        cv_std: float,
        feature_count: int,
        train_rows: int,
    ) -> Path:
        """Persist CV metrics as JSON alongside the model."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe = self._safe_name(ticker)
        path = self._output_dir / f"{safe}_metrics.json"

        class_name = self._default_model_cfg.get("class", "GradientBoostingClassifier")

        metrics = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "model_type": class_name,
            "cv_mean": round(cv_mean, 6),
            "cv_std": round(cv_std, 6),
            "feature_count": feature_count,
            "train_rows": train_rows,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("[%s] Metrics saved to %s", ticker, path)
        return path

    @staticmethod
    def _safe_name(ticker: str) -> str:
        """Convert a ticker to a filesystem-safe name."""
        return ticker.replace(".", "_").replace("^", "idx_")

    @staticmethod
    def _skip_result(ticker: str, reason: str) -> dict[str, Any]:
        """Return a skip/failure result dict."""
        return {
            "ticker": ticker,
            "cv_mean": 0.0,
            "cv_std": 0.0,
            "model_path": None,
            "feature_count": 0,
            "skipped": True,
            "reason": reason,
        }
