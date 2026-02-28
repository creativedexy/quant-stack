"""Time-series aware cross-validation and scoring.

Provides :func:`walk_forward_cv` which splits data temporally using an
expanding window, inserts a configurable *gap* between the training and
test periods to avoid autocorrelation leakage, and returns per-fold
metrics.

**CRITICAL**: sklearn's default random splitters must *never* be used for
financial time-series.  All splitting in this module respects strict
chronological ordering.

Usage:
    from src.models.evaluation import walk_forward_cv
    results = walk_forward_cv(model, X, y, n_splits=5, gap=5)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

from src.models.base import QuantModel, _is_classification_target
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _eval_cfg() -> dict[str, Any]:
    """Return the evaluation section from config."""
    return load_config().get("models", {}).get("evaluation", {})


# ------------------------------------------------------------------
# Walk-forward cross-validation
# ------------------------------------------------------------------


def walk_forward_cv(
    model: QuantModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
    gap: int | None = None,
    min_train_size: int | None = None,
) -> dict[str, Any]:
    """Expanding-window walk-forward cross-validation.

    The dataset is divided into *n_splits* test folds.  For each fold the
    model is trained on all data **before** the test period (expanding
    window) and evaluated on the next contiguous block.  A *gap* of
    trading days is inserted between the end of training and the start of
    testing to prevent information leakage from autocorrelated returns.

    ::

        |---- train (expanding) ----|-- gap --|---- test fold k ----|

    Args:
        model: An *unfitted* :class:`QuantModel` instance.  A fresh copy
            of the underlying estimator is created for each fold so that
            folds are independent.
        X: Feature matrix with DatetimeIndex, sorted chronologically.
        y: Target series aligned to *X*.
        n_splits: Number of test folds.  Default from config.
        gap: Trading days between train end and test start.  Default
            from config.
        min_train_size: Minimum rows in the first training set.  Default
            from config.

    Returns:
        Dictionary with keys:

        - ``fold_metrics`` — list of per-fold metric dicts
        - ``mean_metrics`` — averaged across folds
        - ``splits`` — list of ``(train_end_date, test_start_date,
          test_end_date)`` tuples for auditability
    """
    cfg = _eval_cfg()
    n_splits = n_splits if n_splits is not None else cfg.get("n_splits", 5)
    gap = gap if gap is not None else cfg.get("gap", 5)
    min_train_size = (
        min_train_size if min_train_size is not None
        else cfg.get("min_train_size", 252)
    )

    if not X.index.is_monotonic_increasing:
        raise ValueError("X must be sorted chronologically (ascending DatetimeIndex)")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n = len(X)
    # Reserve enough room: min_train + gap + at least 1 row per fold
    usable = n - min_train_size - gap
    if usable < n_splits:
        raise ValueError(
            f"Not enough data for {n_splits} folds: {n} rows, "
            f"min_train_size={min_train_size}, gap={gap}"
        )

    fold_size = usable // n_splits

    fold_metrics: list[dict[str, float]] = []
    splits: list[tuple[str, str, str]] = []

    for k in range(n_splits):
        test_start_idx = min_train_size + gap + k * fold_size
        test_end_idx = test_start_idx + fold_size
        if k == n_splits - 1:
            # Last fold absorbs any remaining rows.
            test_end_idx = n
        train_end_idx = test_start_idx - gap

        X_train = X.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]

        if len(X_test) == 0:
            logger.warning(f"Fold {k}: empty test set — skipping")
            continue

        # Train a fresh copy of the model for this fold.
        fold_model = _clone_model(model)
        fold_model.fit(X_train, y_train)

        # Evaluate
        metrics = fold_model.evaluate(X_test, y_test)

        # Sharpe of predictions: annualised Sharpe of the strategy return
        # sign(prediction) * actual_return.
        if _is_classification_target(y_test):
            preds = fold_model.predict(X_test)
            strategy_returns = preds.astype(float) * y_test.astype(float)
        else:
            preds = fold_model.predict(X_test)
            strategy_returns = preds * y_test

        if strategy_returns.std() > 0:
            metrics["sharpe"] = float(
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            )
        else:
            metrics["sharpe"] = 0.0

        fold_metrics.append(metrics)
        splits.append((
            str(X_train.index[-1].date()),
            str(X_test.index[0].date()),
            str(X_test.index[-1].date()),
        ))

        logger.info(
            f"Fold {k}: train → {splits[-1][0]}, "
            f"test {splits[-1][1]} → {splits[-1][2]}, "
            f"metrics={_fmt_metrics(metrics)}"
        )

    # Aggregate
    mean_metrics = _average_metrics(fold_metrics) if fold_metrics else {}

    logger.info(f"Walk-forward CV complete: {len(fold_metrics)} folds, mean={_fmt_metrics(mean_metrics)}")

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "splits": splits,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _clone_model(model: QuantModel) -> QuantModel:
    """Create a fresh, unfitted copy of a QuantModel.

    Uses the model's class and stored hyperparams to reconstruct it,
    rather than deepcopy, so the underlying sklearn estimator is reset.
    """
    cls = type(model)
    params = dict(model.metadata.get("hyperparams", {}))
    return cls(**params)


def _average_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dicts, handling missing keys gracefully."""
    if not fold_metrics:
        return {}
    all_keys = set()
    for m in fold_metrics:
        all_keys.update(m.keys())
    result = {}
    for key in sorted(all_keys):
        values = [m[key] for m in fold_metrics if key in m]
        result[key] = float(np.mean(values)) if values else 0.0
    return result


def _fmt_metrics(metrics: dict[str, float]) -> str:
    """Format metrics dict for logging."""
    return ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
