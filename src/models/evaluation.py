"""Time-series aware cross-validation and model comparison.

Provides :func:`time_series_split` (expanding-window generator),
:func:`walk_forward_cv` (full walk-forward validation),
:func:`compare_models`, and :func:`plot_cv_results`.

**CRITICAL**: sklearn's default random splitters must *never* be used for
financial time-series.  All splitting in this module respects strict
chronological ordering with a configurable gap to prevent autocorrelation
leakage.

Usage:
    from src.models.evaluation import walk_forward_cv, compare_models
    results = walk_forward_cv(model, X, y, n_splits=5, gap=5)
    comparison = compare_models([model_a, model_b], X, y)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.base import QuantModel, _is_classification_target
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Custom exception
# ------------------------------------------------------------------


class DataValidationError(Exception):
    """Raised when temporal data integrity checks fail."""


# ------------------------------------------------------------------
# Config helper
# ------------------------------------------------------------------


def _eval_cfg(config: dict | None = None) -> dict[str, Any]:
    """Return the evaluation section from config."""
    if config is not None:
        return config.get("models", {}).get("evaluation", {})
    return load_config().get("models", {}).get("evaluation", {})


# ------------------------------------------------------------------
# Leakage verification
# ------------------------------------------------------------------


def verify_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    gap: int,
) -> None:
    """Assert that no data leaks between train and test sets.

    Checks that ``max(train_idx) + gap < min(test_idx)`` to guarantee
    a clean separation with the required gap.

    Args:
        train_idx: Positional indices of the training set.
        test_idx: Positional indices of the test set.
        gap: Number of observations that must be excluded between
            the training and test sets.

    Raises:
        DataValidationError: If the leakage check fails.
    """
    train_max = int(np.max(train_idx))
    test_min = int(np.min(test_idx))

    if train_max + gap >= test_min:
        raise DataValidationError(
            f"Data leakage detected: max(train_idx)={train_max} + "
            f"gap={gap} = {train_max + gap} >= min(test_idx)={test_min}. "
            f"Expected max(train_idx) + gap < min(test_idx)."
        )


# ------------------------------------------------------------------
# Time-series splitting generator
# ------------------------------------------------------------------


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 5,
    min_train_size: int = 252,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Expanding-window time-series split generator.

    Yields ``(train_idx, test_idx)`` positional index arrays for each
    fold.  Train grows each fold; test is always the next contiguous
    chunk.  *gap* trading days are excluded between train end and test
    start to prevent information leakage from autocorrelated returns.

    ::

        |---- train (expanding) ----|-- gap --|---- test fold k ----|

    **NEVER shuffles.  NEVER randomises.  Order is strictly temporal.**

    Args:
        X: Feature matrix (used only for length).
        y: Target series (used only for length).
        n_splits: Number of test folds.
        gap: Trading days excluded between train end and test start.
        min_train_size: Minimum rows in the first training set.

    Yields:
        Tuple of ``(train_indices, test_indices)`` as integer arrays.

    Raises:
        ValueError: If there is not enough data for the requested splits.
        DataValidationError: If a fold fails the no-leakage check.
    """
    n = len(X)
    usable = n - min_train_size - gap
    if usable < n_splits:
        raise ValueError(
            f"Not enough data for {n_splits} folds: {n} rows, "
            f"min_train_size={min_train_size}, gap={gap}"
        )

    fold_size = usable // n_splits

    for k in range(n_splits):
        test_start_idx = min_train_size + gap + k * fold_size
        test_end_idx = test_start_idx + fold_size
        if k == n_splits - 1:
            test_end_idx = n
        train_end_idx = test_start_idx - gap

        train_idx = np.arange(train_end_idx)
        test_idx = np.arange(test_start_idx, test_end_idx)

        verify_no_leakage(train_idx, test_idx, gap)

        yield train_idx, test_idx


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
    config: dict | None = None,
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
            of the underlying estimator is created for each fold.
        X: Feature matrix with DatetimeIndex, sorted chronologically.
        y: Target series aligned to *X*.
        n_splits: Number of test folds.  Default from config.
        gap: Trading days between train end and test start.  Default
            from config.
        min_train_size: Minimum rows in the first training set.
        config: Config dict for evaluation parameters.

    Returns:
        Dictionary with keys:

        - ``per_fold`` — list of per-fold dicts with fold index,
          train_size, test_size, and metrics.
        - ``aggregate`` — mean and std of each metric across folds.
        - ``model_name`` — name of the model.
        - ``n_splits`` — number of folds used.
        - ``gap`` — gap used.
    """
    cfg = _eval_cfg(config or getattr(model, "config", None))
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

    per_fold: list[dict[str, Any]] = []

    for fold_num, (train_idx, test_idx) in enumerate(
        time_series_split(X, y, n_splits=n_splits, gap=gap, min_train_size=min_train_size)
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(X_test) == 0:
            logger.warning("Fold %d: empty test set — skipping", fold_num)
            continue

        fold_model = _clone_model(model)
        fold_model.fit(X_train, y_train)
        metrics = fold_model.evaluate(X_test, y_test)

        fold_info = {
            "fold": fold_num + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_end": str(X_train.index[-1].date()),
            "test_start": str(X_test.index[0].date()),
            "test_end": str(X_test.index[-1].date()),
            "metrics": metrics,
        }
        per_fold.append(fold_info)

        logger.info(
            "Fold %d: train → %s (%d rows), test %s → %s (%d rows), "
            "metrics=%s",
            fold_num + 1, fold_info["train_end"], len(X_train),
            fold_info["test_start"], fold_info["test_end"], len(X_test),
            _fmt_metrics(metrics),
        )

    aggregate = _aggregate_metrics(per_fold)

    logger.info(
        "Walk-forward CV complete: %d folds, %s",
        len(per_fold), _fmt_metrics(aggregate),
    )

    return {
        "per_fold": per_fold,
        "aggregate": aggregate,
        "model_name": model.name,
        "n_splits": n_splits,
        "gap": gap,
    }


# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------


def compare_models(
    models: list[QuantModel],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 5,
    min_train_size: int = 252,
) -> pd.DataFrame:
    """Run walk-forward CV on each model and return a comparison table.

    Args:
        models: List of unfitted QuantModel instances.
        X: Feature matrix with DatetimeIndex.
        y: Target series aligned to *X*.
        n_splits: Number of test folds.
        gap: Gap between train and test.
        min_train_size: Minimum training set size.

    Returns:
        DataFrame with one row per model, columns for each aggregate
        metric.  Sorted by mean information coefficient descending.
    """
    rows: list[dict[str, Any]] = []

    for model in models:
        cv_result = walk_forward_cv(
            model, X, y,
            n_splits=n_splits, gap=gap, min_train_size=min_train_size,
        )
        row = {"model": model.name}
        row.update(cv_result["aggregate"])
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("model")

    # Sort by mean_ic descending (if present)
    if "mean_ic" in comparison.columns:
        comparison = comparison.sort_values("mean_ic", ascending=False)

    return comparison


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------


def plot_cv_results(
    cv_results: dict[str, Any],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of per-fold IC with error bars.

    Useful for spotting if a model works in some regimes but not others.

    Args:
        cv_results: Output from :func:`walk_forward_cv`.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    per_fold = cv_results["per_fold"]
    model_name = cv_results.get("model_name", "Model")

    fold_labels = [f"Fold {f['fold']}" for f in per_fold]
    ic_values = [f["metrics"].get("information_coefficient", 0.0) for f in per_fold]

    fig, ax = plt.subplots(figsize=(10, 5))

    colours = ["steelblue" if v >= 0 else "salmon" for v in ic_values]
    ax.bar(fold_labels, ic_values, color=colours, alpha=0.8, edgecolor="grey")
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--")

    mean_ic = cv_results["aggregate"].get("mean_ic", 0.0)
    std_ic = cv_results["aggregate"].get("std_ic", 0.0)
    ax.axhline(
        y=mean_ic, color="orange", linewidth=1.2, linestyle="-",
        label=f"Mean IC = {mean_ic:.4f} ± {std_ic:.4f}",
    )

    ax.set_title(f"{model_name} — Per-Fold Information Coefficient")
    ax.set_ylabel("IC (Spearman)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info("CV results plot saved to %s", save_path)
        plt.close(fig)

    return fig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _clone_model(model: QuantModel) -> QuantModel:
    """Create a fresh, unfitted copy of a QuantModel.

    Uses the model's class and stored hyperparameters to reconstruct it.
    """
    cls = type(model)
    params = dict(model.metadata.get("hyperparameters", {}))
    config = getattr(model, "config", None)
    target_type = model.metadata.get("target_type")

    # Remove params that aren't sklearn constructor args
    clone_kwargs: dict[str, Any] = {}
    if target_type is not None:
        clone_kwargs["target_type"] = target_type

    # Pass sklearn hyperparameters as **kwargs
    return cls(name=model.name, config=config, **clone_kwargs, **params)


def _aggregate_metrics(per_fold: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean and std of each metric across folds."""
    if not per_fold:
        return {}

    all_keys: set[str] = set()
    for fold in per_fold:
        all_keys.update(fold["metrics"].keys())

    result: dict[str, float] = {}
    for key in sorted(all_keys):
        values = [f["metrics"][key] for f in per_fold if key in f["metrics"]]
        if values:
            result[f"mean_{key}"] = float(np.mean(values))
            result[f"std_{key}"] = float(np.std(values))

    return result


def _fmt_metrics(metrics: dict[str, float]) -> str:
    """Format metrics dict for logging."""
    return ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
