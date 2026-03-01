"""PyCaret AutoML wrapper for rapid model comparison.

**Prototyping tool, NOT production.**  The purpose is fast iteration:
"does any model family find signal in these features?"  Use
:func:`quick_compare` to screen a handful of algorithms with
time-series-aware CV, then promote winners to the production
:class:`~src.models.base.QuantModel` wrappers in ``classical.py`` for
proper walk-forward validation.

Requires the ``ml-extended`` optional dependency group::

    pip install "quant-stack[ml-extended]"

Usage:
    from src.features.pipeline import FeaturePipeline
    from src.models.automl import quick_compare

    pipe = FeaturePipeline()
    features = pipe.run(ohlcv_df)
    y = (features["ret_1d"].shift(-1) > 0).astype(int).dropna()
    X = features.loc[y.index]

    summary, best_model = quick_compare(X, y)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import QuantModel
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Model families to compare — a mix of linear, tree, and ensemble methods.
_DEFAULT_CLF_MODELS = ["lr", "rf", "gbc", "et", "lightgbm"]
_DEFAULT_REG_MODELS = ["lr", "rf", "gbr", "et", "lightgbm"]


def _automl_cfg() -> dict[str, Any]:
    """Return the automl section from config."""
    return load_config().get("models", {}).get("automl", {})


# ------------------------------------------------------------------
# quick_compare
# ------------------------------------------------------------------


def quick_compare(
    X: pd.DataFrame,
    y: pd.Series,
    target_type: str = "classification",
    *,
    n_folds: int | None = None,
    include: list[str] | None = None,
    sort: str | None = None,
    save_path: str | Path | None = None,
    session_id: int | None = None,
) -> tuple[pd.DataFrame, "PyCaretModel"]:
    """Screen model families for signal using PyCaret with temporal CV.

    This is a **prototyping** convenience — it runs PyCaret's
    ``compare_models`` under time-series fold settings (no shuffle,
    ``fold_strategy='timeseries'``) and wraps the winner in a
    :class:`PyCaretModel` so it can be saved through the standard
    :meth:`QuantModel.save` interface.

    Args:
        X: Feature matrix with DatetimeIndex.  Must be sorted
            chronologically and free of NaN.
        y: Target series aligned to *X*.
        target_type: ``"classification"`` or ``"regression"``.
        n_folds: Number of temporal CV folds.  Default from config.
        include: PyCaret model IDs to compare.  Default is 5 families
            (logistic/linear, RF, gradient boosting, extra trees,
            LightGBM).
        sort: Metric to rank models by.  Defaults to ``"F1"`` for
            classification or ``"MAE"`` for regression.
        save_path: If provided, the best model is saved here via
            :meth:`QuantModel.save`.
        session_id: PyCaret session seed.  Default from config
            ``general.random_seed``.

    Returns:
        Tuple of ``(leaderboard_df, best_model)`` where
        *leaderboard_df* is a DataFrame of per-model metrics ranked by
        *sort*, and *best_model* is a :class:`PyCaretModel` wrapping the
        winning estimator.

    Raises:
        ImportError: If PyCaret is not installed.
        ValueError: If *target_type* is unrecognised.
    """
    try:
        if target_type == "classification":
            import pycaret.classification as pc
        elif target_type == "regression":
            import pycaret.regression as pc
        else:
            raise ValueError(
                f"target_type must be 'classification' or 'regression', got '{target_type}'"
            )
    except ImportError as exc:
        raise ImportError(
            "PyCaret is required for quick_compare.  "
            "Install with: pip install 'quant-stack[ml-extended]'"
        ) from exc

    cfg = _automl_cfg()
    general_cfg = load_config().get("general", {})

    n_folds = n_folds if n_folds is not None else cfg.get("n_folds", 3)
    session_id = session_id if session_id is not None else general_cfg.get("random_seed", 42)

    if include is None:
        include = (
            cfg.get("clf_models", _DEFAULT_CLF_MODELS) if target_type == "classification"
            else cfg.get("reg_models", _DEFAULT_REG_MODELS)
        )

    if sort is None:
        sort = "F1" if target_type == "classification" else "MAE"

    # ---- Temporal train / test split (80/20) ----
    split_idx = int(len(X) * 0.8)
    train_df = X.iloc[:split_idx].copy()
    train_df["__target__"] = y.iloc[:split_idx].values
    test_df = X.iloc[split_idx:].copy()
    test_df["__target__"] = y.iloc[split_idx:].values

    logger.info(
        f"quick_compare: {target_type}, {len(include)} models, "
        f"{n_folds} folds, sort={sort}, "
        f"train={len(train_df)}, test={len(test_df)}"
    )

    # ---- PyCaret setup — time-series aware, no shuffling ----
    pc.setup(
        data=train_df,
        test_data=test_df,
        target="__target__",
        fold_strategy="timeseries",
        fold=n_folds,
        data_split_shuffle=False,
        fold_shuffle=False,
        session_id=session_id,
        html=False,
        verbose=False,
    )

    # ---- Compare models ----
    best_estimator = pc.compare_models(
        include=include,
        n_select=1,
        sort=sort,
        verbose=False,
    )
    leaderboard = pc.pull()

    # Finalise: re-train winner on the full training set (train + test).
    final_estimator = pc.finalize_model(best_estimator)

    logger.info(
        f"Best model: {type(final_estimator).__name__} "
        f"(sorted by {sort})"
    )

    # ---- Wrap in QuantModel interface ----
    wrapped = PyCaretModel(
        estimator=final_estimator,
        target_type=target_type,
        feature_names=list(X.columns),
        train_index=X.index,
        pycaret_module=pc,
    )

    if save_path is not None:
        wrapped.save(save_path)

    return leaderboard, wrapped


# ------------------------------------------------------------------
# PyCaretModel — QuantModel adapter
# ------------------------------------------------------------------


class PyCaretModel(QuantModel):
    """Thin wrapper that adapts a PyCaret estimator to the QuantModel interface.

    Not intended for direct construction — use :func:`quick_compare` which
    returns an instance of this class.
    """

    def __init__(
        self,
        estimator: Any = None,
        target_type: str = "classification",
        feature_names: list[str] | None = None,
        train_index: pd.DatetimeIndex | None = None,
        pycaret_module: Any = None,
    ) -> None:
        super().__init__(name="pycaret_model")
        self._estimator = estimator
        self._target_type = target_type
        self._pycaret_module = pycaret_module

        if train_index is not None and len(train_index) > 0:
            self.metadata["train_date_range"] = (
                str(train_index.min().date()),
                str(train_index.max().date()),
            )
        self.metadata["trained"] = estimator is not None
        self.metadata["feature_names"] = feature_names or []
        self.metadata["hyperparameters"] = {
            "target_type": target_type,
            "estimator_class": type(estimator).__name__ if estimator else None,
        }

    # -- Pickle support: drop the unpicklable module reference ------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_pycaret_module", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._pycaret_module = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PyCaretModel":
        """Re-fit the underlying estimator.

        For most prototyping workflows, the model is already fitted by
        :func:`quick_compare`.  This method exists to satisfy the
        :class:`QuantModel` interface and to allow re-training on new data.
        """
        if self._estimator is None:
            raise RuntimeError("No estimator set — use quick_compare() first")

        self._estimator.fit(X_train, y_train)
        self.metadata["trained"] = True
        self.metadata["train_date_range"] = (
            str(X_train.index.min().date()),
            str(X_train.index.max().date()),
        )
        self.metadata["feature_names"] = list(X_train.columns)
        self.metadata["n_train_samples"] = len(X_train)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions using the PyCaret estimator.

        Args:
            X: Feature matrix with DatetimeIndex.

        Returns:
            Series of predictions indexed to match *X*.
        """
        if self._estimator is None:
            raise RuntimeError("No estimator set — use quick_compare() first")

        preds = self._estimator.predict(X)
        return pd.Series(
            np.asarray(preds).ravel(),
            index=X.index,
            name="prediction",
        )
