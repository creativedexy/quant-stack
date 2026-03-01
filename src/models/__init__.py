"""ML models — abstract base, scikit-learn wrappers, AutoML, and evaluation."""

from src.models.base import QuantModel, ModelRegistry, model_registry
from src.models.classical import RandomForestModel, GradientBoostingModel
from src.models.evaluation import (
    DataValidationError,
    time_series_split,
    verify_no_leakage,
    walk_forward_cv,
    compare_models,
    plot_cv_results,
)
from src.models.targets import (
    create_direction_target,
    create_return_target,
    align_features_and_target,
)

# AutoML wrapper is imported lazily because PyCaret is an optional dependency.
# Use: from src.models.automl import quick_compare, PyCaretModel

# Prepopulate the module-level model registry with all concrete models.
model_registry.register("random_forest", RandomForestModel)
model_registry.register("gradient_boosting", GradientBoostingModel)

__all__ = [
    "QuantModel",
    "ModelRegistry",
    "model_registry",
    "RandomForestModel",
    "GradientBoostingModel",
    "DataValidationError",
    "time_series_split",
    "verify_no_leakage",
    "walk_forward_cv",
    "compare_models",
    "plot_cv_results",
    "create_direction_target",
    "create_return_target",
    "align_features_and_target",
]
