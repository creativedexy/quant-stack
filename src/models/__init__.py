"""ML models — abstract base, scikit-learn wrappers, AutoML, and evaluation."""

from src.models.base import QuantModel
from src.models.classical import RandomForestModel, GradientBoostingModel
from src.models.evaluation import walk_forward_cv

# AutoML wrapper is imported lazily because PyCaret is an optional dependency.
# Use: from src.models.automl import quick_compare, PyCaretModel

__all__ = [
    "QuantModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "walk_forward_cv",
]
