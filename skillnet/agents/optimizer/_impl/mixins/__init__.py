"""
Optimizer Mixins

Module splitting for better maintainability.
Added ApplyOptimizationMixin, EditContextMixin, EditAnalysisMixin.
Added OptimizationUtilsMixin, OptimizationLifecycleMixin, QuickOptimizationMixin.
Contains mixin classes extracted from optimizer_impl.py.
"""

from .code_edit import CodeEditMixin
from .validation import ValidationMixin
from .feedback import FeedbackMixin
from .interface_mgmt import InterfaceManagementMixin
from .optimization import OptimizationMixin
from .apply_optimization import ApplyOptimizationMixin
from .edit_context import EditContextMixin
from .edit_analysis import EditAnalysisMixin
from .optimization_utils import OptimizationUtilsMixin
from .optimization_lifecycle import OptimizationLifecycleMixin
from .quick_optimization import QuickOptimizationMixin

__all__ = [
    "CodeEditMixin",
    "ValidationMixin",
    "FeedbackMixin",
    "InterfaceManagementMixin",
    "OptimizationMixin",
    "ApplyOptimizationMixin",
    "EditContextMixin",
    "EditAnalysisMixin",
    "OptimizationUtilsMixin",
    "OptimizationLifecycleMixin",
    "QuickOptimizationMixin",
]
