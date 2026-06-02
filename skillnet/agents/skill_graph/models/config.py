"""
Configuration class definitions

Includes Value Function parameter configs and failure categorization.
"""

from enum import Enum
from dataclasses import dataclass


class FailureCategory(Enum):
    """Failure reason categorization, used for more precise precondition inference"""
    PRECONDITION_NOT_MET = "precondition_not_met"  # Precondition not satisfied
    EXECUTION_ERROR = "execution_error"  # Runtime error (e.g. block not found)
    CODE_BUG = "code_bug"  # Bug in the code itself (e.g. undefined variable)
    EXTERNAL_FAILURE = "external_failure"  # External factor (network, server)
    TIMEOUT = "timeout"  # Timeout
    UNKNOWN = "unknown"  # Unknown cause


@dataclass
class PreconditionValueFunctionConfig:
    """
    Precondition Value Function parameter config

    Value formula:
    V(precond) = p_combined - λ * uncertainty - γ * (1 - code_verified)

    where:
    - p_combined = w_s * P(precond|success) + w_f * P(missing|failure)
    - uncertainty = (n + 1)^(-0.5)
    """
    # Bayesian prior parameters
    alpha: float = 1.0  # Success prior
    beta: float = 2.0   # Failure prior (more conservative than skills; initial prob = 1/3)

    # Uncertainty penalty coefficient (higher than the 1.0 used for skills)
    lambda_uncertainty: float = 1.5

    # Code-verification penalty (when the code does not check for it)
    gamma_code_unverified: float = 0.4

    # Success/failure weights
    weight_success: float = 0.6  # Weight for presence in successful samples
    weight_failure: float = 0.4  # Weight for absence in failed samples

    # Decision thresholds
    threshold_high: float = 0.5    # High-confidence threshold
    threshold_medium: float = 0.3  # Medium-confidence threshold
    threshold_low: float = 0.15    # Low-confidence threshold (do not add below this)


@dataclass
class EffectValueFunctionConfig:
    """
    Effect Value Function parameter config

    Value formula:
    V(effect) = p_s - λ * uncertainty

    where:
    - p_s = Bayesian-smoothed effect frequency
    - uncertainty = (n + 1)^(-0.5)
    """
    # Bayesian prior parameters (effects can be slightly looser)
    alpha: float = 1.0
    beta: float = 1.5

    # Uncertainty penalty coefficient
    lambda_uncertainty: float = 1.2

    # Decision thresholds
    threshold_high: float = 0.5
    threshold_medium: float = 0.3
    threshold_low: float = 0.1  # Effect threshold can be lower
