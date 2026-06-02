"""
Value Functions Module

Compute confidence values for Preconditions and Effects.

v5.0 architecture reorganization - migrated from graph_manager_impl.py
"""

import math
from typing import Any, Dict, Optional

from skillnet.agents.skill_graph.models import (
    PreconditionValueFunctionConfig,
    EffectValueFunctionConfig,
)


def calculate_precondition_value(
    n_success_with: int,
    n_success_total: int,
    n_failure_missing: int,
    n_failure_total: int,
    code_verified: bool,
    config: Optional[PreconditionValueFunctionConfig] = None,
) -> Dict[str, Any]:
    """
    Compute the Value Function value for a Precondition.

    Formula: V(precond) = p_combined - λ * uncertainty - γ * (1 - code_verified)

    Args:
        n_success_with: number of successful samples where the condition holds
        n_success_total: total number of successful samples
        n_failure_missing: number of failed samples where the condition is missing
        n_failure_total: total number of failed samples
        code_verified: whether the code contains a corresponding check
        config: Value Function configuration

    Returns:
        Dict containing:
        - value: float, Value Function value
        - confidence_level: str, confidence level ("high", "medium", "low", "uncertain")
        - should_add: bool, whether to add as a precondition
        - details: Dict, computation details
    """
    if config is None:
        config = PreconditionValueFunctionConfig()

    # 1. Bayesian-smoothed success presence rate
    if n_success_total > 0:
        p_success = (n_success_with + config.alpha) / (
            n_success_total + config.alpha + config.beta
        )
    else:
        p_success = config.alpha / (config.alpha + config.beta)

    # 2. Bayesian-smoothed failure missing rate
    if n_failure_total > 0:
        p_failure = (n_failure_missing + config.alpha) / (
            n_failure_total + config.alpha + config.beta
        )
    else:
        p_failure = 0.5  # neutral when there are no failure samples

    # 3. Combined probability
    combined_p = config.weight_success * p_success + config.weight_failure * p_failure

    # 4. Uncertainty penalty (based on total sample size)
    total_n = n_success_total + n_failure_total
    uncertainty = config.lambda_uncertainty * math.pow(total_n + 1, -0.5)

    # 5. Code verification penalty
    code_penalty = 0 if code_verified else config.gamma_code_unverified

    # 6. Final value
    value = combined_p - uncertainty - code_penalty

    # 7. Determine confidence level
    if value >= config.threshold_high:
        confidence_level = "high"
    elif value >= config.threshold_medium:
        confidence_level = "medium"
    elif value >= config.threshold_low:
        confidence_level = "low"
    else:
        confidence_level = "uncertain"

    return {
        "value": value,
        "confidence_level": confidence_level,
        "should_add": value >= config.threshold_low,
        "details": {
            "p_success": p_success,
            "p_failure": p_failure,
            "combined_p": combined_p,
            "uncertainty": uncertainty,
            "code_penalty": code_penalty,
            "n_success_with": n_success_with,
            "n_success_total": n_success_total,
            "n_failure_missing": n_failure_missing,
            "n_failure_total": n_failure_total,
            "code_verified": code_verified,
        },
    }


def calculate_effect_value(
    n_occurrences: int,
    n_total: int,
    config: Optional[EffectValueFunctionConfig] = None,
) -> Dict[str, Any]:
    """
    Compute the Value Function value for an Effect.

    Formula: V(effect) = p_s - λ * uncertainty

    Args:
        n_occurrences: number of times the effect appears
        n_total: total number of samples
        config: Value Function configuration

    Returns:
        Dict containing:
        - value: float, Value Function value
        - confidence_level: str, confidence level ("high", "medium", "low", "uncertain")
        - should_add: bool, whether to add as an effect
        - details: Dict, computation details
    """
    if config is None:
        config = EffectValueFunctionConfig()

    # 1. Bayesian-smoothed occurrence rate
    if n_total > 0:
        p_s = (n_occurrences + config.alpha) / (n_total + config.alpha + config.beta)
    else:
        p_s = config.alpha / (config.alpha + config.beta)

    # 2. Uncertainty penalty
    uncertainty = config.lambda_uncertainty * math.pow(n_total + 1, -0.5)

    # 3. Final value
    value = p_s - uncertainty

    # 4. Determine confidence level
    if value >= config.threshold_high:
        confidence_level = "high"
    elif value >= config.threshold_medium:
        confidence_level = "medium"
    elif value >= config.threshold_low:
        confidence_level = "low"
    else:
        confidence_level = "uncertain"

    return {
        "value": value,
        "confidence_level": confidence_level,
        "should_add": value >= config.threshold_low,
        "details": {
            "p_s": p_s,
            "uncertainty": uncertainty,
            "n_occurrences": n_occurrences,
            "n_total": n_total,
        },
    }
