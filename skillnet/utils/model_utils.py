"""
Utilities for handling OpenAI model configurations.

Delegates to ModelProfile for model-specific settings.
"""

from skillnet.core.model_profile import detect_model_profile


def get_temperature_for_model(model_name: str, requested_temperature: float) -> float:
    """
    Get the appropriate temperature value for a given model.

    Models with fixed_temperature in their ModelProfile always use that value.
    Others use the requested temperature.

    Args:
        model_name: The model name (e.g., "gpt-5-mini", "Qwen/Qwen3-Coder-Next-FP8")
        requested_temperature: The requested temperature value

    Returns:
        The temperature to use.
    """
    profile = detect_model_profile(model_name)
    if profile.fixed_temperature is not None:
        return profile.fixed_temperature
    return requested_temperature
