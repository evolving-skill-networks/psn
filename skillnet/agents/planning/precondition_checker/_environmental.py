"""
Environmental Feedback (Layer B)

Environmental-feedback layer: environmental feedback dataclass and error-inference function.

Classes:
- EnvironmentalFeedback: structured environmental feedback container

Functions:
- infer_environmental_feedback_from_error: infer environmental feedback from an error message
"""

from typing import Any, Dict, List, Optional

from skillnet.core.dk_registry import get_domain_knowledge


class EnvironmentalFeedback:
    """
    Structured environmental feedback that provides environment context for LLM reasoning.

    Generated when a biome_resource check fails. The LLM derives a solution
    from the Knowledge layer (e.g., lava_location) using this context, instead
    of receiving a pre-baked suggestion.
    """

    def __init__(
        self,
        feedback_type: str = "environment_mismatch",
        current_biome: str = "",
        missing_resources: List[str] = None,
        available_resources: List[str] = None,
        bot_position: Dict[str, float] = None,
    ):
        self.feedback_type = feedback_type
        self.current_biome = current_biome
        self.missing_resources = missing_resources or []
        self.available_resources = available_resources or []
        self.bot_position = bot_position or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict form"""
        return {
            "feedback_type": self.feedback_type,
            "current_biome": self.current_biome,
            "missing_resources": self.missing_resources,
            "available_resources": self.available_resources,
            "bot_position": self.bot_position,
        }

    def to_prompt_context(self) -> str:
        """
        Generate a context string for the optimization prompt. Supplies
        environmental context only; the LLM derives the solution from the
        ENVIRONMENT KNOWLEDGE section.
        """
        lines = [
            f"## Environmental Context (ENVIRONMENT_MISMATCH)",
            f"- Current biome: {self.current_biome}",
            f"- Missing resources in this biome: {', '.join(self.missing_resources)}",
            f"- Available resources: {', '.join(self.available_resources[:10])}",  # cap length
        ]

        if self.bot_position:
            y_level = self.bot_position.get("y", 0)
            lines.append(f"- Bot Y level: {y_level:.0f}")

        lines.append("")
        lines.append("(Refer to ENVIRONMENT KNOWLEDGE section for resource distribution facts)")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_prompt_context()


def infer_environmental_feedback_from_error(
    error_message: str,
    current_state: Dict[str, Any] = None,
) -> Optional[EnvironmentalFeedback]:
    """
    Infer environmental feedback from an error message.

    When a skill has not explicitly declared biome_resource preconditions,
    infer environmental problems from the runtime error.

    Args:
        error_message: error message
        current_state: current state (contains biome, position)

    Returns:
        EnvironmentalFeedback: feedback if an environmental issue is detected, otherwise None
    """
    if not error_message:
        return None

    error_lower = error_message.lower()
    current_state = current_state or {}
    current_biome = current_state.get("biome", "unknown")
    bot_position = current_state.get("position", {})

    missing_resources = []

    # Get resource-not-found patterns from domain knowledge
    _dk = get_domain_knowledge()
    if (
        _dk
        and hasattr(_dk, 'get_resource_not_found_patterns')
    ):
        resource_patterns = _dk.get_resource_not_found_patterns()
    else:
        resource_patterns = {}

    for resource, patterns in resource_patterns.items():
        for pattern in patterns:
            if pattern in error_lower:
                missing_resources.append(resource)
                break

    # Detect pathfinder timeouts (often caused by the target resource being too far away)
    if "pathfinder timeout" in error_lower or "path to" in error_lower:
        # Detect target resources via the keys of resource_patterns
        for resource in resource_patterns:
            if resource in error_lower and resource not in missing_resources:
                missing_resources.append(resource)

    if not missing_resources:
        return None

    # Fetch available resources (via DI)
    dk = get_domain_knowledge()
    available_resources = dk.get_biome_resources(current_biome) if dk else []

    return EnvironmentalFeedback(
        feedback_type="environment_mismatch_inferred",
        current_biome=current_biome,
        missing_resources=missing_resources,
        available_resources=available_resources,
        bot_position=bot_position if isinstance(bot_position, dict) else {},
    )
