"""
Model-Aware Configuration Profiles.

Centralizes model-specific defaults (temperature, include_skill_code, thinking)
into a single registry. Replaces scattered hardcoded lists in model_utils.py
and action.py.

Usage:
    from skillnet.core.model_profile import detect_model_profile

    profile = detect_model_profile("Qwen/Qwen3-Coder-Next-FP8")
    # profile.include_skill_code == False
    # profile.fixed_temperature == 1.0
    # profile.enable_thinking == False
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelProfile:
    """Configuration defaults for a model family.

    All fields are defaults — CLI flags and explicit config always override.
    """
    name: str = "default"
    include_skill_code: bool = False
    enable_thinking: bool = True
    enable_sanitizer: bool = False
    fixed_temperature: Optional[float] = None  # None = use requested
    use_structured_output: bool = True  # False → delimited sections (easier for weak models)
    iterative_gradients: bool = False  # True → apply gradients one-at-a-time in Phase 2
    # Plan v3-rev Fix 2B: how to enrich object-parameter schemas in
    # parameters.py:extract(). Babel produces a non-planner schema format
    # `{destructured: bool, fields: [...]}` that the planner can't consume.
    # Strategies:
    # "llm_always":     always call extract_object_schemas_llm (cleanest;
    # ~1 extra LLM call per new object-param skill)
    # "babel_then_llm": try deterministic Babel→planner translation via
    # a known field-name hint table; LLM fallback for
    # unresolved fields (cheaper for weak/local models)
    # "babel_only":     no LLM enrichment (fragile; test/debug only)
    object_schema_strategy: str = "llm_always"


# Registry: pattern (matched against model_name.lower()) → profile
# Order matters: first match wins.
_PROFILES = [
    ("gpt-5", ModelProfile(
        name="gpt-5",
        include_skill_code=True,
        enable_thinking=True,
        fixed_temperature=1.0,
        object_schema_strategy="llm_always",
    )),
    ("qwen3-coder", ModelProfile(
        name="qwen3-coder",
        include_skill_code=False,
        enable_thinking=False,
        enable_sanitizer=True,
        fixed_temperature=1.0,
        use_structured_output=False,
        iterative_gradients=True,
        # Qwen3-Coder is slower at object schema inference; deterministic
        # translation handles 70-80% of common patterns (count/item/amount)
        # for free; LLM only fires for novel field names.
        object_schema_strategy="babel_then_llm",
    )),
]

_DEFAULT_PROFILE = ModelProfile()


def detect_model_profile(model_name: str) -> ModelProfile:
    """Detect model profile from model name string.

    Matches against known patterns (case-insensitive).
    Returns default profile if no match.

    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen3-Coder-Next-FP8", "gpt-5-mini")

    Returns:
        Matching ModelProfile or default.
    """
    if not model_name:
        return _DEFAULT_PROFILE

    lower = model_name.lower()
    for pattern, profile in _PROFILES:
        if pattern in lower:
            return profile

    return _DEFAULT_PROFILE
