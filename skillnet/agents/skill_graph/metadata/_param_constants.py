"""
Parameter Extraction Constants & Prompts

Shared constants used by both ParameterExtractionMixin and ParameterSemanticMixin.
Extracted to avoid circular imports between parameters.py (facade) and mixin files.
"""

import re
from typing import List

from skillnet.agents.planning.inference import (
    DirectionSemantic,
    TransformHint,
)
from skillnet.core.dk_registry import get_domain_knowledge


def _get_mine_keywords():
    """Return mining-related keywords from domain knowledge."""
    dk = get_domain_knowledge()
    return dk.get_mine_keywords() if dk else []


def _get_func_prefix_to_direction():
    """Return {prefix: DirectionSemantic} mapping from domain knowledge."""
    dk = get_domain_knowledge()
    if not dk:
        return {}
    raw = dk.get_func_prefix_to_direction()
    return {prefix: DirectionSemantic(direction) for prefix, direction in raw.items()}


def _get_param_suffix_to_transform():
    """Return {suffix: TransformHint} mapping from domain knowledge."""
    dk = get_domain_knowledge()
    if not dk:
        return {}
    raw = dk.get_param_suffix_to_transform()
    return {suffix: TransformHint(**hint_dict) for suffix, hint_dict in raw}


# ============================================================================
# Constants
# ============================================================================

# List of count-parameter names (used for semantic inference)
QUANTITY_PARAM_NAMES = ["count", "amount", "num", "quantity", "number", "total", "target"]

# Code patterns for target_total semantics
TARGET_TOTAL_PATTERNS = [
    # Pattern: have >= count or variants
    r"(have|current|inventory|existing)\s*>=\s*(count|quantity|amount|num|number)",
    r"if\s*\(\s*(count|quantity|amount|num)\s*<=\s*(have|current|inventory)",
    # Pattern: related to "already have"
    r"already\s+have",
    r"I\s+already\s+have\s+\d+",
    # Pattern: "Nothing to do"
    r"[Nn]othing\s+to\s+do",
]

# Mining-related keywords — now provided by DK via _get_mine_keywords()

# Function-name prefixes for delta semantics
DELTA_FUNC_PREFIXES = ["craft", "mine", "collect", "get", "gather", "harvest", "produce"]

# Function-name prefixes for ensure semantics
ENSURE_PREFIXES = ["ensure"]

# ============================================================================
# Direction Inference Constants
# ============================================================================

# INPUT direction indicators (parameter names containing these words indicate input materials)
INPUT_DIRECTION_INDICATORS = [
    "input", "source", "material", "raw", "ingredient",
    "logtype", "log_type", "oretype", "ore_type",
    "fueltype", "fuel_type", "inputitem", "input_item",
]

# OUTPUT direction indicators (parameter names containing these words indicate output products)
OUTPUT_DIRECTION_INDICATORS = [
    "output", "result", "target", "product",
    "resulttype", "result_type", "targettype", "target_type",
    "outputtype", "output_type", "producttype", "product_type",
]

# CONFIG direction indicators (parameter names containing these words indicate config parameters)
CONFIG_DIRECTION_INDICATORS = [
    "priority", "preference", "prefer", "option", "options",
    "config", "timeout", "distance", "fallback", "default",
    "selection", "choices", "allowed", "setting", "mode",
    "max", "min", "limit", "threshold", "radius",
]

# Fuel parameter names (special-cased)
FUEL_PARAM_NAMES = ["fuel", "fueltype", "fuel_type", "fuelpriority", "fuel_priority"]

# Function-name prefix → direction / parameter suffix → TransformHint — now via DK
# Use _get_func_prefix_to_direction() and _get_param_suffix_to_transform()


# ============================================================================
# System Prompts (moved to domain-owned prompt files)
# ============================================================================


def _get_parameter_extraction_prompt():
    """Load parameter extraction prompt from domain knowledge."""
    dk = get_domain_knowledge()
    return dk.get_prompt("parameter_extraction") if dk else ""


# Backward-compatible constant — consumers should migrate to the function
PARAMETER_EXTRACTION_SYSTEM_PROMPT = ""  # loaded from domain at runtime

def _get_object_schema_extraction_prompt():
    """Load object schema extraction prompt from domain knowledge."""
    dk = get_domain_knowledge()
    return dk.get_prompt("object_schema_extraction") if dk else ""


OBJECT_SCHEMA_EXTRACTION_PROMPT = ""  # loaded from domain at runtime

SEMANTIC_INFERENCE_PROMPT = """You are a code analysis expert. Analyze the parameter semantic and respond with exactly one word."""


# ============================================================================
# Pure Functions
# ============================================================================

def determine_delta_prefix(func_name: str) -> str:
    """
    Determine the prefix to use when renaming in reverse (ensure -> craft/mine).

    Based on function-name traits, decide whether to use 'mine' or 'craft':
    - Contains keywords like ore/log/block/sand/gravel/stone/dirt/coal -> mine
    - Otherwise -> craft (default)

    Args:
        func_name: function name

    Returns:
        Recommended prefix: "mine" or "craft"
    """
    func_name_lower = func_name.lower()

    for keyword in _get_mine_keywords():
        if keyword in func_name_lower:
            return "mine"

    return "craft"
