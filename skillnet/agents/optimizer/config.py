"""
Optimizer Configuration

Centralized optimizer configuration parameters.
"""

from dataclasses import dataclass


@dataclass
class BloatPreventionConfig:
    """
    Bloat-prevention rule configuration.

    Controls code growth and prevents unbounded skill-code bloat.
    """
    # Growth-ratio limits
    GROWTH_HARD_LIMIT_RATIO: float = 3.0   # Hard limit: reject growth above 300%
    GROWTH_SOFT_LIMIT_RATIO: float = 2.0   # Soft limit: growth above 200% triggers a warning

    # Type-specific growth limits (Check 3)
    COMPLEX_GROWTH_LIMIT_RATIO: float = 1.75  # Max growth for complex skills (>100 lines)
    NORMAL_GROWTH_LIMIT_RATIO: float = 2.0    # Max growth for normal skills (25-100 lines)

    # Override cap for MEDIUM-priority fixes
    MEDIUM_FIX_OVERRIDE_RATIO: float = 3.0    # Max growth allowed for MEDIUM fixes (matches hard limit)

    # Line-count limits
    ABSOLUTE_MAX_LINES: int = 800          # Absolute limit: no skill exceeds 800 lines
    WRAPPER_MAX_LINES: int = 80            # Wrappers cap at 80 lines (relaxed)
    WRAPPER_LINE_THRESHOLD: int = 25       # Lines below this count are considered a wrapper
    WRAPPER_AWAIT_THRESHOLD: int = 2       # Maximum awaits in a wrapper

    # Helper-function limits
    MAX_NEW_HELPERS: int = 2               # At most 2 new helper functions

    # Covered-wrapper special limits (stricter)
    COVERED_WRAPPER_MAX_LINES: int = 20    # Covered wrappers cap at 20 lines
    COVERED_WRAPPER_MAX_GROWTH: float = 1.5  # Covered wrapper growth at most 50%

    # Retry config
    RETRY_ENABLED: bool = True             # Enable retry mechanism
    RETRY_MAX_DIFF_LINES: int = 10         # Maximum diff lines on retry

    # Stats config
    STATS_ENABLED: bool = True             # Enable stats tracking
    STATS_FILE: str = "bloat_stats.json"   # Stats file name


@dataclass
class SkipOptimizationConfig:
    """
    Skip-optimization configuration.

    Decides when to skip optimizing a skill.
    """
    # Maximum versions; beyond this, skip
    MAX_VERSIONS_BEFORE_SKIP: int = 20

    # === Probabilistic-skip parameters ===
    # P(optimize(s)) = (1-epsilon) * sigma(gamma(threshold - V(s))) + epsilon
    # where sigma is the sigmoid function

    # Minimum optimization probability (epsilon).
    # Even when V(s) is high, the skill still has this probability of being optimized.
    OPTIMIZATION_EPSILON: float = 0.05

    # Sigmoid slope (gamma).
    # Larger values make probability more sensitive to V(s) changes.
    OPTIMIZATION_GAMMA: float = 8.0

    # Value-function threshold.
    # Below this value the optimization probability is high; above it the probability is low.
    OPTIMIZATION_THRESHOLD: float = 0.6

    # Whether to enable probabilistic skipping (can be disabled for debugging)
    ENABLE_PROBABILITY_SKIP: bool = True


# Default configuration instances
DEFAULT_BLOAT_CONFIG = BloatPreventionConfig()
DEFAULT_SKIP_CONFIG = SkipOptimizationConfig()
