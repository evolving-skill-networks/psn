"""
Loop Manager — optimization-loop detection and breaking

Loop management extracted from optimizer_impl.py. Responsibilities:
1. Detect optimization loops (the same error pattern repeating)
2. Classify strategy-level issue types
3. Provide optimization statistics
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..feedback.types import OptimizationRecord


# Domain-agnostic strategy patterns (api_hints injected at runtime)
_STRATEGY_PATTERNS = {
    "placement_failure": {
        "patterns": [
            "could not find a valid position to place",
            "no valid adjacent",
            "cannot place.*block",
            "failed to place",
            "placeitem failed",
        ],
        "diagnosis": "Placement-location search has been optimized many times but still fails; the bot may need to move to a more suitable area",
    },
    "pathfinding_stuck": {
        "patterns": [
            "path.*blocked",
            "cannot reach",
            "no path found",
            "pathfinder.*failed",
            "goal.*unreachable",
        ],
        "diagnosis": "Pathfinding fails repeatedly; may need to change the goal or clear obstacles",
    },
    "mining_obstructed": {
        "patterns": [
            "cannot mine",
            "block.*unreachable",
            "no.*to mine",
            "mining.*failed",
        ],
        "diagnosis": "Mining target is unreachable; may need to adjust position or target selection",
    },
}


class LoopManager:
    """
    Optimization-loop manager

    Detects and handles loop issues during optimization, and provides loop-breaking suggestions.
    """

    def __init__(
        self,
        history: Dict[str, List["OptimizationRecord"]],
        logger: Optional[logging.Logger] = None,
        api_hints: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
            history: mapping from skill name to its list of optimization records
            logger: optional logger
            api_hints: domain-specific API hints per issue type (e.g. from DomainKnowledge)
        """
        self.history = history
        self.logger = logger or logging.getLogger(__name__)
        self._api_hints = api_hints or {}

    def detect_optimization_loop(
        self,
        skill_name: str,
        current_error_pattern: str,
        window_size: int = 5,
    ) -> Tuple[bool, str]:
        """
        Detect whether we are stuck in an optimization loop

        Args:
            skill_name: skill name
            current_error_pattern: current error pattern
            window_size: detection window size

        Returns:
            Tuple[bool, str]: (whether a loop is detected, reason)
        """
        if skill_name not in self.history:
            return False, ""

        recent = self.history[skill_name][-window_size:]

        # Check whether the same error pattern recurs
        same_pattern_count = sum(1 for r in recent if r.error_pattern == current_error_pattern)

        if same_pattern_count >= 3:
            # Check whether all such attempts failed
            same_pattern_records = [r for r in recent if r.error_pattern == current_error_pattern]
            all_failed = all(not r.successful for r in same_pattern_records)

            if all_failed:
                return True, (
                    f"Same error pattern '{current_error_pattern}' occurred {same_pattern_count} times in the last "
                    f"{window_size} optimizations and all failed — likely the optimization direction is wrong"
                )

        # Check whether the same strategy keeps failing
        if len(recent) >= 3:
            strategies = [r.strategy_used for r in recent]
            most_common_strategy = max(set(strategies), key=strategies.count)
            same_strategy_records = [r for r in recent if r.strategy_used == most_common_strategy]

            if len(same_strategy_records) >= 3 and all(not r.successful for r in same_strategy_records):
                return True, (
                    f"Strategy '{most_common_strategy}' has been tried {len(same_strategy_records)} times and all failed — "
                    f"need to try a different strategy"
                )

        return False, ""

    def get_optimization_statistics(self, skill_name: str) -> Dict[str, Any]:
        """Get optimization statistics for a skill"""
        if skill_name not in self.history:
            return {
                "total_optimizations": 0,
                "successful_optimizations": 0,
                "success_rate": 0.0,
                "most_common_error": None,
                "most_common_strategy": None,
            }

        records = self.history[skill_name]
        successful = sum(1 for r in records if r.successful)

        error_counts: Dict[str, int] = {}
        strategy_counts: Dict[str, int] = {}
        for r in records:
            error_counts[r.error_pattern] = error_counts.get(r.error_pattern, 0) + 1
            strategy_counts[r.strategy_used] = strategy_counts.get(r.strategy_used, 0) + 1

        most_common_error = max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        most_common_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else None

        return {
            "total_optimizations": len(records),
            "successful_optimizations": successful,
            "success_rate": successful / len(records) if records else 0.0,
            "most_common_error": most_common_error,
            "most_common_strategy": most_common_strategy,
            "error_distribution": error_counts,
            "strategy_distribution": strategy_counts,
        }

