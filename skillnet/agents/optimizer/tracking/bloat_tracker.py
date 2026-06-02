"""
Code Bloat Tracker

Tracks code-bloat statistics, recording code growth for each optimization.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from skillnet.agents.optimizer.config import DEFAULT_BLOAT_CONFIG as BLOAT_CONFIG

logger = logging.getLogger(__name__)


class CodeBloatTracker:
    """Code-bloat statistics tracker.

    Records code growth per optimization and provides summary stats.
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.stats_dir = os.path.join(ckpt_dir, "skill_graph")
        self.stats_file = os.path.join(self.stats_dir, BLOAT_CONFIG.STATS_FILE)
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Any]:
        """Load stats data."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[BloatTracker] failed to load stats: {e}")

        return {
            "optimizations": [],
            "avg_growth_ratio": 0.0,
            "max_growth_ratio": 0.0,
            "total_rejected": 0,
            "session_start": datetime.now().isoformat(),
        }

    def _save_stats(self):
        """Save stats data."""
        if not BLOAT_CONFIG.STATS_ENABLED:
            return

        try:
            os.makedirs(self.stats_dir, exist_ok=True)
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[BloatTracker] failed to save stats: {e}")

    def record_optimization(
        self,
        skill_name: str,
        old_lines: int,
        new_lines: int,
        accepted: bool,
        rejection_reason: str = None
    ):
        """Record code growth for an optimization.

        Args:
            skill_name: skill name
            old_lines: original code line count
            new_lines: new code line count
            accepted: whether accepted
            rejection_reason: rejection reason (when rejected)
        """
        if not BLOAT_CONFIG.STATS_ENABLED:
            return

        growth_ratio = new_lines / max(old_lines, 1)

        record = {
            "timestamp": datetime.now().isoformat(),
            "skill_name": skill_name,
            "old_lines": old_lines,
            "new_lines": new_lines,
            "growth_ratio": round(growth_ratio, 2),
            "accepted": accepted,
            "rejection_reason": rejection_reason,
        }

        self.stats["optimizations"].append(record)
        self._update_aggregates()
        self._save_stats()

        # Emit warning
        if growth_ratio > BLOAT_CONFIG.GROWTH_SOFT_LIMIT_RATIO:
            status = "accepted" if accepted else "rejected"
            logger.warning(f"[Code Growth Warning] {skill_name}: "
                           f"{old_lines} -> {new_lines} lines ({growth_ratio:.0%}), {status}")

    def _update_aggregates(self):
        """Update aggregate stats."""
        records = self.stats["optimizations"]
        if not records:
            return

        growth_ratios = [r["growth_ratio"] for r in records]
        self.stats["avg_growth_ratio"] = round(sum(growth_ratios) / len(growth_ratios), 2)
        self.stats["max_growth_ratio"] = max(growth_ratios)
        self.stats["total_rejected"] = sum(1 for r in records if not r["accepted"])

    def get_summary(self) -> Dict[str, Any]:
        """Get stats summary."""
        records = self.stats["optimizations"]

        if not records:
            return {
                "total_optimizations": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "average_growth_ratio": 0,
                "max_growth_ratio": 0,
                "bloated_skills": [],
                "rejection_reasons": {},
            }

        accepted = [r for r in records if r["accepted"]]
        rejected = [r for r in records if not r["accepted"]]

        # Bucket rejection reasons
        rejection_reasons = {}
        for r in rejected:
            reason = r.get("rejection_reason", "unknown")
            # Reduce reason to a category
            if "Resource" in reason or "resource" in reason.lower():
                category = "resource name mismatch"
            elif "growth" in reason.lower():
                category = "exceeded growth limit"
            elif "lines" in reason.lower():
                category = "exceeded line-count limit"
            else:
                category = "other"
            rejection_reasons[category] = rejection_reasons.get(category, 0) + 1

        # Get severely bloated skills
        bloated_skills = [
            {"skill": r["skill_name"], "growth": r["growth_ratio"]}
            for r in records
            if r["growth_ratio"] > BLOAT_CONFIG.GROWTH_SOFT_LIMIT_RATIO and r["accepted"]
        ]

        return {
            "total_optimizations": len(records),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "average_growth_ratio": self.stats.get("avg_growth_ratio", 0),
            "max_growth_ratio": self.stats.get("max_growth_ratio", 0),
            "bloated_skills": bloated_skills,
            "rejection_reasons": rejection_reasons,
        }

