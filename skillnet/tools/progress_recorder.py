"""
Progress Recorder for PSN

Records iteration-level progress data for visualization and experiment analysis.
Data is stored in JSONL format for efficient append-only writes.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


class ProgressRecorder:
    """
    Records iteration-level progress data for the Progress Viewer.

    Data files created:
    - progress/iteration_log.jsonl: Per-iteration records
    - progress/skill_events.jsonl: Skill lifecycle events
    - progress/resource_stats.jsonl: Resource statistics
    """

    # Milestone definitions — populated from domain knowledge in __init__
    MILESTONE_ITEMS = {}

    def __init__(
        self,
        ckpt_dir: str,
        mode: str = "graph",  # "graph" or "legacy"
        resume: bool = False,
        domain_knowledge=None,
    ):
        """
        Initialize the progress recorder.

        Args:
            ckpt_dir: Checkpoint directory path
            mode: Skill manager mode ("graph" or "legacy")
            resume: Whether to resume from existing data
            domain_knowledge: DomainKnowledge instance for milestone items
        """
        # Build milestone items from domain knowledge
        if domain_knowledge:
            milestone_mapping = domain_knowledge.get_milestone_item_mapping()
            self.MILESTONE_ITEMS = {item: f"first_{item}" for item in milestone_mapping}

        self.ckpt_dir = Path(ckpt_dir)
        self.progress_dir = self.ckpt_dir / "progress"
        self.mode = mode

        # Create progress directory
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.iteration_log_path = self.progress_dir / "iteration_log.jsonl"
        self.skill_events_path = self.progress_dir / "skill_events.jsonl"
        self.resource_stats_path = self.progress_dir / "resource_stats.jsonl"

        # Current iteration state
        self.current_iteration = 0
        self.current_task = ""
        self.iteration_start_time: Optional[datetime] = None
        self.iteration_llm_calls = 0
        self.iteration_llm_tokens = 0

        # Cumulative state
        self.cumulative_inventory: Dict[str, int] = {}
        self.reached_milestones: Set[str] = set()
        self.known_skills: Set[str] = set()  # Skills that existed before this iteration

        # Skills state for this iteration
        self.skills_generated_this_iteration: List[str] = []
        self.skills_executed_this_iteration: List[str] = []
        self.call_tree: Dict[str, List[str]] = {}

        if resume:
            self._load_state()

    def _load_state(self):
        """Load state from existing files for resume."""
        # Load reached milestones from iteration log
        if self.iteration_log_path.exists():
            with open(self.iteration_log_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        self.current_iteration = record.get("iteration", 0)
                        # Track reached milestones
                        milestones = record.get("milestones_reached", [])
                        self.reached_milestones.update(milestones)
                        # Track cumulative inventory
                        inventory = record.get("inventory_snapshot", {})
                        self.cumulative_inventory = inventory.copy()
                    except json.JSONDecodeError:
                        continue

        # Load known skills from skill events
        if self.skill_events_path.exists():
            with open(self.skill_events_path, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get("event_type") == "skill_added":
                            self.known_skills.add(event.get("skill_name", ""))
                    except json.JSONDecodeError:
                        continue

    def _append_jsonl(self, path: Path, data: Dict[str, Any]):
        """Append a JSON record to a JSONL file."""
        with open(path, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def record_iteration_start(self, task: str, iteration: int):
        """
        Called at the start of each iteration.

        Args:
            task: Task description
            iteration: Iteration number
        """
        self.current_iteration = iteration
        self.current_task = task
        self.iteration_start_time = datetime.now()
        self.iteration_llm_calls = 0
        self.iteration_llm_tokens = 0

        # Reset iteration-specific tracking
        self.skills_generated_this_iteration = []
        self.skills_executed_this_iteration = []
        self.call_tree = {}

    def record_llm_call(self, tokens: int = 0):
        """
        Record an LLM call during this iteration.

        Args:
            tokens: Number of tokens used in the call
        """
        self.iteration_llm_calls += 1
        self.iteration_llm_tokens += tokens

    def record_skill_added(self, skill_name: str, version: str = "1.0.0"):
        """
        Record that a new skill was added.

        Args:
            skill_name: Name of the skill
            version: Version string
        """
        event = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "event_type": "skill_added",
            "skill_name": skill_name,
            "version": version,
        }
        self._append_jsonl(self.skill_events_path, event)

        # Track as generated this iteration (new skill)
        if skill_name not in self.known_skills:
            self.skills_generated_this_iteration.append(skill_name)
            self.known_skills.add(skill_name)

    def record_skill_executed(
        self,
        skill_name: str,
        success: bool,
        call_depth: int = 1,
        parent_skill: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ):
        """
        Record a skill execution.

        Args:
            skill_name: Name of the executed skill
            success: Whether execution succeeded
            call_depth: Depth in call stack (1 = top-level)
            parent_skill: Name of parent skill (if nested call)
            duration_ms: Execution duration in milliseconds
        """
        event = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "event_type": "skill_executed",
            "skill_name": skill_name,
            "success": success,
            "call_depth": call_depth,
        }
        if duration_ms is not None:
            event["duration_ms"] = duration_ms

        self._append_jsonl(self.skill_events_path, event)

        # Track executed skills
        if skill_name not in self.skills_executed_this_iteration:
            self.skills_executed_this_iteration.append(skill_name)

        # Build call tree
        if parent_skill:
            if parent_skill not in self.call_tree:
                self.call_tree[parent_skill] = []
            if skill_name not in self.call_tree[parent_skill]:
                self.call_tree[parent_skill].append(skill_name)

    def record_skill_refactored(
        self,
        skill_name: str,
        new_version: str,
        refactor_type: str,
        covered_by: Optional[str] = None,
    ):
        """
        Record a skill refactoring event.

        Args:
            skill_name: Name of the refactored skill
            new_version: New version string
            refactor_type: Type of refactoring (e.g., "parametric", "merge")
            covered_by: Name of skill that now covers this one
        """
        event = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "event_type": "skill_refactored",
            "skill_name": skill_name,
            "new_version": new_version,
            "refactor_type": refactor_type,
        }
        if covered_by:
            event["covered_by"] = covered_by

        self._append_jsonl(self.skill_events_path, event)

    def record_skill_optimized(
        self,
        skill_name: str,
        old_version: str,
        new_version: str,
        triggered_by_skill: Optional[str] = None,
        triggered_by_task: Optional[str] = None,
        optimization_type: str = "backpropagation",
    ):
        """
        Record a skill optimization event (e.g., from backpropagation).

        Args:
            skill_name: Name of the optimized skill
            old_version: Previous version string
            new_version: New version string
            triggered_by_skill: Skill that triggered the optimization
            triggered_by_task: Task that triggered the optimization
            optimization_type: Type of optimization
        """
        event = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "event_type": "skill_optimized",
            "skill_name": skill_name,
            "old_version": old_version,
            "new_version": new_version,
            "optimization_type": optimization_type,
        }
        if triggered_by_skill:
            event["triggered_by_skill"] = triggered_by_skill
        if triggered_by_task:
            event["triggered_by_task"] = triggered_by_task

        self._append_jsonl(self.skill_events_path, event)

    def _check_milestones(self, inventory: Dict[str, int]) -> List[str]:
        """
        Check for new milestones based on inventory.

        Args:
            inventory: Current inventory state

        Returns:
            List of newly reached milestone names
        """
        new_milestones = []

        for item, milestone in self.MILESTONE_ITEMS.items():
            if milestone not in self.reached_milestones:
                if inventory.get(item, 0) > 0:
                    new_milestones.append(milestone)
                    self.reached_milestones.add(milestone)

                    # Record milestone event
                    event = {
                        "iteration": self.current_iteration,
                        "timestamp": datetime.now().isoformat(),
                        "event_type": "milestone",
                        "milestone_type": milestone,
                        "item": item,
                    }
                    self._append_jsonl(self.skill_events_path, event)

        return new_milestones

    def _calculate_resource_changes(
        self,
        old_inventory: Dict[str, int],
        new_inventory: Dict[str, int],
    ) -> tuple:
        """
        Calculate resources collected and consumed.

        Returns:
            (resources_collected, resources_consumed)
        """
        collected = {}
        consumed = {}

        all_items = set(old_inventory.keys()) | set(new_inventory.keys())
        for item in all_items:
            old_count = old_inventory.get(item, 0)
            new_count = new_inventory.get(item, 0)
            delta = new_count - old_count

            if delta > 0:
                collected[item] = delta
            elif delta < 0:
                consumed[item] = -delta

        return collected, consumed

    def record_iteration_end(
        self,
        success: bool,
        final_inventory: Dict[str, int],
        position: Dict[str, float],
        skill_count: int,
        active_skill_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ):
        """
        Called at the end of each iteration.

        Args:
            success: Whether the task succeeded
            final_inventory: Final inventory state
            position: Final position {x, y, z}
            skill_count: Total number of skills
            active_skill_count: Number of non-deprecated skills
            error_message: Error message if failed
        """
        if active_skill_count is None:
            active_skill_count = skill_count

        # Calculate resource changes
        collected, consumed = self._calculate_resource_changes(
            self.cumulative_inventory, final_inventory
        )

        # Check for new milestones
        new_milestones = self._check_milestones(final_inventory)

        # Determine which skills were reused vs generated
        skills_reused = [
            s for s in self.skills_executed_this_iteration
            if s not in self.skills_generated_this_iteration
        ]

        # Record iteration
        record = {
            "iteration": self.current_iteration,
            "timestamp": self.iteration_start_time.isoformat() if self.iteration_start_time else datetime.now().isoformat(),
            "task": self.current_task,
            "success": success,
            "mode": self.mode,

            # Skill statistics
            "skill_count": skill_count,
            "active_skill_count": active_skill_count,
            "skills_executed": self.skills_executed_this_iteration,
            "skills_reused": skills_reused,
            "skills_generated": self.skills_generated_this_iteration,
            "call_tree": self.call_tree,

            # State snapshot
            "inventory_snapshot": final_inventory,
            "position": position,

            # Efficiency metrics
            "llm_calls": self.iteration_llm_calls,
            "llm_tokens": self.iteration_llm_tokens,

            # Milestones
            "milestones_reached": new_milestones,
        }

        if error_message:
            record["error_message"] = error_message

        self._append_jsonl(self.iteration_log_path, record)

        # Record resource stats
        resource_record = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "cumulative_inventory": final_inventory,
            "resources_collected": collected,
            "resources_consumed": consumed,
        }
        self._append_jsonl(self.resource_stats_path, resource_record)

        # Update cumulative inventory
        self.cumulative_inventory = final_inventory.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of recorded progress."""
        iterations = []
        total_success = 0
        total_llm_calls = 0

        if self.iteration_log_path.exists():
            with open(self.iteration_log_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        iterations.append(record)
                        if record.get("success"):
                            total_success += 1
                        total_llm_calls += record.get("llm_calls", 0)
                    except json.JSONDecodeError:
                        continue

        return {
            "total_iterations": len(iterations),
            "successful_iterations": total_success,
            "success_rate": total_success / len(iterations) if iterations else 0,
            "total_llm_calls": total_llm_calls,
            "milestones_reached": list(self.reached_milestones),
            "skill_count": len(self.known_skills),
        }
