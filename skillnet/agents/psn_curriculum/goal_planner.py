"""
GoalPlanner - Long-term goal planning with milestone tracking

Responsibilities:
- Manage stage-based goals (early_game -> mid_game -> late_game)
- Generate prerequisite task chains
- Check task feasibility against current inventory
- Provide next milestone tasks
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

from skillnet.agents.constants.task_semantics import TaskWithSemantic
from skillnet.core.dk_registry import get_domain_knowledge

from .resource_tracker import ResourceTracker


def _build_milestones_from_domain():
    """Build Milestone list from domain knowledge, or return empty list."""
    dk = get_domain_knowledge()
    if not dk:
        return []
    raw = dk.get_progression_milestones()
    if not raw:
        return []
    return [
        Milestone(
            name=m["name"],
            description=m["description"],
            required_items=m["required_items"],
            required_achievements=m.get("required_achievements", []),
            stage=GameStage(m.get("stage", "early")),
        )
        for m in raw
    ]


def _get_group_count(inventory: Dict[str, int], item_or_group: str) -> int:
    """Get count of an item or sum of all items in a group.

    Supports group aliases: e.g., when ENABLE_WOOD_AS_FUEL is True,
    counting 'fuel' also includes items from the 'logs' group.
    """
    dk = get_domain_knowledge()
    item_groups = dk.get_item_groups() if dk else {}
    if item_or_group in item_groups:
        total = sum(inventory.get(item, 0) for item in item_groups[item_or_group])

        # Check group aliases (e.g., fuel -> also count logs)
        group_aliases = dk.get_group_aliases() if dk else {}
        for alias_group in group_aliases.get(item_or_group, []):
            if alias_group in item_groups:
                total += sum(inventory.get(item, 0) for item in item_groups[alias_group])

        return total
    return inventory.get(item_or_group, 0)


class GameStage(Enum):
    """Progression stages"""
    EARLY = "early"      # Wood and stone tools
    MID = "mid"          # Iron tools and basic exploration
    LATE = "late"        # Diamond gear and advanced content
    END_GAME = "end"     # Nether, End dimension


@dataclass
class Milestone:
    """A milestone in the game progression.

    required_items can use item group names (e.g., "logs", "cooked_food")
    which will accept any item from that group.
    """
    name: str
    description: str
    required_items: Dict[str, int]  # Items or groups needed to achieve this milestone
    required_achievements: List[str] = field(default_factory=list)  # Named achievements
    stage: GameStage = GameStage.EARLY

    def is_achieved(self, inventory: Dict[str, int], achievements: Set[str]) -> bool:
        """Check if this milestone has been achieved.

        Supports item groups - "logs" will count all log types together.
        """
        # Check required items (with group support)
        for item_or_group, count in self.required_items.items():
            have = _get_group_count(inventory, item_or_group)
            if have < count:
                return False
        # Check required achievements
        for achievement in self.required_achievements:
            if achievement not in achievements:
                return False
        return True

    def get_missing_items(self, inventory: Dict[str, int]) -> Dict[str, int]:
        """Get items still needed for this milestone.

        Supports item groups - returns group name if it's a group.

        Note: returns the TARGET semantic (how many are needed in total), not a delta (how many are still missing).
        E.g. a milestone needs 4 logs and the inventory has 2 -> returns {logs: 4} (target count).
        This ensures the generated "Ensure you have X" task uses the correct TARGET semantic.
        """
        missing = {}
        for item_or_group, count in self.required_items.items():
            have = _get_group_count(inventory, item_or_group)
            if have < count:
                # Return TARGET count, not delta (count - have).
                # This ensures "Ensure you have {count}" satisfies the milestone once completed.
                missing[item_or_group] = count
        return missing


@dataclass
class FeasibilityReport:
    """Report on whether a task is feasible"""
    task: str
    is_feasible: bool
    confidence: float  # 0-1
    missing_items: Dict[str, int]
    missing_tools: List[str]
    suggested_prerequisites: List[str]
    estimated_steps: int
    already_satisfied: bool = False  # True when "ensure" task is already met by inventory

    def to_dict(self) -> Dict:
        return {
            "task": self.task,
            "is_feasible": self.is_feasible,
            "confidence": self.confidence,
            "missing_items": self.missing_items,
            "missing_tools": self.missing_tools,
            "suggested_prerequisites": self.suggested_prerequisites,
            "estimated_steps": self.estimated_steps,
            "already_satisfied": self.already_satisfied,
        }


@dataclass
class TaskNode:
    """A node in the task dependency graph"""
    task: str
    required_items: Dict[str, int]
    expected_output: Dict[str, int]
    prerequisites: List['TaskNode'] = field(default_factory=list)
    skill_name: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "task": self.task,
            "required_items": self.required_items,
            "expected_output": self.expected_output,
            "prerequisites": [p.to_dict() for p in self.prerequisites]
        }


DEFAULT_MILESTONES: List[Milestone] = []


class GoalPlanner:
    """
    Long-term goal planner for domain progression.

    Features:
    - Milestone tracking and progression
    - Task feasibility checking
    - Prerequisite chain generation
    - Integration with KnowledgeBase for dependencies
    """

    def __init__(
        self,
        knowledge_base,
        resource_tracker: ResourceTracker,
        milestones: Optional[List[Milestone]] = None,
        ckpt_dir: Optional[str] = None
    ):
        """
        Initialize the goal planner.

        Args:
            knowledge_base: Minecraft knowledge base for recipes
            resource_tracker: Resource tracker for inventory state
            milestones: Custom milestones (uses defaults if None)
            ckpt_dir: Directory to save/load progress
        """
        self.knowledge_base = knowledge_base
        self.resource_tracker = resource_tracker
        if milestones is not None:
            self.milestones = milestones
        else:
            built = _build_milestones_from_domain()
            self.milestones = built if built else DEFAULT_MILESTONES.copy()
        self.ckpt_dir = ckpt_dir

        # Track progress
        self.completed_milestones: Set[str] = set()
        self.current_goal: Optional[str] = None
        self.achievements: Set[str] = set()

        # Load saved progress
        if ckpt_dir:
            self._load_progress()

    def _load_progress(self) -> None:
        """Load saved progress from checkpoint"""
        if self.ckpt_dir is None:
            return
        progress_path = os.path.join(self.ckpt_dir, "psn_curriculum", "progress.json")
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    data = json.load(f)
                self.completed_milestones = set(data.get("completed_milestones", []))
                self.current_goal = data.get("current_goal")
                self.achievements = set(data.get("achievements", []))
            except Exception as e:
                print(f"Warning: Failed to load PSN progress: {e}")

    def save_progress(self) -> None:
        """Save progress to checkpoint"""
        if self.ckpt_dir is None:
            return
        progress_dir = os.path.join(self.ckpt_dir, "psn_curriculum")
        os.makedirs(progress_dir, exist_ok=True)
        progress_path = os.path.join(progress_dir, "progress.json")
        data = {
            "completed_milestones": list(self.completed_milestones),
            "current_goal": self.current_goal,
            "achievements": list(self.achievements)
        }
        with open(progress_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_progress(self, inventory: Optional[Dict[str, int]] = None) -> List[str]:
        """
        Update milestone progress based on current inventory.

        Returns:
            List of newly completed milestone names
        """
        if inventory is None:
            inventory = self.resource_tracker.current_inventory

        newly_completed = []
        for milestone in self.milestones:
            if milestone.name not in self.completed_milestones:
                if milestone.is_achieved(inventory, self.achievements):
                    self.completed_milestones.add(milestone.name)
                    newly_completed.append(milestone.name)
                    print(f"[PSN] Milestone achieved: {milestone.name}")

        if newly_completed:
            self.save_progress()

        return newly_completed

    def get_current_stage(self) -> GameStage:
        """Determine current game stage based on completed milestones"""
        # Check milestones in reverse order to find highest stage
        for stage in [GameStage.END_GAME, GameStage.LATE, GameStage.MID, GameStage.EARLY]:
            stage_milestones = [m for m in self.milestones if m.stage == stage]
            if any(m.name in self.completed_milestones for m in stage_milestones):
                return stage
        return GameStage.EARLY

    def get_next_milestone(self) -> Optional[Milestone]:
        """Get the next milestone to work towards"""
        for milestone in self.milestones:
            if milestone.name not in self.completed_milestones:
                return milestone
        return None

    def get_next_milestone_tasks(self) -> List[str]:
        """
        Get tasks needed to complete the next milestone.

        This method checks feasibility and expands prerequisite chains to ensure
        the returned list contains actionable tasks in the correct order.

        Returns:
            List of task descriptions in order they should be done.
            The first task in the list is guaranteed to be feasible (or a base task
            like mining that doesn't require prerequisites).
            Returns empty list only when ALL milestones are completed.
        """
        newly_completed = []

        # Skip already-completed milestones until we find the first unfinished one
        while True:
            milestone = self.get_next_milestone()
            if milestone is None:
                break  # All milestones completed

            inventory = self.resource_tracker.current_inventory

            if not milestone.is_achieved(inventory, self.achievements):
                break  # Found an unfinished milestone

            # The current milestone is already achieved — mark and keep looking
            print(f"\033[32m[PSN Milestone] '{milestone.name}' already achieved by inventory\033[0m")
            self.completed_milestones.add(milestone.name)
            newly_completed.append(milestone.name)

        # Save all newly completed milestones once
        if newly_completed:
            self.save_progress()

        # No unfinished milestones — return empty
        if milestone is None:
            return []

        missing = milestone.get_missing_items(inventory)

        # Collect tasks in execution order
        # We want: [feasible_base_tasks, then feasible_prereqs, then target_tasks]
        base_tasks = []      # Mining/gathering tasks (always feasible in theory)
        prereq_tasks = []    # Intermediate crafting tasks
        target_tasks = []    # Final milestone target tasks

        seen_tasks = set()   # Avoid duplicates

        for item, count in missing.items():
            task = self._item_to_task(item, count)
            if not task or task in seen_tasks:
                continue
            seen_tasks.add(task)

            # Check feasibility and expand prerequisites recursively
            self._expand_task_chain(task, base_tasks, prereq_tasks, target_tasks, seen_tasks)

        # Combine in learning order: prereqs first (learn sub-skills), then targets, then base
        # This ensures progressive skill learning while avoiding redundant gathering
        # Note: base_tasks (mining) are placed last to avoid selecting them when materials are sufficient
        result = prereq_tasks + target_tasks + base_tasks

        # Remove duplicates while preserving order
        final_result = []
        seen_final = set()
        for t in result:
            if t not in seen_final:
                final_result.append(t)
                seen_final.add(t)

        return final_result

    def _expand_task_chain(
        self,
        task: str,
        base_tasks: List[str],
        prereq_tasks: List[str],
        target_tasks: List[str],
        seen_tasks: set,
        depth: int = 0
    ) -> None:
        """
        Recursively expand a task into its prerequisite chain.

        Args:
            task: The task to expand
            base_tasks: List to append base mining/gathering tasks
            prereq_tasks: List to append intermediate tasks
            target_tasks: List to append final target tasks
            seen_tasks: Set of already seen tasks to avoid cycles
            depth: Current recursion depth (max 5)
        """
        if depth > 5:  # Prevent infinite recursion
            return

        task_lower = task.lower()

        # Check if this is a base task (mining/gathering - always "feasible")
        if any(kw in task_lower for kw in ["mine", "obtain", "gather", "collect"]):
            if task not in base_tasks:
                base_tasks.append(task)
            return

        # Check feasibility - create temporary TaskWithSemantic for internal use
        task_obj = TaskWithSemantic.from_legacy_string(task)
        feasibility = self.check_feasibility(task_obj)

        if feasibility.is_feasible and feasibility.already_satisfied:
            # Task already satisfied by current inventory — skip entirely
            return

        if feasibility.is_feasible:
            # Task can be done directly
            if task not in prereq_tasks and task not in target_tasks:
                if depth == 0:
                    target_tasks.append(task)
                else:
                    prereq_tasks.append(task)
        elif feasibility.suggested_prerequisites:
            # Expand prerequisites first
            for prereq in feasibility.suggested_prerequisites:
                if prereq not in seen_tasks:
                    seen_tasks.add(prereq)
                    self._expand_task_chain(
                        prereq, base_tasks, prereq_tasks, target_tasks, seen_tasks, depth + 1
                    )

            # Then add this task
            if task not in prereq_tasks and task not in target_tasks:
                if depth == 0:
                    target_tasks.append(task)
                else:
                    prereq_tasks.append(task)
        else:
            # No feasibility info, treat as base task
            if task not in base_tasks:
                base_tasks.append(task)

    def _item_to_task(self, item: str, count: int) -> Optional[str]:
        """Convert an item requirement to a task description.

        Handles both specific items (e.g. "oak_log") and item groups (e.g. "logs").

        PSN Curriculum semantic design
        ==============================
        All tasks use TARGET semantics ("Ensure you have X" means "ensure you
        eventually have X"). This keeps the system aligned with ResourceTracker's
        threshold-based design.
        """
        # Handle item groups first
        dk = get_domain_knowledge()
        _item_groups = dk.get_item_groups() if dk else {}
        if item in _item_groups:
            if item == "logs":
                return f"Ensure you have {count} wood logs"
            elif item == "planks":
                return f"Ensure you have {count} planks"
            elif item == "cooked_food":
                return f"Ensure you have {count} cooked food"
            elif item == "fuel":
                return f"Ensure you have {count} fuel (coal or wood)"
            elif item in ["pickaxes", "axes", "swords", "shovels", "hoes"]:
                # For tool groups, suggest crafting the basic version
                tool_type = item[:-1] if item.endswith("s") else item  # Remove plural 's'
                return f"Craft 1 wooden {tool_type}"
            else:
                return f"Ensure you have {count} {item.replace('_', ' ')}"

        # Check if item can be mined
        if self.knowledge_base.is_raw_material(item):
            if "log" in item:
                return f"Ensure you have {count} {item.replace('_', ' ')}"
            elif "ore" in item or item in ["coal", "diamond", "emerald", "redstone", "lapis_lazuli"]:
                return f"Ensure you have {count} {item.replace('_', ' ')}"
            elif item in ["cobblestone", "stone"]:
                return f"Ensure you have {count} {item.replace('_', ' ')}"
            else:
                return f"Ensure you have {count} {item.replace('_', ' ')}"

        # Check if item can be smelted (priority: smelting is the "canonical" way to obtain
        # items like iron_ingot, gold_ingot, etc. Checking crafting first would incorrectly
        # suggest decomposing storage blocks like iron_block -> iron_ingot)
        smelt = self.knowledge_base.get_smelt_recipe_for_output(item)
        if smelt:
            return f"Smelt {count} {item.replace('_', ' ')}"

        # Check if item can be crafted
        recipe = self.knowledge_base.get_recipe(item)
        if recipe:
            return f"Craft {count} {item.replace('_', ' ')}"

        # Fallback: use "Ensure you have" for consistency with failed_tasks tracking
        # This ensures task name matching works correctly for failure counting
        return f"Ensure you have {count} {item.replace('_', ' ')}"

    def check_feasibility(self, task: TaskWithSemantic, _visited: Optional[Set[str]] = None) -> FeasibilityReport:
        """
        Check if a task is feasible with current resources.

        Args:
            task: TaskWithSemantic object
            _visited: Internal parameter for cycle detection

        Returns:
            FeasibilityReport with details on feasibility
        """
        task_str = task.task

        # Cycle detection - prevent infinite recursion
        if _visited is None:
            _visited = set()

        task_normalized = task_str.lower().strip()
        if task_normalized in _visited:
            # Cycle detected - return as infeasible with no prerequisites
            return FeasibilityReport(
                task=task_str,
                is_feasible=False,
                confidence=0.0,
                missing_items={},
                missing_tools=[],
                suggested_prerequisites=[],
                estimated_steps=999  # High cost to discourage this path
            )
        _visited.add(task_normalized)

        inventory = self.resource_tracker.current_inventory

        # Parse task
        task_lower = task_str.lower()
        missing_items: Dict[str, int] = {}
        missing_tools: List[str] = []
        prerequisites: List[str] = []

        if "craft" in task_lower:
            # Extract item name and count
            item, count = self._parse_craft_task(task_str)
            if item:
                can_craft, missing = self.knowledge_base.check_can_craft(item, inventory, count)
                missing_items = missing

                # Check if crafting table is needed (use best variant for inventory)
                recipe = self.knowledge_base.get_best_recipe(item, inventory, count)
                if recipe and recipe.requires_crafting_table:
                    from skillnet.core.dk_registry import get_domain_knowledge
                    _dk = get_domain_knowledge()
                    station_cfg = _dk.get_crafting_station_config() if _dk else {}
                    station_item = station_cfg.get("item_name")
                    if station_item and inventory.get(station_item, 0) == 0:
                        can_craft_station, _ = self.knowledge_base.check_can_craft(
                            station_item, inventory
                        )
                        if not can_craft_station:
                            prerequisites.append(
                                station_cfg.get("craft_task", f"Craft 1 {station_item}")
                            )

                # Generate prerequisites for missing items
                # Priority: mining tasks first, then crafting tasks
                # This prevents cycles like: craft planks -> craft planks
                mining_prereqs = []
                crafting_prereqs = []

                # Compute the TARGET amount per ingredient (not delta).
                # E.g. a recipe needs 3 iron_ingot — regardless of current inventory,
                # the task should be "Ensure you have 3". This keeps "Ensure you have X"
                # tasks using the correct TARGET semantics.
                required_amounts = {}
                if recipe:
                    recipes_needed = (count + recipe.result_count - 1) // recipe.result_count
                    for ingredient, ing_count in recipe.ingredients.items():
                        if ingredient in missing:  # Only emit tasks for missing ingredients
                            required_amounts[ingredient] = ing_count * recipes_needed

                for missing_item, _delta in missing.items():
                    # Use the TARGET amount, not the delta
                    target_count = required_amounts.get(missing_item, _delta)
                    prereq = self._item_to_task(missing_item, target_count)
                    if prereq:
                        # Categorize by type
                        prereq_lower = prereq.lower()
                        if "mine" in prereq_lower or "obtain" in prereq_lower:
                            mining_prereqs.append(prereq)
                        else:
                            crafting_prereqs.append(prereq)

                # Add mining tasks first (base tasks), then crafting
                prerequisites.extend(mining_prereqs)
                prerequisites.extend(crafting_prereqs)

                return FeasibilityReport(
                    task=task_str,
                    is_feasible=can_craft and len(prerequisites) == 0,
                    confidence=0.9 if can_craft else 0.5,
                    missing_items=missing_items,
                    missing_tools=missing_tools,
                    suggested_prerequisites=prerequisites,
                    estimated_steps=len(prerequisites) + 1
                )

        elif "mine" in task_lower:
            # Check for required tools
            item = self._extract_item_from_task(task_str)
            if item:
                required_tool = self.knowledge_base.get_required_tool(item)
                min_tier = self.knowledge_base.get_min_tool_tier(item)

                if required_tool == "pickaxe":
                    # Check if we have a suitable pickaxe
                    has_pickaxe = self._has_tool_tier("pickaxe", min_tier, inventory)
                    if not has_pickaxe:
                        tier = min_tier or "wooden"
                        missing_tools.append(f"{tier}_pickaxe")
                        prerequisites.append(f"Craft 1 {tier} pickaxe")

                return FeasibilityReport(
                    task=task_str,
                    is_feasible=len(missing_tools) == 0,
                    confidence=0.8,
                    missing_items=missing_items,
                    missing_tools=missing_tools,
                    suggested_prerequisites=prerequisites,
                    estimated_steps=len(prerequisites) + 1
                )

        elif "smelt" in task_lower or "cook" in task_lower:
            # Check for furnace - this is critical for smelting
            if inventory.get("furnace", 0) == 0:
                can_craft_furnace, furnace_missing = self.knowledge_base.check_can_craft(
                    "furnace", inventory
                )
                if can_craft_furnace:
                    # Can craft furnace, add it as prerequisite
                    prerequisites.append("Craft 1 furnace")
                else:
                    # Cannot craft furnace - need to get materials first
                    # Furnace requires 8 cobblestone
                    cobblestone_count = inventory.get("cobblestone", 0)
                    if cobblestone_count < 8:
                        # Need to ensure we have 8 cobblestone total (TARGET semantic)
                        prerequisites.append("Ensure you have 8 cobblestone")
                    # Then craft the furnace
                    prerequisites.append("Craft 1 furnace")
                    missing_items.update(furnace_missing)

            # Check for fuel
            has_fuel = (
                inventory.get("coal", 0) > 0 or
                inventory.get("charcoal", 0) > 0 or
                any("log" in item or "planks" in item for item in inventory)
            )
            if not has_fuel:
                prerequisites.append("Ensure you have fuel (coal or wood)")

            return FeasibilityReport(
                task=task_str,
                is_feasible=len(prerequisites) == 0,
                confidence=0.8,
                missing_items=missing_items,
                missing_tools=missing_tools,
                suggested_prerequisites=prerequisites,
                estimated_steps=len(prerequisites) + 1
            )

        elif "ensure" in task_lower and "have" in task_lower:
            # Parse "Ensure you have X item" — check inventory satisfaction
            item, count = self._parse_ensure_task(task_str)
            if item:
                current_count = self._count_inventory_item(item, inventory)
                if current_count >= count:
                    # Already satisfied by current inventory
                    return FeasibilityReport(
                        task=task_str,
                        is_feasible=True,
                        confidence=1.0,
                        missing_items={},
                        missing_tools=[],
                        suggested_prerequisites=[],
                        estimated_steps=0,
                        already_satisfied=True,
                    )
                # Not satisfied — generate a prerequisite task
                prereq = self._item_to_task(item, count)  # TARGET semantic
                prereqs = []
                if prereq and prereq.lower().strip() != task_lower.strip():
                    prereqs.append(prereq)
                deficit = count - current_count
                return FeasibilityReport(
                    task=task_str,
                    is_feasible=len(prereqs) == 0,
                    confidence=0.7,
                    missing_items={item: deficit},
                    missing_tools=[],
                    suggested_prerequisites=prereqs,
                    estimated_steps=len(prereqs) + 1,
                )

        # Default: assume feasible
        return FeasibilityReport(
            task=task_str,
            is_feasible=True,
            confidence=0.5,  # Low confidence since we don't know the task
            missing_items={},
            missing_tools=[],
            suggested_prerequisites=[],
            estimated_steps=1
        )

    def _parse_craft_task(self, task: str) -> Tuple[Optional[str], int]:
        """Parse a craft task to extract item and count"""
        import re
        # Match patterns like "Craft 1 iron_pickaxe" or "craft iron pickaxe"
        match = re.match(r"craft\s+(\d+)?\s*(.+)", task.lower())
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).strip().replace(" ", "_")
            return item, count
        return None, 1

    def _parse_ensure_task(self, task: str) -> Tuple[Optional[str], int]:
        """Parse an 'Ensure you have X item' task to extract item and count."""
        import re
        match = re.match(
            r"ensure\s+you\s+have\s+(\d+)?\s*(.+)", task.lower()
        )
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).strip().replace(" ", "_")
            return item, count
        return None, 1

    def _count_inventory_item(self, item: str, inventory: Dict[str, int]) -> int:
        """Count how many of an item the player has, with group matching.

        Handles:
        - Exact match: "cobblestone" → inventory["cobblestone"]
        - Group match via _get_group_count: "wood_logs" → sum of all *_log
        - Fuzzy suffix match: "wood_logs" → try "logs" group
        """
        # Try exact match first
        if item in inventory:
            return inventory[item]

        # Try group match (handles "logs", "planks", etc.)
        count = _get_group_count(inventory, item)
        if count > 0:
            return count

        # Fuzzy: "wood_logs" → try "logs"
        for suffix in ["_logs", "_planks"]:
            if item.endswith(suffix):
                group_name = suffix.lstrip("_")
                count = _get_group_count(inventory, group_name)
                if count > 0:
                    return count

        # Fuzzy: "wood logs" was converted to "wood_logs", check for log variants
        if "log" in item:
            return _get_group_count(inventory, "logs")

        return 0

    def _extract_item_from_task(self, task: str) -> Optional[str]:
        """Extract item name from a task description"""
        import re
        task_lower = task.lower()
        # Match TARGET semantic: "Ensure you have 1 iron_ore"
        match = re.match(r"ensure you have\s+(\d+)?\s*(.+)", task_lower)
        if match:
            item = match.group(2).strip().replace(" ", "_")
            return item
        # Match DELTA semantic (legacy): "Mine 1 iron_ore" or "mine iron ore"
        match = re.match(r"(?:mine|obtain|gather|collect)\s+(\d+)?\s*(.+)", task_lower)
        if match:
            item = match.group(2).strip().replace(" ", "_")
            return item
        return None

    def _has_tool_tier(self, tool_type: str, min_tier: Optional[str],
                       inventory: Dict[str, int]) -> bool:
        """Check if player has a tool of at least the minimum tier"""
        if min_tier is None:
            min_tier = "wooden"

        dk = get_domain_knowledge()
        if dk:
            tier_config = dk.get_tool_tier_config()
            tool_tiers = sorted(
                tier_config.get("tool_tiers", {}).keys(),
                key=lambda t: tier_config["tool_tiers"][t],
            ) if tier_config.get("tool_tiers") else []
        else:
            tool_tiers = []
        min_tier_idx = tool_tiers.index(min_tier) if min_tier in tool_tiers else 0

        for tier_idx, tier in enumerate(tool_tiers):
            if tier_idx >= min_tier_idx:
                tool_name = f"{tier}_{tool_type}"
                if inventory.get(tool_name, 0) > 0:
                    return True
        return False

    def generate_prerequisite_chain(self, target_task: str,
                                      visited: Optional[Set[str]] = None,
                                      max_depth: int = 10) -> List[TaskNode]:
        """
        Generate a full prerequisite chain for a target task.

        Args:
            target_task: The task to generate prerequisites for
            visited: Set of already visited tasks to prevent cycles
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            Ordered list of TaskNodes representing the chain
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        if target_task in visited or max_depth <= 0:
            return []

        visited.add(target_task)

        # Create temporary TaskWithSemantic for internal use
        task_obj = TaskWithSemantic.from_legacy_string(target_task)
        feasibility = self.check_feasibility(task_obj)
        if feasibility.is_feasible:
            return []

        chain: List[TaskNode] = []

        # Recursively build chain for each prerequisite
        for prereq in feasibility.suggested_prerequisites:
            # Skip if already visited (cycle detection)
            if prereq in visited:
                continue

            prereq_chain = self.generate_prerequisite_chain(
                prereq,
                visited.copy(),  # Use copy to allow parallel branches
                max_depth - 1
            )
            chain.extend(prereq_chain)

            # Add the prerequisite itself
            chain.append(TaskNode(
                task=prereq,
                required_items=feasibility.missing_items,
                expected_output={}
            ))

        return chain

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary"""
        lines = ["=== PSN Curriculum Progress ==="]
        lines.append(f"Current Stage: {self.get_current_stage().value}")
        lines.append(f"Completed Milestones: {len(self.completed_milestones)}/{len(self.milestones)}")

        next_milestone = self.get_next_milestone()
        if next_milestone:
            lines.append(f"\nNext Milestone: {next_milestone.name}")
            lines.append(f"Description: {next_milestone.description}")
            missing = next_milestone.get_missing_items(self.resource_tracker.current_inventory)
            if missing:
                lines.append("Missing items:")
                for item, count in missing.items():
                    lines.append(f"  - {item}: {count}")
        else:
            lines.append("\nAll milestones completed!")

        return "\n".join(lines)
