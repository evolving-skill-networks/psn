"""
TaskDecomposer - task decomposer

Uses an LLM to break complex tasks into a sequence of simple, learnable subtasks.

Core responsibilities:
- Use the LLM to decompose complex tasks
- Validate the reasonableness of the decomposition
- Sort subtasks by dependency
- Cache decomposition results
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.constants.task_semantics import TaskWithSemantic
from skillnet.utils.stats_tracker import record_llm_usage


@dataclass
class SubTask:
    """A decomposed subtask"""
    task_description: str      # task description
    task_type: str            # ensure/craft/mine/find/place
    target_item: str          # target item
    target_count: int         # target count
    prerequisites: List[str] = field(default_factory=list)  # prerequisite item requirements


@dataclass
class DecomposedTask:
    """Decomposition result"""
    original_task: str
    subtasks: List[SubTask]
    decomposition_reason: str


class TaskDecomposer:
    """
    Task decomposer

    Core responsibilities:
    - Use the LLM to decompose complex tasks
    - Validate the reasonableness of the decomposition
    - Sort subtasks by dependency
    """

    def __init__(
        self,
        llm,
        knowledge_base: Optional[Any] = None,
        max_subtasks: int = 10,
        cache_enabled: bool = True,
    ):
        """
        Initialize the task decomposer

        Args:
            llm: LangChain LLM instance
            knowledge_base: Minecraft knowledge base (optional)
            max_subtasks: maximum number of subtasks
            cache_enabled: whether caching is enabled
        """
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.max_subtasks = max_subtasks
        self.cache_enabled = cache_enabled

        # DomainKnowledge reference (set externally, has get_tool_unlock_mapping)
        self._domain_knowledge = None

        # Decomposition cache
        self._decomposition_cache: Dict[str, DecomposedTask] = {}

    def decompose(
        self,
        task: TaskWithSemantic,
        inventory: Optional[Dict[str, int]] = None,
        skill_names: Optional[List[str]] = None,
    ) -> DecomposedTask:
        """
        Decompose a task into a sequence of subtasks

        Args:
            task: TaskWithSemantic object
            inventory: current inventory
            skill_names: list of already-learned skill names

        Returns:
            DecomposedTask: decomposition result
        """
        task_str = task.task

        # Check cache
        cache_key = self._get_cache_key(task_str)
        if self.cache_enabled and cache_key in self._decomposition_cache:
            print(f"\033[36m[TaskDecomposer] Using cached decomposition for: {task_str}\033[0m")
            return self._decomposition_cache[cache_key]

        # LLM decomposition
        print(f"\033[36m[TaskDecomposer] Decomposing task: {task_str}\033[0m")
        subtasks = self._decompose_with_llm(task_str, inventory, skill_names)

        # Validate the decomposition result
        if not self._validate_decomposition(subtasks, task_str):
            print(f"\033[33m[TaskDecomposer] Validation failed, keeping original task\033[0m")
            # Phase 10: if we can't extract a target from the parent task,
            # ABORT the decomposition entirely (return None) rather than
            # emit a "unknown" placeholder target that downstream would
            # surface as "Ensure you have 1 unknown". Caller falls back to
            # the original task_str via its own path.
            extracted_target = self._extract_target(task_str)
            if not extracted_target:
                print(
                    f"\033[33m[TaskDecomposer] Could not extract target from "
                    f"'{task_str}'; aborting decomposition (no 'unknown' fallback)\033[0m"
                )
                return None
            return DecomposedTask(
                original_task=task_str,
                subtasks=[SubTask(
                    task_description=task_str,
                    task_type="ensure",
                    target_item=extracted_target,
                    target_count=1,
                    prerequisites=[]
                )],
                decomposition_reason="Decomposition validation failed, keeping original task"
            )

        # Sort by dependencies
        subtasks = self._sort_by_dependencies(subtasks)

        result = DecomposedTask(
            original_task=task_str,
            subtasks=subtasks,
            decomposition_reason="LLM decomposition"
        )

        # Cache the result
        if self.cache_enabled:
            self._decomposition_cache[cache_key] = result

        print(f"\033[32m[TaskDecomposer] Decomposed into {len(subtasks)} subtasks\033[0m")
        for i, st in enumerate(subtasks):
            print(f"  {i+1}. {st.task_description} (target: {st.target_item})")

        return result

    def _decompose_with_llm(
        self,
        task: str,
        inventory: Optional[Dict[str, int]],
        skill_names: Optional[List[str]],
    ) -> List[SubTask]:
        """Decompose a task using the LLM"""
        # Inject domain knowledge (tool tiers) if available.
        # Without this, the LLM may generate incorrect prerequisites
        # (e.g., requiring iron_pickaxe to mine iron_ore, when stone_pickaxe suffices).
        domain_knowledge_section = ""
        try:
            # Try DomainKnowledge (has get_tool_unlock_mapping)
            dk = self._domain_knowledge
            if dk is None:
                # Fallback: try central registry
                from skillnet.core.dk_registry import get_domain_knowledge
                dk = get_domain_knowledge()
            if dk:
                tool_unlock = dk.get_tool_unlock_mapping()
                if tool_unlock:
                    lines = ["## Tool tier rules (CRITICAL for prerequisites)"]
                    for tool, ores in tool_unlock.items():
                        if ores:
                            lines.append(f"- {tool} can mine: {', '.join(ores)}")
                    lines.append("- Do NOT require a higher-tier tool if the current tier already suffices")
                    domain_knowledge_section = "\n".join(lines)
        except Exception:
            pass  # Non-critical

        # Load the system prompt (task-planner instructions + example) from
        # the active DomainKnowledge. This was previously a Minecraft-flavored
        # string literal hardcoded in this module; routing it through DK lets
        # each domain ship its own decomposition rubric (e.g., a dict-event domain's
        # collect/place/eat/sleep vocabulary instead of mine/smelt).
        template = ""
        dk_for_prompt = self._domain_knowledge
        if dk_for_prompt is None:
            try:
                from skillnet.core.dk_registry import get_domain_knowledge
                dk_for_prompt = get_domain_knowledge()
            except Exception:
                dk_for_prompt = None
        if dk_for_prompt is not None:
            template = dk_for_prompt.get_prompt("task_decomposition") or ""
        if not template:
            raise RuntimeError(
                "task_decomposition prompt is not provided by the active "
                "DomainKnowledge; supply skillnet/domains/<domain>/prompts/"
                "task_decomposition.txt"
            )

        # Build the user message (per-call state — not a domain template)
        user_content = f"""## Task to decompose
{task}

## Current state
- Inventory: {inventory or {}}
- Already learned skills: {skill_names or []}

{domain_knowledge_section}

Now decompose the given task. Return ONLY valid JSON, no other text."""

        messages = [
            SystemMessage(content=template),
            HumanMessage(content=user_content),
        ]

        try:
            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="curriculum", function_name="task_decomposer._decompose_with_llm", task=task)
            response = _llm_resp.content
            return self._parse_decomposition_response(response)
        except Exception as e:
            print(f"\033[31m[TaskDecomposer] LLM call failed: {e}\033[0m")
            return []

    def _parse_decomposition_response(self, response: str) -> List[SubTask]:
        """Parse the LLM's decomposition response"""
        try:
            # Extract the JSON portion
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                print(f"\033[31m[TaskDecomposer] No JSON found in response\033[0m")
                return []

            data = json.loads(json_match.group())
            subtasks_data = data.get("subtasks", [])

            subtasks = []
            placeholder_targets = {
                "unknown", "item", "items", "thing", "things",
                "something", "object", "objects", "resource", "resources",
                "stuff", "", None,
            }
            for st_data in subtasks_data:
                # Phase 10: skip malformed subtasks with missing or placeholder
                # target. Previously defaulted to "unknown", which propagated
                # into tasks like "Ensure you have 1 unknown" that always fail.
                raw_target = st_data.get("target")
                target = (raw_target or "").strip().lower() if isinstance(raw_target, str) else raw_target
                if target in placeholder_targets:
                    print(
                        f"\033[33m[TaskDecomposer] Skipping malformed subtask "
                        f"(target={raw_target!r}): {st_data.get('task', '')[:60]}\033[0m"
                    )
                    continue
                subtask = SubTask(
                    task_description=st_data.get("task", ""),
                    task_type=st_data.get("type", "ensure"),
                    target_item=raw_target,
                    target_count=st_data.get("count", 1),
                    prerequisites=st_data.get("prerequisites", []),
                )
                subtasks.append(subtask)

            return subtasks

        except json.JSONDecodeError as e:
            print(f"\033[31m[TaskDecomposer] JSON parse error: {e}\033[0m")
            return []
        except Exception as e:
            print(f"\033[31m[TaskDecomposer] Parse error: {e}\033[0m")
            return []

    def _validate_decomposition(self, subtasks: List[SubTask], original_task: str) -> bool:
        """
        Validate the reasonableness of the decomposition.

        Checks:
        1. Subtask count is within the reasonable range (1-max_subtasks)
        2. No circular dependencies
        3. Subtasks are related to the original task
        """
        if not subtasks:
            return False

        if len(subtasks) > self.max_subtasks:
            print(f"\033[33m[TaskDecomposer] Too many subtasks: {len(subtasks)} > {self.max_subtasks}\033[0m")
            return False

        # Check for circular dependencies
        if self._has_circular_dependency(subtasks):
            print(f"\033[33m[TaskDecomposer] Circular dependency detected\033[0m")
            return False

        # Relevance check: at least one subtask should involve the target item of the original task
        original_item = self._extract_target(original_task)
        if original_item:
            has_related = any(
                original_item in st.target_item or st.target_item in original_item
                for st in subtasks
            )
            if not has_related:
                print(f"\033[33m[TaskDecomposer] No related subtask found for {original_item}\033[0m")
                return False

        return True

    def _sort_by_dependencies(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        Sort subtasks by dependency.

        Ensures prerequisite tasks come before dependent tasks (topological sort).
        """
        if not subtasks:
            return subtasks

        # Build the dependency graph
        task_map = {st.target_item: st for st in subtasks}
        sorted_tasks: List[SubTask] = []
        visited: set = set()

        def visit(subtask: SubTask):
            if subtask.target_item in visited:
                return
            visited.add(subtask.target_item)

            # Visit prerequisites first
            for prereq in subtask.prerequisites:
                if prereq in task_map:
                    visit(task_map[prereq])

            sorted_tasks.append(subtask)

        for subtask in subtasks:
            visit(subtask)

        return sorted_tasks

    def _has_circular_dependency(self, subtasks: List[SubTask]) -> bool:
        """Check whether there is a circular dependency"""
        targets = {st.target_item for st in subtasks}

        for subtask in subtasks:
            for prereq in subtask.prerequisites:
                if prereq in targets:
                    # Check the reverse dependency
                    prereq_task = next(
                        (st for st in subtasks if st.target_item == prereq),
                        None
                    )
                    if prereq_task and subtask.target_item in prereq_task.prerequisites:
                        return True
        return False

    def _extract_target(self, task: str) -> Optional[str]:
        """Extract the target item from the task description"""
        task_lower = task.lower()

        # Match "Ensure you have X <item>"
        match = re.search(r"ensure you have\s+\d+\s+(.+?)(?:\s*$|\s*\()", task_lower)
        if match:
            return match.group(1).strip().replace(" ", "_")

        # Match "Craft X <item>"
        match = re.search(r"craft\s+\d+\s+(.+?)(?:\s*$|\s*\()", task_lower)
        if match:
            return match.group(1).strip().replace(" ", "_")

        # Match "Mine X <item>"
        match = re.search(r"mine\s+\d+\s+(.+?)(?:\s*$|\s*\()", task_lower)
        if match:
            return match.group(1).strip().replace(" ", "_")

        return None

    def _get_cache_key(self, task: str) -> str:
        """Generate a cache key"""
        # Use the normalized task description as the cache key
        return task.lower().strip()

    def clear_cache(self) -> None:
        """Clear the decomposition cache"""
        self._decomposition_cache.clear()
