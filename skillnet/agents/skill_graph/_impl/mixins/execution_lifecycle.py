"""
Execution Lifecycle Mixin for SkillGraphManager

Refactor lifecycle management: delayed refactor triggering, refactored skill
failure handling, rollback, and experimental skill cleanup on task completion.

Extracted from skill_execution.py for modularity.
Contains 4 methods:
  - _trigger_delayed_refactor: Trigger refactor after experimental skill verification
  - _handle_refactored_skill_failure: Handle refactored skill runtime failure
  - _rollback_skill_refactor: Rollback single skill's refactor
  - on_task_completed: Callback when task completes, manage experimental skills
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class ExecutionLifecycleMixin:
    """Execution Lifecycle Mixin - Refactor lifecycle and experimental skill management.

    Methods:
        _trigger_delayed_refactor: Trigger delayed refactor after verification
        _handle_refactored_skill_failure: Handle refactored skill runtime failure
        _rollback_skill_refactor: Rollback single skill's refactor
        on_task_completed: Callback when task completes

    Attributes (from SkillGraphManager):
        graph: SkillGraph instance
        vectordb: ChromaDB instance
        logger: Logger instance
        enable_refactor: Whether refactor is enabled
    """

    def _trigger_delayed_refactor(self: "SkillGraphManager", skill_name: str) -> bool:
        """
        [Fix 7 supplement] Trigger the delayed refactor.

        After an experimental skill is validated, run the previously-skipped refactor detection.

        Args:
            skill_name: the skill that just succeeded validation

        Returns:
            bool: whether the refactor was successfully triggered
        """
        # Check if refactor is enabled
        if not self.enable_refactor:
            print(f"\033[36m[Delayed Refactor] Refactoring disabled, skipping for {skill_name}\033[0m")
            return False

        if skill_name not in self.graph.nodes:
            print(f"\033[33m[Delayed Refactor] Skill '{skill_name}' is not in graph; skipping\033[0m")
            return False

        node = self.graph.get_node(skill_name)
        if not node:
            return False

        # [Bug Fix] Check whether this is a task-specific skill; task-specific skills should not participate in refactor
        if getattr(node, 'is_task_specific', False) or self._is_task_specific_skill(skill_name, node.code or ""):
            print(f"\033[33m[Delayed Refactor] Skill '{skill_name}' is task-specific; skipping refactor\033[0m")
            return False

        # Get the skill description and code
        skill_description = node.description or ""
        skill_code = node.code or ""

        print(f"\033[36m[Delayed Refactor] Starting delayed refactor detection for '{skill_name}'...\033[0m")

        try:
            # Step a: prescreen candidates
            candidates = self._prescreen_refactor_candidates(
                skill_name,
                skill_description,
                top_k=5
            )

            if not candidates:
                print(f"\033[36m[Delayed Refactor] No candidate skills found; skipping refactor\033[0m")
                return False

            self.logger.info(f"\033[36m[Delayed Refactor] Prescreened candidate skills: {', '.join([name for name, _ in candidates])}\033[0m")

            # Step b: LLM detection and classification
            refactor_info = self._detect_refactor_relationships(
                new_skill_name=skill_name,
                new_skill_code=skill_code,
                new_skill_description=skill_description,
                new_skill_params=node.parameters,
                new_skill_preconditions=node.preconditions,
                new_skill_effects=node.expected_effects,
                candidate_skills=candidates
            )

            if refactor_info and refactor_info.get("refactor_type"):
                refactor_type = refactor_info.get("refactor_type")
                self.logger.info(f"\033[36m[Delayed Refactor] Detected refactor type: {refactor_type}\033[0m")

                # Step c: execute refactor
                refactor_success = self._execute_refactor(
                    skill_name,
                    skill_code,
                    refactor_info
                )

                if refactor_success:
                    print(f"\033[32m[Delayed Refactor] Successfully executed refactor: {refactor_type}\033[0m")
                    return True
                else:
                    print(f"\033[33m[Delayed Refactor] Refactor execution failed\033[0m")
                    return False
            else:
                print(f"\033[36m[Delayed Refactor] No refactor-eligible relationships detected\033[0m")
                return False

        except Exception as e:
            print(f"\033[31m[Delayed Refactor] Error during execution: {e}\033[0m")
            import traceback
            print(f"\033[31m{traceback.format_exc()}\033[0m")
            return False

    def _handle_refactored_skill_failure(
        self: "SkillGraphManager",
        skill_name: str,
        error_message: str = None,
        context: str = None,
        task: str = None
    ) -> None:
        """
        Part 18: handle runtime failure of a refactored skill.

        If a wrapper skill fails consecutively many times, it may be a refactor-induced issue;
        consider rolling back to the original code.
        """
        node = self.graph.get_node(skill_name)
        if not node:
            return

        # Initialize failure records
        if not hasattr(node, 'refactor_failures') or node.refactor_failures is None:
            node.refactor_failures = []

        # Record the failure
        failure_record = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "context": context,
            "task": task,
            "covered_by": node.covered_by,
            "coverage_type": node.coverage_type.value if node.coverage_type else None,
        }
        node.refactor_failures.append(failure_record)

        print(f"\033[33m[Refactor Runtime] Wrapper skill {skill_name} failed "
              f"(attempt {len(node.refactor_failures)}, covered_by={node.covered_by})\033[0m")

        # Check whether auto-rollback is needed
        ROLLBACK_THRESHOLD = 3  # roll back after 3 consecutive failures
        recent_failures = [f for f in node.refactor_failures
                          if f.get("covered_by") == node.covered_by]

        if len(recent_failures) >= ROLLBACK_THRESHOLD:
            print(f"\033[31m[Refactor Runtime] {skill_name} failed {len(recent_failures)} times in a row; "
                  f"triggering auto-rollback\033[0m")
            self._rollback_skill_refactor(skill_name)

    def _rollback_skill_refactor(self: "SkillGraphManager", skill_name: str) -> bool:
        """
        Part 18: roll back a single skill's refactor.

        Restore the wrapper skill to its original pre-refactor code.
        """
        node = self.graph.get_node(skill_name)

        if not node:
            print(f"\033[33m[Rollback] {skill_name} does not exist\033[0m")
            return False

        if not node.is_covered:
            print(f"\033[33m[Rollback] {skill_name} is not a covered skill; nothing to roll back\033[0m")
            return False

        # Check whether the original code is available
        if getattr(node, 'original_code_before_refactor', None):
            print(f"\033[36m[Rollback] Restoring original code for {skill_name}\033[0m")

            old_covered_by = node.covered_by

            # Restore original code
            node.code = node.original_code_before_refactor
            node.original_code_before_refactor = None

            # Clear refactor markers
            node.is_covered = False
            node.covered_by = None
            node.coverage_type = None
            node.refactor_failures = []

            # Remove graph edges
            if old_covered_by and self.graph.has_edge(skill_name, old_covered_by):
                self.graph.remove_edge(skill_name, old_covered_by)

            # Update description and vector database (idempotent deletion)
            original_description = self.generate_skill_description(skill_name, node.code)
            node.description = original_description

            try:
                existing = self.vectordb._collection.get(ids=[skill_name])
                if existing and existing.get("ids"):
                    self.vectordb._collection.delete(ids=[skill_name])
            except:
                pass
            self.vectordb.add_texts(
                texts=[original_description],
                ids=[skill_name],
                metadatas=[{"name": skill_name}],
            )

            print(f"\033[32m[Rollback] Successfully rolled back {skill_name}\033[0m")
            return True
        else:
            print(f"\033[33m[Rollback] Warning: no saved original code for {skill_name}; cannot roll back\033[0m")
            return False

    def on_task_completed(
        self: "SkillGraphManager",
        task: str,
        success: bool,
        error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Callback method invoked on task completion.

        Handle the task's experimental skills:
        - If the task succeeded: keep experimental-skill state (they may succeed via later optimization)
        - If the task failed: mark the task's experimental skills as deprecated

        Args:
            task: completed task name
            success: whether the task succeeded
            error_type: error type (e.g. "preflight")

        Returns:
            Dict containing statistics about the processing result
        """
        result = {
            "task": task,
            "success": success,
            "error_type": error_type,
            "experimental_skills_found": [],
            "deprecated_skills": [],
            "verified_skills": [],
        }

        # Preflight (contract-gate) rejects are code generation/interface issues, not task difficulty.
        # They should NOT trigger experimental-skill deprecation for this task.
        if (not success) and error_type == "preflight":
            print(f"\033[33m[Task Completed] Task '{task}' failure reason=preflight; skipping experimental-skill deprecation handling\033[0m")
            return result

        # Phase 9: collect skills to handle first, to avoid modifying the dict during iteration
        skills_to_delete = []

        # Iterate over all nodes to find experimental skills related to the task
        for skill_name, node in list(self.graph.nodes.items()):
            # Check whether this is an experimental skill belonging to the task
            if not getattr(node, 'is_experimental', False):
                continue

            # Issue 10.3 fix: validated skills should not be deleted
            if getattr(node, 'is_verified', False):
                print(f"\033[32m[Task Completed] Skill '{skill_name}' has been validated; skipping deletion\033[0m")
                continue

            experimental_task = getattr(node, 'experimental_task', None)
            if not experimental_task:
                continue

            # Check whether it is the same task
            # Issue 10.3 fix: remove dangerous partial matching; use exact match only
            # Partial matching can wrongly delete skills reused across tasks
            if experimental_task != task:
                continue

            result["experimental_skills_found"].append(skill_name)

            if success:
                # Task succeeded; check whether the skill is still experimental
                # If the skill helped complete the task, mark_execution_success should already have handled it
                # If it is still experimental, it likely was not used directly; keep state unchanged
                if not node.is_experimental:
                    result["verified_skills"].append(skill_name)
                    print(f"\033[32m[Task Completed] Skill '{skill_name}' validated successfully; experimental flag removed\033[0m")
            else:
                # Issue 10.3 fix: check whether it succeeded in other tasks
                # Note: execution_traces is a list of SkillExecutionTrace objects, not a dict
                # exclude not_executed traces; they should not affect success determination
                successful_in_other_tasks = any(
                    getattr(trace, 'success', False)
                    and getattr(trace, 'task', '') != task
                    and getattr(trace, 'execution_status', 'success') != 'not_executed'
                    for trace in getattr(node.statistics, 'execution_traces', [])
                )
                if successful_in_other_tasks:
                    print(f"\033[33m[Task Completed] Skill '{skill_name}' succeeded in other tasks; skipping deletion\033[0m")
                    continue

                # Phase 9: collect skills to delete
                skills_to_delete.append(skill_name)

        # Phase 9: perform deletion after the loop
        for skill_name in skills_to_delete:
            delete_success = self.delete_skill(
                skill_name,
                reason=f"Task '{task}' ultimately failed; experimental skill failed to be validated"
            )
            if delete_success:
                result["deprecated_skills"].append(skill_name)
                print(f"\033[33m[Task Completed] Task '{task}' failed; deleted experimental skill '{skill_name}'\033[0m")

        # Save changes - delete_skill() already saves; we only need to handle the verified_skills case
        if result["verified_skills"]:
            self._save_to_checkpoint()

        if result["experimental_skills_found"]:
            print(f"\033[36m[Task Completed] Processed {len(result['experimental_skills_found'])} experimental skills:\033[0m")
            if result["deprecated_skills"]:
                print(f"\033[36m  - Deprecated: {', '.join(result['deprecated_skills'])}\033[0m")
            if result["verified_skills"]:
                print(f"\033[36m  - Verified: {', '.join(result['verified_skills'])}\033[0m")

        return result
