"""
Refactor Applier — SkillRefactor Base Class

Base class for skill refactors; provides a common interface and rollback support.

Subclasses (parametric, behavioral, duplication, sibling, subskill_extraction)
extend SkillRefactor to implement the concrete refactor logic.

Extracted from base.py for modularity.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from ..utils import validate_code_syntax
from .models import RefactorOpportunity, RefactorResult


class SkillRefactor:
    """
    Base class for skill refactors.

    Provides a common interface for refactor operations and rollback support.
    """

    def __init__(
        self,
        skill_graph_manager=None,
        llm=None,
        logger=None,
        domain_knowledge=None,
    ):
        """
        Initialize the SkillRefactor

        Args:
            skill_graph_manager: skill graph manager
            llm: LLM instance
            logger: logger
            domain_knowledge: DomainKnowledge object, used for function-reference validation
        """
        self.skill_graph_manager = skill_graph_manager
        self.llm = llm
        self.logger = logger
        self.domain_knowledge = domain_knowledge

        # Rollback-data storage
        self._rollback_stack: List[RefactorResult] = []

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)

    def _to_snake_case(self, name: str) -> str:
        """
        Convert CamelCase to snake_case

        For example: mineOakLogs -> mine_oak_logs
                      WoodenPickaxe -> wooden_pickaxe
        """
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def apply(self, opportunity: RefactorOpportunity) -> RefactorResult:
        """
        Apply the refactor

        Args:
            opportunity: refactor opportunity

        Returns:
            RefactorResult: refactor result
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def rollback(self, result: RefactorResult) -> bool:
        """
        Roll back a refactor

        Args:
            result: the refactor result to roll back

        Returns:
            bool: whether rollback was successful
        """
        if not result.rollback_available or not result.rollback_data:
            self._log(f"[Refactor] Cannot roll back: no rollback data", "warning")
            return False

        try:
            # Restore the original code
            skill_name = result.source_skill
            old_code = result.rollback_data.get("original_code")

            if old_code and self.skill_graph_manager:
                node = self.skill_graph_manager.get_node(skill_name)
                if node:
                    node.code = old_code
                    self._log(f"[Refactor] Rolled back code of '{skill_name}'", "info")
                    return True

            return False

        except Exception as e:
            self._log(f"[Refactor] Rollback failed: {e}", "error")
            return False

    def _save_rollback_data(
        self,
        skill_name: str,
        old_code: str,
        old_effects: List = None
    ) -> Dict[str, Any]:
        """
        Save basic rollback data

        Subclasses should call this to obtain the base data, then use update() to add extra fields.
        For example, parametric.py adds coverage state, and behavioral.py adds target_skill.

        Args:
            skill_name: skill name
            old_code: original code
            old_effects: original effects list

        Returns:
            Dict[str, Any]: base rollback data; subclasses may extend it
        """
        return {
            "skill_name": skill_name,
            "original_code": old_code,
            "original_effects": old_effects,
            "saved_at": datetime.now().isoformat(),
        }

    def _save_multi_skill_rollback(
        self,
        skill_nodes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save rollback data for multiple skills"""
        rollback = {
            "skills": {},
            "saved_at": datetime.now().isoformat(),
        }

        for name, node in skill_nodes.items():
            rollback["skills"][name] = {
                "code": node.code,
                "effects": list(node.expected_effects) if node.expected_effects else [],
                "is_covered": getattr(node, 'is_covered', False),
                "covered_by": getattr(node, 'covered_by', None),
                "coverage_type": getattr(node, 'coverage_type', None),
            }

        return rollback

    def _create_failed_result(
        self,
        opportunity: RefactorOpportunity,
        error_message: str
    ) -> RefactorResult:
        """Create a failed result"""
        return RefactorResult(
            success=False,
            refactor_type=opportunity.refactor_type,
            source_skill=opportunity.source_skill,
            target_skill=opportunity.target_skill,
            error_message=error_message,
            rollback_available=False,
        )

    # ========== Caller-update methods ==========

    def _get_callers(self, skill_name: str) -> List[str]:
        """
        Get every skill that calls the given skill (i.e., its callers)

        Args:
            skill_name: name of the called skill

        Returns:
            List[str]: list of caller skill names
        """
        if not self.skill_graph_manager:
            return []

        # Use graph.get_parents() to get the callers
        return self.skill_graph_manager.get_parents(skill_name)

    def _should_update_caller(
        self,
        caller_name: str,
        refactored_skill: str,
        changes: Dict[str, Any],
    ) -> bool:
        """
        Decide whether a caller needs to be updated

        Args:
            caller_name: caller skill name
            refactored_skill: name of the refactored skill
            changes: change information
                - change_type: change type (wrapper, renamed, signature_changed, deprecated)
                - new_skill_name: new skill name (if renamed or a general version was created)
                - new_signature: new function signature (if the signature changed)

        Returns:
            bool: whether an update is needed
        """
        if not self.skill_graph_manager:
            return False

        caller_node = self.skill_graph_manager.get_node(caller_name)
        if not caller_node or not caller_node.code:
            return False

        change_type = changes.get("change_type", "")

        # Case 1: the refactored skill became a wrapper
        # → the caller does not need to be updated (the interface is unchanged)
        if change_type == "wrapper":
            return False

        # Helper: precisely check whether a skill is being called
        # Use a function-call pattern to avoid false positives like "craft" matching "craftTable"
        def _is_skill_called(skill_name: str, code: str) -> bool:
            import re
            # Match function calls: skillName( or await skillName(
            # Use \b to ensure whole-word matching
            pattern = rf'\b{re.escape(skill_name)}\s*\('
            return bool(re.search(pattern, code))

        # Case 2: a new general version was created
        # → the caller may want to switch to the general version (optional optimization)
        if change_type == "general_created":
            # Check whether the caller directly calls the refactored skill
            if _is_skill_called(refactored_skill, caller_node.code):
                return True

        # Case 3: the skill was renamed
        # → the caller must be updated
        if change_type == "renamed":
            if _is_skill_called(refactored_skill, caller_node.code):
                return True

        # Case 4: the function signature changed
        # → the caller must update its arguments
        if change_type == "signature_changed":
            if _is_skill_called(refactored_skill, caller_node.code):
                return True

        # Case 5: the skill was marked deprecated
        # → the caller should migrate to the replacement
        if change_type == "deprecated":
            if _is_skill_called(refactored_skill, caller_node.code):
                return True

        return False

    def _update_caller_code(
        self,
        caller_name: str,
        refactored_skill: str,
        changes: Dict[str, Any],
    ) -> Optional[str]:
        """
        Update the caller's code

        Args:
            caller_name: caller skill name
            refactored_skill: name of the refactored skill
            changes: change information

        Returns:
            Optional[str]: the updated code, or None if it cannot be updated
        """
        import re

        if not self.skill_graph_manager:
            return None

        caller_node = self.skill_graph_manager.get_node(caller_name)
        if not caller_node or not caller_node.code:
            return None

        code = caller_node.code
        change_type = changes.get("change_type", "")
        new_skill_name = changes.get("new_skill_name", "")

        # Case 1: a general version was created; suggest the caller switch to it
        if change_type == "general_created" and new_skill_name:
            # Look for call patterns: await refactored_skill(bot) or refactored_skill(bot, ...)
            pattern = rf'(await\s+)?{re.escape(refactored_skill)}\s*\(([^)]*)\)'

            def replace_with_general(match):
                await_prefix = match.group(1) or ""
                args = match.group(2)

                # Get the new parameter (if needed)
                new_param = changes.get("param_value", "")
                if new_param:
                    # Append the new parameter after the existing arguments
                    if args.strip():
                        new_args = f"{args}, \"{new_param}\""
                    else:
                        new_args = f"bot, \"{new_param}\""
                    return f"{await_prefix}{new_skill_name}({new_args})"
                else:
                    return f"{await_prefix}{new_skill_name}({args})"

            new_code = re.sub(pattern, replace_with_general, code)

            # Add an explanatory comment
            if new_code != code:
                comment = f"    // Updated: now calls {new_skill_name} instead of {refactored_skill}\n"
                # Insert the comment at the beginning of the function body
                func_match = re.search(r'(async\s+function\s+\w+\s*\([^)]*\)\s*\{)', new_code)
                if func_match:
                    new_code = new_code.replace(
                        func_match.group(1),
                        f"{func_match.group(1)}\n{comment}"
                    )
                return new_code

        # Case 2: the skill was renamed
        if change_type == "renamed" and new_skill_name:
            # Simple name substitution
            new_code = re.sub(
                rf'\b{re.escape(refactored_skill)}\b',
                new_skill_name,
                code
            )
            if new_code != code:
                return new_code

        # Case 3: signature changed
        if change_type == "signature_changed":
            new_signature = changes.get("new_signature", {})
            added_params = new_signature.get("added_params", [])

            if added_params:
                # Update the call to include the new parameters
                pattern = rf'(await\s+)?{re.escape(refactored_skill)}\s*\(([^)]*)\)'

                def add_new_params(match):
                    await_prefix = match.group(1) or ""
                    existing_args = match.group(2)
                    # Add default values for the new parameters
                    new_param_values = ", ".join(
                        f"undefined /* {p} */" for p in added_params
                    )
                    if existing_args.strip():
                        new_args = f"{existing_args}, {new_param_values}"
                    else:
                        new_args = new_param_values
                    return f"{await_prefix}{refactored_skill}({new_args})"

                new_code = re.sub(pattern, add_new_params, code)
                if new_code != code:
                    return new_code

        # Case 4: the skill was deprecated
        if change_type == "deprecated":
            replacement = changes.get("replacement", "")
            if replacement:
                # Substitute the recommended replacement
                new_code = re.sub(
                    rf'\b{re.escape(refactored_skill)}\b',
                    replacement,
                    code
                )
                if new_code != code:
                    # Add a migration comment
                    comment = f"    // MIGRATED: {refactored_skill} -> {replacement}\n"
                    func_match = re.search(r'(async\s+function\s+\w+\s*\([^)]*\)\s*\{)', new_code)
                    if func_match:
                        new_code = new_code.replace(
                            func_match.group(1),
                            f"{func_match.group(1)}\n{comment}"
                        )
                    return new_code

        return None

    def propagate_to_callers(
        self,
        refactored_skill: str,
        changes: Dict[str, Any],
        auto_update: bool = False,
    ) -> Dict[str, Any]:
        """
        Propagate changes to callers

        Args:
            refactored_skill: name of the refactored skill
            changes: change information
                - change_type: change type
                - new_skill_name: new skill name
                - param_value: parameter value (used for the general-version call)
                - replacement: replacement skill (used for deprecated)
            auto_update: whether to auto-update (False = detect only, do not modify)

        Returns:
            Dict[str, Any]: propagation result
                - callers: list of callers
                - needs_update: callers that need updating
                - updated: callers that have been updated (if auto_update=True)
                - update_details: update details
        """
        result = {
            "callers": [],
            "needs_update": [],
            "updated": [],
            "update_details": {},
            "rollback_data": {},
            "failed": [],
        }

        if not self.skill_graph_manager:
            return result

        # Get all callers
        callers = self._get_callers(refactored_skill)
        result["callers"] = callers

        if not callers:
            self._log(f"[CallerUpdate] {refactored_skill} has no callers", "info")
            return result

        self._log(
            f"[CallerUpdate] {refactored_skill} has {len(callers)} callers: {callers}",
            "info"
        )

        # Inspect each caller
        for caller in callers:
            if self._should_update_caller(caller, refactored_skill, changes):
                result["needs_update"].append(caller)

                if auto_update:
                    # Save the original code for rollback
                    caller_node = self.skill_graph_manager.get_node(caller)
                    if caller_node:
                        result["rollback_data"][caller] = caller_node.code

                        # Update the code
                        new_code = self._update_caller_code(
                            caller, refactored_skill, changes
                        )

                        if new_code:
                            # Validate the updated code syntax
                            is_valid, syntax_error = validate_code_syntax(new_code)
                            if not is_valid:
                                self._log(
                                    f"[CallerUpdate] ✗ JS syntax validation failed (caller: {caller}): {syntax_error}",
                                    "error"
                                )
                                result["failed"].append(caller)
                                continue

                            # Use the unified interface to update code (syntax already validated, skip duplicate validation)
                            update_success = self.skill_graph_manager.update_skill_code(
                                skill_name=caller,
                                new_code=new_code,
                                change_log=f"Updated to use {changes.get('new_skill_name', refactored_skill)}",
                                source=f"refactor:{self.__class__.__name__}:caller_update",
                                skip_validation=True,  # syntax already validated
                                skip_metadata=True,    # caller updates do not need re-extracting metadata
                                skip_interface_check=True,  # caller updates do not trigger cascade
                                create_version=True,
                            )
                            if not update_success:
                                self._log(f"[CallerUpdate] ✗ Update failed: {caller}", "error")
                                result["failed"].append(caller)
                                continue

                            result["updated"].append(caller)
                            result["update_details"][caller] = (
                                f"Updated to use {changes.get('new_skill_name', refactored_skill)}"
                            )
                            self._log(
                                f"[CallerUpdate] ✓ Updated {caller}",
                                "info"
                            )
                        else:
                            self._log(
                                f"[CallerUpdate] ⚠ Cannot auto-update {caller}; manual handling required",
                                "warning"
                            )

        if result["needs_update"] and not auto_update:
            self._log(
                f"[CallerUpdate] {len(result['needs_update'])} callers need updating: "
                f"{result['needs_update']}",
                "info"
            )

        return result

    def rollback_caller_updates(
        self,
        rollback_data: Dict[str, str],
    ) -> bool:
        """
        Roll back caller updates

        Args:
            rollback_data: {caller_name: original_code}

        Returns:
            bool: whether rollback succeeded
        """
        if not self.skill_graph_manager or not rollback_data:
            return False

        success = True
        for caller, original_code in rollback_data.items():
            node = self.skill_graph_manager.get_node(caller)
            if node:
                node.code = original_code
                self._log(f"[CallerUpdate] Rolled back {caller}", "info")
            else:
                success = False
                self._log(f"[CallerUpdate] Cannot roll back {caller}: node does not exist", "warning")

        return success
