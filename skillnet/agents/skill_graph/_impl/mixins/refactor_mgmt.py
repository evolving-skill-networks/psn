"""
Refactor Management Mixin for SkillGraphManager

This module contains methods for refactor detection, execution, and relationship analysis.
These methods are mostly thin wrappers that delegate to the skillnet.agents.refactor module.

Extracted from graph_manager_impl.py for better modularity.
"""

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple

from skillnet.agents.skill_graph.models import (
    SkillPrecondition,
    SkillEffect,
    SkillVersion,
    CodeSource,
)
from skillnet.agents.skill_graph.utils import validate_code_syntax

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class RefactorManagementMixin:
    """Refactor Management Mixin - Refactor detection, execution, and relationship analysis.

    Methods:
        _prescreen_refactor_candidates: Prescreen refactor candidates
        _prescreen_refactor_candidates_detailed: Detailed prescreening
        _detect_refactor_relationships: Detect refactor relationships
        _validate_refactor_direction: Validate refactor direction
        _execute_refactor: Execute refactor
        _handle_covered_skills: Handle covered skills

    Note:
        Refactor logic has been partially extracted to skillnet.agents.refactor module.
        Methods here are mainly coordination and adaptation.

    Attributes (from SkillGraphManager):
        _refactor_prescreener: RefactorPrescreener instance
        _refactor_analyzer: RefactorRelationshipAnalyzer instance
        _refactor_executor: RefactorExecutor instance
        graph: SkillGraph instance
        logger: Logger instance
    """

    def _prescreen_refactor_candidates(
        self: "SkillGraphManager",
        new_skill_name: str,
        new_skill_description: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Prescreen candidates using embedding similarity + function name similarity.

        Delegates to RefactorPrescreener.

        Args:
            new_skill_name: Name of the new skill
            new_skill_description: Description of the new skill
            top_k: Number of candidates to return (default 5)

        Returns:
            List[Tuple[str, float]]: [(skill_name, similarity_score), ...] candidate list
        """
        return self._refactor_prescreener.prescreen_candidates(
            new_skill_name=new_skill_name,
            new_skill_description=new_skill_description,
            top_k=top_k
        )

    def _detect_refactor_relationships(
        self: "SkillGraphManager",
        new_skill_name: str,
        new_skill_code: str,
        new_skill_description: str,
        new_skill_params: Dict[str, Dict[str, Any]],
        new_skill_preconditions: List[SkillPrecondition],
        new_skill_effects: List[SkillEffect],
        candidate_skills: List[Tuple[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect and classify refactor relationships using LLM.

        Delegates to RefactorRelationshipAnalyzer.

        Args:
            new_skill_name: Name of the new skill
            new_skill_code: Code of the new skill
            new_skill_description: Description of the new skill
            new_skill_params: Parameter metadata of the new skill
            new_skill_preconditions: Preconditions of the new skill
            new_skill_effects: Effects of the new skill
            candidate_skills: Candidate skills list [(name, similarity), ...]

        Returns:
            Optional[Dict]: Refactor information
        """
        return self._refactor_analyzer.detect_relationships(
            new_skill_name=new_skill_name,
            new_skill_code=new_skill_code,
            new_skill_description=new_skill_description,
            new_skill_params=new_skill_params,
            new_skill_preconditions=new_skill_preconditions,
            new_skill_effects=new_skill_effects,
            candidate_skills=candidate_skills
        )

    def _execute_refactor(
        self: "SkillGraphManager",
        new_skill_name: str,
        new_skill_code: str,
        refactor_info: Dict[str, Any],
        use_new_executor: bool = True,
    ) -> bool:
        """
        Execute refactor action: set node G and local subgraph rewriting.

        Args:
            new_skill_name: Name of the new skill
            new_skill_code: Code of the new skill
            refactor_info: Refactor information (from _detect_refactor_relationships)
            use_new_executor: Whether to use new RefactorExecutor (default True)

        Returns:
            bool: Whether refactor was executed successfully
        """
        if use_new_executor:
            try:
                self.logger.info(
                    f"\033[36m[Refactor] Using RefactorExecutor for "
                    f"{refactor_info.get('refactor_type')} refactor\033[0m"
                )
                result = self._refactor_executor.execute(
                    refactor_info=refactor_info,
                    new_skill_name=new_skill_name,
                    new_skill_code=new_skill_code,
                    auto_update_callers=False,
                )

                # Record to RefactorHistoryTracker (read by LearningDynamicsRecorder)
                # For BEHAVIORAL Case B batch refactors, sub_results contains
                # an independent result for each covered_skill. We record one history entry per sub_result
                # individually instead of only recording the merged result (avoids the r34 batch
                # refactor info-loss bug — see the same fix in transaction_adapter.py).
                try:
                    if result.sub_results:
                        for sub_result in result.sub_results:
                            self._refactor_tracker.record(
                                result=sub_result,
                                trigger="delayed_refactor",
                            )
                    else:
                        self._refactor_tracker.record(
                            result=result,
                            trigger="delayed_refactor",
                        )
                except Exception as e:
                    self.logger.warning(f"[Refactor] Failed to record refactor history: {e}")

                if result.success:
                    self.logger.info(
                        f"\033[32m[Refactor] RefactorExecutor succeeded: "
                        f"{result.refactor_type.value}\033[0m"
                    )
                    if result.changes_made:
                        for change in result.changes_made:
                            self.logger.info(f"\033[36m  - {change}\033[0m")
                    self._save_to_checkpoint()
                    return True
                else:
                    self.logger.warning(
                        f"\033[33m[Refactor] RefactorExecutor failed: "
                        f"{result.error_message}\033[0m"
                    )
                    return False

            except Exception as e:
                self.logger.error(f"\033[31m[Refactor] RefactorExecutor exception: {e}\033[0m")
                import traceback
                traceback.print_exc()
                return False

        self.logger.warning(f"\033[33m[Refactor] No available RefactorExecutor\033[0m")
        return False

    def _handle_covered_skills(
        self: "SkillGraphManager",
        program_name: str,
        program_code: str,
        covered_skills: List[str]
    ) -> None:
        """
        Handle skills covered by the new skill.

        Converts covered skills to wrappers that call the new skill,
        maintaining backward compatibility.

        Args:
            program_name: Name of the new skill
            program_code: Code of the new skill
            covered_skills: List of covered skill names
        """
        if not covered_skills:
            return

        # Converting covered skills to wrappers is a refactor operation; honor
        # the global toggle (--no-refactor) so it stays off when disabled.
        if not getattr(self, "enable_refactor", True):
            return

        print(
            f"\033[36mDetected new skill {program_name} covers existing skills: "
            f"{', '.join(covered_skills)}\033[0m"
        )

        for covered_skill_name in covered_skills:
            if not self.graph.has_node(covered_skill_name):
                continue

            covered_node = self.graph.get_node(covered_skill_name)
            print(
                f"\033[33m  - Skill '{covered_skill_name}' covered by new skill "
                f"'{program_name}', marking as redundant\033[0m"
            )

            try:
                # Extract new skill's parameter signature
                new_func_match = re.search(
                    r'async\s+function\s+(\w+)\s*\(([^)]*)\)', program_code
                )
                if not new_func_match:
                    continue

                new_params = new_func_match.group(2).split(',')
                param_names = [
                    p.strip().split('=')[0].strip()
                    for p in new_params if p.strip() and p.strip() != 'bot'
                ]

                # Extract old skill's parameter signature
                old_func_match = re.search(
                    r'async\s+function\s+(\w+)\s*\(([^)]*)\)', covered_node.code
                )
                if not old_func_match:
                    continue

                old_params = old_func_match.group(2).split(',')
                old_param_names = [
                    p.strip().split('=')[0].strip()
                    for p in old_params if p.strip() and p.strip() != 'bot'
                ]

                # Build call code
                if len(old_param_names) == 0:
                    call_code = f"await {program_name}(bot);"
                else:
                    if len(param_names) >= len(old_param_names):
                        param_calls = ', '.join([
                            f"{old_param_names[i] if i < len(old_param_names) else param_names[i]}"
                            for i in range(len(param_names))
                        ])
                        call_code = f"await {program_name}(bot, {param_calls});"
                    else:
                        call_code = f"await {program_name}(bot);"

                # Generate wrapper code
                params_str = ', '.join(old_param_names) if old_param_names else ''
                wrapper_code = f"""// This skill has been generalized by {program_name}
// Keeping this wrapper for backward compatibility
async function {covered_skill_name}(bot{', ' + params_str if params_str else ''}) {{
  // Call the generalized skill
  {call_code}
}}"""

                # Validate wrapper code syntax
                is_valid, error_msg = validate_code_syntax(wrapper_code)
                if not is_valid:
                    print(f"\033[31m  Error: Generated wrapper code has invalid syntax - {error_msg}\033[0m")
                    print(f"\033[33m  Skipping '{covered_skill_name}', keeping original code\033[0m")
                    continue

                # Preserve original metadata
                original_description = covered_node.description
                original_preconditions = covered_node.preconditions if covered_node.preconditions else []
                original_effects = covered_node.expected_effects if covered_node.expected_effects else []
                original_params = covered_node.parameters if covered_node.parameters else {}

                # Create new version
                latest_version = covered_node.get_latest_version()
                if latest_version:
                    version_parts = latest_version.version.split(".")
                    version_parts[-1] = str(int(version_parts[-1]) + 1)
                    new_version = ".".join(version_parts)
                else:
                    new_version = "1.0.1"

                # Use unified code update interface
                result = covered_node.update_code(wrapper_code, CodeSource.REFACTOR)

                if not result.success:
                    print(f"\033[33m  Warning: Failed to update wrapper code: {result.reason}\033[0m")
                    continue

                # Set wrapper metadata
                covered_node.is_covered = True
                covered_node.covered_by = program_name

                # Generate wrapper description
                wrapper_description = self._generate_wrapper_description(
                    covered_skill_name,
                    program_name,
                    wrapper_code,
                    original_description
                )
                covered_node.description = wrapper_description

                # Create new version record
                wrapper_version = SkillVersion(
                    version=new_version,
                    code=wrapper_code,
                    description=wrapper_description,
                    change_log=f"Refactored to wrapper for {program_name}",
                    preconditions=original_preconditions,
                    effects=original_effects,
                    parameters=original_params,
                    update_source="refactor",
                    update_reason=f"Converted to wrapper for {program_name}"
                )
                covered_node.add_version(wrapper_version)

                # Update vectordb with new description
                if hasattr(self, 'vectordb') and self.vectordb:
                    try:
                        self.vectordb.update_skill(covered_skill_name, wrapper_description)
                        print(f"\033[32m  Updated vectordb description for '{covered_skill_name}'\033[0m")
                    except Exception as vdb_err:
                        print(f"\033[33m  Warning: Failed to update vectordb: {vdb_err}\033[0m")

                # Add graph edge
                if not self.graph.has_edge(covered_skill_name, program_name):
                    self.graph.add_edge(covered_skill_name, program_name)
                    print(f"\033[32m  Added graph edge: {covered_skill_name} -> {program_name}\033[0m")

                print(f"\033[32m  Updated '{covered_skill_name}' to wrapper for '{program_name}' (is_covered=True)\033[0m")

            except Exception as e:
                print(f"\033[33m  Warning: Error updating covered skill '{covered_skill_name}': {e}\033[0m")

    def _generate_wrapper_description(
        self: "SkillGraphManager",
        covered_skill_name: str,
        target_skill_name: str,
        wrapper_code: str,
        original_description: str,
    ) -> str:
        """
        Generate description for a wrapper skill.

        Creates a clear description that indicates this skill delegates to another,
        while preserving the core functionality description.

        Args:
            covered_skill_name: Name of the covered skill (now wrapper)
            target_skill_name: Name of the general skill it delegates to
            wrapper_code: The wrapper code
            original_description: Original description of the covered skill

        Returns:
            str: New description for the wrapper skill
        """
        # Extract core description (first line or first sentence)
        core_description = ""
        if original_description:
            # Try to get first meaningful line
            lines = original_description.strip().split('\n')
            for line in lines:
                stripped = line.strip()
                # Skip comment markers and empty lines
                if stripped and not stripped.startswith('//') and not stripped.startswith('*'):
                    core_description = stripped
                    break

            # If still empty, use first 100 chars
            if not core_description:
                core_description = original_description[:100].strip()
                if len(original_description) > 100:
                    core_description += "..."

        # Build wrapper description
        if core_description:
            wrapper_desc = f"Wrapper that delegates to {target_skill_name}. {core_description}"
        else:
            wrapper_desc = f"Wrapper that delegates to {target_skill_name}."

        # Extract function signature from wrapper code
        sig_match = re.search(
            rf'async\s+function\s+{re.escape(covered_skill_name)}\s*\([^)]*\)',
            wrapper_code
        )
        signature = sig_match.group(0) if sig_match else f"async function {covered_skill_name}(bot)"

        return f"{signature} {{\n    // {wrapper_desc}\n}}"
