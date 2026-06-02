"""
Graph Manager Adapter

Bridges the old refactor detection results from graph_manager to the new refactor module classes.

This adapter:
1. Converts refactor_info (Dict) into a RefactorOpportunity
2. Selects and runs the corresponding Refactor class
3. Returns a unified RefactorResult
4. Triggers callbacks (e.g. on_skill_code_changed)
"""

from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime

from .base import (
    RefactorType,
    RefactorOpportunity,
    RefactorResult,
)
from .parametric import ParametricRefactor
from .behavioral import BehavioralRefactor
from .sibling import SiblingRefactor
from .duplication import DuplicationRefactor
from .subskill_extraction import SubskillExtractionRefactor

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillGraphManager


class RefactorExecutor:
    """
    Refactor executor

    Converts graph_manager's refactor_info and runs the new refactor module.

    Usage:
        executor = RefactorExecutor(skill_graph_manager=manager, llm=llm)
        result = executor.execute(refactor_info, new_skill_name, new_skill_code)
    """

    # Mapping from refactor_type strings to RefactorType enum values
    TYPE_MAPPING = {
        "parametric": RefactorType.PARAMETRIC,
        "behavioral": RefactorType.BEHAVIORAL,
        "sibling": RefactorType.MERGE_SIBLINGS,
        "merge_siblings": RefactorType.MERGE_SIBLINGS,
        "duplication": RefactorType.DUPLICATION,
        "functional_superset": RefactorType.BEHAVIORAL,  # Merged into BEHAVIORAL
        "extract_common": RefactorType.EXTRACT_COMMON,
        "extract_common_subskill": RefactorType.EXTRACT_COMMON,
    }

    def __init__(
        self,
        skill_graph_manager: 'SkillGraphManager' = None,
        llm=None,
        logger=None,
        on_code_changed_callback: Optional[Callable] = None,
    ):
        """
        Initialize RefactorExecutor

        Args:
            skill_graph_manager: skill graph manager
            llm: LLM instance (used for code generation)
            logger: logger
            on_code_changed_callback: code-change callback with signature: (skill_name, old_code, new_code, change_source) -> None
        """
        self.skill_graph_manager = skill_graph_manager
        self.llm = llm
        self.logger = logger
        self.on_code_changed_callback = on_code_changed_callback

        # Initialize refactor instances for each type
        self._refactors = {
            RefactorType.PARAMETRIC: ParametricRefactor(
                skill_graph_manager=skill_graph_manager,
                llm=llm,
                logger=logger,
            ),
            RefactorType.BEHAVIORAL: BehavioralRefactor(
                skill_graph_manager=skill_graph_manager,
                llm=llm,
                logger=logger,
            ),
            RefactorType.MERGE_SIBLINGS: SiblingRefactor(
                skill_graph_manager=skill_graph_manager,
                llm=llm,
                logger=logger,
            ),
            RefactorType.DUPLICATION: DuplicationRefactor(
                skill_graph_manager=skill_graph_manager,
                llm=llm,
                logger=logger,
            ),
            RefactorType.EXTRACT_COMMON: SubskillExtractionRefactor(
                skill_graph_manager=skill_graph_manager,
                llm=llm,
                logger=logger,
            ),
        }

        # Rollback history
        self._rollback_history: List[RefactorResult] = []

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            log_fn = getattr(self.logger, level, self.logger.info)
            log_fn(message)
        else:
            print(f"[RefactorExecutor] {message}")

    def execute(
        self,
        refactor_info: Dict[str, Any],
        new_skill_name: str,
        new_skill_code: str,
        auto_update_callers: bool = False,
    ) -> RefactorResult:
        """
        Execute a refactor

        Convert graph_manager's refactor_info to a RefactorOpportunity,
        then run the corresponding Refactor class.

        Args:
            refactor_info: information returned by graph_manager's _detect_refactor_relationships
                format: {
                    "refactor_type": "parametric" | "behavioral" | "sibling" | "duplication",
                    "general_skill": "skill_name",
                    "covered_skills": ["skill1", "skill2"],
                    "parameter_mapping": {...},
                    "reason": "explanation",
                    "general_skill_name": "suggested_name",  # used by sibling
                }
            new_skill_name: name of the new skill
            new_skill_code: code of the new skill
            auto_update_callers: whether to automatically update caller code

        Returns:
            RefactorResult: execution result
        """
        # 1. Parse refactor_type
        type_str = refactor_info.get("refactor_type", "")
        refactor_type = self.TYPE_MAPPING.get(type_str.lower())

        if not refactor_type:
            return RefactorResult(
                success=False,
                refactor_type=RefactorType.PARAMETRIC,  # Default
                source_skill=new_skill_name,
                target_skill="",
                error_message=f"Unknown refactor type: {type_str}",
            )

        self._log(f"Executing {refactor_type.value} refactor", "info")

        # 2. Get the corresponding refactor instance
        refactor = self._refactors.get(refactor_type)
        if not refactor:
            return RefactorResult(
                success=False,
                refactor_type=refactor_type,
                source_skill=new_skill_name,
                target_skill="",
                error_message=f"No refactor implementation for type: {refactor_type.value}",
            )

        # 3. Detect BEHAVIORAL Case B: new skill is the small skill; multiple covered_skills need refactoring
        general_skill = refactor_info.get("general_skill", "")
        covered_skills = refactor_info.get("covered_skills", [])

        is_behavioral_case_b = (
            refactor_type == RefactorType.BEHAVIORAL
            and general_skill == new_skill_name
            and covered_skills
        )

        if is_behavioral_case_b:
            # Case B: iterate over all covered_skills and refactor each
            # Pre-filter skills that do not exist in the graph
            valid_covered_skills = [
                skill for skill in covered_skills
                if self.skill_graph_manager.get_node(skill) is not None
            ]
            invalid_skills = set(covered_skills) - set(valid_covered_skills)
            if invalid_skills:
                self._log(
                    f"[BEHAVIORAL Case B] Skipping non-existent skills: {invalid_skills}",
                    "warning"
                )

            # Filter out unverified experimental skills (skills that have never executed successfully shouldn't participate in refactor)
            verified_covered_skills = []
            for skill in valid_covered_skills:
                node = self.skill_graph_manager.get_node(skill)
                # Keep: non-experimental skills, or experimental skills that have been verified
                if node and (not getattr(node, 'is_experimental', False) or getattr(node, 'is_verified', False)):
                    verified_covered_skills.append(skill)
                else:
                    self._log(
                        f"[Refactor] Skipping unverified skill '{skill}' (experimental but never succeeded)",
                        "warning"
                    )
            valid_covered_skills = verified_covered_skills

            if not valid_covered_skills:
                return RefactorResult(
                    success=False,
                    refactor_type=refactor_type,
                    source_skill=new_skill_name,
                    target_skill=new_skill_name,
                    error_message=f"No valid covered skills found in graph. Original: {covered_skills}",
                )

            # Log differently depending on the original type (functional_superset and behavioral have different semantics)
            if type_str == "functional_superset":
                self._log(
                    f"[FUNCTIONAL_SUPERSET] New skill '{new_skill_name}' is the more mature version; "
                    f"it will cover {len(valid_covered_skills)} older versions: {valid_covered_skills}",
                    "info"
                )
            else:
                self._log(
                    f"[BEHAVIORAL Case B] New skill '{new_skill_name}' is a foundational component; "
                    f"it will be called by {len(valid_covered_skills)} broader skills: {valid_covered_skills}",
                    "info"
                )

            all_changes = []
            all_updated_callers = []
            failed_skills = []
            # Collect every individual single_result so the history tracker
            # can persist them independently. Without this, N refactors
            # collapse into 1 history record (r34 bug: 3 craftFurnace
            # sub-refactors became 1 record at refactor_fe95ef019073.json).
            sub_results: List[RefactorResult] = []

            for skill_to_refactor in valid_covered_skills:
                # Create an individual opportunity for each covered_skill
                single_opportunity = RefactorOpportunity(
                    refactor_type=refactor_type,
                    source_skill=skill_to_refactor,  # The broader skill that needs refactoring
                    target_skill=new_skill_name,     # The smaller skill being called
                    reason=refactor_info.get("reason", ""),
                    confidence=0.8,
                    parameter_mapping=refactor_info.get("parameter_mapping"),
                    covered_skills=[],  # Single-target refactors don't need covered_skills
                    analysis_id=f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{skill_to_refactor}",
                )

                try:
                    single_result = refactor.apply(single_opportunity)
                    sub_results.append(single_result)  # always record (success or failure)
                    if single_result.success:
                        self._notify_code_changed(single_result)
                        self._rollback_history.append(single_result)
                        all_changes.extend(single_result.changes_made or [])
                        all_updated_callers.extend(single_result.updated_callers or [])
                        self._log(f"  ✓ Successfully refactored '{skill_to_refactor}'", "info")
                    else:
                        failed_skills.append((skill_to_refactor, single_result.error_message))
                        self._log(f"  ✗ Failed to refactor '{skill_to_refactor}': {single_result.error_message}", "warning")
                except Exception as e:
                    failed_skills.append((skill_to_refactor, str(e)))
                    # Synthesize a failed single_result so the exception is also
                    # persisted in history (otherwise it's invisible to auditors)
                    sub_results.append(RefactorResult(
                        success=False,
                        refactor_type=refactor_type,
                        source_skill=skill_to_refactor,
                        target_skill=new_skill_name,
                        error_message=str(e),
                    ))
                    self._log(f"  ✗ Exception while refactoring '{skill_to_refactor}': {e}", "error")

            # Return the combined result (with sub_results for the history tracker to record individually)
            if failed_skills:
                error_msg = "; ".join([f"{s}: {e}" for s, e in failed_skills])
                return RefactorResult(
                    success=len(failed_skills) < len(covered_skills),  # Partial success
                    refactor_type=refactor_type,
                    source_skill=new_skill_name,
                    target_skill=new_skill_name,
                    changes_made=all_changes,
                    updated_callers=all_updated_callers,
                    error_message=f"Some refactors failed: {error_msg}" if failed_skills else None,
                    sub_results=sub_results,
                )
            else:
                return RefactorResult(
                    success=True,
                    refactor_type=refactor_type,
                    source_skill=new_skill_name,
                    target_skill=new_skill_name,
                    changes_made=all_changes,
                    updated_callers=all_updated_callers,
                    sub_results=sub_results,
                )

        # 4. Standard path: convert to a single RefactorOpportunity and execute
        opportunity = self._convert_to_opportunity(
            refactor_info, refactor_type, new_skill_name
        )

        # 4.1 Check whether it was skipped due to an existing call relation, etc.
        if opportunity is None:
            self._log(
                f"[GraphAdapter] Refactor skipped: call relation already exists or data invalid",
                "info"
            )
            return RefactorResult(
                success=True,  # Skipping is not a failure; it is normal business logic
                refactor_type=refactor_type,
                source_skill=new_skill_name,
                target_skill=refactor_info.get("general_skill", ""),
                error_message=None,
            )

        # 5. Run the refactor
        try:
            result = refactor.apply(opportunity)
        except Exception as e:
            self._log(f"Refactor execution exception: {e}", "error")
            import traceback
            traceback.print_exc()
            return RefactorResult(
                success=False,
                refactor_type=refactor_type,
                source_skill=opportunity.source_skill,
                target_skill=opportunity.target_skill,
                error_message=str(e),
            )

        # 6. On success, trigger callback notification
        if result.success:
            self._notify_code_changed(result)
            self._rollback_history.append(result)

            # Auto-update callers (if enabled)
            if auto_update_callers and result.updated_callers:
                self._log(
                    f"Updated {len(result.updated_callers)} callers: {result.updated_callers}",
                    "info"
                )

        return result

    def _convert_to_opportunity(
        self,
        refactor_info: Dict[str, Any],
        refactor_type: RefactorType,
        new_skill_name: str,
    ) -> Optional[RefactorOpportunity]:
        """
        Convert refactor_info to a RefactorOpportunity

        Different refactor_types have different field interpretations:
        - PARAMETRIC: source_skill = covered skill, target_skill = general skill
        - BEHAVIORAL: source_skill = broader skill, target_skill = narrower skill
        - MERGE_SIBLINGS: source_skill = first sibling, covered_skills = all siblings
        - DUPLICATION: source_skill = the one to mark as deprecated, target_skill = the one to keep

        Returns:
            RefactorOpportunity: the converted opportunity object
            None: if the call relation already exists or data is invalid, returns None to indicate skip
        """
        general_skill = refactor_info.get("general_skill", "")
        covered_skills = refactor_info.get("covered_skills", [])
        parameter_mapping = refactor_info.get("parameter_mapping")
        reason = refactor_info.get("reason", "")
        general_skill_name = refactor_info.get("general_skill_name", "")  # Name suggested by sibling

        # Determine source and target based on type
        if refactor_type == RefactorType.PARAMETRIC:
            # PARAMETRIC: specialized skill -> general skill
            # source = the covered specialized skill (or new skill)
            # target = the existing general skill
            if general_skill and general_skill != new_skill_name:
                # New skill is covered by an existing general skill
                source = new_skill_name
                target = general_skill
            elif covered_skills:
                # The new skill is general and covers an existing skill
                source = covered_skills[0] if covered_skills else new_skill_name
                target = new_skill_name
            else:
                source = new_skill_name
                target = general_skill or ""

        elif refactor_type == RefactorType.BEHAVIORAL:
            # BEHAVIORAL: a broader skill calls a narrower skill
            # general_skill = the narrower skill (the one to be called)
            #
            # Case A: new skill is the broader skill (general_skill != new_skill_name, covered_skills empty)
            # source = new_skill_name (broader skill, needs refactoring)
            # target = general_skill (narrower skill, called)
            #
            # Case B: new skill is the narrower skill (general_skill == new_skill_name, covered_skills non-empty)
            # source = a skill in covered_skills (broader skill, needs refactoring)
            # target = new_skill_name (narrower skill, called)

            if general_skill == new_skill_name and covered_skills:
                # Case B: new skill is the narrower one; refactor the broader skills in covered_skills
                # Only set the first one here; the execute method handles all covered_skills
                source = covered_skills[0]
                target = new_skill_name
            else:
                # Case A: new skill is the broader one; refactor it to call general_skill
                source = new_skill_name
                target = general_skill or ""

            # check whether a call relation already exists (to avoid pointless refactors)
            # If a source -> target or target -> source edge already exists, skip this refactor
            if source and target and self.skill_graph_manager:
                # Check whether source -> target already exists (call relation established)
                if self.skill_graph_manager.has_edge(source, target):
                    self._log(
                        f"[GraphAdapter] Skipping BEHAVIORAL: {source} already calls {target}",
                        "info"
                    )
                    return None
                # Check whether the reverse target -> source exists
                if self.skill_graph_manager.has_edge(target, source):
                    self._log(
                        f"[GraphAdapter] Skipping BEHAVIORAL: reverse call relation already exists {target} -> {source}",
                        "info"
                    )
                    return None

        elif refactor_type == RefactorType.MERGE_SIBLINGS:
            # MERGE_SIBLINGS: merge multiple siblings into a single general skill
            # source = new skill (first sibling)
            # covered_skills = other siblings (excluding source)
            # Note: sibling.py's apply() prepends source_skill to the siblings list
            source = new_skill_name
            target = general_skill_name or general_skill or ""

        elif refactor_type == RefactorType.DUPLICATION:
            # DUPLICATION: functional duplication; keep the higher-maturity one
            # source = the one to mark as deprecated
            # target = the one to keep
            source = new_skill_name
            target = covered_skills[0] if covered_skills else ""

        elif refactor_type == RefactorType.EXTRACT_COMMON:
            # EXTRACT_COMMON: extract common code
            # source = new skill
            # target = name of the extracted common skill
            source = new_skill_name
            target = general_skill_name or general_skill or ""

        else:
            source = new_skill_name
            target = general_skill or ""

        # Make sure covered_skills does not contain source_skill (avoid duplicates)
        # The covered_skills returned by LLM detection may contain source; filter it
        if source in covered_skills:
            covered_skills = [s for s in covered_skills if s != source]

        return RefactorOpportunity(
            refactor_type=refactor_type,
            source_skill=source,
            target_skill=target,
            reason=reason,
            confidence=0.8,  # From LLM detection; relatively high confidence
            parameter_mapping=parameter_mapping,
            covered_skills=covered_skills,
            analysis_id=f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def _notify_code_changed(self, result: RefactorResult):
        """
        Notify that code has changed

        Triggers on_code_changed_callback so components like the optimizer know the code has changed.
        """
        if not self.on_code_changed_callback:
            return

        if not result.success or not result.new_code:
            return

        try:
            self.on_code_changed_callback(
                skill_name=result.source_skill,
                old_code=result.old_code or "",
                new_code=result.new_code,
                change_source="refactor",
            )
            self._log(
                f"Notified code change: {result.source_skill} ({result.refactor_type.value})",
                "info"
            )
        except Exception as e:
            self._log(f"Failed to notify code change: {e}", "warning")

    def rollback(self, result: RefactorResult) -> bool:
        """
        Roll back the specified refactor

        Args:
            result: RefactorResult to roll back

        Returns:
            bool: whether the rollback succeeded
        """
        refactor = self._refactors.get(result.refactor_type)
        if refactor:
            return refactor.rollback(result)
        return False


def create_executor_for_graph_manager(
    skill_graph_manager: 'SkillGraphManager',
    llm=None,
    logger=None,
) -> RefactorExecutor:
    """
    Create a RefactorExecutor for a SkillGraphManager

    This is a convenience function that wires up the callback automatically.

    Args:
        skill_graph_manager: skill graph manager
        llm: LLM instance
        logger: logger

    Returns:
        RefactorExecutor: a configured executor
    """
    # Fetch the on_code_changed callback
    callback = getattr(skill_graph_manager, '_notify_code_changed', None)

    return RefactorExecutor(
        skill_graph_manager=skill_graph_manager,
        llm=llm,
        logger=logger,
        on_code_changed_callback=callback,
    )
