"""
Subskill Extraction Refactor

Common subskill extraction strategy: extract shared logic from multiple skills
into a standalone skill.

Examples:
- craftOakBoat and craftBirchBoat both contain logic for placing a crafting table
  -> extract into ensureCraftingTablePlaced(bot)

- mineOakLogs and mineBirchLogs both contain logic for checking the tool
  -> extract into ensureAxeEquipped(bot)

This refactor will:
1. Create a new shared skill
2. Modify the original skills to call the new skill
3. Add dependency edges
"""

import math
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from .base import (
    RefactorType,
    RefactorOpportunity,
    RefactorResult,
    SkillRefactor,
    analyze_code_dependencies,
    inject_dependencies,
    ensure_bot_parameter,
)
from ..skill_graph.models.coverage import CoverageType
from ..skill_graph.models.node import SkillNode
from ..utils import validate_code_syntax
from skillnet.utils.stats_tracker import record_llm_usage


@dataclass
class CommonCodeBlock:
    """Common code block."""
    code: str
    source_skills: List[str]
    similarity_score: float
    start_lines: Dict[str, int]  # skill_name -> start_line
    end_lines: Dict[str, int]    # skill_name -> end_line
    suggested_name: str = ""
    description: str = ""


@dataclass
class ExtractionPlan:
    """Extraction plan."""
    common_block: CommonCodeBlock
    new_skill_name: str
    new_skill_code: str
    new_skill_description: str
    affected_skills: Dict[str, str]  # skill_name -> modified_code


class SubskillExtractionRefactor(SkillRefactor):
    """
    Common subskill extraction refactorer.

    Extract shared logic from multiple skills into a standalone subskill.

    Refactor workflow:
    1. Analyze code of multiple skills and identify common code blocks
    2. Generate a new shared skill
    3. Modify the original skills to call the new skill
    4. Add dependency edges
    """

    # Minimum number of common code lines
    MIN_COMMON_LINES = 3

    # Minimum similarity threshold
    MIN_SIMILARITY = 0.7

    def apply(self, opportunity: RefactorOpportunity) -> RefactorResult:
        """
        Apply the common subskill extraction refactor.

        Args:
            opportunity: Refactor opportunity

        Returns:
            RefactorResult: Refactor result
        """
        # Note: RefactorType.EXTRACT_COMMON_SUBSKILL is an alias for EXTRACT_COMMON; they are equivalent
        if opportunity.refactor_type != RefactorType.EXTRACT_COMMON:
            return RefactorResult(
                success=False,
                refactor_type=opportunity.refactor_type,
                source_skill=opportunity.source_skill,
                target_skill=opportunity.target_skill,
                error_message="Wrong refactor type for SubskillExtractionRefactor"
            )

        # For EXTRACT_COMMON_SUBSKILL, source_skill is one of them, and
        # covered_skills contains all the skills to extract from
        skills_to_extract = [opportunity.source_skill]
        if opportunity.covered_skills:
            skills_to_extract.extend(opportunity.covered_skills)

        self._log(
            f"[SubskillExtraction] Starting common subskill extraction: {skills_to_extract}",
            "info"
        )

        if not self.skill_graph_manager:
            return self._create_failed_result(
                opportunity, "No skill_graph_manager available"
            )

        # Get all skill nodes
        skill_nodes = {}
        for name in skills_to_extract:
            node = self.skill_graph_manager.get_node(name)
            if node:
                skill_nodes[name] = node

        if len(skill_nodes) < 2:
            return self._create_failed_result(
                opportunity, "Need at least 2 skills to extract common subskill"
            )

        # Check whether code is empty
        for name, node in skill_nodes.items():
            if not node.code:
                return self._create_failed_result(
                    opportunity, f"Skill '{name}' has empty code"
                )

        # Save rollback data
        rollback_data = self._save_multi_skill_rollback(skill_nodes)

        try:
            # Identify common code blocks
            common_blocks = self._find_common_code_blocks(skill_nodes)

            if not common_blocks:
                return self._create_failed_result(
                    opportunity, "No common code blocks found"
                )

            # Select the best common block
            best_block = self._select_best_block(common_blocks)

            if not best_block:
                return self._create_failed_result(
                    opportunity, "No valid common blocks to extract"
                )

            # Generate the extraction plan
            # v3.H+ Layer 3: wrap plan-build to capture LLM raw response
            from ._retry_helpers import (
                capture_llm_responses,
                save_refactor_failure_forensics,
                build_retry_message,
            )

            with capture_llm_responses(self) as cap_first:
                extraction_plan = self._create_extraction_plan(best_block, skill_nodes)
            first_raw = cap_first.raw_responses[-1] if cap_first.raw_responses else ""
            first_messages = cap_first.last_messages

            if not extraction_plan:
                return self._create_failed_result(
                    opportunity, "Failed to create extraction plan"
                )

            # Create the new skill
            new_skill_name = extraction_plan.new_skill_name
            new_skill_code = extraction_plan.new_skill_code

            # [Fix] Validate new_skill_code syntax
            is_valid, syntax_error = validate_code_syntax(new_skill_code)

            # v3.H+ Layer 3 — single-shot retry + forensics on validation failure
            if not is_valid:
                save_refactor_failure_forensics(
                    skill_graph_manager=self.skill_graph_manager,
                    refactor_type="extract_common",
                    skill_names=list(skill_nodes.keys()),
                    attempt=1,
                    raw_response=first_raw,
                    babel_error=syntax_error or "",
                    prompt_messages=first_messages,
                    logger=getattr(self, "logger", None),
                )
                self._log(
                    f"[SubskillExtraction] ⟳ Layer 3 retry — first attempt failed validation: "
                    f"{(syntax_error or '')[:120]}",
                    "warning",
                )
                with capture_llm_responses(self) as cap_retry:
                    plan2 = self._create_plan_with_llm(
                        best_block, skill_nodes,
                        retry_hint=build_retry_message(syntax_error or ""),
                    )
                retry_raw = cap_retry.raw_responses[-1] if cap_retry.raw_responses else ""
                if plan2 is not None:
                    extraction_plan = plan2
                    new_skill_name = extraction_plan.new_skill_name
                    new_skill_code = extraction_plan.new_skill_code
                    is_valid, syntax_error = validate_code_syntax(new_skill_code)
                    if is_valid:
                        self._log("[SubskillExtraction] ✓ Layer 3 retry succeeded", "info")
                    else:
                        save_refactor_failure_forensics(
                            skill_graph_manager=self.skill_graph_manager,
                            refactor_type="extract_common",
                            skill_names=list(skill_nodes.keys()),
                            attempt=2,
                            raw_response=retry_raw,
                            babel_error=syntax_error or "",
                            prompt_messages=cap_retry.last_messages,
                            logger=getattr(self, "logger", None),
                        )

            if not is_valid:
                self._log(
                    f"[SubskillExtraction] ✗ Syntax validation failed for new skill code: {syntax_error}",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Syntax error in new skill code: {syntax_error}"
                )

            # Function-reference validation: detect whether the LLM-extracted subskill calls non-existent functions
            from skillnet.agents.optimizer.validators.code_validator._references import (
                validate_function_references,
            )
            available_skills = set(self.skill_graph_manager.get_all_skill_names())
            # Get domain function sets for validation (if domain configured)
            _dk = getattr(self, 'domain_knowledge', None)
            _domain_fns = _dk.get_known_functions() if _dk else None
            _domain_fns = _domain_fns or None
            ref_result = validate_function_references(
                new_skill_code,
                available_skills=available_skills,
                domain_functions=_domain_fns,
                strict_mode=True,
            )
            if not ref_result["valid"]:
                undefined_names = [f["name"] for f in ref_result["undefined_functions"]]
                self._log(
                    f"[SubskillExtraction] ✗ Function-reference validation failed: {new_skill_name} calls non-existent functions {undefined_names}",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Function reference validation failed for {new_skill_name}: "
                    f"undefined functions {undefined_names}"
                )

            # Add the new skill to the graph
            new_node = SkillNode(
                name=new_skill_name,
                code=new_skill_code,
                description=extraction_plan.new_skill_description,
            )

            # ───────────────────────────────────────────────────────────────────
            # Plan v3 Part A: extract parameter metadata for the new helper
            # skill BEFORE add_skill_node. add_skill_node is a low-level graph
            # insert that bypasses the parameter_extractor invocation used in
            # add_new_skill. Without this, planner.plan() sees parameters={}
            # for the new helper and falls through to code-based extraction
            # where object params resolve to undefined → trim → wrong call.
            # ───────────────────────────────────────────────────────────────────
            try:
                extracted_params = self.skill_graph_manager._extract_parameters(
                    new_node.code,
                    new_node.description,
                    new_skill_name,
                )
                if extracted_params:
                    new_node.parameters = extracted_params
                    self._log(
                        f"[SubskillExtraction] PartA: extracted {len(extracted_params)} "
                        f"parameter schemas for new helper {new_skill_name}: "
                        f"{list(extracted_params.keys())}",
                        "info",
                    )
            except Exception as e:
                self._log(
                    f"[SubskillExtraction] PartA: parameter extraction failed for "
                    f"{new_skill_name}: {e!r}",
                    "warning",
                )

            self.skill_graph_manager.add_skill_node(new_node)

            # Set attributes of the new skill: this is a standalone valuable common subskill
            new_node = self.skill_graph_manager.get_node(new_skill_name)
            if new_node:
                new_node.is_general_skill = True
                new_node.coverage_type = CoverageType.COMMON_SUBSKILL
                # Note: is_covered = False, because this is an extracted standalone skill

            # Modify the original skills
            changes_made = [f"Created new skill: {new_skill_name}"]

            for skill_name, modified_code in extraction_plan.affected_skills.items():
                node = skill_nodes[skill_name]

                # Phase 4: semantic equivalence validation
                if self.skill_graph_manager:
                    from skillnet.agents.optimizer.validators.semantic_compatibility import (
                        SemanticEquivalenceValidator,
                    )
                    semantic_validator = SemanticEquivalenceValidator(
                        skill_graph=self.skill_graph_manager,
                        llm=None,
                        enable_llm_validation=False,
                        enable_call_chain_analysis=True,
                    )
                    old_code = node.code if node else ""
                    old_effects = [str(e) for e in node.expected_effects] if node and node.expected_effects else []

                    is_semantically_equivalent = semantic_validator.quick_check(old_code, modified_code)
                    if not is_semantically_equivalent:
                        self._log(
                            f"[SubskillExtraction] ⚠️ Semantic-equivalence quick check did not pass (skill: {skill_name}); continuing with other validations",
                            "warning"
                        )
                    else:
                        self._log(
                            f"[SubskillExtraction] ✓ Semantic-equivalence quick check passed (skill: {skill_name})",
                            "info"
                        )

                # Phase 2: naming-conflict check (before syntax validation)
                if self.skill_graph_manager:
                    from skillnet.agents.optimizer.validators import check_naming_conflicts
                    existing_skill_names = set(self.skill_graph_manager.get_all_skill_names(include_task_specific=True))
                    # The skill legitimately redefines its own function; only shadowing
                    # a DIFFERENT skill is a conflict.
                    existing_skill_names.discard(skill_name)
                    no_conflict, conflict_messages = check_naming_conflicts(modified_code, existing_skill_names)
                    if not no_conflict:
                        self._log(
                            f"[SubskillExtraction] ✗ Naming-conflict detection failed (skill: {skill_name}): {conflict_messages}",
                            "error"
                        )
                        return self._create_failed_result(
                            opportunity,
                            f"Naming conflict in modified code for {skill_name}: {'; '.join(conflict_messages)}"
                        )

                # Validate the syntax of the modified code
                is_valid, syntax_error = validate_code_syntax(modified_code)
                if not is_valid:
                    self._log(
                        f"[SubskillExtraction] ✗ JS syntax validation failed (skill: {skill_name}): {syntax_error}",
                        "error"
                    )
                    return self._create_failed_result(
                        opportunity,
                        f"Syntax error in modified code for {skill_name}: {syntax_error}"
                    )

                # Update code through the unified interface (syntax already validated; skip duplicate validation)
                update_success = self.skill_graph_manager.update_skill_code(
                    skill_name=skill_name,
                    new_code=modified_code,
                    change_log=f"Modified to call extracted subskill {new_skill_name}",
                    source="refactor:subskill_extraction",
                    skip_validation=True,  # Syntax already validated
                    skip_metadata=False,   # Issue 9 fix: capabilities may change after extraction; metadata must be updated
                    skip_interface_check=True,  # Do not trigger cascade
                    create_version=True,
                )
                if not update_success:
                    return self._create_failed_result(
                        opportunity,
                        f"Failed to update code for {skill_name}"
                    )

                # Add dependency edge
                self.skill_graph_manager.add_edge(skill_name, new_skill_name)
                changes_made.append(f"Modified {skill_name} to call {new_skill_name}")

            self._log(
                f"[SubskillExtraction] ✓ Successfully extracted common subskill {new_skill_name}",
                "info"
            )

            # Notify callers that a new shared skill is available
            updated_callers = []
            caller_update_details = {}
            all_caller_rollback_data = {}  # Aggregate rollback data from all callers

            for skill_name in skill_nodes.keys():
                # Check which skills called the skills involved in the extraction
                # SubskillExtraction only modifies internal implementation without changing the interface; callers do not need updates
                propagation_result = self.propagate_to_callers(
                    refactored_skill=skill_name,
                    changes={
                        "change_type": "general_created",
                        "new_skill_name": new_skill_name,
                    },
                    auto_update=False,  # Disable auto-update to avoid wrongly replacing function calls
                )

                if propagation_result["needs_update"]:
                    for caller in propagation_result["needs_update"]:
                        if caller not in updated_callers and caller not in skill_nodes:
                            updated_callers.append(caller)
                            caller_update_details[caller] = (
                                f"Could directly use {new_skill_name}"
                            )

                # Aggregate caller rollback data
                if propagation_result.get("rollback_data"):
                    all_caller_rollback_data.update(propagation_result["rollback_data"])

            # Save caller rollback data (if any)
            if all_caller_rollback_data:
                rollback_data["caller_rollback_data"] = all_caller_rollback_data

            if updated_callers:
                changes_made.append(
                    f"Detected {len(updated_callers)} callers that could use {new_skill_name}"
                )

            # Save to disk
            self.skill_graph_manager.save()

            return RefactorResult(
                success=True,
                refactor_type=RefactorType.EXTRACT_COMMON,
                source_skill=opportunity.source_skill,
                target_skill=new_skill_name,
                old_code=None,  # Multiple skills were modified
                new_code=new_skill_code,
                changes_made=changes_made,
                rollback_available=True,
                rollback_data=rollback_data,
                updated_callers=updated_callers,
                caller_update_details=caller_update_details,
            )

        except Exception as e:
            self._log(f"[SubskillExtraction] Refactor failed: {e}", "error")
            return self._create_failed_result(opportunity, str(e))

    def _find_common_code_blocks(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> List[CommonCodeBlock]:
        """
        Find common code blocks.

        Prefer LLM-based semantic analysis; fall back to rules if it fails.

        Args:
            skill_nodes: Dictionary of skill nodes

        Returns:
            List[CommonCodeBlock]: Common code blocks found
        """
        # Prefer LLM-based semantic analysis
        if self.llm:
            result = self._find_common_code_blocks_with_llm(skill_nodes)
            if result:
                return result
            self._log("[SubskillExtraction] LLM analysis failed; falling back to rules", "warning")

        return self._find_common_code_blocks_simple(skill_nodes)

    def _find_common_code_blocks_with_llm(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[List[CommonCodeBlock]]:
        """
        Use the LLM for semantic common-code detection.

        Can identify:
        - Code with different variable names but the same logic
        - Code with similar functionality but slightly different implementations
        - Setup/teardown patterns across skills
        """
        try:
            import json
            from langchain.schema import HumanMessage, SystemMessage
            from .prompts import PromptLoader

            # Build the skill-code block string
            skill_code_parts = []
            for name, node in skill_nodes.items():
                if node.code:
                    skill_code_parts.append(
                        f"**{name}**:\n```javascript\n{node.code}\n```\n"
                    )

            if len(skill_code_parts) < 2:
                return None

            skill_code_blocks = "\n".join(skill_code_parts)

            template = PromptLoader.load("subskill_common_detection")
            system_prompt, human_prompt = template.format(
                skill_code_blocks=skill_code_blocks,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.subskill_extraction._find_common_code_blocks_with_llm")
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse the JSON response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                if not data.get("has_common_code", False):
                    self._log("[SubskillExtraction] LLM did not detect common code", "info")
                    return []

                common_blocks = []
                for block_data in data.get("common_blocks", []):
                    skills = block_data.get("skills_involved", [])
                    if len(skills) < 2:
                        continue

                    confidence = block_data.get("confidence", 0.7)
                    if confidence < self.MIN_SIMILARITY:
                        continue

                    # Prefer the code snippet returned by the LLM (O(n) complexity)
                    code_snippet = block_data.get("code_snippet", "")

                    if code_snippet and len(code_snippet.strip()) >= 10:
                        # Option A: use the LLM code snippet + O(n) position lookup
                        positions = self._find_code_positions(skill_nodes, skills, code_snippet)

                        if positions:
                            common_blocks.append(CommonCodeBlock(
                                code=code_snippet,
                                source_skills=skills,
                                similarity_score=confidence,
                                start_lines={s: pos[0] for s, pos in positions.items()},
                                end_lines={s: pos[1] for s, pos in positions.items()},
                                suggested_name=block_data.get("suggested_name", "commonHelper"),
                                description=block_data.get("description", ""),
                            ))

                            self._log(
                                f"[SubskillExtraction] LLM found common code: "
                                f"{block_data.get('suggested_name')} (confidence={confidence:.2f})",
                                "info"
                            )
                            continue

                    # Option B: fall back to rule-based validation (but limit window size)
                    common_code = self._extract_common_code_for_skills(
                        skill_nodes, skills
                    )

                    if common_code:
                        common_blocks.append(CommonCodeBlock(
                            code=common_code,
                            source_skills=skills,
                            similarity_score=confidence,
                            start_lines={s: 0 for s in skills},
                            end_lines={s: len(common_code.split('\n')) for s in skills},
                            suggested_name=block_data.get("suggested_name", "commonHelper"),
                            description=block_data.get("description", ""),
                        ))

                        self._log(
                            f"[SubskillExtraction] LLM found common code (rule-based validation): "
                            f"{block_data.get('suggested_name')} (confidence={confidence:.2f})",
                            "info"
                        )

                return common_blocks if common_blocks else None

            return None

        except Exception as e:
            self._log(f"[SubskillExtraction] LLM analysis exception: {e}", "warning")
            return None

    def _extract_common_code_for_skills(
        self,
        skill_nodes: Dict[str, 'SkillNode'],
        skills: List[str]
    ) -> Optional[str]:
        """
        Extract common code from the specified skills.

        Uses the rule-based method as a fallback to retrieve the actual code lines.
        """
        # Collect code from the specified skills
        skill_lines = {}
        for name in skills:
            if name in skill_nodes and skill_nodes[name].code:
                lines = self._normalize_code_lines(skill_nodes[name].code)
                if lines:
                    skill_lines[name] = lines

        if len(skill_lines) < 2:
            return None

        # Use the rule-based method to find common sequences
        common_sequences = self._find_common_sequences(skill_lines)
        if common_sequences:
            # Return the longest common sequence
            best_seq = max(common_sequences, key=lambda x: len(x[0]))
            return '\n'.join(best_seq[0])

        return None

    def _find_code_positions(
        self,
        skill_nodes: Dict[str, 'SkillNode'],
        skills: List[str],
        code_snippet: str
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Use string matching to locate the code snippet position within each skill.

        Complexity: O(n) per skill; overall O(n * k) where k is the number of skills.

        Args:
            skill_nodes: Dictionary of skill nodes
            skills: List of skills to search
            code_snippet: Code snippet returned by the LLM

        Returns:
            {skill_name: (start_line, end_line)} or None (if not enough matches are found)
        """
        positions = {}

        # Normalize the code snippet for matching
        snippet_normalized = self._normalize_for_matching(code_snippet)
        snippet_lines = snippet_normalized.split('\n')

        for skill_name in skills:
            if skill_name not in skill_nodes:
                continue

            node = skill_nodes[skill_name]
            if not node.code:
                continue

            # Try exact matching
            pos = self._find_snippet_position(node.code, code_snippet)

            if pos is None:
                # Try matching after normalization
                code_normalized = self._normalize_for_matching(node.code)
                pos = self._find_snippet_position_normalized(
                    code_normalized, snippet_normalized, snippet_lines
                )

            if pos:
                positions[skill_name] = pos

        # Need to find at least 70% of the skills
        min_required = math.ceil(len(skills) * self.MIN_SIMILARITY)
        if len(positions) >= min_required:
            return positions

        return None

    def _normalize_for_matching(self, code: str) -> str:
        """Normalize code for fuzzy matching (strip whitespace and comments)."""
        lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('//'):
                lines.append(stripped)
        return '\n'.join(lines)

    def _find_snippet_position(
        self,
        full_code: str,
        snippet: str
    ) -> Optional[Tuple[int, int]]:
        """Locate position via exact string matching; O(n) complexity."""
        idx = full_code.find(snippet)
        if idx >= 0:
            start_line = full_code[:idx].count('\n')
            end_line = start_line + snippet.count('\n') + 1
            return (start_line, end_line)
        return None

    def _find_snippet_position_normalized(
        self,
        code_normalized: str,
        snippet_normalized: str,
        snippet_lines: List[str]
    ) -> Optional[Tuple[int, int]]:
        """
        Locate position via matching after normalization.

        Used to handle whitespace/indentation differences.
        """
        code_lines = code_normalized.split('\n')
        snippet_len = len(snippet_lines)

        # Sliding window with fixed window size equal to snippet length; thus O(n)
        for i in range(len(code_lines) - snippet_len + 1):
            if code_lines[i:i + snippet_len] == snippet_lines:
                return (i, i + snippet_len)

        return None

    def _find_common_code_blocks_simple(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> List[CommonCodeBlock]:
        """
        Find common code blocks using rules (simple version).

        Sliding-window algorithm based on exact line-level matching.
        """
        common_blocks = []

        # Simple implementation: line-based comparison
        skill_lines = {}
        for name, node in skill_nodes.items():
            # Skip nodes with empty code
            if not node.code:
                self._log(f"[SubskillExtraction] Skipping skill with empty code: {name}", "warning")
                continue
            # Clean code lines
            lines = self._normalize_code_lines(node.code)
            skill_lines[name] = lines

        # Find line sequences common to all skills
        common_sequences = self._find_common_sequences(skill_lines)

        for seq, locations in common_sequences:
            if len(seq) >= self.MIN_COMMON_LINES:
                # Compute similarity
                similarity = len(locations) / len(skill_nodes)

                if similarity >= self.MIN_SIMILARITY:
                    common_blocks.append(CommonCodeBlock(
                        code='\n'.join(seq),
                        source_skills=list(locations.keys()),
                        similarity_score=similarity,
                        start_lines={k: v[0] for k, v in locations.items()},
                        end_lines={k: v[1] for k, v in locations.items()},
                    ))

        return common_blocks

    def _normalize_code_lines(self, code: str) -> List[str]:
        """Normalize code lines for comparison."""
        lines = []
        for line in code.split('\n'):
            # Strip whitespace
            normalized = line.strip()
            # Skip empty lines and pure comment lines
            if normalized and not normalized.startswith('//'):
                lines.append(normalized)
        return lines

    # Maximum iteration limit to prevent excessive computation on large files
    MAX_SEQUENCE_SEARCH_ITERATIONS = 10000

    def _find_common_sequences(
        self,
        skill_lines: Dict[str, List[str]]
    ) -> List[Tuple[List[str], Dict[str, Tuple[int, int]]]]:
        """
        Find common line sequences.

        Returns: [(sequence, {skill_name: (start_line, end_line)}), ...]
        """
        sequences = []

        skill_names = list(skill_lines.keys())
        if len(skill_names) < 2:
            return sequences

        # Use the first skill as the baseline
        base_name = skill_names[0]
        base_lines = skill_lines[base_name]

        # Sliding window to find common sequences
        # Limit the maximum window size to avoid O(n²) complexity timeouts
        MAX_WINDOW_SIZE = 50  # Common code blocks typically do not exceed 50 lines
        max_size = min(len(base_lines), MAX_WINDOW_SIZE)
        iteration_count = 0
        for window_size in range(max_size, self.MIN_COMMON_LINES - 1, -1):
            for start in range(len(base_lines) - window_size + 1):
                iteration_count += 1
                if iteration_count > self.MAX_SEQUENCE_SEARCH_ITERATIONS:
                    self._log(
                        f"[SubskillExtraction] Reached max iteration count {self.MAX_SEQUENCE_SEARCH_ITERATIONS}; terminating search early",
                        "debug"  # Downgraded to debug since LLM is now the primary path
                    )
                    return sequences
                window = base_lines[start:start + window_size]

                # Check whether other skills have the same sequence
                locations = {base_name: (start, start + window_size)}

                for other_name in skill_names[1:]:
                    other_lines = skill_lines[other_name]
                    match_start = self._find_sequence_in_lines(window, other_lines)
                    if match_start >= 0:
                        locations[other_name] = (match_start, match_start + window_size)

                # If most skills have this sequence (use ceil to ensure correct integer comparison)
                min_required = math.ceil(len(skill_names) * self.MIN_SIMILARITY)
                if len(locations) >= min_required:
                    # Check whether it overlaps with an existing sequence
                    is_subset = False
                    for existing_seq, _ in sequences:
                        if self._is_subsequence(window, existing_seq):
                            is_subset = True
                            break

                    if not is_subset:
                        sequences.append((window, locations))

        return sequences

    def _find_sequence_in_lines(
        self,
        sequence: List[str],
        lines: List[str]
    ) -> int:
        """Search for a sequence within a list of lines; return start index or -1."""
        seq_len = len(sequence)
        for i in range(len(lines) - seq_len + 1):
            if lines[i:i + seq_len] == sequence:
                return i
        return -1

    def _is_subsequence(self, short: List[str], long: List[str]) -> bool:
        """Check whether `short` is a subsequence of `long`."""
        if len(short) > len(long):
            return False
        return self._find_sequence_in_lines(short, long) >= 0

    def _select_best_block(
        self,
        common_blocks: List[CommonCodeBlock]
    ) -> Optional[CommonCodeBlock]:
        """Select the best common code block."""
        if not common_blocks:
            return None

        # Sort by code length and similarity
        return max(
            common_blocks,
            key=lambda b: len(b.code) * b.similarity_score
        )

    def _create_extraction_plan(
        self,
        common_block: CommonCodeBlock,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[ExtractionPlan]:
        """Create the extraction plan."""
        if self.llm:
            return self._create_plan_with_llm(common_block, skill_nodes)
        else:
            return self._create_plan_simple(common_block, skill_nodes)

    def _create_plan_simple(
        self,
        common_block: CommonCodeBlock,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> ExtractionPlan:
        """Create a simple extraction plan (without the LLM)."""
        # Generate the new skill name
        skill_names = list(skill_nodes.keys())
        base_name = self._extract_common_prefix(skill_names)
        new_skill_name = f"{base_name}Common" if base_name else "commonHelper"

        # Generate the new skill code
        new_skill_code = f"""async function {new_skill_name}(bot) {{
    // Common logic extracted from: {', '.join(skill_names)}
{self._indent_code(common_block.code, 4)}
}}"""

        # Run dependency analysis and bot-parameter handling on the new skill code
        new_skill_code = self._finalize_new_skill_code(new_skill_code)

        # Generate the modified skills
        affected_skills = {}
        for name, node in skill_nodes.items():
            if name in common_block.source_skills:
                modified_code = self._replace_common_block(
                    node.code,
                    common_block.code,
                    f"await {new_skill_name}(bot);"
                )
                affected_skills[name] = modified_code

        return ExtractionPlan(
            common_block=common_block,
            new_skill_name=new_skill_name,
            new_skill_code=new_skill_code,
            new_skill_description=f"Common logic extracted from {', '.join(skill_names)}",
            affected_skills=affected_skills,
        )

    def _finalize_new_skill_code(self, code: str) -> str:
        """
        Apply final processing to newly extracted skill code.

        1. Ensure the `bot` parameter exists (prevents autoInjectBot from misaligning parameters)
        2. Analyze and inject missing dependencies (e.g. GoalPlaceBlock)

        Args:
            code: New skill code

        Returns:
            str: Processed code
        """
        # 1. Ensure the bot parameter exists
        code = ensure_bot_parameter(code)

        # 2. Analyze and inject missing dependencies (only inject INJECTABLE_DEPENDENCIES,
        # not those already provided by globalDepsCode)
        missing_deps = analyze_code_dependencies(code)
        if missing_deps:
            self._log(
                f"[SubskillExtraction] Injecting missing dependencies: {missing_deps}",
                "info"
            )
            code = inject_dependencies(code, missing_deps)

        return code

    def _create_plan_with_llm(
        self,
        common_block: CommonCodeBlock,
        skill_nodes: Dict[str, 'SkillNode'],
        retry_hint: Optional[str] = None,
    ) -> Optional[ExtractionPlan]:
        """Use the LLM to create the extraction plan.

        v3.H+ Layer 3: when retry_hint is set, append it as an extra
        HumanMessage so the LLM can correct an earlier validation failure.
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage
            import json

            from .prompts import PromptLoader

            skills_info = []
            for name, node in skill_nodes.items():
                skills_info.append(f"### {name}\n```javascript\n{node.code}\n```")

            template = PromptLoader.load("subskill_extraction")
            system_prompt, human_prompt = template.format(
                common_code=common_block.code,
                skills_info=''.join(skills_info),
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            if retry_hint:
                messages.append(HumanMessage(content=retry_hint))

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.subskill_extraction._create_plan_with_llm")
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                plan_data = json.loads(json_match.group())
                # Also run dependency analysis and bot-parameter handling on the new skill code generated by the LLM
                new_skill_code = self._finalize_new_skill_code(plan_data["new_skill_code"])
                return ExtractionPlan(
                    common_block=common_block,
                    new_skill_name=plan_data["new_skill_name"],
                    new_skill_code=new_skill_code,
                    new_skill_description=plan_data["new_skill_description"],
                    affected_skills=plan_data["affected_skills"],
                )

            return None

        except Exception as e:
            self._log(f"[SubskillExtraction] LLM planning failed: {e}", "warning")
            return self._create_plan_simple(common_block, skill_nodes)

    def _extract_common_prefix(self, names: List[str]) -> str:
        """Extract the common prefix of the names."""
        if not names:
            return ""

        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix) and prefix:
                prefix = prefix[:-1]

        # Clean the prefix (preserve complete camelCase words)
        if prefix:
            # Find the position before the last uppercase letter
            for i in range(len(prefix) - 1, -1, -1):
                if prefix[i].isupper():
                    prefix = prefix[:i]
                    break

        return prefix

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line for line in lines)

    def _replace_common_block(
        self,
        original_code: str,
        common_code: str,
        replacement: str
    ) -> str:
        """Replace the common code block within the original code."""
        # Normalize for comparison
        common_lines = [line.strip() for line in common_code.split('\n') if line.strip()]
        original_lines = original_code.split('\n')

        # Find the matching position
        start_idx = -1
        for i, line in enumerate(original_lines):
            if line.strip() == common_lines[0]:
                # Check subsequent lines
                match = True
                for j, common_line in enumerate(common_lines):
                    if i + j >= len(original_lines):
                        match = False
                        break
                    if original_lines[i + j].strip() != common_line:
                        match = False
                        break
                if match:
                    start_idx = i
                    break

        if start_idx >= 0:
            # Preserve indentation
            indent = len(original_lines[start_idx]) - len(original_lines[start_idx].lstrip())
            indent_str = ' ' * indent

            # Replace
            new_lines = (
                original_lines[:start_idx] +
                [indent_str + replacement] +
                original_lines[start_idx + len(common_lines):]
            )
            return '\n'.join(new_lines)

        return original_code

    def rollback(self, result: RefactorResult) -> bool:
        """Roll back the extraction operation."""
        if not result.rollback_available or not result.rollback_data:
            self._log(f"[SubskillExtraction] Cannot roll back: no rollback data", "warning")
            return False

        try:
            rollback_data = result.rollback_data

            # Restore the original skills' code
            for skill_name, data in rollback_data.get("skills", {}).items():
                if self.skill_graph_manager:
                    node = self.skill_graph_manager.get_node(skill_name)
                    if node:
                        node.code = data["code"]
                        self._log(f"[SubskillExtraction] Restored code for '{skill_name}'", "info")

            # Delete the newly created skill
            new_skill = result.target_skill
            if self.skill_graph_manager and new_skill:
                if self.skill_graph_manager.get_node(new_skill):
                    self.skill_graph_manager.remove_skill_node(new_skill)
                    self._log(f"[SubskillExtraction] Deleted '{new_skill}'", "info")

            # Roll back caller updates (if any)
            caller_rollback_data = rollback_data.get("caller_rollback_data")
            if caller_rollback_data:
                self.rollback_caller_updates(caller_rollback_data)

            # Persist the rolled-back state to disk
            self.skill_graph_manager.save()

            self._log(f"[SubskillExtraction] ✓ Rollback complete", "info")
            return True

        except Exception as e:
            self._log(f"[SubskillExtraction] Rollback failed: {e}", "error")
            return False

def detect_extraction_opportunities(
    skill_graph_manager,
    skill_names: Optional[List[str]] = None,
    min_common_lines: int = 3,
    min_similarity: float = 0.7,
    logger=None,
) -> List[RefactorOpportunity]:
    """
    Detect common-subskill extraction opportunities.

    Args:
        skill_graph_manager: Skill graph manager
        skill_names: List of skills to check (None means all)
        min_common_lines: Minimum number of common code lines
        min_similarity: Minimum similarity threshold
        logger: Logger

    Returns:
        List[RefactorOpportunity]: Detected extraction opportunities
    """
    opportunities = []

    if not skill_graph_manager:
        return opportunities

    # Determine which skills to check
    if skill_names is None:
        skill_names = skill_graph_manager.get_all_skill_names(include_task_specific=True)

    # Plan v3-rev Fix 1.B: filter out skills already covered by a prior refactor.
    # Re-extracting from a wrapper produces nothing useful (the wrapper IS the
    # common subskill call) and causes "Failed to update code" errors in
    # multi-round optimization loops.
    skill_names = [
        n for n in skill_names
        if not getattr(skill_graph_manager.get_node(n), 'is_covered', False)
    ]

    if len(skill_names) < 2:
        return opportunities

    # Use SubskillExtractionRefactor to find common blocks
    refactor = SubskillExtractionRefactor(
        skill_graph_manager=skill_graph_manager,
        logger=logger,
    )
    refactor.MIN_COMMON_LINES = min_common_lines
    refactor.MIN_SIMILARITY = min_similarity

    # Fetch nodes
    skill_nodes = {}
    for name in skill_names:
        node = skill_graph_manager.get_node(name)
        if node:
            skill_nodes[name] = node

    # Find common blocks
    common_blocks = refactor._find_common_code_blocks(skill_nodes)

    for block in common_blocks:
        # Ensure source_skills is non-empty
        if not block.source_skills:
            continue
        opportunities.append(RefactorOpportunity(
            refactor_type=RefactorType.EXTRACT_COMMON,
            source_skill=block.source_skills[0],
            target_skill="",  # New skill, not yet created
            reason=f"Found common code block ({len(block.code.split(chr(10)))} lines) "
                   f"in {len(block.source_skills)} skills",
            confidence=block.similarity_score,
            covered_skills=block.source_skills[1:],
        ))

    return opportunities
