"""
Duplication Refactor

Duplicate-code refactor strategy: when two skills' code is highly duplicated (>70% similarity),
merge them or extract the common parts into a single general skill.

For example:
- mineOakLogs() and mineBirchLogs() are 90% identical
  -> extract the common code into mineLogs(type)
  -> the two original skills become wrappers that call mineLogs

Difference from PARAMETRIC:
- PARAMETRIC: a general skill already exists; we only need to make the specific call it
- DUPLICATION: we must first create the general skill, then make both duplicates call it

Difference from SIBLING:
- DUPLICATION: focuses on code duplication; needs merging
- SIBLING: focuses on a functional family; code may not be fully identical but belongs to the same category
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import difflib

from .base import (
    RefactorType,
    RefactorOpportunity,
    RefactorResult,
    SkillRefactor,
)
from ..skill_graph.models.coverage import CoverageType
from ..skill_graph.models.node import SkillNode
from ..utils import validate_code_syntax
from skillnet.utils.stats_tracker import record_llm_usage


@dataclass
class DuplicationAnalysis:
    """Duplicate-code analysis result"""
    skill_a: str
    skill_b: str
    similarity_ratio: float           # similarity (0-1)
    common_lines: List[str]           # common code lines
    diff_lines_a: List[str]           # lines unique to A
    diff_lines_b: List[str]           # lines unique to B
    suggested_general_name: str       # suggested general-skill name
    parameter_candidates: List[str]   # possible parameterization candidates


@dataclass
class DuplicationPlan:
    """Duplicate-code refactor plan"""
    general_skill_name: str
    general_skill_code: str
    general_skill_description: str
    wrapper_a_code: str               # wrapper code for skill A
    wrapper_b_code: str               # wrapper code for skill B
    parameter_mapping_a: Dict[str, str]  # parameter mapping from A to general
    parameter_mapping_b: Dict[str, str]  # parameter mapping from B to general


class DuplicationRefactor(SkillRefactor):
    """
    Duplicate-code refactor.

    When two skills' code is highly duplicated, create a general skill and have both call it.

    Refactor flow:
    1. Analyze the code similarity of the two skills
    2. Identify the common code and the differences
    3. Generate the general skill (parameterize the differences)
    4. Convert the original skills into wrappers that call the general skill
    5. Add dependency edges
    """

    # Minimum similarity threshold
    MIN_SIMILARITY = 0.7

    def apply(self, opportunity: RefactorOpportunity) -> RefactorResult:
        """
        Apply the duplicate-code refactor

        Args:
            opportunity: refactor opportunity
                - source_skill: the first duplicate skill
                - target_skill: the second duplicate skill
                - or covered_skills contains all duplicate skills

        Returns:
            RefactorResult: refactor result
        """
        if opportunity.refactor_type != RefactorType.DUPLICATION:
            return RefactorResult(
                success=False,
                refactor_type=opportunity.refactor_type,
                source_skill=opportunity.source_skill,
                target_skill=opportunity.target_skill,
                error_message="Wrong refactor type for DuplicationRefactor"
            )

        # Gather all duplicate skills
        skills_to_merge = [opportunity.source_skill, opportunity.target_skill]
        if opportunity.covered_skills:
            skills_to_merge.extend(
                s for s in opportunity.covered_skills
                if s not in skills_to_merge
            )

        self._log(
            f"[Duplication] Starting duplicate-code refactor: {skills_to_merge}",
            "info"
        )

        if not self.skill_graph_manager:
            return self._create_failed_result(
                opportunity, "No skill_graph_manager available"
            )

        # Fetch the skill nodes
        skill_nodes = {}
        for name in skills_to_merge:
            node = self.skill_graph_manager.get_node(name)
            if node:
                skill_nodes[name] = node

        if len(skill_nodes) < 2:
            return self._create_failed_result(
                opportunity, "Need at least 2 skills for duplication refactor"
            )

        # Check whether any code is empty
        for name, node in skill_nodes.items():
            if not node.code:
                return self._create_failed_result(
                    opportunity, f"Skill '{name}' has empty code"
                )

        # Save rollback data
        rollback_data = self._save_multi_skill_rollback(skill_nodes)

        try:
            # Analyze the duplicate code
            analysis = self._analyze_duplication(skill_nodes)

            if not analysis or analysis.similarity_ratio < self.MIN_SIMILARITY:
                return self._create_failed_result(
                    opportunity,
                    f"Similarity too low: {analysis.similarity_ratio if analysis else 0:.2f}"
                )

            # Generate the refactor plan (with response capture for forensics + retry)
            from ._retry_helpers import (
                capture_llm_responses,
                save_refactor_failure_forensics,
                build_retry_message,
            )

            with capture_llm_responses(self) as cap_first:
                plan = self._create_duplication_plan(analysis, skill_nodes)
            first_raw = cap_first.raw_responses[-1] if cap_first.raw_responses else ""
            first_messages = cap_first.last_messages

            if not plan:
                return self._create_failed_result(
                    opportunity, "Failed to create duplication plan"
                )

            # [Fix] Validate the syntax of general_skill_code
            is_valid, syntax_error = validate_code_syntax(plan.general_skill_code)

            # v3.H+ Layer 3 — single-shot retry + forensics on validation failure
            if not is_valid:
                save_refactor_failure_forensics(
                    skill_graph_manager=self.skill_graph_manager,
                    refactor_type="duplication",
                    skill_names=list(skill_nodes.keys()),
                    attempt=1,
                    raw_response=first_raw,
                    babel_error=syntax_error or "",
                    prompt_messages=first_messages,
                    logger=getattr(self, "logger", None),
                )
                self._log(
                    f"[Duplication] ⟳ Layer 3 retry — first attempt failed validation: "
                    f"{(syntax_error or '')[:120]}",
                    "warning",
                )
                with capture_llm_responses(self) as cap_retry:
                    plan2 = self._create_plan_with_llm(
                        analysis, skill_nodes,
                        retry_hint=build_retry_message(syntax_error or ""),
                    )
                retry_raw = cap_retry.raw_responses[-1] if cap_retry.raw_responses else ""
                if plan2 is not None:
                    plan = plan2
                    is_valid, syntax_error = validate_code_syntax(plan.general_skill_code)
                    if is_valid:
                        self._log("[Duplication] ✓ Layer 3 retry succeeded", "info")
                    else:
                        save_refactor_failure_forensics(
                            skill_graph_manager=self.skill_graph_manager,
                            refactor_type="duplication",
                            skill_names=list(skill_nodes.keys()),
                            attempt=2,
                            raw_response=retry_raw,
                            babel_error=syntax_error or "",
                            prompt_messages=cap_retry.last_messages,
                            logger=getattr(self, "logger", None),
                        )

            if not is_valid:
                self._log(
                    f"[Duplication] ✗ General-skill code failed syntax validation: {syntax_error}",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Syntax error in general skill code: {syntax_error}"
                )

            # Create the general skill
            general_name = plan.general_skill_name
            general_node = SkillNode(
                name=general_name,
                code=plan.general_skill_code,
                description=plan.general_skill_description,
            )

            # v3-rev Fix 2A (extended to duplication, v3.H+):
            # add_skill_node is a bare graph insert — it does NOT invoke
            # parameter extraction, so the new general skill would be born
            # with parameters={} and planner would later be unable to map
            # task target_effects to its options object. Mirror sibling.py
            # and subskill_extraction.py: extract parameters before insert.
            # Wrapped in try/except so refactor still proceeds if extraction
            # fails (extraction failure is a separate diagnostic problem,
            # should not block the wrapper-conversion architectural win).
            try:
                extracted_params = self.skill_graph_manager._extract_parameters(
                    general_node.code,
                    general_node.description,
                    general_name,
                )
                if extracted_params:
                    general_node.parameters = extracted_params
                    self._log(
                        f"[Duplication] Fix2A: extracted {len(extracted_params)} "
                        f"parameter schemas for general skill '{general_name}': "
                        f"{list(extracted_params.keys())}",
                        "info",
                    )
            except Exception as e:
                self._log(
                    f"[Duplication] Fix2A: parameter extraction failed for "
                    f"'{general_name}' (refactor will still proceed): {e!r}",
                    "warning",
                )

            self.skill_graph_manager.add_skill_node(general_node)

            changes_made = [f"Created general skill: {general_name}"]

            # Update the original skills to be wrappers
            skill_names = list(skill_nodes.keys())
            wrapper_codes = [plan.wrapper_a_code, plan.wrapper_b_code]

            for i, (name, node) in enumerate(skill_nodes.items()):
                if i < len(wrapper_codes):
                    wrapper_code = wrapper_codes[i]

                    # Validate wrapper-code syntax
                    is_valid, syntax_error = validate_code_syntax(wrapper_code)
                    if not is_valid:
                        self._log(
                            f"[Duplication] ✗ JS syntax validation failed (skill: {name}): {syntax_error}",
                            "error"
                        )
                        return self._create_failed_result(
                            opportunity,
                            f"Syntax error in wrapper code for {name}: {syntax_error}"
                        )

                    # Use the unified interface to update code (syntax already validated, skip duplicate validation)
                    update_success = self.skill_graph_manager.update_skill_code(
                        skill_name=name,
                        new_code=wrapper_code,
                        change_log=f"Converted to wrapper calling {general_name}",
                        source="refactor:duplication",
                        skip_validation=True,  # syntax already validated
                        skip_metadata=True,    # Refactor does not update metadata
                        skip_interface_check=True,  # do not trigger cascade
                        create_version=True,
                    )
                    if not update_success:
                        return self._create_failed_result(
                            opportunity,
                            f"Failed to update code for {name}"
                        )

                    # Mark as covered
                    node.is_covered = True
                    node.covered_by = general_name
                    node.coverage_type = CoverageType.DUPLICATION

                    # Add dependency edge
                    self.skill_graph_manager.add_edge(name, general_name)

                    changes_made.append(f"Converted {name} to wrapper")

            rollback_data["general_skill_name"] = general_name

            self._log(
                f"[Duplication] ✓ Refactor succeeded; created {general_name}",
                "info"
            )

            # Propagate changes to callers
            updated_callers = []
            caller_update_details = {}
            all_caller_rollback_data = {}  # aggregate all callers' rollback data

            for name in skill_nodes.keys():
                propagation_result = self.propagate_to_callers(
                    refactored_skill=name,
                    changes={
                        "change_type": "general_created",
                        "new_skill_name": general_name,
                    },
                    auto_update=True,  # enable auto-update of callers to avoid broken references
                )

                if propagation_result["needs_update"]:
                    for caller in propagation_result["needs_update"]:
                        if caller not in updated_callers:
                            updated_callers.append(caller)
                            caller_update_details[caller] = (
                                f"Could use {general_name} instead of {name}"
                            )

                # Aggregate caller rollback data
                if propagation_result.get("rollback_data"):
                    all_caller_rollback_data.update(propagation_result["rollback_data"])

            # Save caller rollback data (if any)
            if all_caller_rollback_data:
                rollback_data["caller_rollback_data"] = all_caller_rollback_data

            if updated_callers:
                changes_made.append(
                    f"Detected {len(updated_callers)} callers that could use {general_name}"
                )

            return RefactorResult(
                success=True,
                refactor_type=RefactorType.DUPLICATION,
                source_skill=opportunity.source_skill,
                target_skill=general_name,
                old_code=None,
                new_code=plan.general_skill_code,
                changes_made=changes_made,
                rollback_available=True,
                rollback_data=rollback_data,
                updated_callers=updated_callers,
                caller_update_details=caller_update_details,
            )

        except Exception as e:
            self._log(f"[Duplication] Refactor failed: {e}", "error")
            return self._create_failed_result(opportunity, str(e))

    def _analyze_duplication(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[DuplicationAnalysis]:
        """
        Analyze code duplication.

        Prefers semantic LLM analysis; falls back to rules if it fails.

        Args:
            skill_nodes: dict of skill nodes

        Returns:
            DuplicationAnalysis: analysis result
        """
        names = list(skill_nodes.keys())
        if len(names) < 2:
            return None

        # Take the first two skills for analysis
        skill_a = names[0]
        skill_b = names[1]
        node_a = skill_nodes[skill_a]
        node_b = skill_nodes[skill_b]

        # Check whether the code is empty
        code_a = node_a.code
        code_b = node_b.code
        if not code_a or not code_a.strip():
            self._log(f"[Duplication] Skill '{skill_a}' code is empty or whitespace-only", "warning")
            return None
        if not code_b or not code_b.strip():
            self._log(f"[Duplication] Skill '{skill_b}' code is empty or whitespace-only", "warning")
            return None

        # Prefer LLM semantic analysis
        if self.llm:
            result = self._analyze_duplication_with_llm(
                skill_a, skill_b, node_a, node_b
            )
            if result:
                return result
            self._log("[Duplication] LLM analysis failed; falling back to rules", "warning")

        return self._analyze_duplication_simple(skill_a, skill_b, node_a, node_b)

    def _analyze_duplication_with_llm(
        self,
        skill_a: str,
        skill_b: str,
        node_a: 'SkillNode',
        node_b: 'SkillNode'
    ) -> Optional[DuplicationAnalysis]:
        """
        Use the LLM for semantic duplicate analysis.

        Can identify:
        - Code that has different variable names but the same logic
        - Syntactic variants (obj['key'] vs obj.key)
        - Independent operations in a different order
        """
        try:
            import json
            from langchain.schema import HumanMessage, SystemMessage
            from .prompts import PromptLoader

            template = PromptLoader.load("duplication_analysis")
            system_prompt, human_prompt = template.format(
                skill_a=skill_a,
                skill_b=skill_b,
                code_a=node_a.code,
                code_b=node_b.code,
                desc_a=node_a.description or "No description",
                desc_b=node_b.description or "No description",
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.duplication._analyze_duplication_with_llm", skill_name=skill_a)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse the JSON response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                # Validate required fields
                if not data.get("is_duplicate", False):
                    self._log(
                        f"[Duplication] LLM judged {skill_a} and {skill_b} are not semantic duplicates",
                        "info"
                    )
                    # Return the analysis result even when not duplicates, for downstream decisions
                    similarity = data.get("similarity", 0.0)
                else:
                    similarity = data.get("similarity", 0.8)

                self._log(
                    f"[Duplication] LLM semantic analysis: {skill_a} vs {skill_b}, "
                    f"similarity={similarity:.2f}",
                    "info"
                )

                # Use difflib for detailed line-level info (LLM analysis + rule supplement)
                lines_a = node_a.code.split('\n')
                lines_b = node_b.code.split('\n')
                matcher = difflib.SequenceMatcher(None, lines_a, lines_b)

                common_lines = []
                diff_lines_a = []
                diff_lines_b = []

                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        common_lines.extend(lines_a[i1:i2])
                    elif tag == 'replace':
                        diff_lines_a.extend(lines_a[i1:i2])
                        diff_lines_b.extend(lines_b[j1:j2])
                    elif tag == 'delete':
                        diff_lines_a.extend(lines_a[i1:i2])
                    elif tag == 'insert':
                        diff_lines_b.extend(lines_b[j1:j2])

                return DuplicationAnalysis(
                    skill_a=skill_a,
                    skill_b=skill_b,
                    similarity_ratio=similarity,
                    common_lines=common_lines,
                    diff_lines_a=diff_lines_a,
                    diff_lines_b=diff_lines_b,
                    suggested_general_name=data.get(
                        "suggested_general_name",
                        self._suggest_general_name(skill_a, skill_b)
                    ),
                    parameter_candidates=data.get("parameter_candidates", []),
                )

            return None

        except Exception as e:
            self._log(f"[Duplication] LLM analysis exception: {e}", "warning")
            return None

    def _analyze_duplication_simple(
        self,
        skill_a: str,
        skill_b: str,
        node_a: 'SkillNode',
        node_b: 'SkillNode'
    ) -> DuplicationAnalysis:
        """
        Rule-based duplicate-code analysis (simple version).

        Uses difflib string similarity and line-level comparison.
        """
        code_a = node_a.code
        code_b = node_b.code

        # Compute similarity with difflib
        similarity = difflib.SequenceMatcher(None, code_a, code_b).ratio()

        # Get common and differing lines
        lines_a = code_a.split('\n')
        lines_b = code_b.split('\n')

        matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
        common_lines = []
        diff_lines_a = []
        diff_lines_b = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                common_lines.extend(lines_a[i1:i2])
            elif tag == 'replace':
                diff_lines_a.extend(lines_a[i1:i2])
                diff_lines_b.extend(lines_b[j1:j2])
            elif tag == 'delete':
                diff_lines_a.extend(lines_a[i1:i2])
            elif tag == 'insert':
                diff_lines_b.extend(lines_b[j1:j2])

        # Infer a general-skill name
        suggested_name = self._suggest_general_name(skill_a, skill_b)

        # Identify parameterization candidates
        param_candidates = self._identify_parameter_candidates(
            diff_lines_a, diff_lines_b
        )

        return DuplicationAnalysis(
            skill_a=skill_a,
            skill_b=skill_b,
            similarity_ratio=similarity,
            common_lines=common_lines,
            diff_lines_a=diff_lines_a,
            diff_lines_b=diff_lines_b,
            suggested_general_name=suggested_name,
            parameter_candidates=param_candidates,
        )

    def _suggest_general_name(self, name_a: str, name_b: str) -> str:
        """Infer the general-skill name"""
        # Find common prefix
        prefix = ""
        for i, (c1, c2) in enumerate(zip(name_a, name_b)):
            if c1 == c2:
                prefix += c1
            else:
                break

        # Trim the prefix (keep whole words)
        if prefix:
            # Find the last uppercase letter
            for i in range(len(prefix) - 1, -1, -1):
                if prefix[i].isupper() and i > 0:
                    prefix = prefix[:i]
                    break

        if prefix and len(prefix) >= 3:
            return f"{prefix}General"

        # Extract a verb from the names
        verbs = ['mine', 'craft', 'smelt', 'collect', 'find', 'get', 'make']
        for verb in verbs:
            if name_a.lower().startswith(verb) or name_b.lower().startswith(verb):
                return f"{verb}Item"

        return "generalHelper"

    def _identify_parameter_candidates(
        self,
        diff_a: List[str],
        diff_b: List[str]
    ) -> List[str]:
        """Identify possible parameterization candidates"""
        candidates = []

        # Simple heuristic: look for differing string literals
        pattern = r'"([^"]+)"'

        strings_a = set()
        strings_b = set()

        for line in diff_a:
            strings_a.update(re.findall(pattern, line))
        for line in diff_b:
            strings_b.update(re.findall(pattern, line))

        # Find the differing strings
        diff_strings = strings_a.symmetric_difference(strings_b)
        candidates.extend(diff_strings)

        return list(candidates)[:5]  # cap the count

    def _create_duplication_plan(
        self,
        analysis: DuplicationAnalysis,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[DuplicationPlan]:
        """Create the duplicate-code refactor plan"""
        if self.llm:
            return self._create_plan_with_llm(analysis, skill_nodes)
        else:
            return self._create_plan_simple(analysis, skill_nodes)

    def _create_plan_simple(
        self,
        analysis: DuplicationAnalysis,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> DuplicationPlan:
        """Create a simple refactor plan (no LLM)"""
        general_name = analysis.suggested_general_name
        node_a = skill_nodes[analysis.skill_a]
        node_b = skill_nodes[analysis.skill_b]

        # Identify parameters
        param_name = "itemType"
        if analysis.parameter_candidates:
            # Infer the parameter name from a candidate
            first_candidate = analysis.parameter_candidates[0]
            if 'log' in first_candidate.lower():
                param_name = "logType"
            elif 'ore' in first_candidate.lower():
                param_name = "oreType"
            elif 'wood' in first_candidate.lower():
                param_name = "woodType"

        # Generate the general-skill code
        # Use skill A's code as the base and parameterize the differences
        general_code = node_a.code

        # Simple substitution: replace A's unique strings with the parameter
        if analysis.parameter_candidates:
            for candidate in analysis.parameter_candidates[:1]:
                general_code = general_code.replace(
                    f'"{candidate}"',
                    param_name
                )

        # Modify the function signature
        sig_match = re.search(
            rf'async\s+function\s+{re.escape(analysis.skill_a)}\s*\(([^)]*)\)',
            general_code
        )
        if sig_match:
            old_sig = sig_match.group(0)
            params = sig_match.group(1)
            if params.strip():
                new_params = f"{params}, {param_name} = undefined"
            else:
                new_params = f"bot, {param_name} = undefined"
            new_sig = f'async function {general_name}({new_params})'
            general_code = general_code.replace(old_sig, new_sig)

        # Generate wrapper code
        wrapper_a = self._generate_wrapper(
            analysis.skill_a,
            general_name,
            param_name,
            analysis.parameter_candidates[0] if analysis.parameter_candidates else "undefined"
        )

        # Safely get the parameter value: the second wrapper uses the second candidate, or undefined if absent
        param_b = (
            analysis.parameter_candidates[1]
            if len(analysis.parameter_candidates) > 1
            else "undefined"
        )
        wrapper_b = self._generate_wrapper(
            analysis.skill_b,
            general_name,
            param_name,
            param_b
        )

        return DuplicationPlan(
            general_skill_name=general_name,
            general_skill_code=general_code,
            general_skill_description=f"General skill extracted from {analysis.skill_a} and {analysis.skill_b}",
            wrapper_a_code=wrapper_a,
            wrapper_b_code=wrapper_b,
            parameter_mapping_a={param_name: analysis.parameter_candidates[0] if analysis.parameter_candidates else "undefined"},
            parameter_mapping_b={param_name: analysis.parameter_candidates[1] if len(analysis.parameter_candidates) > 1 else "undefined"},
        )

    def _generate_wrapper(
        self,
        skill_name: str,
        general_name: str,
        param_name: str,
        param_value: str
    ) -> str:
        """Generate wrapper code"""
        return f"""async function {skill_name}(bot) {{
    // Wrapper: calls {general_name} with specific {param_name}
    return await {general_name}(bot, "{param_value}");
}}"""

    def _create_plan_with_llm(
        self,
        analysis: DuplicationAnalysis,
        skill_nodes: Dict[str, 'SkillNode'],
        retry_hint: Optional[str] = None,
    ) -> Optional[DuplicationPlan]:
        """Create a refactor plan with the LLM.

        v3.H+ Layer 3: when retry_hint is set, append it as an extra
        HumanMessage so the LLM can correct an earlier validation failure.
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage
            import json

            from .prompts import PromptLoader

            node_a = skill_nodes[analysis.skill_a]
            node_b = skill_nodes[analysis.skill_b]

            template = PromptLoader.load("duplication_refactor")
            system_prompt, human_prompt = template.format(
                skill_a_code=node_a.code,
                skill_b_code=node_b.code,
                similarity_ratio=f"{analysis.similarity_ratio:.2f}",
                parameter_candidates=analysis.parameter_candidates,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            if retry_hint:
                messages.append(HumanMessage(content=retry_hint))

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.duplication._create_plan_with_llm", skill_name=analysis.skill_a)
            content = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    # Validate required fields exist
                    required_fields = ["general_skill_name", "general_skill_code",
                                       "general_skill_description", "wrapper_a_code", "wrapper_b_code"]
                    missing = [f for f in required_fields if f not in data]
                    if missing:
                        self._log(f"[Duplication] LLM response missing required fields: {missing}", "warning")
                    else:
                        return DuplicationPlan(
                            general_skill_name=data["general_skill_name"],
                            general_skill_code=data["general_skill_code"],
                            general_skill_description=data["general_skill_description"],
                            wrapper_a_code=data["wrapper_a_code"],
                            wrapper_b_code=data["wrapper_b_code"],
                            parameter_mapping_a=data.get("parameter_mapping_a", {}),
                            parameter_mapping_b=data.get("parameter_mapping_b", {}),
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    self._log(f"[Duplication] Failed to parse LLM response: {e}", "warning")
            else:
                self._log("[Duplication] No valid JSON found in LLM response", "warning")

        except Exception as e:
            self._log(f"[Duplication] LLM planning failed: {e}", "warning")

        return self._create_plan_simple(analysis, skill_nodes)

    def rollback(self, result: RefactorResult) -> bool:
        """Roll back the duplicate-code refactor"""
        if not result.rollback_available or not result.rollback_data:
            self._log(f"[Duplication] Cannot roll back: no rollback data", "warning")
            return False

        try:
            rollback_data = result.rollback_data

            # Restore the original skills
            for skill_name, data in rollback_data.get("skills", {}).items():
                if self.skill_graph_manager:
                    node = self.skill_graph_manager.get_node(skill_name)
                    if node:
                        node.code = data["code"]
                        node.is_covered = data.get("is_covered", False)
                        node.covered_by = data.get("covered_by")
                        node.coverage_type = data.get("coverage_type")
                        self._log(f"[Duplication] Restored '{skill_name}'", "info")

            # Remove the created general skill
            general_name = rollback_data.get("general_skill_name")
            if general_name and self.skill_graph_manager:
                if self.skill_graph_manager.get_node(general_name):
                    self.skill_graph_manager.remove_skill_node(general_name)
                    self._log(f"[Duplication] Removed '{general_name}'", "info")

            # Roll back caller updates (if any)
            caller_rollback_data = rollback_data.get("caller_rollback_data")
            if caller_rollback_data:
                self.rollback_caller_updates(caller_rollback_data)

            self._log(f"[Duplication] ✓ Rollback complete", "info")
            return True

        except Exception as e:
            self._log(f"[Duplication] Rollback failed: {e}", "error")
            return False

def detect_duplication_opportunities(
    skill_graph_manager,
    skill_names: Optional[List[str]] = None,
    min_similarity: float = 0.7,
    logger=None,
) -> List[RefactorOpportunity]:
    """
    Detect duplicate-code refactor opportunities

    Args:
        skill_graph_manager: skill graph manager
        skill_names: list of skills to check (None means all)
        min_similarity: minimum similarity threshold
        logger: logger

    Returns:
        List[RefactorOpportunity]: detected duplicate-code opportunities
    """
    opportunities = []

    if not skill_graph_manager:
        return opportunities

    # Determine which skills to check
    if skill_names is None:
        skill_names = skill_graph_manager.get_all_skill_names(include_task_specific=True)

    # Plan v3-rev Fix 1.B: filter out skills already covered by a prior refactor.
    skill_names = [
        n for n in skill_names
        if not getattr(skill_graph_manager.get_node(n), 'is_covered', False)
    ]

    if len(skill_names) < 2:
        return opportunities

    # Pairwise comparison
    checked_pairs = set()

    for i, name_a in enumerate(skill_names):
        for name_b in skill_names[i+1:]:
            pair_key = tuple(sorted([name_a, name_b]))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            node_a = skill_graph_manager.get_node(name_a)
            node_b = skill_graph_manager.get_node(name_b)

            if not node_a or not node_b:
                continue

            # Check whether either code is empty
            if not node_a.code or not node_b.code:
                continue

            # Compute similarity
            similarity = difflib.SequenceMatcher(
                None, node_a.code, node_b.code
            ).ratio()

            if similarity >= min_similarity:
                opportunities.append(RefactorOpportunity(
                    refactor_type=RefactorType.DUPLICATION,
                    source_skill=name_a,
                    target_skill=name_b,
                    reason=f"Code similarity: {similarity:.1%}",
                    confidence=similarity,
                ))

    return opportunities
