"""
Refactor Relationship Analyzer

Methods migrated from graph_manager_impl.py:
- _detect_refactor_relationships
- _validate_refactor_direction
- _estimate_functional_scope
- _estimate_implementation_maturity
- _estimate_parameterization_level
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from skillnet.utils.stats_tracker import record_llm_usage

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillNode, SkillPrecondition, SkillEffect

logger = logging.getLogger(__name__)


# ============================================================================
# Pure functions - dimension estimation
# ============================================================================

def estimate_functional_scope(
    code: str,
    effects: Optional[List] = None,
    node: Optional["SkillNode"] = None
) -> int:
    """
    Estimate a skill's functional scope.

    Measures how many "different things" the skill does, used to judge
    behavioral/subgraph relationships. A skill with larger functional scope
    should **call** skills with smaller functional scope.

    Metrics:
    1. Number of produced item types
    2. Distinct operation types (mining, crafting, placing, exploring, etc.)
    3. Number of sub-skills

    Args:
        code: skill code
        effects: list of effects (optional)
        node: SkillNode instance (optional, used to obtain effects)

    Returns:
        int: functional-scope score
    """
    scope = 0

    # 1. Count produced item types from effects
    unique_outputs = set()
    if effects:
        for effect in effects:
            desc = getattr(effect, 'description', '') if hasattr(effect, 'description') else str(effect)
            # Extract item name (e.g. "Adds 4 oak_planks to inventory" -> oak_planks)
            item_match = re.search(r'Adds?\s+\d+\s+(\w+)', desc, re.IGNORECASE)
            if item_match:
                unique_outputs.add(item_match.group(1))
            # Check affected_items in the effect structure
            if hasattr(effect, 'affected_items'):
                for item in effect.affected_items:
                    unique_outputs.add(item)
    elif node and hasattr(node, 'expected_effects') and node.expected_effects:
        for effect in node.expected_effects:
            desc = getattr(effect, 'description', '') if hasattr(effect, 'description') else str(effect)
            item_match = re.search(r'Adds?\s+\d+\s+(\w+)', desc, re.IGNORECASE)
            if item_match:
                unique_outputs.add(item_match.group(1))
            if hasattr(effect, 'affected_items'):
                for item in effect.affected_items:
                    unique_outputs.add(item)
    scope += len(unique_outputs)

    # 2. Count operation types based on code (no duplicate counting)
    operations = set()
    if code:
        # Mining operations
        if re.search(r'\bmineBlock\b|\bbot\.dig\b|\bmine\w*\(', code, re.IGNORECASE):
            operations.add("mining")
        # Crafting operations
        if re.search(r'\bbot\.craft\b|\bcraftItem\b|\bcraft\w*\(', code, re.IGNORECASE):
            operations.add("crafting")
        # Placing operations
        if re.search(r'\bbot\.place\b|\bplaceBlock\b|\bplace\w*Block\b', code, re.IGNORECASE):
            operations.add("placing")
        # Exploring operations
        if re.search(r'\bexploreUntil\b|\bexplore\w*\(', code, re.IGNORECASE):
            operations.add("exploring")
        # Smelting operations
        if re.search(r'\bsmelt\w*\(|\bbot\.activateEntity', code, re.IGNORECASE):
            operations.add("smelting")
        # Equipping operations
        if re.search(r'\bbot\.equip\b|\bequip\w*\(', code, re.IGNORECASE):
            operations.add("equipping")
        # Collecting operations (e.g. picking up drops)
        if re.search(r'\bcollect\w*\(', code, re.IGNORECASE):
            operations.add("collecting")
    scope += len(operations) * 2  # Operation types weighted higher

    # 3. Count sub-skill calls (await calls)
    if code:
        await_calls = re.findall(r'\bawait\s+(\w+)\s*\(', code)
        # Exclude bot methods
        child_calls = [c for c in await_calls if not c.startswith('bot')]
        scope += len(set(child_calls))

    return scope


def estimate_implementation_maturity(code: str) -> int:
    """
    Estimate a skill's implementation maturity.

    Measures how "robust" the implementation of the same function is — used
    for judging functional_superset relationships. Skills with higher maturity
    should **cover** lower-maturity skills with the same function.

    Metrics:
    1. Input validation (parameter checks)
    2. Precondition checks (material/tool checks)
    3. Error handling (try-catch)
    4. Retry logic
    5. Fallback logic
    6. Progress reporting
    7. Result verification

    Args:
        code: skill code

    Returns:
        int: implementation-maturity score
    """
    if not code:
        return 0

    maturity = 0

    # 1. Input validation (parameter checks)
    if re.search(r'if\s*\(\s*!?\s*(bot|count|type|name)\s*(===|!==|==|!=|<|>|<=|>=)', code):
        maturity += 1
    if re.search(r'typeof\s+(count|type|name)', code):
        maturity += 1
    if re.search(r'(count|amount)\s*(<=?|>=?|===?)\s*(0|1)\s*\)', code):
        maturity += 1

    # 2. Precondition checks (materials/tools)
    inventory_checks = len(re.findall(r'inventory\.count\(|inventory\.items\(|bot\.inventory', code))
    maturity += min(inventory_checks, 2)  # Cap at 2 points

    # Check whether tools/materials are sufficient
    if re.search(r'(has|have|need|require)\w*\s*(material|resource|tool|item)', code, re.IGNORECASE):
        maturity += 1

    # 3. Error handling (try-catch)
    try_catch_count = len(re.findall(r'\btry\s*\{', code))
    maturity += min(try_catch_count * 2, 4)  # Cap at 4 points

    # 4. Retry logic
    if re.search(r'\bretry\b|\battempt\b|\bfor\s*\(\s*let\s+\w+\s*=\s*0', code, re.IGNORECASE):
        maturity += 2
    if re.search(r'while\s*\([^)]*<\s*\d+[^)]*attempt', code, re.IGNORECASE):
        maturity += 1

    # 5. Fallback logic (fallback/alternative)
    if re.search(r'\bfallback\b|\balternative\b', code, re.IGNORECASE):
        maturity += 2
    # else blocks also count as a kind of fallback
    else_count = len(re.findall(r'\}\s*else\s*\{', code))
    maturity += min(else_count, 2)  # Cap at 2 points

    # 6. Progress reporting (bot.chat)
    chat_count = len(re.findall(r'bot\.chat\(', code))
    if chat_count > 0:
        maturity += 1
    if chat_count > 3:
        maturity += 1  # More reports indicates more detailed output

    # 7. Result verification (whether the operation succeeded)
    if re.search(r'return\s+(true|false|success)', code, re.IGNORECASE):
        maturity += 1
    if re.search(r'(verify|check|confirm|ensure)\w*\s*\(', code, re.IGNORECASE):
        maturity += 1

    return maturity


def estimate_parameterization_level(code: str, metadata: Optional[Dict] = None) -> int:
    """
    Estimate a skill's parameterization level.

    Used to judge parametric relationships. Skills with higher
    parameterization should **cover** lower-parameterized specialized
    versions.

    Metrics:
    1. Number of function parameters (excluding bot)
    2. Whether parameters have defaults
    3. Whether type parameters exist (e.g. logType, oreName)

    Args:
        code: skill code
        metadata: metadata dict (optional, includes parameters info)

    Returns:
        int: parameterization-level score
    """
    level = 0

    # 1. Extract parameters from the function signature
    func_match = re.search(r'async\s+function\s+\w+\s*\(([^)]*)\)', code)
    if func_match:
        params_str = func_match.group(1)
        # Split parameters
        params = [p.strip() for p in params_str.split(',') if p.strip()]
        # Exclude bot parameter
        non_bot_params = [p for p in params if not p.startswith('bot')]
        level += len(non_bot_params)

        # Check for defaults
        params_with_default = len(re.findall(r'=\s*["\'\w]+', params_str))
        level += params_with_default

    # 2. Check for type parameters
    type_param_patterns = [
        r'\b(type|Type)\b',  # Generic type parameters
        r'\b(logType|oreType|woodType|stoneType|plankType)\b',  # Specific type parameters
        r'\b(itemName|blockName|toolName)\b',  # Name parameters
    ]
    for pattern in type_param_patterns:
        if re.search(pattern, code):
            level += 2  # Type parameters weighted higher

    # 3. Pull parameter info from metadata
    if metadata and 'parameters' in metadata:
        params = metadata.get('parameters', {})
        if isinstance(params, dict):
            level += len(params)

    return level


# ============================================================================
# RefactorRelationshipAnalyzer class
# ============================================================================

class RefactorRelationshipAnalyzer:
    """
    Refactor relationship analyzer.

    Responsibilities:
    1. Use the LLM to detect refactor relationship types.
    2. Validate that the refactor direction is correct.
    """

    def __init__(
        self,
        graph,  # SkillGraph
        llm,  # LangChain LLM
        prescreener: Optional["RefactorPrescreener"] = None,
        custom_logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the relationship analyzer.

        Args:
            graph: SkillGraph or SkillGraphManager instance (duck typing)
            llm: LangChain LLM instance
            prescreener: RefactorPrescreener instance (used for prescreening)
            custom_logger: optional custom logger
        """
        self.graph = graph
        self.llm = llm
        self.prescreener = prescreener
        self.logger = custom_logger or logger

    def detect_relationships(
        self,
        new_skill_name: str,
        new_skill_code: str,
        new_skill_description: str,
        new_skill_params: Dict[str, Dict[str, Any]],
        new_skill_preconditions: List["SkillPrecondition"],
        new_skill_effects: List["SkillEffect"],
        candidate_skills: List[Tuple[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect and classify refactor relationships using the LLM (two-step detection).

        Step 1: prescreen candidate skills using metadata.
        Step 2: run a detailed analysis on the prescreened candidates using full code.

        Args:
            new_skill_name: name of the new skill
            new_skill_code: code of the new skill
            new_skill_description: description of the new skill
            new_skill_params: parameter metadata of the new skill
            new_skill_preconditions: preconditions of the new skill
            new_skill_effects: effects of the new skill
            candidate_skills: list of candidate skills [(name, similarity), ...]

        Returns:
            Optional[Dict]: refactor info, format:
            {
                "refactor_type": "parametric" | "behavioral" | "sibling" | "duplication",
                "general_skill": "skill_name",
                "covered_skills": ["skill1", "skill2"],
                "parameter_mapping": {...},
                "reason": "explanation"
            }
            Returns None if no refactorable relationship is detected.
        """
        if not candidate_skills:
            return None

        try:
            from langchain.schema import HumanMessage, SystemMessage

            # Get the new skill's children (skeleton)
            new_skill_node = self.graph.get_node(new_skill_name)
            new_skill_children = new_skill_node.children if new_skill_node else []

            # Step 1: prescreen candidate skills using metadata
            if self.prescreener:
                prescreened_candidates = self.prescreener.prescreen_candidates_detailed(
                    new_skill_name=new_skill_name,
                    new_skill_description=new_skill_description,
                    new_skill_params=new_skill_params,
                    new_skill_preconditions=new_skill_preconditions,
                    new_skill_effects=new_skill_effects,
                    new_skill_children=new_skill_children,
                    candidate_skills=candidate_skills
                )
            else:
                # No prescreener — use all candidates
                prescreened_candidates = [name for name, _ in candidate_skills]

            if not prescreened_candidates:
                self.logger.info(
                    "[Refactor Detection] no candidate skills after prescreening, skipping refactor detection"
                )
                return None

            # Step 2: detailed analysis on prescreened candidates using full code
            #
            # Defense-in-depth: re-validate is_covered before sending
            # candidates to the LLM. The upstream prescreener already filters
            # is_covered=True wrappers, but this analyzer is the ONLY detection
            # path for behavioral and parametric refactors (those types have no
            # detect_*_opportunities function of their own), so a single
            # defensive filter here covers all refactor types in one place.
            # Cases this catches:
            # (a) is_covered flag set on a node AFTER prescreening
            # (b) future call sites that bypass prescreener
            # (c) prescreener backstop re-adding a stale-flagged wrapper
            # (d) is_covered flag-loss during checkpoint serialization
            # Under healthy conditions this is a no-op. When it fires, it
            # blocks the wrapper-balloon cycle where the LLM recommends
            # covering an already-covered wrapper.
            from skillnet.agents.skill_graph.models.graph import should_skip_covered_skill

            candidate_info = []
            skipped_covered: List[str] = []
            for skill_name in prescreened_candidates:
                if not self.graph.has_node(skill_name):
                    continue
                node = self.graph.get_node(skill_name)

                if should_skip_covered_skill(node):
                    skipped_covered.append(skill_name)
                    continue

                candidate_info.append({
                    "name": skill_name,
                    "description": node.description,
                    "parameters": node.parameters,
                    "preconditions": [{"description": p.description} for p in node.preconditions],
                    "effects": [{"description": e.description} for e in node.expected_effects],
                    "children": node.children,  # skeleton
                    "code": node.code  # full code
                })

            if skipped_covered:
                self.logger.info(
                    f"[Refactor Detection] Skipped {len(skipped_covered)} covered "
                    f"wrapper(s) that slipped past prescreener: {skipped_covered}"
                )

            if not candidate_info:
                return None

            # Build the LLM prompt
            system_prompt = self._get_detection_system_prompt()

            # Prepare the new skill's full info
            new_skill_info = {
                "name": new_skill_name,
                "description": new_skill_description,
                "parameters": new_skill_params,
                "preconditions": [{"description": p.description} for p in new_skill_preconditions],
                "effects": [{"description": e.description} for e in new_skill_effects],
                "children": new_skill_children,
                "code": new_skill_code
            }

            new_skill_json = json.dumps(new_skill_info, indent=2, ensure_ascii=False)
            candidate_json = json.dumps(candidate_info, indent=2, ensure_ascii=False)

            human_prompt = self._get_detection_human_prompt(new_skill_json, candidate_json)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            self.logger.info(
                f"[Refactor Detection] detecting refactor relationships between new skill {new_skill_name} and candidate skills..."
            )
            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="refactor", function_name="refactor.analyzer.detect_relationships", skill_name=new_skill_name)
            response = _llm_resp.content

            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                needs_refactor = result.get("needs_refactor", False)

                if needs_refactor:
                    refactor_type = result.get("refactor_type")
                    covered_skills = result.get("covered_skills", [])
                    general_skill = result.get("general_skill")
                    general_skill_name = result.get("general_skill_name")
                    parameter_mapping = result.get("parameter_mapping")
                    reason = result.get("reason", "")

                    self.logger.info(f"[Refactor Detection] detected refactor relationship: {refactor_type}")
                    self.logger.info(
                        f"[Refactor Detection] General skill: {general_skill or general_skill_name}"
                    )
                    self.logger.info(
                        f"[Refactor Detection] covered skills: {', '.join(covered_skills)}"
                    )
                    self.logger.info(f"[Refactor Detection] reason: {reason}")

                    # Validate refactor direction
                    if refactor_type in ("behavioral", "functional_superset", "parametric", "duplication"):
                        if general_skill:
                            validation_result = self.validate_direction(
                                refactor_type=refactor_type,
                                general_skill=general_skill,
                                covered_skills=covered_skills if covered_skills else [],
                                new_skill_name=new_skill_name,
                                new_skill_code=new_skill_code,
                            )

                            if not validation_result["valid"]:
                                self.logger.warning(
                                    f"[Refactor Detection] direction validation failed: {validation_result['reason']}"
                                )

                                correction = validation_result.get("suggested_correction")
                                if correction:
                                    self.logger.warning(
                                        f"[Refactor Detection] suggested correction: {correction}"
                                    )

                                    if correction == "clear_covered_skills":
                                        covered_skills = []
                                    elif correction == "swap_direction":
                                        # No longer auto-swap direction — it can flip a correct LLM judgment.
                                        # Let the downstream behavioral.py guards catch true errors.
                                        self.logger.warning(
                                            f"[Refactor Detection] direction validation suggests swap, but not auto-swapping"
                                        )
                                        return None  # Reject this detection result
                                    elif correction == "change_to_behavioral":
                                        refactor_type = "behavioral"
                                        covered_skills = []
                                    elif correction == "reconsider_relationship":
                                        return None
                                else:
                                    return None

                    return {
                        "refactor_type": refactor_type,
                        "general_skill": general_skill,
                        "general_skill_name": general_skill_name,
                        "covered_skills": covered_skills,
                        "parameter_mapping": parameter_mapping,
                        "reason": reason
                    }
                else:
                    self.logger.info("[Refactor Detection] no refactorable relationship detected")
            else:
                self.logger.warning(f"[Refactor Detection] could not parse LLM response: {response[:200]}...")

        except Exception as e:
            self.logger.warning(f"[Refactor Detection] detection failed: {e}")
            import traceback
            self.logger.warning(f"[Refactor Detection] Traceback: {traceback.format_exc()}")

        return None

    def validate_direction(
        self,
        refactor_type: str,
        general_skill: str,
        covered_skills: List[str],
        new_skill_name: str,
        new_skill_code: str,
    ) -> Dict[str, Any]:
        """
        Validate whether the refactor direction is correct.

        Uses different validation dimensions for different relationship types:
        - behavioral: compare functional scope (larger scope calls smaller scope)
        - functional_superset: compare implementation maturity (higher maturity covers lower)
        - parametric: compare parameterization level (more parameterized covers less)
        - duplication: select via value function

        Args:
            refactor_type: refactor type
            general_skill: the skill labeled "general"
            covered_skills: list of skills labeled "covered"
            new_skill_name: name of the new skill
            new_skill_code: code of the new skill

        Returns:
            Dict: {
                "valid": bool,
                "reason": str,
                "suggested_correction": str or None,
                "dimension_analysis": Dict
            }
        """
        # Get general skill info
        general_node = self.graph.get_node(general_skill)
        if general_skill == new_skill_name:
            general_code = new_skill_code
            general_node_for_scope = None
        elif general_node:
            general_code = general_node.code or ""
            general_node_for_scope = general_node
        else:
            return {"valid": True, "reason": "General skill not found, skipping validation"}

        # ==========================================================================
        # Unified semantic relevance check.
        #
        # Catches LLM-hallucinated refactor relationships, e.g.:
        # - craftFurnace ← [craftWoodenAxe, craftStonePickaxe, craftWoodenPickaxe]
        # - ensureDiamonds ← [ensureRawIron, ensureCobblestone] (mislabeled
        #   functional_superset)
        # - ensureCobblestone ← ensureRawIron (granularity error)
        #
        # These share a root cause: the LLM picks a target whose products are not
        # referenced by the source's code. This single check rejects all of them.
        #
        # Runs BEFORE the per-type checks below — those are preserved as a second
        # layer for cases that pass relevance but fail dimension validation.
        # ==========================================================================
        # Always pass general_node (whether the skill is new or existing).
        # The node may have populated effects/preconditions even when it's
        # the new skill, since add_skill is typically called before detection.
        relevant, relevance_reason = self._validate_semantic_relevance(
            refactor_type=refactor_type,
            general_skill=general_skill,
            general_code=general_code,
            general_node=general_node,
            covered_skills=covered_skills,
            new_skill_name=new_skill_name,
            new_skill_code=new_skill_code,
        )
        if not relevant:
            return {
                "valid": False,
                "reason": relevance_reason,
                "suggested_correction": "reconsider_relationship",
                "dimension_analysis": None,
            }

        # Compute dimension scores for the general skill
        general_scope = estimate_functional_scope(general_code, node=general_node_for_scope)
        general_maturity = estimate_implementation_maturity(general_code)
        general_param = estimate_parameterization_level(general_code)

        validation_results = []

        # Check each covered skill
        for covered_skill in covered_skills:
            if covered_skill == new_skill_name:
                covered_code = new_skill_code
                covered_node = None
            else:
                covered_node = self.graph.get_node(covered_skill)
                if not covered_node:
                    continue
                covered_code = covered_node.code or ""

            # Compute dimension scores for the covered skill
            covered_scope = estimate_functional_scope(covered_code, node=covered_node)
            covered_maturity = estimate_implementation_maturity(covered_code)
            covered_param = estimate_parameterization_level(covered_code)

            dimension_analysis = {
                "general_skill": general_skill,
                "covered_skill": covered_skill,
                "functional_scope": {
                    "general": general_scope,
                    "covered": covered_scope
                },
                "implementation_maturity": {
                    "general": general_maturity,
                    "covered": covered_maturity
                },
                "parameterization_level": {
                    "general": general_param,
                    "covered": covered_param
                }
            }

            # Validate based on refactor type
            if refactor_type == "behavioral":
                # For behavioral, trust the LLM's semantic judgment; skip code-complexity validation.
                # Reasoning:
                # - The LLM judges "functional abstraction level" (who is the building block).
                # - estimate_functional_scope measures "code complexity" (number of operations).
                # - These dimensions are orthogonal: a skill may be code-complex yet be a low-level operation.
                # - behavioral.py's triple-cycle guard catches true circular dependencies.
                pass

            elif refactor_type == "functional_superset":
                if general_maturity < covered_maturity:
                    return {
                        "valid": False,
                        "reason": (
                            f"Functional superset direction is wrong: implementation maturity of '{general_skill}' ({general_maturity}) < "
                            f"that of '{covered_skill}' ({covered_maturity}). "
                            f"The more-mature (more-robust) skill should cover the less-mature one, not the other way around."
                        ),
                        "suggested_correction": "swap_direction",
                        "dimension_analysis": dimension_analysis
                    }
                scope_diff = abs(general_scope - covered_scope)
                if scope_diff > 3:
                    return {
                        "valid": False,
                        "reason": (
                            f"Functional superset relationship does not hold: '{general_skill}' and '{covered_skill}' "
                            f"have too large a functional-scope difference ({scope_diff}). Functional superset should be the same function but more robust. "
                            f"This may actually be a behavioral relationship."
                        ),
                        "suggested_correction": "change_to_behavioral",
                        "dimension_analysis": dimension_analysis
                    }

            elif refactor_type == "parametric":
                if general_param < covered_param:
                    return {
                        "valid": False,
                        "reason": (
                            f"Parametric direction is wrong: parameterization of '{general_skill}' ({general_param}) < "
                            f"that of '{covered_skill}' ({covered_param}). "
                            f"More-parameterized skills should cover specialized versions, not the other way around."
                        ),
                        "suggested_correction": "swap_direction",
                        "dimension_analysis": dimension_analysis
                    }

            elif refactor_type == "duplication":
                total_diff = (
                    abs(general_scope - covered_scope) +
                    abs(general_maturity - covered_maturity) +
                    abs(general_param - covered_param)
                )
                if total_diff > 5:
                    return {
                        "valid": False,
                        "reason": (
                            f"Duplication relationship is suspect: total dimensional difference between "
                            f"'{general_skill}' and '{covered_skill}' ({total_diff}) is large; may not be a true duplicate."
                        ),
                        "suggested_correction": "reconsider_relationship",
                        "dimension_analysis": dimension_analysis
                    }

            validation_results.append(dimension_analysis)

        return {
            "valid": True,
            "reason": "Direction validation passed",
            "dimension_analysis": validation_results[0] if validation_results else None
        }

    # ==========================================================================
    # Semantic relevance check (rejects hallucinated refactor relationships)
    # ==========================================================================

    def _validate_semantic_relevance(
        self,
        refactor_type: str,
        general_skill: str,
        general_code: str,
        general_node: Optional[Any],
        covered_skills: List[str],
        new_skill_name: str,
        new_skill_code: str,
    ) -> Tuple[bool, str]:
        """
        Verify that source skills actually use what target skill produces.

        Domain-agnostic semantic check that catches LLM-hallucinated refactor
        relationships where the target's products have nothing to do with the
        source's logic. Examples:

          - craftFurnace produces {furnace}; craftWoodenAxe doesn't reference
            "furnace" → REJECT
          - ensureDiamonds produces {diamond}; ensureRawIron references "iron_ore"
            not "diamond" → REJECT
          - ensureCobblestone produces {cobblestone}; ensureRawIron doesn't
            reference "cobblestone" → REJECT

        Per-type semantics:
          - behavioral (incl. functional_superset alias): each covered skill's
            code/preconditions must reference at least one of general's items
          - parametric: covered_items must be a subset of general_items
          - duplication: covered_items must equal general_items

        Returns:
            (valid, reason)
            valid=True if the relationship is semantically plausible (or items
            cannot be extracted reliably — fail open to avoid blocking utility
            refactors).
        """
        # Step 1: Extract general skill's products
        general_items = self._extract_target_items(
            skill_name=general_skill,
            code=general_code,
            node=general_node,
        )

        if not general_items:
            return (
                True,
                f"semantic check skipped: cannot extract products of '{general_skill}'",
            )

        # Step 2: For each covered skill, check item relationship
        for covered_skill in covered_skills:
            # Resolve covered skill's code/preconditions/items
            if covered_skill == new_skill_name:
                covered_code = new_skill_code
                covered_node = None
            else:
                covered_node = self.graph.get_node(covered_skill)
                if not covered_node:
                    # Skill not in graph — can't validate, skip this one
                    continue
                covered_code = covered_node.code or ""

            covered_pre_text = ""
            if covered_node and getattr(covered_node, "preconditions", None):
                covered_pre_text = " ".join(
                    getattr(p, "description", "") or "" for p in covered_node.preconditions
                )

            # Per-type relevance check
            if refactor_type in ("behavioral", "functional_superset"):
                # Source must reference at least one of target's items
                source_text = (covered_code + " " + covered_pre_text).lower()
                if not any(item.lower() in source_text for item in general_items):
                    return (
                        False,
                        f"Semantic relevance check failed: '{covered_skill}' "
                        f"does not reference any of '{general_skill}'s products "
                        f"{sorted(general_items)}. The {refactor_type} refactor "
                        f"is likely an LLM-hallucinated relationship.",
                    )

            elif refactor_type == "parametric":
                # covered_items should be ⊆ general_items (specialized version's
                # outputs are a subset of the general version's outputs).
                #
                # Use canonicalized item sets (singular form) for the comparison
                # to avoid false positives when one skill's items include the
                # plural variant added by function-name parsing (e.g., ensureOakLogs
                # produces {oak_log, oak_logs} while mineWoodLogs produces {oak_log}
                # — the plural mismatch should NOT block a legitimate parametric
                # refactor).
                covered_items = self._extract_target_items(
                    skill_name=covered_skill,
                    code=covered_code,
                    node=covered_node,
                )
                if covered_items:
                    canonical_covered = self._canonicalize_items(covered_items)
                    canonical_general = self._canonicalize_items(general_items)
                    if not canonical_covered.issubset(canonical_general):
                        return (
                            False,
                            f"Parametric check failed: '{covered_skill}' produces "
                            f"{sorted(covered_items)}, which is not a subset of "
                            f"'{general_skill}' products {sorted(general_items)}. "
                            f"Parametric coverage requires the specialized skill's "
                            f"outputs to be a subset of the general skill's outputs.",
                        )

            elif refactor_type == "duplication":
                # covered_items should == general_items (literally same skill).
                # Same canonicalization as parametric to handle plural/singular.
                covered_items = self._extract_target_items(
                    skill_name=covered_skill,
                    code=covered_code,
                    node=covered_node,
                )
                if covered_items:
                    canonical_covered = self._canonicalize_items(covered_items)
                    canonical_general = self._canonicalize_items(general_items)
                    if canonical_covered != canonical_general:
                        return (
                            False,
                            f"Duplication check failed: '{covered_skill}' produces "
                            f"{sorted(covered_items)} but '{general_skill}' produces "
                            f"{sorted(general_items)}. Duplication requires identical "
                            f"product sets.",
                        )

        return (True, "semantic relevance check passed")

    @staticmethod
    def _canonicalize_items(items: set) -> set:
        """
        Normalize an item set for set-based comparisons (parametric/duplication).

        Collapses singular/plural variants by stripping a trailing 's' (when not
        '-ss'). This prevents false positives where the function-name parser at
        `_extract_target_items` adds both singular and plural forms (e.g.,
        `ensureOakLogs` → `{oak_log, oak_logs}`) and one set has the plural while
        the other only has the singular.

        The relevance check (substring match) does NOT use this canonicalization
        because there variant expansion is HELPFUL (catches more text mentions).
        """
        canonical = set()
        for item in items:
            s = item.lower()
            # Strip trailing 's' but not 'ss' (e.g., 'pass' stays as 'pass')
            if s.endswith("s") and not s.endswith("ss") and len(s) > 1:
                s = s[:-1]
            canonical.add(s)
        return canonical

    def _extract_target_items(
        self,
        skill_name: str,
        code: str,
        node: Optional[Any],
    ) -> set:
        """
        Extract the set of items a skill produces/affects.

        Strategy:
          1. If `expected_effects` is populated, use ONLY structured effects
             (ground truth from the system's own effect extractor).
          2. Otherwise, combine code-based parsing + function name parsing
             (best-effort fallback for skills detected before effects are
             populated).

        Code-based extraction uses `craftItem(bot, 'X', N)` and
        `mineBlock(bot, 'X', N)` calls. `placeItem(bot, 'X', pos)` is
        intentionally EXCLUDED — `placeItem` is ambiguous: in
        `placeCraftingTableNearby` it's the primary effect, but in
        `craftFurnace` it's just workspace setup. The function name
        fallback handles the legitimate place* cases.

        Function name parsing recognizes verb-prefixed names
        (`craft<X>`, `mine<X>`, `ensure<X>`, `place<X>`, etc.) and extracts
        the noun in snake_case form.

        Returns empty set if no signal is available — caller treats empty
        as "skip the check" to avoid blocking utility refactors.
        """
        # Source 1: structured effects (preferred when populated)
        items = set()
        effects = getattr(node, "expected_effects", None) if node else None
        if effects:
            for eff in effects:
                # Try state_representation.item first (structured)
                sr = getattr(eff, "state_representation", None)
                if isinstance(sr, dict):
                    item = sr.get("item")
                    if isinstance(item, str) and item:
                        items.add(item.lower())
                elif isinstance(sr, list):
                    for entry in sr:
                        if isinstance(entry, dict):
                            item = entry.get("item")
                            if isinstance(item, str) and item:
                                items.add(item.lower())
                # Fallback: parse description "Adds N <item>"
                desc = getattr(eff, "description", "") or ""
                m = re.search(r"adds?\s+\d+\s+([a-z_][a-z0-9_]+)", desc, re.IGNORECASE)
                if m:
                    items.add(m.group(1).lower())
            if items:
                return items
            # If effects existed but yielded nothing extractable, fall through
            # to code/name fallbacks rather than returning empty.

        # Source 2: code-based extraction (when effects unavailable)
        # craftItem and mineBlock are the unambiguous signals.
        if code:
            for m in re.finditer(
                r"\bcraftItem\s*\(\s*bot\s*,\s*['\"]([a-z_][a-z0-9_]*)['\"]",
                code,
                re.IGNORECASE,
            ):
                items.add(m.group(1).lower())
            for m in re.finditer(
                r"\bmineBlock\s*\(\s*bot\s*,\s*['\"]([a-z_][a-z0-9_]*)['\"]",
                code,
                re.IGNORECASE,
            ):
                items.add(m.group(1).lower())

        # Source 3: function name parsing (always combine with code-based,
        # to maximize coverage when effects are missing)
        name_match = re.match(
            r"(?:craft|mine|ensure|place|smelt|get|make|collect|gather)([A-Z]\w*)",
            skill_name,
        )
        if name_match:
            noun = name_match.group(1)
            # Convert CamelCase → snake_case
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", noun).lower()
            # Drop trailing positional adverbs ("_nearby", "_safely", etc.)
            snake = re.sub(r"_(nearby|safely|carefully|quickly)$", "", snake)
            items.add(snake)
            # Also add singular form (very rough: drop trailing 's' if not 'ss')
            if snake.endswith("s") and not snake.endswith("ss"):
                items.add(snake[:-1])

        return items

    def _get_detection_system_prompt(self) -> str:
        """Return the detection system prompt (large; kept in its own method)."""
        return """You are a code analysis expert specializing in skill refactoring. Your task is to detect if a new skill has refactoring relationships with existing skills.

================================================================================
                        BIDIRECTIONAL REFACTORING RELATIONSHIPS
================================================================================

IMPORTANT: All refactoring relationships can be BIDIRECTIONAL. This means:
- The NEW skill might generalize/cover EXISTING skills, OR
- EXISTING skills might generalize/cover the NEW skill
- Both directions are possible; you must determine which is correct

================================================================================
                        THREE EVALUATION DIMENSIONS
================================================================================

Use these THREE independent dimensions to determine the correct refactoring direction:

**1. FUNCTIONAL SCOPE** (Used for: behavioral/subgraph)
   Measures HOW MANY DIFFERENT THINGS a skill does:
   - Number of distinct operations (mining, crafting, placing, smelting, etc.)
   - Number of different item types produced
   - Complexity of the overall task

   RULE: For behavioral refactoring, the skill with LARGER scope should CALL the skill with SMALLER scope

**2. IMPLEMENTATION MATURITY** (Used for: functional_superset)
   Measures HOW ROBUST the implementation is FOR THE SAME FUNCTION:
   - Input validation / parameter checking
   - Precondition checking (material/tool availability)
   - Error handling (try-catch)
   - Retry logic / fallback logic

   RULE: For functional_superset, the skill with HIGHER maturity COVERS the skill with LOWER maturity

**3. PARAMETERIZATION LEVEL** (Used for: parametric)
   Measures HOW GENERALIZED the parameters are:
   - Number of configurable parameters
   - Presence of type parameters (logType, oreName, etc.)

   RULE: For parametric, the skill with HIGHER parameterization COVERS specialized versions

================================================================================
                        SIX TYPES OF REFACTORING RELATIONSHIPS
================================================================================

1. **PARAMETRIC COVERAGE** - Different parameterization levels
2. **BEHAVIORAL/SUBGRAPH COVERAGE** - One contains the other's logic
3. **SIBLING SPECIALIZATIONS** - Same operation, different types, no general exists
4. **EXTRACT COMMON SUB-SKILL** - Different operations, shared sub-operation
5. **DUPLICATION** - Functionally identical
6. **FUNCTIONAL SUPERSET** - Same function, different maturity

Return ONLY a JSON object in this format:
{
  "needs_refactor": true/false,
  "refactor_type": "parametric" | "behavioral" | "sibling" | "extract_common_subskill" | "duplication" | "functional_superset" | null,
  "general_skill": "skill_name" or null,
  "general_skill_name": "suggested_name" or null,
  "covered_skills": ["skill_name1", "skill_name2"] or [],
  "parameter_mapping": {...} or null,
  "reason": "brief explanation"
}

================================================================================
                 CRITICAL: FIELD SEMANTICS FOR DIFFERENT REFACTOR TYPES
================================================================================

The meaning of "general_skill" and "covered_skills" varies by refactor type:

- **PARAMETRIC/FUNCTIONAL_SUPERSET/DUPLICATION**:
  - general_skill = the skill with HIGHER parameterization/maturity (the "better" one)
  - covered_skills = skills that can be REPLACED by general_skill

- **BEHAVIORAL**: (IMPORTANT - Different semantics!)
  - general_skill = the SMALLER-SCOPE skill that should be CALLED (the reusable component)
  - covered_skills = LARGER-SCOPE skills that should CALL the general_skill
  - Example: If "craftCraftingTable" calls "ensureLogs", then:
    - general_skill = "ensureLogs" (smaller scope, reusable)
    - covered_skills = ["craftCraftingTable"] (larger scope, caller)

- **SIBLING/EXTRACT_COMMON_SUBSKILL**:
  - general_skill = null (no existing general skill)
  - general_skill_name = suggested name for NEW skill to be extracted
  - covered_skills = skills that share common logic

If no refactoring is needed, return {"needs_refactor": false}."""

    def _get_detection_human_prompt(self, new_skill_json: str, candidate_json: str) -> str:
        """Return the detection human prompt."""
        return f"""Analyze if the new skill needs refactoring with the prescreened candidate skills. Use the FULL CODE to make a detailed analysis.

NEW SKILL (FULL INFORMATION):
{new_skill_json}

PRESCREENED CANDIDATE EXISTING SKILLS (FULL INFORMATION):
{candidate_json}

Based on the full code and metadata, determine if any refactoring relationship exists. Return only JSON."""
