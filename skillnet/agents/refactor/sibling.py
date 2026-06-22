"""
Sibling Refactor

Sibling-relation refactor strategy: when multiple skills are different variants
of the same functional family, create a generic parent skill and turn all
variants into wrappers that call it.

For example:
- craftOakBoat(), craftBirchBoat(), craftSpruceBoat()
  are all specializations of craftBoat(woodType)
  -> create craftBoat(woodType)
  -> each wrapper calls craftBoat with the specific parameter

Difference from DUPLICATION:
- DUPLICATION: highly duplicated code (>70%), focuses on the code level
- SIBLING: same functional family but code may not be identical; focuses on the semantic level

Identifying features:
- Similar name patterns (e.g. craft*Boat, mine*Logs)
- Similar descriptions/functionality
- Similar parameter structures (only values differ)
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

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
class SiblingGroup:
    """Sibling skill group."""
    pattern: str                     # Name pattern (e.g. "craft*Boat")
    members: List[str]               # Member skill names
    variants: Dict[str, str]         # {skill_name: variant_value}
    suggested_general_name: str      # Suggested generic skill name
    suggested_param_name: str        # Suggested parameter name
    confidence: float                # Confidence score


@dataclass
class SiblingPlan:
    """Sibling refactor plan."""
    general_skill_name: str
    general_skill_code: str
    general_skill_description: str
    param_name: str
    wrappers: Dict[str, str]         # {skill_name: wrapper_code}
    param_values: Dict[str, str]     # {skill_name: param_value}


class SiblingRefactor(SkillRefactor):
    """
    Sibling-relation refactorer.

    Identify skill variants of the same functional family and create a generic parent skill.

    Refactor workflow:
    1. Identify sibling skill groups (based on name patterns, description similarity)
    2. Extract variant differences (e.g. Oak/Birch/Spruce)
    3. Create a parameterized generic skill
    4. Convert each variant into a wrapper that calls the generic skill
    5. Add dependency relationships
    """

    # Common variant patterns (module-level fallback; overridden by domain_knowledge)
    VARIANT_PATTERNS = [
        # Wood types
        (r'(oak|birch|spruce|jungle|acacia|dark_?oak|mangrove|cherry)', 'woodType'),
        # Ore/material types
        (r'(iron|gold|diamond|netherite|copper|coal|redstone|lapis|emerald)', 'materialType'),
        # Tool types
        (r'(pickaxe|axe|shovel|hoe|sword)', 'toolType'),
        # Directions
        (r'(north|south|east|west|up|down)', 'direction'),
        # Colors
        (r'(white|orange|magenta|light_?blue|yellow|lime|pink|gray|light_?gray|cyan|purple|blue|brown|green|red|black)', 'color'),
    ]

    def _get_variant_patterns(self):
        """Return variant patterns — domain-injected or class-level fallback."""
        dk = getattr(self, 'domain_knowledge', None)
        if dk:
            patterns = dk.get_variant_patterns()
            if patterns:
                return patterns
        return self.VARIANT_PATTERNS

    def apply(self, opportunity: RefactorOpportunity) -> RefactorResult:
        """
        Apply the sibling-relation refactor.

        Args:
            opportunity: Refactor opportunity
                - source_skill: First sibling skill
                - covered_skills: Other sibling skills

        Returns:
            RefactorResult: Refactor result
        """
        # Note: RefactorType.SIBLING is an alias for MERGE_SIBLINGS; they are equivalent
        if opportunity.refactor_type != RefactorType.MERGE_SIBLINGS:
            return RefactorResult(
                success=False,
                refactor_type=opportunity.refactor_type,
                source_skill=opportunity.source_skill,
                target_skill=opportunity.target_skill,
                error_message="Wrong refactor type for SiblingRefactor"
            )

        # Get all sibling skills
        siblings = [opportunity.source_skill]
        if opportunity.covered_skills:
            siblings.extend(opportunity.covered_skills)

        self._log(
            f"[Sibling] Starting refactor of sibling skills: {siblings}",
            "info"
        )

        if not self.skill_graph_manager:
            return self._create_failed_result(
                opportunity, "No skill_graph_manager available"
            )

        # Get the skill nodes
        skill_nodes = {}
        for name in siblings:
            node = self.skill_graph_manager.get_node(name)
            if node:
                skill_nodes[name] = node

        if len(skill_nodes) < 2:
            return self._create_failed_result(
                opportunity, "Need at least 2 siblings for sibling refactor"
            )

        # Save rollback data
        rollback_data = self._save_multi_skill_rollback(skill_nodes)

        try:
            # Identify the sibling group
            sibling_group = self._identify_sibling_group(skill_nodes)

            if not sibling_group or len(sibling_group.members) < 2:
                return self._create_failed_result(
                    opportunity, "Could not identify sibling relationship"
                )

            # Create the refactor plan (with response capture for forensics + retry)
            from ._retry_helpers import (
                capture_llm_responses,
                save_refactor_failure_forensics,
                build_retry_message,
            )

            with capture_llm_responses(self) as cap_first:
                plan = self._create_sibling_plan(sibling_group, skill_nodes)
            first_raw = cap_first.raw_responses[-1] if cap_first.raw_responses else ""
            first_messages = cap_first.last_messages

            if not plan:
                return self._create_failed_result(
                    opportunity, "Failed to create sibling plan"
                )

            # [Fix] Validate general_skill_code syntax
            is_valid, syntax_error = validate_code_syntax(plan.general_skill_code)

            # v3.H+ Layer 3 — single-shot retry + forensics on validation failure
            if not is_valid:
                save_refactor_failure_forensics(
                    skill_graph_manager=self.skill_graph_manager,
                    refactor_type="merge_siblings",
                    skill_names=list(skill_nodes.keys()),
                    attempt=1,
                    raw_response=first_raw,
                    babel_error=syntax_error or "",
                    prompt_messages=first_messages,
                    logger=getattr(self, "logger", None),
                )
                self._log(
                    f"[Sibling] ⟳ Layer 3 retry — first attempt failed validation: "
                    f"{(syntax_error or '')[:120]}",
                    "warning",
                )
                with capture_llm_responses(self) as cap_retry:
                    plan2 = self._create_plan_with_llm(
                        sibling_group, skill_nodes,
                        retry_hint=build_retry_message(syntax_error or ""),
                    )
                retry_raw = cap_retry.raw_responses[-1] if cap_retry.raw_responses else ""
                if plan2 is not None:
                    plan = plan2
                    is_valid, syntax_error = validate_code_syntax(plan.general_skill_code)
                    if is_valid:
                        self._log("[Sibling] ✓ Layer 3 retry succeeded", "info")
                    else:
                        save_refactor_failure_forensics(
                            skill_graph_manager=self.skill_graph_manager,
                            refactor_type="merge_siblings",
                            skill_names=list(skill_nodes.keys()),
                            attempt=2,
                            raw_response=retry_raw,
                            babel_error=syntax_error or "",
                            prompt_messages=cap_retry.last_messages,
                            logger=getattr(self, "logger", None),
                        )

            if not is_valid:
                self._log(
                    f"[Sibling] ✗ Syntax validation failed for general skill code: {syntax_error}",
                    "error"
                )
                return self._create_failed_result(
                    opportunity,
                    f"Syntax error in general skill code: {syntax_error}"
                )

            # Create the generic skill
            general_name = plan.general_skill_name
            general_node = SkillNode(
                name=general_name,
                code=plan.general_skill_code,
                description=plan.general_skill_description,
            )

            # ───────────────────────────────────────────────────────────────────
            # Layer 1 fix (Phase B validation): propagate union of sibling
            # effects/preconditions to the new general skill so the planner's
            # effect_matcher can find it for specific-product queries (e.g.
            # "Craft 1 wooden axe" → wooden_axe target → matches general skill
            # because we propagated craftAxe's wooden_axe effect into it).
            #
            # Without this, general skill is born with empty expected_effects
            # which makes it invisible to effect_matcher — planner falls back
            # to action_agent which regenerates inline + overwrites wrapper.
            # ───────────────────────────────────────────────────────────────────
            import copy as _copy
            import json as _json

            propagated_effects = []
            seen_effect_keys = set()
            propagated_preconds = []
            seen_precond_keys = set()
            for sib_name, sib_node in skill_nodes.items():
                sib_param_value = plan.param_values.get(sib_name)
                # Effects union (deduplicated by description + state_representation)
                for eff in (sib_node.expected_effects or []):
                    sr = getattr(eff, 'state_representation', None) or {}
                    key = (
                        getattr(eff, 'description', '') or '',
                        _json.dumps(sr, sort_keys=True) if isinstance(sr, dict) else str(sr),
                    )
                    if key not in seen_effect_keys:
                        seen_effect_keys.add(key)
                        eff_copy = _copy.deepcopy(eff)
                        # Annotate with specialization condition so downstream
                        # parameter inference can recover the binding.
                        # Stored as attribute (not part of SkillEffect schema)
                        # so existing serialization/dedup logic is unchanged.
                        if sib_param_value is not None:
                            eff_copy.specialized_when = {plan.param_name: sib_param_value}
                        propagated_effects.append(eff_copy)
                # Preconditions union (deduplicated by description + check code)
                for pre in (sib_node.preconditions or []):
                    key = (
                        getattr(pre, 'description', '') or '',
                        getattr(pre, 'code', '') or '',
                    )
                    if key not in seen_precond_keys:
                        seen_precond_keys.add(key)
                        propagated_preconds.append(_copy.deepcopy(pre))

            # ───────────────────────────────────────────────────────────────────
            # Synthesize the general skill's PRIMARY product effect.
            #
            # The siblings' own product effects (e.g. +wooden_axe / +wooden_pickaxe)
            # were propagated above, but they are FIXED items on a PARAMETRIC skill:
            # downstream _validate_effects_against_code strips them (the parametric
            # code crafts via the `toolType` parameter and contains no literal
            # product name), leaving the general skill with NO primary effect. That
            # lets EffectMatcher's no-primary fallback mis-match an intermediate
            # by-product.
            #
            # Record the product as a parameter-bound OR over the siblings' products
            # so it survives validation (_effect_is_param_implemented) and the
            # matcher prefers it. Replace the fixed sibling primaries with this one.
            # (Repair-on-load reconstructs the same primary for already-broken
            # checkpoints, per _repair_missing_primary_effects.)
            # ───────────────────────────────────────────────────────────────────
            sibling_products = []
            for sib_node in skill_nodes.values():
                for eff in (sib_node.expected_effects or []):
                    if not getattr(eff, 'is_primary', False):
                        continue
                    sr = getattr(eff, 'state_representation', None) or {}
                    if isinstance(sr, dict):
                        if isinstance(sr.get('conditions'), list):
                            sibling_products += [
                                c.get('item') for c in sr['conditions'] if isinstance(c, dict)
                            ]
                        elif sr.get('item'):
                            sibling_products.append(sr.get('item'))
            sibling_products = [p for p in dict.fromkeys(sibling_products) if p]
            if sibling_products:
                from skillnet.agents.skill_graph.metadata.effects import (
                    synthesize_general_primary_effect,
                )
                general_primary = synthesize_general_primary_effect(
                    sibling_products, plan.param_name
                )
                if general_primary is not None:
                    propagated_effects = [
                        e for e in propagated_effects if not getattr(e, 'is_primary', False)
                    ]
                    propagated_effects.append(general_primary)
                    self._log(
                        f"[Sibling] Layer1: synthesized param-bound product primary for "
                        f"{general_name}: {sibling_products} via '{plan.param_name}'",
                        "info",
                    )

            if propagated_effects:
                general_node.expected_effects = propagated_effects
                self._log(
                    f"[Sibling] Layer1: propagated {len(propagated_effects)} effects "
                    f"from {len(skill_nodes)} siblings to {general_name}",
                    "info",
                )
            if propagated_preconds:
                general_node.preconditions = propagated_preconds
                self._log(
                    f"[Sibling] Layer1: propagated {len(propagated_preconds)} preconditions "
                    f"from {len(skill_nodes)} siblings to {general_name}",
                    "info",
                )

            # ───────────────────────────────────────────────────────────────────
            # Plan v3 Part A: extract parameter metadata for the new general
            # skill BEFORE add_skill_node. Without this, planner.plan() sees
            # general_node.parameters == {} and falls through to a code-based
            # extraction path where object params (e.g. `options = {}`) resolve
            # to undefined, Layer 1.5 trim pops them, and the wrapper gets
            # called with only positional args. For ensureResource that meant
            # bot used internal defaults (targetTotal=3) instead of task-
            # requested value (16). See v11 ckpt iter 10 for the exact bug.
            # ───────────────────────────────────────────────────────────────────
            try:
                extracted_params = self.skill_graph_manager._extract_parameters(
                    general_node.code,
                    general_node.description,
                    general_name,
                )
                if extracted_params:
                    general_node.parameters = extracted_params
                    self._log(
                        f"[Sibling] PartA: extracted {len(extracted_params)} parameter "
                        f"schemas for general skill {general_name}: {list(extracted_params.keys())}",
                        "info",
                    )
            except Exception as e:
                # Non-fatal: refactor proceeds even if parameter extraction
                # fails. The downstream planner will fall back to code-based
                # extraction (same path as today). Logged so the issue surfaces.
                self._log(
                    f"[Sibling] PartA: parameter extraction failed for {general_name}: {e!r}",
                    "warning",
                )

            self.skill_graph_manager.add_skill_node(general_node)

            changes_made = [f"Created general skill: {general_name}"]

            # Update each sibling into a wrapper
            for name, wrapper_code in plan.wrappers.items():
                node = skill_nodes.get(name)
                if node:
                    # Validate the wrapper code's syntax
                    is_valid, syntax_error = validate_code_syntax(wrapper_code)
                    if not is_valid:
                        self._log(
                            f"[Sibling] ✗ JS syntax validation failed (skill: {name}): {syntax_error}",
                            "error"
                        )
                        return self._create_failed_result(
                            opportunity,
                            f"Syntax error in wrapper code for {name}: {syntax_error}"
                        )

                    # Update code through the unified interface (syntax already validated; skip duplicate validation)
                    update_success = self.skill_graph_manager.update_skill_code(
                        skill_name=name,
                        new_code=wrapper_code,
                        change_log=f"Converted to wrapper calling {general_name}",
                        source="refactor:sibling",
                        skip_validation=True,  # Syntax already validated
                        skip_metadata=True,    # Refactor does not update metadata
                        skip_interface_check=True,  # Do not trigger cascade
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
                    node.coverage_type = CoverageType.SIBLING

                    # Add dependency edge
                    self.skill_graph_manager.add_edge(name, general_name)

                    changes_made.append(
                        f"Converted {name} to wrapper ({plan.param_name}={plan.param_values.get(name)})"
                    )

            rollback_data["general_skill_name"] = general_name

            self._log(
                f"[Sibling] ✓ Refactor succeeded; created {general_name}",
                "info"
            )

            # Propagate changes to callers (optional auto-update)
            updated_callers = []
            caller_update_details = {}
            all_caller_rollback_data = {}  # Aggregate rollback data from all callers

            for name in skill_nodes.keys():
                # Check which skills called the sibling skills now converted to wrappers
                propagation_result = self.propagate_to_callers(
                    refactored_skill=name,
                    changes={
                        "change_type": "general_created",
                        "new_skill_name": general_name,
                        "param_value": plan.param_values.get(name, ""),
                    },
                    auto_update=True,  # Enable auto-updating callers to avoid broken references
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
                refactor_type=RefactorType.MERGE_SIBLINGS,
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
            self._log(f"[Sibling] Refactor failed: {e}", "error")
            return self._create_failed_result(opportunity, str(e))

    def _identify_sibling_group(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[SiblingGroup]:
        """
        Identify the sibling skill group.

        Prefer LLM-based semantic analysis; fall back to rules if it fails.

        Args:
            skill_nodes: Dictionary of skill nodes

        Returns:
            SiblingGroup: The identified sibling group
        """
        # Prefer LLM-based semantic analysis
        if self.llm:
            result = self._identify_sibling_group_with_llm(skill_nodes)
            if result:
                return result
            self._log("[Sibling] LLM identification failed; falling back to rules", "warning")

        return self._identify_sibling_group_simple(skill_nodes)

    def _identify_sibling_group_with_llm(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[SiblingGroup]:
        """
        Use the LLM to identify the sibling group.

        Can identify:
        - Skills with unobvious name patterns but similar functionality
        - Complex variant patterns (e.g. harvestWheat vs collectPotatoes)
        - Semantic grouping based on descriptions
        """
        try:
            import json
            from langchain.schema import HumanMessage, SystemMessage
            from .prompts import PromptLoader

            names = list(skill_nodes.keys())
            if len(names) < 2:
                return None

            # Build description information
            descriptions = []
            for name, node in skill_nodes.items():
                desc = node.description or "No description"
                descriptions.append(f"- {name}: {desc}")

            template = PromptLoader.load("sibling_identification")
            system_prompt, human_prompt = template.format(
                skill_names=", ".join(names),
                skill_descriptions="\n".join(descriptions),
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.sibling._identify_sibling_group_with_llm")
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse the JSON response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                if not data.get("is_sibling_group", False):
                    self._log("[Sibling] LLM judged this is not a sibling group", "info")
                    return None

                confidence = data.get("confidence", 0.7)
                variants = data.get("variants", {})

                # Ensure all names have a variant value
                for name in names:
                    if name not in variants:
                        # Try to extract a variant from the name
                        variants[name] = self._to_snake_case(name)

                self._log(
                    f"[Sibling] LLM identified sibling group: {data.get('pattern')} "
                    f"(confidence={confidence:.2f})",
                    "info"
                )

                return SiblingGroup(
                    pattern=data.get("pattern", f"{names[0][:3]}*"),
                    members=names,
                    variants=variants,
                    suggested_general_name=data.get(
                        "suggested_general_name", f"{names[0][:3]}General"
                    ),
                    suggested_param_name=data.get("suggested_param_name", "variant"),
                    confidence=confidence,
                )

            return None

        except Exception as e:
            self._log(f"[Sibling] LLM identification exception: {e}", "warning")
            return None

    def _identify_sibling_group_simple(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[SiblingGroup]:
        """
        Identify the sibling group using rules (simple version).

        Based on predefined name patterns and prefix matching.
        """
        names = list(skill_nodes.keys())

        # Try to identify from name patterns
        for pattern, param_name in self._get_variant_patterns():
            variants = {}
            for name in names:
                match = re.search(pattern, name, re.IGNORECASE)
                if match:
                    variants[name] = match.group(1).lower()

            if len(variants) >= 2:
                # Found variants
                # Extract a generic name (replace variant part with a placeholder)
                sample_name = names[0]
                general_name = re.sub(pattern, '', sample_name, flags=re.IGNORECASE)
                general_name = re.sub(r'([a-z])([A-Z])', r'\1\2', general_name)  # Keep camelCase
                general_name = general_name.strip('_') or 'general'

                # Build the pattern string
                pattern_str = re.sub(pattern, '*', sample_name, flags=re.IGNORECASE)

                return SiblingGroup(
                    pattern=pattern_str,
                    members=list(variants.keys()),
                    variants=variants,
                    suggested_general_name=f"{general_name}General",
                    suggested_param_name=param_name,
                    confidence=len(variants) / len(names),
                )

        # Try to identify based on name similarity
        return self._identify_by_name_similarity(skill_nodes)

    def _identify_by_name_similarity(
        self,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[SiblingGroup]:
        """Identify the sibling group based on name similarity."""
        names = list(skill_nodes.keys())

        # Find the common prefix
        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix) and prefix:
                prefix = prefix[:-1]

        if len(prefix) >= 3:
            # Extract the variant part
            variants = {}
            for name in names:
                variant = name[len(prefix):]
                if variant:
                    variants[name] = self._to_snake_case(variant)

            if len(variants) >= 2:
                return SiblingGroup(
                    pattern=f"{prefix}*",
                    members=list(variants.keys()),
                    variants=variants,
                    suggested_general_name=f"{prefix}General",
                    suggested_param_name="variant",
                    confidence=len(variants) / len(names),
                )

        return None

    # _to_snake_case is inherited from the SkillRefactor base class

    def _create_sibling_plan(
        self,
        group: SiblingGroup,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[SiblingPlan]:
        """Create the sibling refactor plan."""
        if self.llm:
            return self._create_plan_with_llm(group, skill_nodes)
        else:
            return self._create_plan_simple(group, skill_nodes)

    def _create_plan_simple(
        self,
        group: SiblingGroup,
        skill_nodes: Dict[str, 'SkillNode']
    ) -> Optional[SiblingPlan]:
        """Create a simple refactor plan (without the LLM)."""
        general_name = group.suggested_general_name
        param_name = group.suggested_param_name

        # Validate group validity
        if not group.members:
            self._log("[Sibling] Empty sibling group", "warning")
            return None

        # Use the first member as a template
        template_name = group.members[0]
        template_node = skill_nodes.get(template_name)
        if not template_node:
            self._log(f"[Sibling] Template skill '{template_name}' does not exist", "warning")
            return None

        template_code = template_node.code
        if not template_code:
            self._log(f"[Sibling] Template skill '{template_name}' code is empty", "warning")
            return None

        # Parameterize the template code
        variant_value = group.variants.get(template_name)
        if not variant_value:
            self._log(f"[Sibling] Template skill '{template_name}' has no variant value", "warning")
            return None

        # Ensure variant_value is not a blank string
        variant_value = variant_value.strip()
        if not variant_value:
            self._log(f"[Sibling] Template skill '{template_name}' variant value is blank", "warning")
            return None

        general_code = template_code.replace(f'"{variant_value}"', param_name)

        # Replace the variant anywhere else in the code as well
        for possible_form in [variant_value, variant_value.replace('_', ' '), variant_value.title()]:
            general_code = general_code.replace(f'"{possible_form}"', param_name)

        # Verify whether the variant value was successfully replaced
        if general_code == template_code:
            self._log(
                f"[Sibling] Warning: could not find variant value '{variant_value}' in template code; "
                f"the generated generic skill may need manual inspection",
                "warning"
            )

        # Modify the function signature
        sig_match = re.search(
            rf'async\s+function\s+{re.escape(template_name)}\s*\(([^)]*)\)',
            general_code
        )
        if sig_match:
            old_sig = sig_match.group(0)
            params = sig_match.group(1)
            if params.strip():
                new_params = f"{params}, {param_name}"
            else:
                new_params = f"bot, {param_name}"
            new_sig = f'async function {general_name}({new_params})'
            general_code = general_code.replace(old_sig, new_sig)

        # Generate each wrapper
        wrappers = {}
        param_values = {}

        for name, variant in group.variants.items():
            wrappers[name] = self._generate_wrapper(name, general_name, param_name, variant)
            param_values[name] = variant

        return SiblingPlan(
            general_skill_name=general_name,
            general_skill_code=general_code,
            general_skill_description=f"General skill for {group.pattern} variants",
            param_name=param_name,
            wrappers=wrappers,
            param_values=param_values,
        )

    def _generate_wrapper(
        self,
        skill_name: str,
        general_name: str,
        param_name: str,
        param_value: str
    ) -> str:
        """Generate wrapper code."""
        return f"""async function {skill_name}(bot) {{
    // Sibling wrapper: calls {general_name} with {param_name}="{param_value}"
    return await {general_name}(bot, "{param_value}");
}}"""

    def _create_plan_with_llm(
        self,
        group: SiblingGroup,
        skill_nodes: Dict[str, 'SkillNode'],
        retry_hint: Optional[str] = None,
    ) -> Optional[SiblingPlan]:
        """Use the LLM to create the refactor plan.

        v3.H+ Layer 3: when retry_hint is set (i.e. a previous attempt's
        general_skill_code failed Babel validation), append the hint as an
        extra HumanMessage so the LLM can correct its previous output.
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage
            import json

            # Prepare skills info
            skills_info = []
            for name in group.members:
                node = skill_nodes.get(name)
                if node:
                    skills_info.append(f"### {name} (variant: {group.variants.get(name, '?')})\n```javascript\n{node.code}\n```")

            from .prompts import PromptLoader

            template = PromptLoader.load("sibling_unification")
            system_prompt, human_prompt = template.format(
                pattern=group.pattern,
                skills_info=chr(10).join(skills_info),
                variants=group.variants,
                suggested_param_name=group.suggested_param_name,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            if retry_hint:
                messages.append(HumanMessage(content=retry_hint))

            response = self.llm.invoke(messages)
            record_llm_usage(response, process_type="refactor", function_name="refactor.sibling._create_plan_with_llm")
            content = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    # Verify required fields exist
                    required_fields = ["general_skill_name", "general_skill_code",
                                       "general_skill_description", "param_name", "wrappers", "param_values"]
                    missing = [f for f in required_fields if f not in data]
                    if missing:
                        self._log(f"[Sibling] LLM response missing required fields: {missing}", "warning")
                    else:
                        return SiblingPlan(
                            general_skill_name=data["general_skill_name"],
                            general_skill_code=data["general_skill_code"],
                            general_skill_description=data["general_skill_description"],
                            param_name=data["param_name"],
                            wrappers=data["wrappers"],
                            param_values=data["param_values"],
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    self._log(f"[Sibling] Failed to parse LLM response: {e}", "warning")
            else:
                self._log("[Sibling] No valid JSON found in LLM response", "warning")

        except Exception as e:
            self._log(f"[Sibling] LLM planning failed: {e}", "warning")

        return self._create_plan_simple(group, skill_nodes)

    def rollback(self, result: RefactorResult) -> bool:
        """Roll back the sibling-relation refactor."""
        if not result.rollback_available or not result.rollback_data:
            self._log(f"[Sibling] Cannot roll back: no rollback data", "warning")
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
                        self._log(f"[Sibling] Restored '{skill_name}'", "info")

            # Delete the created general skill
            general_name = rollback_data.get("general_skill_name")
            if general_name and self.skill_graph_manager:
                if self.skill_graph_manager.get_node(general_name):
                    self.skill_graph_manager.remove_skill_node(general_name)
                    self._log(f"[Sibling] Deleted '{general_name}'", "info")

            # Roll back caller updates (if any)
            caller_rollback_data = rollback_data.get("caller_rollback_data")
            if caller_rollback_data:
                self.rollback_caller_updates(caller_rollback_data)

            self._log(f"[Sibling] ✓ Rollback complete", "info")
            return True

        except Exception as e:
            self._log(f"[Sibling] Rollback failed: {e}", "error")
            return False

def detect_sibling_opportunities(
    skill_graph_manager,
    skill_names: Optional[List[str]] = None,
    min_group_size: int = 2,
    logger=None,
    domain_knowledge=None,
) -> List[RefactorOpportunity]:
    """
    Detect sibling-relation refactor opportunities.

    Args:
        skill_graph_manager: Skill graph manager
        skill_names: List of skills to check (None means all)
        min_group_size: Minimum group size
        logger: Logger
        domain_knowledge: DomainKnowledge instance (optional)

    Returns:
        List[RefactorOpportunity]: Detected sibling-relation opportunities
    """
    opportunities = []

    if not skill_graph_manager:
        return opportunities

    # Determine which skills to check
    if skill_names is None:
        skill_names = skill_graph_manager.get_all_skill_names(include_task_specific=True)

    # Plan v3-rev Fix 1.B: filter out skills that are already covered by a
    # prior refactor (is_covered=True). Re-grouping these as siblings causes
    # the v11 cycle: optimizer expands a wrapper → wrapper protection misses
    # → refactor scans again → re-wraps → optimizer expands again → ...
    # Already-covered skills should not be re-considered for refactor.
    filtered_skill_names = []
    skipped = []
    for name in skill_names:
        node = skill_graph_manager.get_node(name)
        if node and getattr(node, 'is_covered', False):
            skipped.append(name)
            continue
        filtered_skill_names.append(name)
    if skipped and logger:
        logger.info(
            f"[detect_sibling_opportunities] Fix 1.B: filtered out "
            f"{len(skipped)} already-covered skills: {skipped[:5]}{'...' if len(skipped) > 5 else ''}"
        )
    skill_names = filtered_skill_names

    if len(skill_names) < min_group_size:
        return opportunities

    # Group by name pattern (domain-injected patterns or class-level fallback)
    pattern_groups = defaultdict(list)

    variant_patterns = SiblingRefactor.VARIANT_PATTERNS
    if domain_knowledge:
        dk_patterns = domain_knowledge.get_variant_patterns()
        if dk_patterns:
            variant_patterns = dk_patterns

    for pattern, _ in variant_patterns:
        for name in skill_names:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                # Extract the pattern (replace variant part)
                pattern_key = re.sub(pattern, '*', name, flags=re.IGNORECASE)
                pattern_groups[pattern_key].append(name)

    # Group by name prefix
    prefix_groups = defaultdict(list)
    for name in skill_names:
        # Extract the prefix (up to the first uppercase letter or underscore)
        prefix = ""
        for i, c in enumerate(name):
            if c.isupper() and i > 0:
                break
            prefix += c
        if len(prefix) >= 3:
            prefix_groups[prefix].append(name)

    # Merge results
    all_groups = {}
    for pattern, members in pattern_groups.items():
        if len(members) >= min_group_size:
            all_groups[pattern] = members

    for prefix, members in prefix_groups.items():
        if len(members) >= min_group_size:
            key = f"{prefix}*"
            if key not in all_groups:
                all_groups[key] = members

    # Generate opportunities
    for pattern, members in all_groups.items():
        if len(members) >= min_group_size:
            opportunities.append(RefactorOpportunity(
                refactor_type=RefactorType.MERGE_SIBLINGS,
                source_skill=members[0],
                target_skill="",  # Not yet created
                reason=f"Sibling pattern: {pattern} ({len(members)} members)",
                confidence=min(1.0, len(members) / 5),  # More members = higher confidence
                covered_skills=members[1:],
            ))

    return opportunities
