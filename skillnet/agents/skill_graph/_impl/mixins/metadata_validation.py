"""
Metadata Validation Mixin for SkillGraphManager

Validation and load-time repair operations for skill effects.
Ensures effects are consistent with code and repairs mismatches on checkpoint load.

Extracted from metadata_mgmt.py for modularity.
Contains 6 methods:
  - _validate_and_clean_effects_on_load: Clean hallucinated effects on load
  - _repair_missing_primary_effects: Repair wrapper skills missing primary effects
  - _fix_function_name_mismatch_on_load: Fix function name vs registered name mismatches
  - _validate_effects_against_code: Validate effects have code implementation
  - _validate_or_effect_conditions: Validate OR logic effect conditions
  - _code_has_effect_implementation: Delegate to code_verifier module
"""

import copy
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from skillnet.agents.skill_graph.models import SkillEffect
from skillnet.agents.skill_graph.utils import (
    validate_code_syntax,
    code_has_effect_implementation,
)
import skillnet.utils as U

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class MetadataValidationMixin:
    """Metadata Validation Mixin - Effects validation and load-time repair.

    Methods:
        _validate_and_clean_effects_on_load: Clean hallucinated effects on load
        _repair_missing_primary_effects: Repair wrapper skills missing primary effects
        _fix_function_name_mismatch_on_load: Fix function name mismatches
        _validate_effects_against_code: Validate effects against code
        _validate_or_effect_conditions: Validate OR logic effect conditions
        _code_has_effect_implementation: Delegate to code_verifier module

    Attributes (from SkillGraphManager):
        graph: SkillGraph instance
        ckpt_dir: Checkpoint directory
        logger: Logger instance
    """

    # =========================================================================
    # Validation Methods - Effects against code
    # =========================================================================

    def _validate_and_clean_effects_on_load(self: "SkillGraphManager") -> Dict[str, Any]:
        """
        Validate and clean all skill effects on checkpoint load.

        Iterates all skills, validates each skill's effects against code,
        removes effects hallucinated by LLM but not implemented in code.

        Returns:
            Dict[str, Any]: {
                "cleaned_skills": [{"skill": name, "before": n, "after": m}, ...],
                "total_effects_removed": int
            }
        """
        result = {
            "cleaned_skills": [],
            "total_effects_removed": 0
        }

        for name, node in self.graph.nodes.items():
            # [Bug Fix] Skip task-specific skills (their effects don't need long-term maintenance)
            if getattr(node, 'is_task_specific', False):
                continue

            if not node.expected_effects:
                continue

            # Load skill code
            code_file = f"{self.ckpt_dir}/skill_graph/code/{name}{self._get_file_extension()}"
            if not os.path.exists(code_file):
                continue

            try:
                with open(code_file, 'r') as f:
                    code = f.read()
            except (IOError, OSError):
                continue

            before_count = len(node.expected_effects)

            # Use existing validation method
            validated, warnings = self._validate_effects_against_code(
                node.expected_effects, code
            )

            after_count = len(validated)

            if after_count < before_count:
                # Effects were removed
                node.expected_effects = validated
                result["cleaned_skills"].append({
                    "skill": name,
                    "before": before_count,
                    "after": after_count,
                    "removed": before_count - after_count
                })
                result["total_effects_removed"] += (before_count - after_count)

                # Also update the separate effects file. Use the shared serializer
                # so is_primary / condition are preserved here too -- the minimal
                # dict this path used to write dropped them, which would strip the
                # primary/by-product distinction (and the param-bound marker) from
                # any skill whose effects were trimmed on load.
                try:
                    effects_file = f"{self.ckpt_dir}/skill_graph/effects/{name}.json"
                    effects_data = self._serialize_effects(validated)
                    U.dump_json(effects_data, effects_file)
                except (IOError, OSError, TypeError) as e:
                    print(f"\033[31m[Effect Cleanup] Failed to save {name} effects: {e}\033[0m")

        return result

    def _repair_missing_primary_effects(self: "SkillGraphManager") -> Dict[str, Any]:
        """
        Repair skills that are missing primary effects on checkpoint load.

        Two cases:

        1. GENERAL skills produced by sibling refactoring (e.g. craftWoodenTool,
           covering craftWoodenAxe / craftWoodenPickaxe). Their product is
           parameter-selected, so the fixed sibling product effects get stripped
           by ``_validate_effects_against_code`` and name inference cannot recover
           it (``craftWoodenTool`` -> ``wooden_tool``, not a real item). Here we
           reconstruct an OR product primary from the *covered siblings' own*
           primary products -- the reliable source -- bound to the product
           parameter so it survives validation. This is what keeps EffectMatcher
           from mis-matching by-products.

        2. Ordinary wrapper/leaf skills that lost their primary during LLM-based
           effect inference. Fall back to name inference (the original behaviour).

        Returns:
            Dict with "repaired_skills" list of {"skill": name, "item": item(s)}
        """
        from skillnet.agents.skill_graph.metadata.effects import (
            infer_primary_effect_from_name,
            synthesize_general_primary_effect,
        )

        result = {"repaired_skills": []}

        for name, node in self.graph.nodes.items():
            if getattr(node, 'is_task_specific', False):
                continue

            # Check if skill already has a primary effect
            has_primary = any(
                getattr(e, 'is_primary', False) for e in node.expected_effects
            )
            if has_primary:
                continue

            # Case 1: a general skill -> reconstruct from the covered siblings.
            inferred = None
            sibling_products, param_name = self._collect_sibling_products(name, node)
            if sibling_products:
                inferred = synthesize_general_primary_effect(sibling_products, param_name)

            # Case 2: fall back to name inference for ordinary skills.
            if inferred is None:
                inferred = infer_primary_effect_from_name(name)
            if not inferred:
                continue

            # Idempotency: skip if the existing effects already cover the item(s).
            inferred_items = self._effect_items(inferred)
            existing_items = set()
            for e in node.expected_effects:
                existing_items |= self._effect_items(e)
            if inferred_items and inferred_items <= existing_items:
                continue

            node.expected_effects.append(inferred)
            result["repaired_skills"].append({
                "skill": name,
                "item": sorted(inferred_items) if inferred_items else None,
            })

        return result

    def _collect_sibling_products(self, general_name: str, node) -> "Tuple[List[str], Optional[str]]":
        """For a GENERAL skill, gather the union of its covered siblings' primary
        product items and the product-selecting parameter name.

        Siblings are the skills whose ``covered_by`` points at this general skill;
        post-refactor each retains its own ``is_primary`` product effect (e.g.
        craftWoodenAxe keeps ``+wooden_axe``). Returns ``([], None)`` when this is
        not a general skill or no sibling products can be found.
        """
        siblings = [
            n for n in self.graph.nodes.values()
            if getattr(n, 'covered_by', None) == general_name
        ]
        if not getattr(node, 'is_general_skill', False) and not siblings:
            return [], None

        from skillnet.agents.skill_graph.metadata.effects import infer_primary_effect_from_name

        products: List[str] = []
        for sib in siblings:
            sib_items = []
            for e in (getattr(sib, 'expected_effects', None) or []):
                if getattr(e, 'is_primary', False):
                    sib_items.extend(self._effect_items(e))
            if not sib_items:
                # The sibling is a wrapper now; its primary may have been
                # transiently stripped by effect-validation and not yet re-added
                # (repair order is not guaranteed). Recover its product from its
                # OWN name -- which, unlike the general skill's name, is a valid
                # item (craftWoodenAxe -> wooden_axe). This makes reconstruction
                # independent of the order repair visits general vs sibling skills.
                inf = infer_primary_effect_from_name(getattr(sib, 'name', '') or '')
                if inf is not None:
                    sib_items.extend(self._effect_items(inf))
            products.extend(sib_items)
        products = list(dict.fromkeys(p for p in products if p))
        if not products:
            return [], None
        return products, self._infer_product_param_name(node)

    @staticmethod
    def _infer_product_param_name(node) -> str:
        """Pick the parameter that selects a general skill's product (e.g.
        ``toolType``). Prefer an input-semantic parameter that the code actually
        uses (so the reconstructed primary survives ``_effect_is_param_implemented``
        on the next validation pass); fall back to parsing the function signature."""
        code = getattr(node, 'code', '') or ''
        params = getattr(node, 'parameters', None) or {}
        _CONFIGISH = {
            'count', 'amount', 'quantity', 'total', 'targettotal',
            'maxdistance', 'craftingtablemaxdistance', 'preferredplank', 'plankvariant',
        }
        best, best_score = None, -1
        for pname, meta in params.items():
            if not isinstance(pname, str) or not pname:
                continue
            md = meta if isinstance(meta, dict) else {}
            blob = f"{md.get('semantic', '')} {md.get('direction', '')}".lower()
            score = 0
            if 'input' in blob:
                score += 2
            if pname.lower() not in _CONFIGISH:
                score += 1
            if code and re.search(rf'\b{re.escape(pname)}\b', code):
                score += 3
            if score > best_score:
                best, best_score = pname, score
        if best is not None:
            return best
        m = re.search(r'function\s+\w+\s*\(\s*bot\s*,\s*(\w+)', code)
        if m:
            return m.group(1)
        return 'toolType'

    def _fix_function_name_mismatch_on_load(self: "SkillGraphManager") -> Dict[str, Any]:
        """
        Fix function name vs registered name mismatches on checkpoint load.

        Iterates all skills, checks if code's main function name matches skill registered name,
        auto-corrects function name in code if mismatched.

        This fixes legacy inconsistencies (caused by LLM modifying function names during optimization).

        Returns:
            Dict[str, Any]: {
                "fixed_skills": [{"skill": name, "old_func_name": x, "new_func_name": y}, ...],
                "total_fixed": int
            }
        """
        result = {
            "fixed_skills": [],
            "total_fixed": 0
        }

        for name, node in self.graph.nodes.items():
            # Load skill code
            code_file = f"{self.ckpt_dir}/skill_graph/code/{name}{self._get_file_extension()}"
            if not os.path.exists(code_file):
                continue

            try:
                with open(code_file, 'r') as f:
                    code = f.read()
            except (IOError, OSError):
                continue

            # Extract main function name from code
            func_match = re.search(r'async\s+function\s+(\w+)\s*\(', code)
            if not func_match:
                continue

            actual_func_name = func_match.group(1)

            # Check if matches registered name
            if actual_func_name != name:
                # Step 1: Fix function definition name
                new_code = re.sub(
                    rf'(async\s+function\s+){re.escape(actual_func_name)}(\s*\()',
                    rf'\g<1>{name}\g<2>',
                    code,
                    count=1  # Only fix first match (main function definition)
                )

                # Step 2: Replace all variable references to old function name
                # This includes export parts like: module.exports.xxx = craftCraftingTable
                # Use word boundary to match complete identifiers only
                new_code = re.sub(
                    rf'\b{re.escape(actual_func_name)}\b',
                    name,
                    new_code
                )

                # Validate fixed syntax
                is_valid, error_msg = validate_code_syntax(new_code)
                if not is_valid:
                    print(f"\033[31m[Function Name Fix] Syntax error after fixing {name}, skipping: {error_msg}\033[0m")
                    continue

                # Save fixed code
                try:
                    with open(code_file, 'w') as f:
                        f.write(new_code)

                    # Also update in-memory node.code
                    node.code = new_code

                    result["fixed_skills"].append({
                        "skill": name,
                        "old_func_name": actual_func_name,
                        "new_func_name": name
                    })
                    result["total_fixed"] += 1
                except (IOError, OSError) as e:
                    print(f"\033[31m[Function Name Fix] Failed to save {name}: {e}\033[0m")
            else:
                # Edge case: function name correct, but may have stale references
                # Common case: function renamed from craftX to ensureX, but exports still reference craftX
                export_refs = re.findall(r'module\.exports\.\w+\s*=\s*(\w+)', code)
                globalthis_refs = re.findall(r'globalThis\.\w+\s*=\s*(\w+)', code)
                all_refs = set(export_refs + globalthis_refs)

                # Collect all function names defined in code (including internal helpers)
                defined_funcs = set(re.findall(r'(?:async\s+)?function\s+(\w+)\s*\(', code))

                # Find referenced but undefined function names
                undefined_refs = all_refs - defined_funcs
                if undefined_refs:
                    # Try to fix: replace undefined references with main function name
                    new_code = code
                    for old_ref in undefined_refs:
                        new_code = re.sub(
                            rf'\b{re.escape(old_ref)}\b',
                            name,
                            new_code
                        )

                    # Validate fixed syntax
                    is_valid, error_msg = validate_code_syntax(new_code)
                    if is_valid and new_code != code:
                        try:
                            with open(code_file, 'w') as f:
                                f.write(new_code)
                            node.code = new_code
                            result["fixed_skills"].append({
                                "skill": name,
                                "old_func_name": list(undefined_refs)[0],
                                "new_func_name": name,
                                "fix_type": "stale_reference"
                            })
                            result["total_fixed"] += 1
                            print(f"\033[33m[Function Name Fix] Fixed stale references in {name}: {undefined_refs}\033[0m")
                        except Exception as e:
                            print(f"\033[31m[Function Name Fix] Failed to save {name}: {e}\033[0m")

        return result

    def _validate_effects_against_code(
        self: "SkillGraphManager",
        effects: List[SkillEffect],
        code: str
    ) -> Tuple[List[SkillEffect], List[str]]:
        """
        Validate effects have corresponding implementation in code.

        Key fix: No longer adds effects not implemented in code, prevents LLM hallucination errors.

        For OR logic effects, validates each condition and keeps only implemented ones.

        Args:
            effects: Effects to validate
            code: Skill code

        Returns:
            Tuple[List[SkillEffect], List[str]]: (Validated effects, warnings list)
        """
        validated = []
        warnings = []

        for effect in effects:
            # Parameter-bound product effects on GENERAL skills (e.g.
            # craftWoodenTool's primary, an OR over {wooden_axe, wooden_pickaxe}
            # selected by `toolType`) never contain the product literal in the
            # parametric code -- the crafted item is chosen via the parameter. The
            # literal-implementation checks below would always discard them,
            # leaving the general skill with no primary effect. Exempt such effects
            # when their bound parameter is actually used by the code.
            if self._effect_is_param_implemented(effect, code):
                validated.append(effect)
                continue

            # Check OR logic in state_representation
            state_repr = getattr(effect, 'state_representation', None)

            if state_repr and isinstance(state_repr, dict) and state_repr.get('logic') == 'OR':
                # For OR logic, validate each condition and keep only implemented ones
                validated_effect = self._validate_or_effect_conditions(effect, code)
                if validated_effect:
                    validated.append(validated_effect)
                else:
                    warnings.append(
                        f"[DISCARDED] Effect '{effect.description}' has no implemented conditions in code"
                    )
                    print(f"\033[33m[Effect Validation] Discarding invalid OR effect: {effect.description[:60]}...\033[0m")
            else:
                # Normal effect, check if code has corresponding implementation
                has_implementation = self._code_has_effect_implementation(code, effect)

                if has_implementation:
                    validated.append(effect)
                else:
                    # Key fix: don't add unimplemented effects, prevent hallucination
                    warnings.append(
                        f"[DISCARDED] Effect '{effect.description}' is not implemented in code"
                    )
                    print(f"\033[33m[Effect Validation] Discarding unimplemented effect: {effect.description[:60]}...\033[0m")

        return validated, warnings

    @staticmethod
    def _effect_items(effect) -> set:
        """Collect the item names an effect concerns (handles scalar and OR/AND
        logic state_representations)."""
        sr = getattr(effect, 'state_representation', None)
        if not isinstance(sr, dict):
            return set()
        if "conditions" in sr and isinstance(sr.get("conditions"), list):
            items = set()
            for cond in sr["conditions"]:
                if isinstance(cond, dict) and cond.get("item"):
                    items.add(cond["item"])
            return items
        return {sr["item"]} if sr.get("item") else set()

    @staticmethod
    def _effect_is_param_implemented(effect, code: str) -> bool:
        """Whether an effect is a parameter-bound product whose selecting
        parameter is actually used by the code.

        General skills produced by sibling refactoring carry a primary product
        effect bound to a parameter (``effect.condition = {param_name: [...]}``);
        the product item is chosen at call time via that parameter, so the
        parametric code never contains the product literal. Such an effect is
        "implemented via the parameter" when the bound parameter name appears as
        an identifier in the code. This is intentionally narrow: effects without a
        ``condition`` (the overwhelming majority) are never exempted, so genuine
        hallucinations are still discarded by the literal-implementation checks.
        """
        cond = getattr(effect, 'condition', None)
        if not isinstance(cond, dict) or not cond:
            return False
        code = code or ""
        for param_name in cond:
            if isinstance(param_name, str) and param_name and \
                    re.search(rf'\b{re.escape(param_name)}\b', code):
                return True
        return False

    def _validate_or_effect_conditions(
        self: "SkillGraphManager",
        effect: SkillEffect,
        code: str
    ) -> Optional[SkillEffect]:
        """
        Validate each condition in an OR logic effect, keep only implemented ones.

        E.g., if effect claims can produce raw_iron OR iron_ingot OR iron_ore,
        but code only handles raw_iron, keep only raw_iron condition.

        Improvement: Supports dynamic pattern detection, e.g., endsWith("_log") keeps all log types.

        Args:
            effect: Effect with OR logic
            code: Skill code

        Returns:
            Optional[SkillEffect]: Validated effect, None if no valid conditions
        """
        state_repr = effect.state_representation
        conditions = state_repr.get('conditions', [])

        if not conditions:
            return None

        code_lower = code.lower()

        # === New: Detect dynamic item acquisition patterns ===
        # Build suffix→group mapping from domain knowledge
        suffix_to_group = {}
        if hasattr(self, '_domain_knowledge') and self._domain_knowledge:
            suffix_to_group = self._domain_knowledge.get_suffix_to_group()

        DYNAMIC_PATTERN_TO_GROUP = {}
        for suffix, group_name in suffix_to_group.items():
            pattern = rf'endswith\s*\(\s*["\']' + re.escape(suffix) + r'["\']\s*\)'
            DYNAMIC_PATTERN_TO_GROUP[pattern] = group_name

        # Collect valid items from detected dynamic patterns
        item_groups = {}
        if hasattr(self, '_domain_knowledge') and self._domain_knowledge:
            item_groups = self._domain_knowledge.get_item_groups()

        valid_items_from_patterns = set()
        detected_groups = []
        for pattern, group_name in DYNAMIC_PATTERN_TO_GROUP.items():
            if re.search(pattern, code_lower):
                if group_name and group_name in item_groups:
                    valid_items_from_patterns.update(
                        item.lower() for item in item_groups[group_name]
                    )
                    detected_groups.append(group_name)

        if detected_groups:
            print(f"\033[36m[Effect Validation] Detected dynamic patterns, categories: {detected_groups}\033[0m")
        # === End dynamic pattern detection ===

        validated_conditions = []

        for condition in conditions:
            item = condition.get('item', '')
            if not item:
                continue

            # Handle item being a list (OR logic)
            items_to_check = item if isinstance(item, list) else [item]
            item_found = False

            for single_item in items_to_check:
                if not isinstance(single_item, str):
                    continue
                item_lower = single_item.lower()

                # Check if item name appears in code
                # Use stricter matching: item name should appear as string constant
                item_patterns = [
                    f"'{item_lower}'",      # Single quote string
                    f'"{item_lower}"',      # Double quote string
                    f"name === '{item_lower}'",  # Name comparison
                    f'name === "{item_lower}"',
                    f"=== '{item_lower}'",
                    f'=== "{item_lower}"',
                ]

                item_in_code = any(pattern in code_lower for pattern in item_patterns)

                # New: Check if item belongs to detected dynamic category
                item_in_dynamic_category = item_lower in valid_items_from_patterns

                if item_in_code or item_in_dynamic_category:
                    item_found = True
                    if item_in_dynamic_category and not item_in_code:
                        print(f"\033[36m[Effect Validation] OR condition '{single_item}' kept via dynamic category match\033[0m")
                    break

            if item_found:
                validated_conditions.append(condition)
            else:
                item_str = item if isinstance(item, str) else str(item)
                print(f"\033[33m[Effect Validation] OR condition '{item_str}' not found in code, removing\033[0m")

        if not validated_conditions:
            return None

        # Create new effect with only validated conditions
        new_effect = copy.deepcopy(effect)

        if len(validated_conditions) == 1:
            # Only one condition, simplify to normal effect
            new_effect.state_representation = validated_conditions[0]
            # Update description
            item = validated_conditions[0].get('item', '')
            # Handle item being a list
            item_str = item if isinstance(item, str) else (item[0] if isinstance(item, list) and item else str(item))
            count = validated_conditions[0].get('count', 1)
            new_effect.description = f"Adds {count} {item_str} to inventory"
        else:
            # Multiple conditions, keep OR logic
            new_effect.state_representation = {
                'logic': 'OR',
                'conditions': validated_conditions
            }

        return new_effect

    def _code_has_effect_implementation(
        self: "SkillGraphManager",
        code: str,
        effect: SkillEffect
    ) -> bool:
        """Delegate to code_verifier module (v4.0 refactor)"""
        return code_has_effect_implementation(code, effect)
