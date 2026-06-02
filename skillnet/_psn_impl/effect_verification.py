"""
EffectVerificationMixin for PSNAgent

Methods for checking skill effect achievement via rule-based and LLM-based
verification.
"""

import re
from typing import TYPE_CHECKING

from skillnet.agents.planning import normalize_operation
from skillnet.utils.stats_tracker import record_llm_usage

if TYPE_CHECKING:
    from ..psn import PSNAgent


class EffectVerificationMixin:
    """Skill effect verification (rule-based + LLM fallback)."""

    def _check_skill_effect_achieved(
        self,
        skill_name: str,
        pre_state: dict,
        post_state: dict
    ) -> tuple:
        """
        Check whether the skill's expected effects are reflected in state changes.

        By comparing pre- and post-execution state, validates whether the skill's expected_effects are achieved.
        This is more accurate than just checking whether the code raised, since code may complete without achieving the intended effects.

        Args:
            skill_name: skill name
            pre_state: pre-execution state {"inventory": {...}, "position": {...}, "equipment": {...}}
            post_state: post-execution state {"inventory": {...}, "position": {...}, "equipment": {...}}

        Returns:
            Tuple[bool, str, dict]: (success, reason, detailed state changes)
        """
        # Cannot verify without state info
        if not pre_state and not post_state:
            return True, "No state information available for verification", {}

        # Check whether the skill is in the graph
        if not self.skill_manager.has_node(skill_name):
            return True, f"Skill '{skill_name}' not in graph (cannot verify effects)", {}

        node = self.skill_manager.get_node(skill_name)

        # Check whether expected_effects exist
        if not node.expected_effects:
            return True, f"Skill '{skill_name}' has no expected_effects defined", {}

        # Record state-change details
        state_changes = {
            "inventory_changes": {},
            "equipment_changes": {},
            "position_changes": {},
        }

        # Compute inventory changes
        pre_inv = pre_state.get("inventory", {}) if pre_state else {}
        post_inv = post_state.get("inventory", {}) if post_state else {}

        # Ensure inventory is a dict
        if not isinstance(pre_inv, dict):
            pre_inv = {}
        if not isinstance(post_inv, dict):
            post_inv = {}

        all_items = set(pre_inv.keys()) | set(post_inv.keys())
        for item in all_items:
            pre_count = pre_inv.get(item, 0)
            post_count = post_inv.get(item, 0)
            if pre_count != post_count:
                state_changes["inventory_changes"][item] = {
                    "before": pre_count,
                    "after": post_count,
                    "delta": post_count - pre_count
                }

        # Compute equipment changes
        pre_equip = pre_state.get("equipment", {}) if pre_state else {}
        post_equip = post_state.get("equipment", {}) if post_state else {}

        if not isinstance(pre_equip, dict):
            pre_equip = {}
        if not isinstance(post_equip, dict):
            post_equip = {}

        all_slots = set(pre_equip.keys()) | set(post_equip.keys())
        for slot in all_slots:
            pre_item = pre_equip.get(slot)
            post_item = post_equip.get(slot)
            if pre_item != post_item:
                state_changes["equipment_changes"][slot] = {
                    "before": pre_item,
                    "after": post_item
                }

        # Check each expected_effect's state_representation.
        # An effect counts as achieved only when the observed delta meets the
        # declared count (not on any positive delta). is_primary effects are
        # tracked separately: skills with any is_primary effect must achieve
        # ALL primaries; skills with none fall back to any-one-wins.
        effects_checked = 0
        effects_achieved = 0
        primary_checked = 0
        primary_achieved = 0
        effect_details = []

        for effect in node.expected_effects:
            state_repr = effect.state_representation
            if not state_repr or not isinstance(state_repr, dict):
                continue

            effect_type = state_repr.get("type", "")
            is_primary = bool(getattr(effect, "is_primary", False))

            if effect_type == "inventory":
                effects_checked += 1
                if is_primary:
                    primary_checked += 1
                item = state_repr.get("item", "")
                # use normalize_operation to unify "ensure" -> "add"
                operation = normalize_operation(state_repr.get("operation", "add"))
                # LLM-generated effect metadata sometimes emits count as a
                # string (e.g. "1") instead of int. Without coercion, the
                # downstream `delta >= expected_count` comparison raises
                # TypeError: '>=' not supported between instances of 'int'
                # and 'str' — observed in ckpt_diamond_postB0_b iter 1 +
                # 8 where it killed two consecutive wooden_axe rollouts
                # and was not propagated to the optimizer as feedback, so
                # the next attempt regenerated the same broken effect and
                # crashed identically. Coerce defensively; on malformed
                # values fall back to the V0 default of 1 (so verification
                # proceeds with a conservative "any positive delta wins"
                # rather than crashing the rollout).
                _raw_count = state_repr.get("count", 1)
                try:
                    expected_count = int(_raw_count)
                except (TypeError, ValueError):
                    expected_count = 1

                # Handle the case where item may be a list
                items_to_check = [item] if isinstance(item, str) else (item if isinstance(item, list) else [])

                item_achieved = False
                for check_item in items_to_check:
                    if not check_item:
                        continue

                    pre_count = pre_inv.get(check_item, 0)
                    post_count = post_inv.get(check_item, 0)
                    delta = post_count - pre_count

                    if operation == "add":
                        if delta >= expected_count:
                            item_achieved = True
                            effect_details.append(f"{check_item} +{delta}")
                            break
                    elif operation == "remove":
                        if -delta >= expected_count:
                            item_achieved = True
                            effect_details.append(f"{check_item} {delta}")
                            break

                if item_achieved:
                    effects_achieved += 1
                    if is_primary:
                        primary_achieved += 1

            elif effect_type == "equipment":
                effects_checked += 1
                if is_primary:
                    primary_checked += 1
                slot = state_repr.get("slot", "hand")
                expected_item = state_repr.get("item", "")

                # Handle the case where expected_item may be a list
                if isinstance(expected_item, list):
                    expected_item = expected_item[0] if expected_item else ""
                elif not isinstance(expected_item, str):
                    expected_item = str(expected_item) if expected_item else ""

                post_equipped = post_equip.get(slot)
                equip_achieved = False

                # Check whether the expected item is equipped
                if expected_item:
                    if post_equipped and expected_item.lower() in str(post_equipped).lower():
                        effects_achieved += 1
                        effect_details.append(f"equipped {expected_item}")
                        equip_achieved = True
                else:
                    # As long as anything is equipped, count as success
                    if post_equipped:
                        effects_achieved += 1
                        effect_details.append(f"equipped {post_equipped}")
                        equip_achieved = True

                if equip_achieved and is_primary:
                    primary_achieved += 1

            elif effect_type == "block":
                # block-type effect verification
                # Note: block-change data lives in environment_events; the current method cannot access it directly
                # Therefore block effects fall through to LLM verification
                block_item = state_repr.get("item", "")
                block_operation = normalize_operation(state_repr.get("operation", "place"))
                # Do not count block effects in effects_checked; let them fall through to LLM verification
                # This is intentional: block effects need more sophisticated verification logic
                effect_details.append(f"block:{block_item}:{block_operation} (LLM verified)")

            elif effect_type == "nearby_block":
                # find-type effect verification: check whether the target block exists in post_state
                effects_checked += 1
                if is_primary:
                    primary_checked += 1
                target_item = state_repr.get("item", "")
                post_nearby = post_state.get("nearby_blocks", []) if post_state else []
                if target_item and any(target_item in block.lower() for block in post_nearby):
                    effects_achieved += 1
                    effect_details.append(f"found {target_item} nearby")
                    if is_primary:
                        primary_achieved += 1
                else:
                    effect_details.append(f"nearby_block:{target_item} not found in voxels")

        # Determine the result
        if effects_checked == 0:
            # No rule-checkable effects; use the LLM for semantic verification
            if getattr(self, 'use_llm_effect_verification', True):
                # Even without structured state_representation, expected_effects may have descriptions
                effect_descriptions = [e.description for e in node.expected_effects if e.description]
                if effect_descriptions or state_changes.get("inventory_changes"):
                    llm_result = self._llm_verify_skill_effect(
                        skill_name=skill_name,
                        skill_description=node.description if hasattr(node, 'description') else "",
                        expected_effects=effect_descriptions,
                        state_changes=state_changes,
                        pre_state=pre_state,
                        post_state=post_state
                    )
                    if llm_result is not None:
                        achieved, llm_reason = llm_result
                        return achieved, f"[LLM Verified] {llm_reason}", state_changes
            # Cannot verify; assume success
            return True, f"Skill '{skill_name}' has no verifiable state_representation effects", state_changes

        # Decision policy: if any effect is is_primary, require ALL primaries
        # to be achieved; otherwise any single achievement wins.
        if primary_checked > 0:
            rule_passed = (primary_achieved == primary_checked)
        else:
            rule_passed = (effects_achieved > 0)

        if rule_passed:
            reason = (
                f"Effect achieved: {', '.join(effect_details)} "
                f"({effects_achieved}/{effects_checked} effects"
                + (f", primary {primary_achieved}/{primary_checked}"
                   if primary_checked > 0 else "")
                + ")"
            )
            return True, reason, state_changes
        else:
            # No effects achieved; try LLM verification
            if getattr(self, 'use_llm_effect_verification', True):
                llm_result = self._llm_verify_skill_effect(
                    skill_name=skill_name,
                    skill_description=node.description if hasattr(node, 'description') else "",
                    expected_effects=[e.description for e in node.expected_effects],
                    state_changes=state_changes,
                    pre_state=pre_state,
                    post_state=post_state
                )
                if llm_result is not None:
                    achieved, llm_reason = llm_result
                    if achieved:
                        return True, f"[LLM Verified] {llm_reason}", state_changes
                    else:
                        return False, f"[LLM Verified] {llm_reason}", state_changes

            reason = f"Expected effects not achieved (0/{effects_checked} effects). State changes: {state_changes['inventory_changes']}"
            return False, reason, state_changes

    def _llm_verify_skill_effect(
        self,
        skill_name: str,
        skill_description: str,
        expected_effects: list,
        state_changes: dict,
        pre_state: dict,
        post_state: dict
    ) -> tuple:
        """
        Use the LLM to verify whether the skill's effects were achieved.

        When rule-based verification cannot determine the result, use the LLM for semantic analysis.
        The LLM analyzes the skill's expected effects and actual state changes to decide whether the skill completed its job.

        Args:
            skill_name: skill name
            skill_description: skill description
            expected_effects: list of expected effects (string descriptions)
            state_changes: detailed state changes
            pre_state: pre-execution state
            post_state: post-execution state

        Returns:
            Tuple[bool, str] or None: (success, reason) or None (if the LLM call fails)
        """
        try:
            # Build state-change description
            inv_changes = state_changes.get("inventory_changes", {})
            equip_changes = state_changes.get("equipment_changes", {})

            inv_desc_parts = []
            for item, change in inv_changes.items():
                delta = change.get("delta", 0)
                if delta > 0:
                    inv_desc_parts.append(f"+{delta} {item}")
                elif delta < 0:
                    inv_desc_parts.append(f"{delta} {item}")
            inv_desc = ", ".join(inv_desc_parts) if inv_desc_parts else "No inventory changes"

            equip_desc_parts = []
            for slot, change in equip_changes.items():
                before = change.get("before", "nothing")
                after = change.get("after", "nothing")
                equip_desc_parts.append(f"{slot}: {before} -> {after}")
            equip_desc = ", ".join(equip_desc_parts) if equip_desc_parts else "No equipment changes"

            # Build expected-effects description
            effects_desc = "\n".join([f"- {e}" for e in expected_effects]) if expected_effects else "No expected effects defined"

            # Build prompt
            prompt = f"""Analyze whether a Minecraft skill successfully achieved its intended effect.

**Skill Name:** {skill_name}
**Skill Description:** {skill_description or "No description available"}

**Expected Effects:**
{effects_desc}

**Actual State Changes:**
- Inventory: {inv_desc}
- Equipment: {equip_desc}

**Pre-execution Inventory:** {pre_state.get('inventory', {}) if pre_state else 'Unknown'}
**Post-execution Inventory:** {post_state.get('inventory', {}) if post_state else 'Unknown'}

**Question:** Based on the skill's name, description, and expected effects, did the actual state changes indicate that the skill successfully achieved its primary purpose?

Consider:
1. The skill name often indicates its purpose (e.g., "ensureLogs" should result in having logs)
2. A skill can be successful even if not all expected effects are achieved, as long as the primary purpose is fulfilled
3. If the inventory already contained the target items before execution, the skill may have correctly detected this and skipped unnecessary actions (which is still a success)

Respond in this exact JSON format:
{{"success": true/false, "reason": "brief explanation"}}"""

            # Call the LLM
            from langchain.schema import HumanMessage

            # Use a smaller model to save cost
            response = self.action_agent.llm.invoke([HumanMessage(content=prompt)])
            record_llm_usage(response, process_type="effect_verification", function_name="effect_verification._llm_verify_skill_effect", skill_name=skill_name)
            response_text = response.content.strip()

            # Parse JSON response
            import json

            # Try extracting JSON
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                success = result.get("success", False)
                reason = result.get("reason", "No reason provided")

                print(f"\033[36m[LLM Effect Verification] {skill_name}: {'Success' if success else 'Failed'} - {reason}\033[0m")
                return (success, reason)
            else:
                print(f"\033[33m[LLM Effect Verification] Failed to parse LLM response for {skill_name}\033[0m")
                return None

        except Exception as e:
            print(f"\033[33m[LLM Effect Verification] Error for {skill_name}: {e}\033[0m")
            return None
