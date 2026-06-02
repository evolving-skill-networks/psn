"""
ResponsibilityValidationMixin - Skill responsibility boundary validation.

Extracted from validation.py for modularity.
Contains 3 methods for skill responsibility validation, feedback relevance
filtering, and responsibility fulfillment checking.
"""

import re
import logging
from typing import Any, List, Tuple, TYPE_CHECKING

from langchain.schema import SystemMessage, HumanMessage

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer

logger = logging.getLogger(__name__)

# Fallback patterns when no domain knowledge is available.
# Empty dict means no keyword-based validation — all checks are LLM-only.
_FALLBACK_RESPONSIBILITY_PATTERNS = {}


class ResponsibilityValidationMixin:
    """
    Skill responsibility boundary validation methods.

    Provides validation for skill responsibility boundaries, feedback relevance
    filtering, and responsibility fulfillment checking.

    Required self attributes:
        - self.logger: Logger instance
        - self.skill_graph_manager: SkillGraphManager instance
        - self._invoke_llm_with_stats(): Method for LLM invocation
        - self._robust_json_parse(): Method for JSON parsing
    """

    # ========== Skill Responsibility Validation ==========

    def _validate_skill_responsibility(
        self: "SkillGraphOptimizer",
        skill_name: str,
        skill_description: str,
        original_code: str,
        new_code: str
    ) -> Tuple[bool, str]:
        """
        Use the LLM to verify that the optimized code respects the skill's
        responsibility boundary.

        Checks whether the optimization steps outside the skill's
        responsibility, e.g.:
        - mineLogs was given crafting logic (should not happen).
        - craftPickaxe was given mining logic (potentially problematic).

        Args:
            skill_name: skill name
            skill_description: skill description
            original_code: code before optimization
            new_code: code after optimization

        Returns:
            Tuple[bool, str]: (is_valid, reason)
            - is_valid: True if optimization is within responsibility.
            - reason: explanation when invalid.
        """
        # Get resource matcher from domain knowledge (if available)
        _dk = getattr(self, '_domain_knowledge', None)
        resource_matcher = _dk.get_resource_matcher() if _dk else None

        # Read Phase 1 gradients (set by optimizer_callback before quick_optimize_skill).
        # When non-empty, inject root-cause + suggested-fix into the LLM prompt so the
        # validator can distinguish "helper implements diagnosed fix" (accept) from
        # "helper introduces unrelated-skill logic" (still reject). The structure-only
        # boundary check otherwise rejects critical correctness fixes that happen to
        # introduce structural helpers (cycle-42 ensureRawIron smelt-removal regression).
        phase1_gradients = getattr(self, '_current_phase1_gradients', None) or []
        phase1_diagnosis_text = ""
        if phase1_gradients:
            parts = []
            for g in phase1_gradients[:4]:  # cap to first 4 gradients
                gt = getattr(g, 'gradient_type', None)
                gt_str = gt.value if hasattr(gt, 'value') else str(gt)
                direction = (getattr(g, 'direction', '') or '')[:300]
                fix = (getattr(g, 'suggested_fix', '') or '')[:300]
                parts.append(f"  [{gt_str}] direction: {direction}\n    suggested fix: {fix}")
            phase1_diagnosis_text = (
                "Root cause + recommended fix from upstream Phase 1 reflection:\n"
                + "\n".join(parts)
            )

        # Extract function names from old and new code
        def extract_function_names(code: str) -> set:
            """Extract function names defined in the code."""
            # Match async function name() and function name()
            async_funcs = set(re.findall(r'async\s+function\s+(\w+)\s*\(', code))
            regular_funcs = set(re.findall(r'(?<!async\s)function\s+(\w+)\s*\(', code))
            # Match const name = async () => and const name = () =>
            arrow_funcs = set(re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>', code))
            return async_funcs | regular_funcs | arrow_funcs

        original_funcs = extract_function_names(original_code)
        new_funcs = extract_function_names(new_code)
        added_funcs = new_funcs - original_funcs

        # Check whether code length changed significantly (may indicate unrelated logic)
        original_lines = len(original_code.split('\n'))
        new_lines = len(new_code.split('\n'))
        significant_growth = (new_lines - original_lines) > 50  # Added more than 50 lines

        # ========== Phase 1: resource-name consistency check ==========
        # Use the domain-injected resource matcher to verify that resources in
        # the code are consistent with the skill name.
        if resource_matcher is not None:
            try:
                expected_resources = resource_matcher.extract_resource_from_skill_name(skill_name)

                if expected_resources:
                    unexpected_resources = resource_matcher.find_unexpected_resources(new_code, expected_resources)

                    if unexpected_resources:
                        # Resource-name mismatch is a serious error — reject outright
                        violation_msg = (
                            f"Resource name mismatch: {skill_name} should handle {expected_resources}, "
                            f"but the code references {unexpected_resources}"
                        )
                        print(f"\033[31m[Resource Consistency] {violation_msg}\033[0m")
                        self.logger.warning(f"[Resource Consistency] {skill_name}: outright rejection - {violation_msg}")
                        return False, f"Resource mismatch: expected {expected_resources}, found {unexpected_resources}"
                    else:
                        print(f"\033[32m[Resource Consistency] {skill_name}: resource-name check passed\033[0m")
                else:
                    print(f"\033[33m[Resource Consistency] {skill_name}: could not extract expected resources, skipping check\033[0m")
            except Exception as e:
                print(f"\033[33m[Resource Consistency] check error: {e}, skipping\033[0m")
        # ========== End Phase 1 ==========

        # Check whether keywords inconsistent with the skill name were added
        def check_unrelated_keywords(skill_name: str, code: str) -> List[str]:
            """Check whether the code contains keywords outside the skill's responsibility."""
            unrelated = []
            skill_lower = skill_name.lower()

            print(f"\033[36m[Responsibility Check] checking keywords: skill_lower='{skill_lower}'\033[0m")

            # Use domain-injected validation patterns
            validation_patterns = (
                _dk.get_responsibility_validation_patterns() if _dk else {}
            ) or _FALLBACK_RESPONSIBILITY_PATTERNS

            for _key, config in validation_patterns.items():
                match_fn = config.get("match")
                if match_fn and match_fn(skill_lower):
                    # Pattern-based check
                    patterns = config.get("patterns")
                    if patterns:
                        print(f"\033[36m[Responsibility Check] detected {_key}-style skill, checking unrelated items...\033[0m")
                        for pattern, desc in patterns:
                            if re.search(pattern, code, re.IGNORECASE):
                                print(f"\033[31m[Responsibility Check] detected unrelated keyword: {desc}\033[0m")
                                unrelated.append(desc)
                    # Custom check function
                    check_fn = config.get("check_fn")
                    if check_fn:
                        extra = check_fn(skill_lower, code)
                        if extra:
                            for desc in extra:
                                print(f"\033[31m[Responsibility Check] detected unrelated logic: {desc}\033[0m")
                            unrelated.extend(extra)
                    break  # Only match first applicable pattern group

            if not unrelated:
                print(f"\033[32m[Responsibility Check] no unrelated keywords detected\033[0m")

            return unrelated

        unrelated_keywords = check_unrelated_keywords(skill_name, new_code)

        # On clear responsibility breaches, reject directly (without involving the LLM)
        if unrelated_keywords:
            # Check whether this is a severe breach (crafting logic for a different item type)
            severe_keywords = (_dk.get_severe_responsibility_violation_keywords() if _dk else set())
            severe_violations = [kw for kw in unrelated_keywords
                                 if any(sk in kw.lower() for sk in severe_keywords)]
            if severe_violations:
                violation_msg = f"Detected unrelated logic in {skill_name}: {', '.join(severe_violations)}"
                self.logger.warning(f"\033[31m[Responsibility Check] {skill_name}: outright rejection - {violation_msg}\033[0m")
                print(f"\033[31m[Responsibility Check] outright rejection: unrelated crafting logic appeared in {skill_name}\033[0m")
                print(f"\033[31m[Responsibility Check] detected: {severe_violations}\033[0m")
                return False, f"Direct rejection: {violation_msg}"

        # If no functions were added, growth is not significant, and no unrelated keywords, skip validation
        if not added_funcs and not significant_growth and not unrelated_keywords:
            return True, "No new functions added and no significant changes"

        # If unrelated keywords were detected, force LLM validation
        force_validate_reason = ""
        if unrelated_keywords:
            force_validate_reason = f"Detected potentially unrelated logic: {', '.join(unrelated_keywords)}"
        elif significant_growth:
            force_validate_reason = f"Significant code growth: {original_lines} -> {new_lines} lines"

        # Use the LLM to validate.
        # Build domain-specific prompt context
        domain_prompt_context = _dk.get_responsibility_prompt_context() if _dk else ""

        system_prompt = """You are validating if code optimization respects skill responsibility boundaries.

SKILL RESPONSIBILITY PRINCIPLE:
Each skill has a PRIMARY responsibility based on its name, but may CALL other skills as dependencies.
"""
        if domain_prompt_context:
            system_prompt += domain_prompt_context
        system_prompt += """

Return JSON:
{
  "valid": true/false,
  "reason": "explanation of why it's valid or invalid",
  "violations": ["list of functions that violate responsibility boundary"],
  "severity": "high" | "medium" | "low"  // Only if invalid
}"""

        # Analyze each newly added function's line count and characteristics
        def get_function_info(code: str, func_name: str) -> dict:
            """Get detailed info about the function (line count, whether it looks like a standalone skill)."""
            # Try locating the function definition
            patterns = [
                rf'async\s+function\s+{func_name}\s*\([^)]*\)\s*\{{',
                rf'function\s+{func_name}\s*\([^)]*\)\s*\{{',
                rf'(?:const|let|var)\s+{func_name}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{'
            ]
            for pattern in patterns:
                match = re.search(pattern, code)
                if match:
                    start = match.start()
                    # Find the function end (simple brace match)
                    brace_count = 0
                    in_func = False
                    end = start
                    for i, char in enumerate(code[start:]):
                        if char == '{':
                            brace_count += 1
                            in_func = True
                        elif char == '}':
                            brace_count -= 1
                            if in_func and brace_count == 0:
                                end = start + i + 1
                                break
                    func_code = code[start:end]
                    line_count = func_code.count('\n') + 1

                    # Heuristic: does it look like a standalone skill (uses bot.chat, await, complex logic)?
                    has_bot_chat = 'bot.chat' in func_code
                    has_await = 'await' in func_code
                    is_async = 'async' in func_code[:50]

                    return {
                        'name': func_name,
                        'lines': line_count,
                        'is_async': is_async,
                        'has_bot_chat': has_bot_chat,
                        'has_await': has_await,
                        'likely_skill': line_count > 15 and is_async and (has_bot_chat or has_await)
                    }
            return {'name': func_name, 'lines': 0, 'likely_skill': False}

        func_infos = [get_function_info(new_code, func) for func in added_funcs]
        func_analysis = "\n".join([
            f"  - {info['name']}: {info['lines']} lines, async={info.get('is_async', False)}, "
            f"has_bot_chat={info.get('has_bot_chat', False)}, likely_standalone_skill={info.get('likely_skill', False)}"
            for info in func_infos
        ])

        # Build warning text
        warnings_text = ""
        if force_validate_reason:
            warnings_text = f"\n**WARNING - DETECTED ISSUE**: {force_validate_reason}\n"
            warnings_text += "Pay extra attention to whether the optimization adds logic that belongs to a DIFFERENT skill!\n"

        # Even without added functions, still need to check when warnings are present
        if not added_funcs:
            func_analysis = "No new functions added, but checking for unrelated logic in existing code."

        # If Phase 1 produced gradients, inject them with a leniency clause for
        # helpers that implement the diagnosed fix. When gradients are missing,
        # phase1_section is empty and the prompt is byte-equivalent to the original.
        phase1_section = ""
        if phase1_diagnosis_text:
            phase1_section = (
                f"\n## UPSTREAM DIAGNOSIS (FROM PHASE 1)\n{phase1_diagnosis_text}\n\n"
                f"## STRUCTURAL ANALYSIS\n"
                f"Helper functions added in the optimization may be structural refactors\n"
                f"required to implement the Phase 1 recommended fix. Do NOT reject solely\n"
                f"because helpers exist — judge whether the OVERALL CHANGE addresses the\n"
                f"root cause Phase 1 identified, regardless of internal helper structure.\n"
                f"HOWEVER: if helpers introduce logic that belongs to a DIFFERENT skill's\n"
                f"primary responsibility (e.g. crafting logic inside a mining skill, smelting\n"
                f"logic inside a crafting skill, or any logic unrelated to Phase 1's root\n"
                f"cause), you SHOULD reject — the diagnosis-injected leniency is ONLY for\n"
                f"helpers that implement the diagnosed fix.\n"
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Skill Name: {skill_name}
Skill Description: {skill_description or "No description available"}
{warnings_text}
New functions added in optimization:
{', '.join(added_funcs) if added_funcs else "None"}

Function Analysis (to help determine if they are helpers or standalone skills):
{func_analysis}

Code size change: {original_lines} lines -> {new_lines} lines

New Code (showing new functions):
```javascript
{new_code[:8000]}
```
{phase1_section}
Is this optimization within the skill's responsibility boundary?
IMPORTANT: If the skill is "craftCraftingTable", it should ONLY craft crafting tables, not stone_pickaxe or other items!
Return only JSON."""),
        ]

        try:
            response = self._invoke_llm_with_stats(
                messages=messages,
                process_type="validation",
                function_name="_validate_skill_responsibility",
                skill_name=skill_name,
            )

            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = self._robust_json_parse(json_match.group(), "Skill Responsibility Validation")
                if result:
                    is_valid = result.get("valid", True)
                    reason = result.get("reason", "Unknown")

                    if not is_valid:
                        violations = result.get("violations", [])
                        severity = result.get("severity", "medium")
                        self.logger.warning(f"\033[33m[Responsibility Check] {skill_name}: {reason}\033[0m")
                        self.logger.warning(f"\033[33m[Responsibility Check] Violations: {violations}, Severity: {severity}\033[0m")
                        return False, f"{reason} (Violations: {violations})"

                    return True, reason

            # If parsing fails, default to passing (conservative)
            return True, "Could not parse validation result"

        except Exception as e:
            self.logger.warning(f"[Responsibility Check] Validation failed: {e}")
            # Conservative default: pass
            return True, f"Validation error: {e}"
