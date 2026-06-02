"""
LLM-Based Helper Extraction for Skill Synthesis

After an optimization succeeds, uses LLM to identify reusable helper functions
in the optimized code. Generic helpers are extracted as independent skills.

Replaces the rule-based approach which could only detect top-level async
functions and missed nested/non-async helpers that LLMs commonly generate.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from skillnet.utils.stats_tracker import record_llm_usage


@dataclass
class ExtractedHelper:
    """A helper function candidate for extraction."""
    name: str
    code: str
    params: str = ""
    line_count: int = 0
    is_generic: bool = True
    registered_name: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of the helper extraction process."""
    extracted: List[ExtractedHelper] = field(default_factory=list)
    parent_updated: bool = False
    parent_new_code: str = ""


_INLINE_DEF_PATTERNS = (
    # `function name(...)` or `async function name(...)` — top-level or nested
    r'(?:^|[\s;])(?:async\s+)?function\s+{name}\s*\(',
    # `const|let|var name = async function ... / function ... / (...) =>`
    r'(?:const|let|var)\s+{name}\s*=\s*(?:async\s+)?'
    r'(?:function\s*\(|\([^)]*\)\s*=>)',
)


def _is_defined_inline(code: str, name: str) -> bool:
    """Return True iff `name` is defined as a function inside `code`.

    Detects function declarations and `const/let/var name = function/arrow`
    bindings — not call sites. Prevents HelperExtractor from accepting an
    LLM-hallucinated helper whose only appearance in the parent is as a
    call (e.g. `await mineLogs(bot, ...)`); without this check, the LLM
    can invent a brand-new body for the callee and overwrite the real skill.
    """
    pat_escaped = re.escape(name)
    for pat_tpl in _INLINE_DEF_PATTERNS:
        if re.search(pat_tpl.format(name=pat_escaped), code):
            return True
    return False


_HELPER_EXTRACTION_PROMPT = """Analyze this JavaScript skill code. The main function is "{main_function_name}".

Identify helper functions defined INSIDE the main function that could be extracted
as independent reusable skills for OTHER tasks (not just this one).

For each extractable helper, return a JSON object with:
- "name": the function name (string)
- "code": a REWRITTEN self-contained version of the function (string) — see rewriting rules below
- "is_generic": true if useful for other skills beyond this specific task (boolean)

REWRITING RULES — the returned code must work as a standalone async function:
1. Add "bot" as the FIRST parameter (the original may use bot from closure scope)
2. Add "const mcData = require('minecraft-data')(bot.version);" at the top if mcData is used
3. Add "const {{ Vec3 }} = require('vec3');" at the top if Vec3 is used
4. Make it "async function" (all extracted skills must be async)
5. The function must NOT rely on ANY closure variables from the parent function
6. Keep the original function name

Extraction criteria:
1. Non-trivial: more than 5 lines of actual logic
2. Generic: would benefit other skills (e.g., finding placement positions, searching for blocks)
3. NOT the main function itself

Do NOT extract:
- Simple one-liner utilities (e.g., item counting wrappers with <5 lines)
- Functions tightly coupled to this specific task's logic
- The main function "{main_function_name}" itself

```javascript
{code}
```

Return ONLY a JSON array. If no helpers worth extracting, return [].
Example: [{{"name": "findPlacementPosition", "code": "async function findPlacementPosition(bot, maxRadius = 2) {{\\n  const {{ Vec3 }} = require(\\"vec3\\");\\n  // ... self-contained logic ...\\n}}", "is_generic": true}}]"""


class HelperExtractor:
    """LLM-based helper function extractor.

    Uses LLM to semantically identify reusable helper functions in optimized
    skill code, regardless of whether they are async/sync, nested/top-level.

    Without LLM, extraction is skipped (returns empty list).
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    def detect_extractable_helpers(
        self,
        code: str,
        main_function_name: str,
    ) -> List[ExtractedHelper]:
        """Use LLM to identify extractable helper functions.

        Args:
            code: The optimized skill code
            main_function_name: The main function name (not a helper)

        Returns:
            List of ExtractedHelper candidates (only generic ones)
        """
        if not self._llm or not code:
            return []

        prompt = _HELPER_EXTRACTION_PROMPT.format(
            main_function_name=main_function_name,
            code=code[:8000],  # Limit code size for LLM context
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self._llm.invoke([HumanMessage(content=prompt)])
            record_llm_usage(response, process_type="optimization_synthesis", function_name="synthesis.extractor.detect_extractable_helpers", skill_name=main_function_name)
            response_text = response.content if hasattr(response, 'content') else str(response)
            return self._parse_llm_response(response_text, code)
        except Exception as e:
            print(f"\033[33m[Synthesis] LLM helper detection failed: {e}\033[0m")
            return []

    def _parse_llm_response(
        self, response_text: str, original_code: str
    ) -> List[ExtractedHelper]:
        """Parse LLM JSON response into ExtractedHelper list."""
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                return []

            data = json.loads(json_match.group())
            if not isinstance(data, list):
                return []

            helpers = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "")
                code = item.get("code", "")
                is_generic = item.get("is_generic", False)

                if not name or not code or not is_generic:
                    continue

                # Verify the helper is actually DEFINED inline in the original
                # code, not just called from there. The previous substring check
                # (`name + "("` in original_code) was satisfied by call sites
                # like `await mineLogs(...)`, so the LLM could hallucinate a
                # brand-new body for any callee and have it pass — leading to
                # overwrites of existing skills (e.g. mineLogs v1.0.3 →
                # v1.0.4 with bot.mineBlock hallucination during the wood-logs
                # task in ckpt_postfix).
                if not _is_defined_inline(original_code, name):
                    continue

                lines = [l for l in code.split('\n') if l.strip()]
                helpers.append(ExtractedHelper(
                    name=name,
                    code=code,
                    line_count=len(lines),
                    is_generic=is_generic,
                ))

            return helpers
        except (json.JSONDecodeError, KeyError) as e:
            print(f"\033[33m[Synthesis] Failed to parse LLM response: {e}\033[0m")
            return []

    def extract_and_register(
        self,
        helpers: List[ExtractedHelper],
        parent_skill_name: str,
        parent_code: str,
        skill_manager: Any,
        task: str = "",
        context: str = "",
    ) -> ExtractionResult:
        """Register extracted helpers as independent skills and update parent.

        For each helper:
        1. Wrap as async function with bot parameter if needed
        2. Register as new skill via skill_manager.add_new_skill()
        3. Remove inline definition from parent code
        """
        result = ExtractionResult()
        updated_code = parent_code

        for helper in helpers:
            # LLM already rewrites as self-contained async function
            skill_code = helper.code

            # Do NOT pass the parent's task / context to the extracted helper.
            # Passing them taints intent-based effect extraction: the helper
            # inherits the parent's target item as a polluted effect even when
            # the helper's own code only does a subset of the parent's work.
            # E.g. an ensureCraftingTable helper extracted from a
            # "Craft 1 diamond pickaxe" parent would get
            # `inventory diamond_pickaxe add` injected as an effect, even
            # though it just places a crafting table — later misleading
            # EffectMatcher into picking the helper as a candidate for
            # diamond_pickaxe targets. The helper's effects should come from
            # its own code / description only.
            info = {
                "program_name": helper.name,
                "program_code": skill_code,
                "task": "",
                "context": "",
                "update_source": f"synthesis:extracted_from_{parent_skill_name}",
            }

            try:
                registered_name = skill_manager.add_new_skill(info)
                if registered_name:
                    helper.registered_name = registered_name
                    result.extracted.append(helper)

                    updated_code = self._remove_inline_definition(
                        updated_code, helper.name
                    )

                    print(
                        f"\033[32m[Synthesis] Extracted helper '{helper.name}' "
                        f"({helper.line_count} lines) from '{parent_skill_name}' "
                        f"→ registered as '{registered_name}'\033[0m"
                    )
                else:
                    print(
                        f"\033[33m[Synthesis] Failed to register helper "
                        f"'{helper.name}' — add_new_skill returned None\033[0m"
                    )
            except Exception as e:
                print(
                    f"\033[33m[Synthesis] Failed to register helper "
                    f"'{helper.name}': {e}\033[0m"
                )

        if result.extracted and updated_code != parent_code:
            try:
                update_success = skill_manager.update_skill_code(
                    skill_name=parent_skill_name,
                    new_code=updated_code,
                    change_log=(
                        f"Extracted helpers: "
                        f"{', '.join(h.registered_name or h.name for h in result.extracted)}"
                    ),
                    source="synthesis:helper_extraction",
                    create_version=True,
                    skip_metadata=False,
                )
                if update_success:
                    result.parent_updated = True
                    result.parent_new_code = updated_code
            except Exception as e:
                print(
                    f"\033[33m[Synthesis] Failed to update parent skill "
                    f"'{parent_skill_name}': {e}\033[0m"
                )

        return result

    def _remove_inline_definition(self, code: str, helper_name: str) -> str:
        """Strip the inline `function helper_name(...)` (sync or async)
        plus its body from the parent code.

        Locates the declaration by NAME via regex, then walks a brace
        counter to find the matching closing brace. Previous version
        substring-matched against the LLM-rewritten standalone version
        (helper.code) which is structurally different from what's in the
        parent (adds `async`, adds bot parameter, etc.); the match always
        failed and the inline def was never removed. Side effect: the
        skill graph gained an async global skill while the parent kept
        the original sync helper, creating a name collision that the
        async validator misreported as crash_risk.
        """
        pattern = re.compile(
            rf'(?:async\s+)?function\s+{re.escape(helper_name)}\s*\([^)]*\)\s*\{{'
        )
        match = pattern.search(code)
        if not match:
            return code
        start = match.start()
        # Balanced-brace scan from just after the opening '{'.
        depth = 1
        i = match.end()
        while i < len(code) and depth > 0:
            c = code[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        if depth != 0:
            return code  # unbalanced — refuse to risk a corrupt edit
        end = i
        updated = code[:start] + code[end:]
        updated = re.sub(r'\n{3,}', '\n\n', updated)
        return updated
