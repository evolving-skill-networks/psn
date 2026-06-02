"""
EditContextMixin - Edit context building for LLM optimization prompts.

Extracted from optimizer_impl.py for better modularity.

Methods included:
- _build_edit_context: Assemble 3-layer edit context (MUST_FIX > LOCALIZATION > CONSTRAINTS)
- _generate_external_skills_warning: Generate warning about external skill dependencies
- _generate_composable_skills_context: Generate composable skills catalog
- _extract_focus_lines: Delegate to helpers.code_context
- _extract_code_with_context: Delegate to helpers.code_context
- _extract_function_signature: Delegate to helpers.code_context

Cross-mixin dependencies (resolved via MRO):
    None
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Set

from skillnet.agents.optimizer._impl.helpers import (
    extract_focus_lines as _extract_focus_lines_impl,
    extract_code_with_context as _extract_code_with_context_impl,
    extract_function_signature as _extract_function_signature_impl,
)

if TYPE_CHECKING:
    from ..optimizer_impl import SkillGraphOptimizer


class EditContextMixin:
    """Edit Context Mixin - Build structured context for LLM code editing.

    Attributes (from SkillGraphOptimizer):
        skill_graph_manager: SkillGraphManager instance
    """

    def _generate_external_skills_warning(self: "SkillGraphOptimizer", skill_name: str) -> str:
        """
        Generate a warning about external skills

        This warning tells the LLM not to redefine external skills that the current skill calls.
        The LLM frequently and incorrectly believes it needs to "fully implement" callee functions
        inside the optimized code, when in fact those functions already exist as independent skills
        in the skill graph.

        Args:
            skill_name: Name of the skill currently being optimized

        Returns:
            str: Warning message (empty string if there are no external skills)
        """
        node = self.skill_graph_manager.get_node(skill_name)
        if not node or not node.children:
            return ""

        # Collect info for all external skills
        external_skills_info = []
        for child_name in node.children:
            child_node = self.skill_graph_manager.get_node(child_name)
            if child_node:
                # Extract function signature
                sig_match = re.search(r'async\s+function\s+\w+\s*\([^)]*\)', child_node.code)
                signature = sig_match.group(0) if sig_match else f"async function {child_name}(bot, ...)"
                external_skills_info.append({
                    "name": child_name,
                    "signature": signature,
                    "description": child_node.description[:200] if child_node.description else "N/A"
                })

        if not external_skills_info:
            return ""

        # Build warning message
        warning = """
**🚨 CRITICAL - EXTERNAL SKILLS WARNING 🚨**

This skill calls the following EXTERNAL skills that ALREADY EXIST in the skill graph:

"""
        for skill_info in external_skills_info:
            warning += f"""- **{skill_info['name']}**
  - Signature: `{skill_info['signature']}`
  - Description: {skill_info['description']}...

"""

        warning += """**DO NOT redefine these functions in your optimized code!**
They are available as external skills and will be loaded automatically at runtime.

What you SHOULD do:
✅ Call these external skills directly (e.g., `await craftPlanks(bot, ...)`)
✅ Trust that they work correctly (they are tested separately)
✅ Focus only on fixing issues in the MAIN function

What you should NOT do:
❌ Define local versions of these functions
❌ Copy their implementation into your code
❌ Add "helper" functions with the same names

If you add local functions with the same names as external skills, they will SHADOW the external skills and cause the code to break!
"""
        return warning

    def _generate_composable_skills_context(self: "SkillGraphOptimizer", skill_name: str) -> str:
        """Generate composable skills context to guide LLM to call existing skills instead of inlining.

        Excludes task-specific wrappers (e.g., craft_1_wooden_pickaxe)
        from the list. Including them caused Qwen3 to call wrappers instead of
        reusable skills (50% placeCraftingTable → 100% after exclusion), and even
        produced recursive calls (craftWoodenPickaxe calling craft_1_wooden_pickaxe
        which calls craftWoodenPickaxe).
        """
        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            return ""

        # exclude task-specific wrappers — they pollute the list
        # and cause LLM to call wrappers instead of reusable skills
        all_skill_names = self.skill_graph_manager.get_all_skill_names(include_task_specific=False)
        current_children = set(node.children or [])
        composable = []
        skipped_task_specific = []
        skipped_cycle = []  # candidates excluded because they would form A→B→A

        for name in all_skill_names:
            if name == skill_name:
                continue
            if name in current_children:
                continue  # Already covered by wrapper_warning

            # Pre-filter cycle-creating candidates: the Cycle Check guard in
            # apply_optimization.py rejects any patch that calls a
            # skill which itself has a path back to skill_name, but it does so
            # AFTER the LLM has already written the patch. Showing such
            # candidates in the composable list invites the LLM to compose
            # them, the apply fails, and the optimizer loops re-proposing the
            # same delegation. Pre-filtering here closes the loop.
            if self.skill_graph_manager.would_create_cycle(skill_name, name):
                skipped_cycle.append(name)
                continue

            other = self.skill_graph_manager.get_node(name)
            if not other or not other.code:
                continue

            # Only include skills with some maturity (at least executed once)
            if other.statistics and other.statistics.total_executions >= 1:
                sig_match = re.search(
                    r'async\s+function\s+\w+\s*\([^)]*\)', other.code
                )
                signature = (sig_match.group(0) if sig_match
                             else f"async function {name}(bot)")
                composable.append({
                    "name": name,
                    "signature": signature,
                    "desc": (other.description or "")[:120],
                })

        if not composable:
            self.logger.info(
                f"\033[33m[Composable Skills] {skill_name}: no composable skills found "
                f"(graph has {len(all_skill_names)} non-task-specific skills)\033[0m"
            )
            return ""

        # Limit count to avoid prompt bloat
        composable = composable[:15]

        # Log what's in the composable list for diagnostics
        composable_names = [s["name"] for s in composable]
        self.logger.info(
            f"\033[36m[Composable Skills] {skill_name}: {len(composable)} skills in prompt: "
            f"{composable_names}\033[0m"
        )
        if skipped_cycle:
            self.logger.info(
                f"\033[33m[Composable Skills] {skill_name}: excluded {len(skipped_cycle)} "
                f"cycle-creating candidate(s): {skipped_cycle}\033[0m"
            )

        lines = [
            "",
            "## COMPOSABLE SKILLS (USE THESE INSTEAD OF INLINING)",
            "",
            "These registered skills can be called via `await skillName(bot, ...)`. "
            "**When you need functionality that an existing skill provides, CALL IT.** "
            "(This list has been pre-filtered to exclude any skill whose call would "
            "create a circular dependency with this one.)",
            "",
        ]
        for s in composable:
            lines.append(f"- `{s['signature']}` — {s['desc']}")

        lines.extend([
            "",
            "**COMPOSITION RULES:**",
            "- If you need to mine/craft/ensure a resource, check if a skill above already does it",
            "- Call directly: `await skillName(bot, args)` — these are global functions already in scope",
            "- Do NOT use require() to import them — they are NOT Node.js modules",
            "- Do NOT copy their implementation into your code",
            "",
        ])
        return "\n".join(lines)

    def _extract_focus_lines(
        self: "SkillGraphOptimizer",
        issues: List[Dict],
        suggested_fixes: List[Dict]
    ) -> Set[int]:
        """Delegates to helpers.code_context"""
        return _extract_focus_lines_impl(issues, suggested_fixes)

    def _extract_code_with_context(
        self: "SkillGraphOptimizer",
        code: str,
        focus_lines: Set[int],
        context: int = 5
    ) -> str:
        """Delegates to helpers.code_context"""
        return _extract_code_with_context_impl(code, focus_lines, context)

    def _extract_function_signature(self: "SkillGraphOptimizer", code: str) -> str:
        """Delegates to helpers.code_context"""
        return _extract_function_signature_impl(code)

    def _build_edit_context(
        self: "SkillGraphOptimizer",
        skill_name: str,
        code: str,
        issues: List[Dict],
        suggested_fixes: List[Dict],
        feedback_requirements: List[str],
    ) -> str:
        """
        Build a structured context for LLM code editing.
        Organized in layers: MUST_FIX > LOCALIZATION > CONSTRAINTS

        Args:
            skill_name: Name of the skill being edited
            code: Current code of the skill
            issues: List of identified issues
            suggested_fixes: List of suggested fixes with line ranges
            feedback_requirements: List of requirements from Phase 1 analysis
                (prefixed with [gradient_type, priority=X.X])

        Returns:
            Formatted context string for LLM prompt
        """
        # Layer 1: MUST FIX — Phase 1 analysis results passed directly
        requirements_list = ""
        if feedback_requirements:
            requirements_list = '\n'.join(
                f'[ ] {i+1}. {req}' for i, req in enumerate(feedback_requirements)
            )
        else:
            requirements_list = "(No specific requirements extracted)"

        must_fix = f"""
## LAYER 1: MUST FIX (Your code will be REJECTED if these are not addressed)

### Phase 1 Analysis Results (Address ALL):
{requirements_list}
"""

        # Layer 2: LOCALIZATION
        if issues:
            issues_text = "\n".join(
                f"- [{iss.get('type', 'issue')}] {iss.get('problem', iss.get('description', 'Unknown issue'))}"
                for iss in issues
            )
        else:
            issues_text = "(No specific issues identified)"

        # FIX: include code_example of suggested_fixes in the prompt
        if suggested_fixes:
            fixes_text_parts = []
            for fix in suggested_fixes:
                fix_entry = f"- **{fix.get('edit_type', 'suggested fix')}**"
                if fix.get('description'):
                    fix_entry += f": {fix['description']}"
                if fix.get('code_example'):
                    fix_entry += f"\n```javascript\n{fix['code_example']}\n```"
                fixes_text_parts.append(fix_entry)
            fixes_text = "\n".join(fixes_text_parts)
        else:
            fixes_text = "(No specific fixes suggested)"

        focus_lines = self._extract_focus_lines(issues, suggested_fixes)
        if focus_lines:
            focus_code = self._extract_code_with_context(code, focus_lines, context=5)
        else:
            focus_code = self._extract_code_with_context(code, set(), context=5)

        localization = f"""
## LAYER 2: PROBLEM LOCALIZATION

### Identified Issues:
{issues_text}

### Suggested Fixes (Analysis-based guidance):
{fixes_text}

### Focus Area (lines to modify):
```javascript
{focus_code}
```
"""

        # Layer 3: CONSTRAINTS
        external_warning = self._generate_external_skills_warning(skill_name)
        function_signature = self._extract_function_signature(code)

        constraints = f"""
## LAYER 3: CONSTRAINTS

{external_warning}

### Function Signature (DO NOT CHANGE):
```javascript
{function_signature}
```

### Code Completeness:
- Return COMPLETE code with ALL brackets matched
- Do NOT truncate any part of the code
"""

        return must_fix + localization + constraints
