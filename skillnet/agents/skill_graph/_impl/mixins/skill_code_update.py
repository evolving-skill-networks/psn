"""
SkillCodeUpdateMixin for SkillGraphManager

Skill code retrieval, update, and code change detection logic.
Includes:
- get_skill_code / update_skill_code: unified code read/write interface
- _update_task_specific_skill: task-specific skill file updates
- _notify_code_changed: code change notification with callback support
- _remove_toplevel_global_deps: strip top-level Vec3/mcData declarations
- _check_parameter_semantic_similarity / _check_functional_equivalence: semantic analysis
- _detect_function_name_conflicts / _fix_function_name_conflicts: conflict resolution
- _detect_code_changes: LLM-assisted code diff analysis

Extracted from graph_manager_impl.py for better maintainability.
"""

import os
import re
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.skill_graph.utils import validate_code_syntax, extract_function_calls
from skillnet.agents.skill_graph.models import CodeSource, UpdateResult, ConfirmResult

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class SkillCodeUpdateMixin:
    """Skill code retrieval, update, and code change detection."""

    def _notify_code_changed(
        self,
        skill_name: str,
        old_code: str,
        new_code: str,
        change_source: str = "unknown",
    ) -> None:
        """
        Notify on code change.

        Call this method when skill code changes; it triggers the registered callback (if any).

        Args:
            skill_name: skill name
            old_code: code before the change
            new_code: code after the change
            change_source: change source ("refactor", "version_update", "manual", "unknown")
        """
        # [P0 Fix] Recursion guard - prevents callback-triggered updates from causing infinite recursion
        if self._is_updating_code:
            self.logger.debug(f"[Code Changed] Skipping callback (recursion guard): {skill_name}")
            return

        if self._on_code_changed_callback:
            try:
                result = self._on_code_changed_callback(
                    skill_name=skill_name,
                    old_code=old_code,
                    new_code=new_code,
                    change_source=change_source,
                )
                if result and result.get("has_interface_change"):
                    print(f"\033[33m[Code Changed] {skill_name}: interface changed; callers handled\033[0m")
            except Exception as e:
                print(f"\033[31m[Code Changed] Callback failed: {e}\033[0m")

    def _remove_toplevel_global_deps(self, code: str) -> str:
        """
        Strip top-level Vec3 and mcData declarations from code to avoid conflicts with globalDepsCode.

        Only removes module-level (outside-function) declarations; keeps in-function declarations.

        Args:
            code: original code

        Returns:
            str: cleaned code
        """
        import re

        # Find the position of the first function definition
        func_match = re.search(r'(async\s+)?function\s+\w+\s*\(', code)
        if not func_match:
            # No function definition; return original code
            return code

        before_func = code[:func_match.start()]
        after_func = code[func_match.start():]

        # Strip top-level declarations from the part before the function
        # Handle multi-line `const { Vec3 } = require("vec3");` form
        multiline_vec3_pattern = r'const\s*\{\s*\n?\s*Vec3\s*\n?\s*\}\s*=\s*require\([\'"]vec3[\'"]\)\s*;?[^\n]*\n?'
        before_func = re.sub(multiline_vec3_pattern, '', before_func, flags=re.MULTILINE | re.DOTALL)

        # Handle single-line forms
        single_line_patterns = [
            # const { Vec3 } = require('vec3');
            r'const\s*\{\s*Vec3\s*\}\s*=\s*require\([\'"]vec3[\'"]\)[^\n]*\n?',
            # const Vec3 = require('vec3').Vec3;
            r'const\s+Vec3\s*=\s*require\([\'"]vec3[\'"]\)[^\n]*\n?',
            # var Vec3 = ...
            r'var\s+Vec3\s*=[^\n]*\n?',
            # const mcData = require('minecraft-data')
            r'const\s+mcData\s*=\s*require\([\'"]minecraft-data[\'"]\)[^\n]*\n?',
            # var mcData = ...
            r'var\s+mcData\s*=[^\n]*\n?',
        ]

        for pattern in single_line_patterns:
            before_func = re.sub(pattern, '', before_func, flags=re.MULTILINE)

        # Clean up extra blank lines and comment lines (e.g. "// keep here for any Vec3 usage")
        before_func = re.sub(r'//\s*keep here for any Vec3 usage[^\n]*\n?', '', before_func)
        before_func = re.sub(r'//\s*main function[^\n]*\n?', '', before_func)
        before_func = re.sub(r'\n{3,}', '\n\n', before_func)

        return before_func + after_func

    def get_skill_code(self, skill_name: str) -> Optional[str]:
        """
        Get skill code, supporting both the main graph and task-specific skills.

        Args:
            skill_name: skill name

        Returns:
            Optional[str]: skill code, or None if it does not exist
        """
        import os

        # First check the main graph
        if skill_name in self.graph.nodes:
            node = self.graph.get_node(skill_name)
            return node.code if node else None

        # Then check task-specific skills
        task_specific_path = os.path.join(
            self.ckpt_dir, "skill_graph", "apply_skills_for_task", f"{skill_name}{self._get_file_extension()}"
        )
        if os.path.exists(task_specific_path):
            try:
                with open(task_specific_path, 'r') as f:
                    return f.read()
            except (IOError, OSError) as e:
                print(f"\033[33m[Skill Graph] Failed to read task-specific skill: {e}\033[0m")
                return None

        return None

    def update_skill_code(
        self,
        skill_name: str,
        new_code: str,
        change_log: str = "",
        source: str = "unknown",
        skip_validation: bool = False,
        skip_metadata: bool = False,
        skip_interface_check: bool = False,
        create_version: bool = True,
        **kwargs
    ) -> bool:
        """
        Unified code-update interface; supports both the main graph and task-specific skills.

        Args:
            skill_name: skill name
            new_code: new code
            change_log: change description
            source: update source (optimizer/refactor/manual, etc.)
            skip_validation: whether to skip syntax validation (used for rollback)
            skip_metadata: whether to skip metadata update
            skip_interface_check: whether to skip interface-change analysis
            create_version: whether to create a new version
            **kwargs: extra parameters forwarded to create_new_version

        Returns:
            bool: whether the update succeeded
        """
        # Recursion guard
        if self._is_updating_code:
            self.logger.warning(f"[Code Update] Detected recursive call; skipping: {skill_name} <- {source}")
            return False

        try:
            self._is_updating_code = True

            # Audit log
            self.logger.info(f"[Code Update] {skill_name} <- {source}")

            # v3.H+ track refactor-driven modifications per step so that
            # psn.py:1002 add_new_skill + Layer 3 wrapper protection can
            # detect "skill was modified by refactor this step" and avoid
            # writing back action_agent's original inline code (covers
            # behavioral Case A and extract_common-callers edge cases
            # where is_covered stays False).
            if source and source.lower().startswith("refactor"):
                self._refactor_modified_skills_this_step.add(skill_name)

            # Wrapper architecture protection check
            if skill_name in self.graph.nodes:
                node = self.graph.get_node(skill_name)

                # Check whether wrapper architecture needs protection
                if node._should_protect_wrapper(new_code):
                    covered_by = getattr(node, 'covered_by', 'unknown')

                    # Convert source string to a type
                    source_lower = source.lower()

                    if source_lower == "optimizer":
                        # OPTIMIZER: redirect to general_skill
                        print(f"\033[33m[Code Update] Wrapper protection triggered: {skill_name} is a wrapper of {covered_by}\033[0m")
                        print(f"\033[33m[Code Update] Suggest modifying {covered_by} instead of {skill_name}\033[0m")
                        self.logger.warning(f"[Code Update] Wrapper redirect: {skill_name} -> {covered_by}")
                        return False
                    elif source_lower not in ["refactor", "manual"]:
                        # Non-trusted source (LLM, etc.): warn but allow (for execution validation)
                        print(f"\033[33m[Code Update] ⚠️ Wrapper architecture may be broken: {skill_name} no longer calls {covered_by}\033[0m")
                        self.logger.warning(f"[Code Update] Wrapper architecture may be broken: {skill_name}")

                # Identical-code check
                if new_code.strip() == node.code.strip():
                    self.logger.info(f"[Code Update] {skill_name} code unchanged")
                    return True

            # Pre-validation of syntax (unless skipped)
            if not skip_validation:
                is_valid, error_msg = validate_code_syntax(new_code)
                if not is_valid:
                    self.logger.error(f"[Code Update] [{source}] Rejecting syntactically invalid code ({skill_name}): {error_msg}")
                    print(f"\033[31m[Update Skill] [{source}] Syntax error: {error_msg}\033[0m")
                    return False

            # First check the main graph
            if skill_name in self.graph.nodes:
                node = self.graph.get_node(skill_name)
                if getattr(node, 'is_task_specific', False):
                    # Task-specific skill: update file and sync in-memory state
                    success = self._update_task_specific_skill(skill_name, new_code, change_log)
                    if success:
                        node.code = new_code  # sync the in-memory code
                        print(f"\033[32m[Task-Specific] Synced in-memory code update: {skill_name}\033[0m")
                    return success

                # Normal skill: use version management
                if create_version:
                    return self.create_new_version(
                        skill_name, new_code,
                        change_log=change_log,
                        update_source=source,
                        auto_update_metadata=not skip_metadata,
                        auto_update_callers=not skip_interface_check,
                        **kwargs
                    )
                else:
                    # No new version; update code directly
                    node.code = new_code
                    if node.versions:
                        node.versions[-1].code = new_code

                    # Update metadata if needed (Issue 9 fix: support metadata updates even when create_version=False)
                    if not skip_metadata:
                        description = getattr(node, 'description', '')
                        self._update_skill_metadata(skill_name, new_code, description, update_source=source)

                    self._save_to_checkpoint()
                    self.logger.info(f"[Code Update] {skill_name} code updated (no new version, metadata-update={not skip_metadata})")
                    return True

            # Then check task-specific skills (not in the in-memory graph but on disk)
            if self.is_task_specific_skill_in_storage(skill_name):
                return self._update_task_specific_skill(skill_name, new_code, change_log)

            print(f"\033[31m[Update Skill] Error: Skill '{skill_name}' not found in graph or task-specific directory\033[0m")
            return False

        finally:
            self._is_updating_code = False

    def _update_task_specific_skill(
        self,
        skill_name: str,
        new_code: str,
        change_log: str = ""
    ) -> bool:
        """
        Update the code of a task-specific skill.

        These skills are stored in the apply_skills_for_task directory.

        Args:
            skill_name: skill name
            new_code: new code
            change_log: change description

        Returns:
            bool: whether the update succeeded
        """
        import os
        import json
        from datetime import datetime

        task_skills_dir = os.path.join(self.ckpt_dir, "skill_graph", "apply_skills_for_task")
        code_file = os.path.join(task_skills_dir, f"{skill_name}{self._get_file_extension()}")
        metadata_file = os.path.join(task_skills_dir, f"{skill_name}.json")

        if not os.path.exists(code_file):
            print(f"\033[31m[Task-Specific] Error: Skill '{skill_name}' not found\033[0m")
            return False

        try:
            # Validate new-code syntax (post-optimization save must pass)
            is_valid, error_msg = validate_code_syntax(new_code)
            if not is_valid:
                print(f"\033[31m[Task-Specific] Syntax error in new code: {error_msg}\033[0m")
                return False

            # [Function-name consistency check] ensure the main function name in code matches the registered skill name
            func_name_match = re.search(r'async\s+function\s+(\w+)\s*\(', new_code)
            if func_name_match:
                actual_func_name = func_name_match.group(1)
                if actual_func_name != skill_name:
                    print(f"\033[33m[Task-Specific] Warning: main function name in code '{actual_func_name}' "
                          f"does not match registered skill name '{skill_name}'\033[0m")
                    print(f"\033[33m[Task-Specific] Auto-correcting function name: '{actual_func_name}' → '{skill_name}'\033[0m")

                    # Step 1: auto-correct the function definition in code
                    new_code = re.sub(
                        rf'(async\s+function\s+){re.escape(actual_func_name)}(\s*\()',
                        rf'\g<1>{skill_name}\g<2>',
                        new_code,
                        count=1
                    )

                    # Step 2: replace all references to the old function name in code
                    # This includes export sections like: module.exports.xxx = oldName
                    new_code = re.sub(
                        rf'\b{re.escape(actual_func_name)}\b',
                        skill_name,
                        new_code
                    )

            # Back up old code
            with open(code_file, 'r') as f:
                old_code = f.read()

            # Write the new code
            with open(code_file, 'w') as f:
                f.write(new_code)

            # Update metadata
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Append optimization history
                if "optimization_history" not in metadata:
                    metadata["optimization_history"] = []

                metadata["optimization_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "change_log": change_log,
                    "old_code_length": len(old_code),
                    "new_code_length": len(new_code),
                })
                metadata["last_updated"] = datetime.now().isoformat()
                # Clear syntax-error markers after optimization
                metadata["has_syntax_error"] = False
                metadata["syntax_error_message"] = ""

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            print(f"\033[32m[Task-Specific] Successfully updated '{skill_name}'\033[0m")
            return True

        except Exception as e:
            print(f"\033[31m[Task-Specific] Error updating '{skill_name}': {e}\033[0m")
            return False

    def _check_parameter_semantic_similarity(self, old_params: List[str], new_params: List[str],
                                            old_param_metadata: Dict[str, Dict[str, Any]] = None,
                                            new_param_metadata: Dict[str, Dict[str, Any]] = None) -> bool:
        """
        Check whether parameters are semantically similar (Improvement 2: parameter matching logic).

        Detects:
        1. Singular/plural semantic similarity (logType vs logTypes)
        2. Array vs single-element equivalence (preferredLogTypes = ["oak_log"] ≈ logType = "oak_log")

        Args:
            old_params: old parameter list
            new_params: new parameter list
            old_param_metadata: old parameter metadata (optional)
            new_param_metadata: new parameter metadata (optional)

        Returns:
            bool: True if parameters are semantically similar
        """
        if not old_params or not new_params:
            return False

        # Extract parameter names (strip default-value parts)
        old_param_names = [p.split('=')[0].strip() for p in old_params if p.strip() and p.strip() != 'bot']
        new_param_names = [p.split('=')[0].strip() for p in new_params if p.strip() and p.strip() != 'bot']

        # If parameter counts differ, check for semantic equivalence
        if len(old_param_names) != len(new_param_names):
            # Check whether an array parameter equates to a single parameter
            # E.g.: preferredLogTypes (array) vs logType (string)
            if len(old_param_names) == len(new_param_names) + 1 or len(old_param_names) + 1 == len(new_param_names):
                # Try matching semantically similar parameters
                for old_param in old_param_names:
                    old_lower = old_param.lower()
                    for new_param in new_param_names:
                        new_lower = new_param.lower()

                        # Check singular/plural forms
                        if old_lower.rstrip('s') == new_lower or old_lower == new_lower.rstrip('s'):
                            # Check whether parameter types are compatible (array vs string)
                            if old_param_metadata and new_param_metadata:
                                old_type = old_param_metadata.get(old_param, {}).get("type", "")
                                new_type = new_param_metadata.get(new_param, {}).get("type", "")
                                # Array vs string can be equivalent (if the array has one element)
                                if (old_type == "array" and new_type == "string") or \
                                   (old_type == "string" and new_type == "array"):
                                    return True

                        # Check semantic-keyword matching
                        # preferredLogTypes vs logType
                        old_keywords = set(re.findall(r'\w+', old_lower))
                        new_keywords = set(re.findall(r'\w+', new_lower))
                        # If both contain keywords related to "log" and "type"
                        if ("log" in old_keywords and "log" in new_keywords) and \
                           (("type" in old_keywords or "types" in old_keywords) and \
                            ("type" in new_keywords or "types" in new_keywords)):
                            return True

        # If parameter counts match, check whether parameter names are semantically similar
        if len(old_param_names) == len(new_param_names):
            # Check whether each parameter is semantically similar
            all_similar = True
            for old_param, new_param in zip(old_param_names, new_param_names):
                old_lower = old_param.lower()
                new_lower = new_param.lower()

                # Exact match
                if old_lower == new_lower:
                    continue

                # Singular/plural form
                if old_lower.rstrip('s') == new_lower or old_lower == new_lower.rstrip('s'):
                    continue

                # Semantic-keyword matching
                old_keywords = set(re.findall(r'\w+', old_lower))
                new_keywords = set(re.findall(r'\w+', new_lower))
                # Check for sufficient keyword overlap
                common_keywords = old_keywords & new_keywords
                if len(common_keywords) >= 2:  # at least 2 matching keywords
                    continue

                all_similar = False
                break

            if all_similar:
                return True

        return False

    def _check_functional_equivalence(self, old_code: str, new_code: str) -> bool:
        """
        Check whether two code versions are functionally equivalent (Improvement 3: code check).

        Identifies "improved" code versions (different implementations with the same behavior).

        Args:
            old_code: old code
            new_code: new code

        Returns:
            bool: True if functionally equivalent
        """
        # Extract function signature
        old_sig_match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', old_code)
        new_sig_match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', new_code)

        if not old_sig_match or not new_sig_match:
            return False

        old_func_name = old_sig_match.group(1)
        new_func_name = new_sig_match.group(1)

        # Function names must match
        if old_func_name != new_func_name:
            return False

        # Check whether the new code just wraps the old (wrapper pattern)
        # If the new code calls itself, it may be a recursive or improved version
        if f"{new_func_name}(" in new_code and "typeof" in new_code:
            # Likely a recursive call; needs further checking
            pass

        # Use LLM to check functional equivalence
        try:
            messages = [
                SystemMessage(content="""You are a code analysis expert. Determine if two versions of code are functionally equivalent.

Two code versions are FUNCTIONALLY EQUIVALENT if:
1. They produce the same outputs for the same inputs
2. They have the same side effects (e.g., mining the same blocks, crafting the same items)
3. The only differences are implementation details (e.g., different helper functions, different error handling, but same core logic)
4. One is a refactored version of the other (e.g., calling a helper function vs inline implementation)

Two code versions are NOT functionally equivalent if:
1. They have different parameters (even if semantically similar)
2. They produce different outputs
3. They have different side effects
4. One has significant new functionality

Return ONLY a JSON object:
{
  "functionally_equivalent": true/false,
  "reason": "brief explanation"
}"""),
                HumanMessage(content=f"Old code:\n{old_code}\n\nNew code:\n{new_code}\n\nAre they functionally equivalent? Return only JSON."),
            ]

            response_obj = self._invoke_llm_with_stats(
                messages=messages,
                process_type="skill_generation",
                function_name="_check_functional_equivalence",
                task=None,
                skill_name=None,
            )
            response = response_obj.content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                is_equivalent = data.get("functionally_equivalent", False)
                reason = data.get("reason", "")
                if is_equivalent:
                    print(f"\033[36m[Code Check] Functionally equivalent code detected: {reason}\033[0m")
                return is_equivalent
        except Exception as e:
            print(f"\033[33mWarning: Failed to check functional equivalence: {e}\033[0m")

        return False

    def _detect_function_name_conflicts(self, code: str, main_function_name: str) -> List[str]:
        """
        Detect whether function names defined in code conflict with existing skill names.

        Problem scenario: LLM-generated code may redefine a local function with the same name as an existing skill,
        causing the local function to shadow the external skill so calls use the local version.

        Args:
            code: code to check
            main_function_name: main function name (excluded; the main function is allowed to share the skill name)

        Returns:
            List[str]: list of conflicting function names
        """
        conflicts = []

        # Extract all function names defined in code
        # Match async function xxx( and function xxx(
        func_pattern = r'(?:async\s+)?function\s+(\w+)\s*\('
        defined_functions = re.findall(func_pattern, code)

        # Exclude the main function name
        defined_functions = [f for f in defined_functions if f != main_function_name]

        # Check whether each defined function name conflicts with an existing skill
        for func_name in defined_functions:
            if self.graph.has_node(func_name):
                conflicts.append(func_name)

        return conflicts

    def _fix_function_name_conflicts(self, code: str, main_function_name: str, conflicts: List[str]) -> Tuple[str, List[str]]:
        """
        Fix local function definitions in code that conflict with existing skill names.

        Fix strategy:
        1. Remove local function definitions that share a name with an existing skill
        2. Keep calls to those functions (calls should resolve to the external skill)

        Args:
            code: original code
            main_function_name: main function name
            conflicts: list of conflicting function names

        Returns:
            Tuple[str, List[str]]: (fixed code, list of removed function names)
        """
        fixed_code = code
        removed_functions = []

        for func_name in conflicts:
            # Match the complete function definition (including the body)
            # Use a more precise regex to match the entire function definition

            # First try matching a function with a leading comment
            patterns = [
                # Match // comment + async function
                rf'(?:^[ \t]*//[^\n]*\n)*[ \t]*async\s+function\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{',
                # Match async function (no comment)
                rf'async\s+function\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{',
                # Match plain function
                rf'(?:^[ \t]*//[^\n]*\n)*[ \t]*function\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{',
                rf'function\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{'
            ]

            for pattern in patterns:
                match = re.search(pattern, fixed_code, re.MULTILINE)
                if match:
                    # Found the function start position
                    start_pos = match.start()

                    # Use brace counting to find the function end position
                    brace_count = 0
                    in_string = False
                    string_char = None
                    end_pos = match.end() - 1  # points to the first {

                    for i in range(match.end() - 1, len(fixed_code)):
                        char = fixed_code[i]

                        # Handle strings
                        if char in ('"', "'", '`') and (i == 0 or fixed_code[i-1] != '\\'):
                            if not in_string:
                                in_string = True
                                string_char = char
                            elif char == string_char:
                                in_string = False
                                string_char = None

                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break

                    # Remove the function definition (including any leading comment)
                    # Look for leading comments
                    comment_start = start_pos
                    lines = fixed_code[:start_pos].split('\n')
                    if lines:
                        # Check whether the preceding lines are comments
                        for j in range(len(lines) - 1, -1, -1):
                            line = lines[j].strip()
                            if line.startswith('//') or line == '':
                                comment_start = sum(len(l) + 1 for l in lines[:j])
                            else:
                                break

                    # Remove the function definition
                    func_def = fixed_code[start_pos:end_pos]
                    # Keep calls; remove only the definition
                    fixed_code = fixed_code[:comment_start] + fixed_code[end_pos:]
                    removed_functions.append(func_name)

                    print(f"\033[33m[Function Conflict] Removed local function definition conflicting with existing skill '{func_name}'\033[0m")
                    break

        return fixed_code, removed_functions

    def _detect_and_fix_function_conflicts(self, code: str, main_function_name: str) -> Tuple[str, List[str], bool]:
        """
        Detect and fix function-name conflicts (composite method).

        Args:
            code: original code
            main_function_name: main function name

        Returns:
            Tuple[str, List[str], bool]: (fixed code, list of conflicting names, whether any fix was applied)
        """
        # 1. Detect conflicts
        conflicts = self._detect_function_name_conflicts(code, main_function_name)

        if not conflicts:
            return code, [], False

        print(f"\033[31m[Function Conflict] Detected {len(conflicts)} function names conflicting with existing skills: {conflicts}\033[0m")
        print(f"\033[31m[Function Conflict] These local functions would shadow external skills; their definitions will be auto-removed\033[0m")

        # 2. Fix conflicts
        fixed_code, removed = self._fix_function_name_conflicts(code, main_function_name, conflicts)

        if removed:
            print(f"\033[32m[Function Conflict] Removed {len(removed)} conflicting local function definitions: {removed}\033[0m")

        return fixed_code, conflicts, len(removed) > 0

    def _detect_code_changes(self, old_code: str, new_code: str,
                            old_param_metadata: Dict[str, Dict[str, Any]] = None,
                            new_param_metadata: Dict[str, Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Detect code changes and decide whether a new version is needed (Improvements 2 and 3).

        Args:
            old_code: old code
            new_code: new code
            old_param_metadata: old parameter metadata (optional, for parameter semantic-similarity checks)
            new_param_metadata: new parameter metadata (optional, for parameter semantic-similarity checks)

        Returns:
            Tuple[bool, str]: (whether there are significant changes, change description)
        """
        # Simple code comparison: compare after collapsing whitespace
        old_normalized = re.sub(r'\s+', ' ', old_code.strip())
        new_normalized = re.sub(r'\s+', ' ', new_code.strip())

        if old_normalized == new_normalized:
            return False, "No changes detected"

        # Improvement 3: check functional equivalence
        if self._check_functional_equivalence(old_code, new_code):
            print(f"\033[36m[Code Check] Detected functionally equivalent code; avoiding duplicate version\033[0m")
            return False, "Functionally equivalent code detected, skipping version creation"

        # Improvement 2: check parameter semantic similarity
        old_sig_match = re.search(r'async\s+function\s+\w+\s*\(([^)]*)\)', old_code)
        new_sig_match = re.search(r'async\s+function\s+\w+\s*\(([^)]*)\)', new_code)

        if old_sig_match and new_sig_match:
            old_params_str = old_sig_match.group(1)
            new_params_str = new_sig_match.group(1)
            old_params = [p.strip() for p in old_params_str.split(',') if p.strip()]
            new_params = [p.strip() for p in new_params_str.split(',') if p.strip()]

            if self._check_parameter_semantic_similarity(
                old_params, new_params, old_param_metadata, new_param_metadata
            ):
                print(f"\033[36m[Parameter Check] Detected semantically similar parameters but different implementations\033[0m")
                # Parameters are semantically similar but code differs; still create a new version but update metadata

        # Use LLM to analyze changes
        try:
            messages = [
                SystemMessage(content="""You are a code change analyzer. Analyze the differences between two versions of code and determine:
1. Whether the changes are significant (logic changes, not just formatting/comments)
2. A brief description of what changed

IMPORTANT: If the code changes are just parameter name changes (e.g., preferredLogTypes to logType) or implementation refactoring but the functionality is the same, mark as significant but note that it's a refactoring.

Return ONLY a JSON object:
{
  "significant": true/false,
  "description": "brief description of changes"
}"""),
                HumanMessage(content=f"Old code:\n{old_code}\n\nNew code:\n{new_code}\n\nAnalyze changes. Return only JSON."),
            ]

            response_obj = self._invoke_llm_with_stats(
                messages=messages,
                process_type="skill_generation",
                function_name="_detect_code_changes",
                task=None,  # code-change detection has no associated task
                skill_name=None,
            )
            response = response_obj.content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("significant", True), data.get("description", "Code changes detected")
        except Exception as e:
            print(f"\033[33mWarning: Failed to analyze code changes: {e}\033[0m")

        # If LLM analysis fails, fall back to simple heuristics
        # Compute change in line count
        old_lines = len([l for l in old_code.split('\n') if l.strip()])
        new_lines = len([l for l in new_code.split('\n') if l.strip()])
        line_diff = abs(new_lines - old_lines)

        # If line count changes by more than 10%, consider it significant
        if old_lines > 0 and line_diff / old_lines > 0.1:
            return True, f"Significant code changes detected ({line_diff} lines changed)"

        return True, "Code changes detected"
