"""
SkillNamingMixin for ParameterizedActionAgent

Methods:
    _normalize_skill_name: Dispatch to rule-based or LLM normalization
    _normalize_skill_name_rule_based: Rule-based skill name normalization
    _normalize_skill_name_with_llm: LLM-based skill name normalization
"""

import re

from skillnet.utils.stats_tracker import record_llm_usage


class SkillNamingMixin:
    """Skill name normalization helpers for ParameterizedActionAgent."""

    def _get_type_config(self):
        """Return (specific_types: set, type_param_patterns: list).

        Uses domain-provided keywords when available via
        ``self._domain_knowledge.get_type_keywords()``, otherwise
        returns empty defaults (no type normalization).
        """
        dk = getattr(self, '_domain_knowledge', None)
        if dk:
            cfg = dk.get_type_keywords()
            if cfg:
                types = set(cfg.get('specific_types', []))
                patterns = cfg.get('type_param_patterns', [])
                if types:
                    return types, patterns
        return set(), []

    def _normalize_skill_name(self, func_name: str, func_params: list, func_body: str) -> str:
        """
        Normalize the function name: ensure the name is consistent with the implementation.

        Bidirectional handling:
        1. Generalize: if the function name contains a specific type but the function supports multiple types via parameterization, produce a more generic name.
        2. Specialize: if the function name is generic but the function body hardcodes a specific type, produce a name that includes the specific type.

        Args:
            func_name: original function name
            func_params: list of function parameters
            func_body: function body code

        Returns:
            Normalized function name (ensured to be consistent with the implementation)
        """
        # Defensive check
        if not func_name or func_name is None:
            return "genericAction"

        # Choose LLM or rule-based method based on configuration
        if self.use_llm_for_normalization:
            normalized_name = self._normalize_skill_name_with_llm(func_name, func_params, func_body)
        else:
            normalized_name = self._normalize_skill_name_rule_based(func_name, func_params, func_body)

        # Recursive-call guard: if the normalized name is called inside the function body, keep the original name.
        # This avoids cases where normalizing killOneCreeper to killOneMob turns a call to killOneMob into a recursive call.
        if normalized_name != func_name and func_body:
            # Check whether the function body calls the normalized name (excluding typeof checks)
            # Remove typeof checks; only detect real function calls
            body_without_typeof = re.sub(r'typeof\s+\w+\s*===\s*["\']function["\']', '', func_body, flags=re.IGNORECASE)
            call_pattern = rf'\b{re.escape(normalized_name)}\s*\('
            if re.search(call_pattern, body_without_typeof):
                print(f"\033[33m[Name Normalization] Warning: normalized name '{normalized_name}' is called inside the function body, "
                      f"which would produce a recursive call; keeping original name '{func_name}'\033[0m")
                return func_name

        return normalized_name

    def _normalize_skill_name_rule_based(self, func_name: str, func_params: list, func_body: str) -> str:
        """
        Normalize the function name using rules (the original implementation)

        Args:
            func_name: original function name
            func_params: list of function parameters
            func_body: function body code

        Returns:
            Normalized function name
        """
        # Defensive check
        if not func_name or func_name is None:
            return "genericAction"

        # Defensive check: ensure func_body is not None
        if func_body is None:
            func_body = ""

        specific_types, type_param_patterns = self._get_type_config()

        # Check whether the function name contains a specific type
        func_name_lower = func_name.lower()
        found_types = [t for t in specific_types if t in func_name_lower]

        if not found_types:
            return func_name

        # Extract parameter names (handle Babel AST nodes or strings)
        param_names = []
        for p in func_params:
            if hasattr(p, 'name') and p.name is not None:
                # Babel AST node
                param_names.append(p.name.lower())
            elif isinstance(p, str) and p:
                # String parameter
                param_name = p.split('=')[0].strip()
                if param_name:
                    param_names.append(param_name.lower())
            elif p is not None:
                # Other cases: convert to string
                param_str = str(p)
                param_name = param_str.split('=')[0].strip()
                if param_name:
                    param_names.append(param_name.lower())
        param_names_str = ' '.join(param_names)

        # Check whether the function body or parameters contain a type parameter
        has_type_param = False
        for pattern in type_param_patterns:
            if re.search(pattern, func_body, re.IGNORECASE) or re.search(pattern, param_names_str, re.IGNORECASE):
                has_type_param = True
                break

        if not has_type_param:
            # The function has no type parameter; keep the original name
            return func_name

        # Generate a generic name: remove specific-type keywords
        normalized_name = func_name
        for specific_type in found_types:
            # Remove the type keyword (preserve camelCase)
            # Handle the various possible naming patterns
            patterns = [
                rf'{specific_type}(\w+)',  # birchLog -> Log
                rf'(\w+){specific_type}',  # craftBirch -> craft
                rf'{specific_type}',        # birch -> remove
            ]

            for pattern in patterns:
                if re.search(pattern, normalized_name, re.IGNORECASE):
                    # Remove the specific type while keeping the rest
                    # Choose the replacement string based on whether the pattern has a capture group
                    if '(' in pattern:
                        # Has a capture group; use \1
                        normalized_name = re.sub(pattern, r'\1', normalized_name, flags=re.IGNORECASE)
                    else:
                        # No capture group; remove the matched portion directly
                        normalized_name = re.sub(pattern, '', normalized_name, flags=re.IGNORECASE)
                    # If the replacement leaves an empty string or only separators, use a generic suffix
                    if not normalized_name or normalized_name in ['_', '-', '']:
                        # Use domain-provided fallbacks for prefix → normalized name
                        dk = getattr(self, '_domain_knowledge', None)
                        fallbacks = dk.get_skill_name_fallbacks() if dk else {}
                        matched = False
                        for prefix, name in fallbacks.items():
                            if func_name_lower.startswith(prefix):
                                normalized_name = name
                                matched = True
                                break
                        if not matched:
                            normalized_name = func_name.replace(specific_type, '').replace(specific_type.capitalize(), '')
                            if not normalized_name:
                                normalized_name = 'genericAction'
                    break

        # Defensive check: ensure normalized_name is neither None nor empty
        if not normalized_name or normalized_name is None:
            return func_name

        # Clean up the name: remove stray underscores and hyphens to keep camelCase
        normalized_name = re.sub(r'[_-]+', '', normalized_name)

        # Check again (re.sub may return an empty string)
        if not normalized_name or normalized_name is None:
            return func_name

        # Ensure the first character is lowercase (JavaScript function-naming convention)
        if normalized_name:
            normalized_name = normalized_name[0].lower() + normalized_name[1:] if len(normalized_name) > 1 else normalized_name.lower()

        # If the normalized name is too short or invalid, keep the original
        if not normalized_name or len(normalized_name) < 3 or normalized_name == func_name.lower():
            return func_name

        return normalized_name

    def _normalize_skill_name_with_llm(self, func_name: str, func_params: list, func_body: str) -> str:
        """
        Normalize the function name using the LLM

        Args:
            func_name: original function name
            func_params: list of function parameters
            func_body: function body code

        Returns:
            Normalized function name
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage

            # Defensive check: ensure func_name is not None
            if not func_name or func_name is None:
                print(f"\033[33m[LLM Normalization] Function name is None; returning default name\033[0m")
                return "genericAction"

            # Defensive check: ensure func_body is not None
            if func_body is None:
                func_body = ""

            # Pre-check: determine whether the function truly supports parameterization across multiple types
            _dk = getattr(self, '_domain_knowledge', None)
            _entry_param = _dk.get_entry_parameter_name() if _dk else "bot"
            func_name_lower = func_name.lower()

            type_keywords, _ = self._get_type_config()

            # Check whether the function name contains a specific type
            found_type_in_name = None
            for type_keyword in type_keywords:
                if type_keyword in func_name_lower:
                    found_type_in_name = type_keyword
                    break

            # If the function name does not contain a specific type, check whether the function body hardcodes one
            # This is a reverse check: if the name is generic but the implementation supports only a specific type,
            # we should suggest specializing the function name
            if not found_type_in_name:
                # Check whether the function body contains hardcoded specific-type strings
                # Build regex alternation from type keywords
                types_alt = '|'.join(re.escape(t) for t in sorted(type_keywords))
                _dk = getattr(self, '_domain_knowledge', None)
                _registry_pats = _dk.get_registry_access_patterns() if _dk else []
                hardcoded_type_patterns = [
                    rf'["\']({types_alt})_[\w]+["\']',
                ]
                for _rp in _registry_pats:
                    hardcoded_type_patterns.append(rf'{_rp}["\']({types_alt})_[\w]+["\']')

                hardcoded_types_found = set()
                for pattern in hardcoded_type_patterns:
                    matches = re.findall(pattern, func_body, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            hardcoded_types_found.update([m.lower() for m in match if m and m is not None])
                        elif match and match is not None:
                            hardcoded_types_found.add(match.lower())

                # If hardcoded specific types are detected, check whether parameters are used to change those types
                if hardcoded_types_found:
                    # Extract parameter names (supports both Babel AST node and string formats)
                    param_names = []
                    for p in func_params:
                        if isinstance(p, dict):
                            # Babel AST node format: may be Identifier or AssignmentPattern
                            if p.get('type') == 'Identifier':
                                param_name = p.get('name')
                                if param_name:
                                    param_names.append(param_name.lower())
                            elif p.get('type') == 'AssignmentPattern':
                                # Handle default parameters: allowedLogTypes = [...]
                                left = p.get('left', {})
                                if isinstance(left, dict) and left.get('type') == 'Identifier':
                                    param_name = left.get('name')
                                    if param_name:
                                        param_names.append(param_name.lower())
                        elif hasattr(p, 'name') and p.name is not None:
                            # Object form (has name attribute)
                            param_names.append(p.name.lower())
                        elif isinstance(p, str) and p:
                            # String form: 'allowedLogTypes = [...]'
                            param_name = p.split('=')[0].strip()
                            if param_name:
                                param_names.append(param_name.lower())

                    # Check whether parameters are used to change the hardcoded types
                    param_used_for_type = False
                    for param_name in param_names:
                        if param_name and len(param_name) > 2 and param_name != _entry_param:
                            param_type_usage_patterns = [
                                # String-concatenation usage
                                rf'\b{re.escape(param_name)}\s*\+\s*["\'_]',
                                rf'\b{re.escape(param_name)}\s*\+\s*["\']_[\w]+',
                                # Array/object access usage
                                rf'\[{re.escape(param_name)}\]',
                                # Array-iteration usage (for...of, forEach, map, etc.)
                                rf'for\s*\([^)]*\bof\s+{re.escape(param_name)}\b',
                                rf'for\s*\([^)]*in\s+{re.escape(param_name)}\b',
                                rf'\b{re.escape(param_name)}\s*\.\s*(forEach|map|filter|some|every|find|findIndex|reduce)\s*\(',
                                rf'\b{re.escape(param_name)}\s*\.\s*(includes|indexOf|join|slice|splice)\s*\(',
                                rf'\b{re.escape(param_name)}\s*\.\s*length\b',
                                # Passed as a function argument
                                rf'\([^)]*\b{re.escape(param_name)}\b[^)]*\)',
                            ]
                            # Domain-specific registry access patterns
                            for _rp in _registry_pats:
                                param_type_usage_patterns.append(rf'{_rp}{re.escape(param_name)}')
                            for usage_pattern in param_type_usage_patterns:
                                if re.search(usage_pattern, func_body, re.IGNORECASE):
                                    param_used_for_type = True
                                    break
                            if param_used_for_type:
                                break

                    # If the function name is generic but the body hardcodes a specific type and the parameter is not used to change the type,
                    # then the name and the implementation are inconsistent and we should suggest specializing the name
                    if not param_used_for_type:
                        # Find the most common hardcoded type (likely the function's primary type)
                        type_counts = {}
                        for hardcoded_type in hardcoded_types_found:
                            # Count occurrences of this type within the function body
                            count = len(re.findall(
                                rf'["\']{re.escape(hardcoded_type)}_[\w]+["\']',
                                func_body,
                                re.IGNORECASE
                            ))
                            type_counts[hardcoded_type] = count

                        if type_counts:
                            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
                            print(f"\033[33m[LLM Normalization] Function name '{func_name}' is generic but the body hardcodes type '{most_common_type}'; will suggest specializing the name\033[0m")
                            # Continue and let the LLM decide whether to specialize the name
                            # Note: we let the LLM decide here because it needs the surrounding context

            # If the function name contains a specific type, check whether the body really supports parameterization
            if found_type_in_name:
                # Extract parameter names (supports both Babel AST node and string formats)
                param_names = []
                for p in func_params:
                    if isinstance(p, dict):
                        # Babel AST node format: may be Identifier or AssignmentPattern
                        if p.get('type') == 'Identifier':
                            param_name = p.get('name')
                            if param_name:
                                param_names.append(param_name.lower())
                        elif p.get('type') == 'AssignmentPattern':
                            # Handle default parameters: allowedLogTypes = [...]
                            left = p.get('left', {})
                            if isinstance(left, dict) and left.get('type') == 'Identifier':
                                param_name = left.get('name')
                                if param_name:
                                    param_names.append(param_name.lower())
                    elif hasattr(p, 'name') and p.name is not None:
                        # Object form (has name attribute)
                        param_names.append(p.name.lower())
                    elif isinstance(p, str) and p:
                        # String form: 'allowedLogTypes = [...]'
                        param_name = p.split('=')[0].strip()
                        if param_name:
                            param_names.append(param_name.lower())

                # Check whether the parameter is used in the body to build type-related values
                # This is the key signal for whether the function actually supports parameterization
                param_used_for_type = False
                _, type_param_patterns = self._get_type_config()

                # Method 1: check for an explicit type parameter (e.g., plankType, logType)
                for param_name in param_names:
                    for pattern in type_param_patterns:
                        if re.search(pattern, param_name, re.IGNORECASE):
                            param_used_for_type = True
                            break
                    if param_used_for_type:
                        break

                # Method 2: check whether the parameter is used inside the body to build a type string
                # e.g., param + "_planks", itemsByName[param], etc.
                if not param_used_for_type:
                    for param_name in param_names:
                        if param_name and len(param_name) > 2 and param_name != _entry_param:
                            # Check whether the parameter appears in a type-construction context
                            param_type_usage_patterns = [
                                rf'\b{re.escape(param_name)}\s*\+\s*["\'_]',  # param + "_"
                                rf'\b{re.escape(param_name)}\s*\+\s*["\']_[\w]+',  # param + "_planks"
                                rf'\[{re.escape(param_name)}\]',  # [param]
                                rf'["\']{re.escape(param_name)}["\']',  # "param"
                            ]
                            # Domain-specific registry access patterns
                            for _rp in _registry_pats:
                                param_type_usage_patterns.append(rf'{_rp}{re.escape(param_name)}')
                            for usage_pattern in param_type_usage_patterns:
                                if re.search(usage_pattern, func_body, re.IGNORECASE):
                                    param_used_for_type = True
                                    break
                            if param_used_for_type:
                                break

                # If the function name contains a specific type but the parameter is not used to support multiple types,
                # we should not normalize. In this case the name accurately reflects the actual behavior (single-type only).
                if not param_used_for_type:
                    print(f"\033[33m[LLM Normalization] Function name contains type '{found_type_in_name}' but the body does not use parameters to support multiple types; skipping normalization: {func_name}\033[0m")
                    return func_name

                # If the parameter is used to support multiple types, let the LLM make the normalization decision
                print(f"\033[36m[LLM Normalization] Function name contains type '{found_type_in_name}' but the body supports multiple types via a parameter; will normalize: {func_name}\033[0m")

            # Extract parameter info
            param_info = []
            for p in func_params:
                if hasattr(p, 'name'):
                    param_name = p.name
                    param_default = getattr(p, 'value', None) if hasattr(p, 'value') else None
                    param_info.append(f"{param_name}" + (f" = {param_default}" if param_default else ""))
                elif isinstance(p, str):
                    param_info.append(p)
                else:
                    param_info.append(str(p))

            param_str = ", ".join(param_info) if param_info else "bot"

            # Load normalization prompt from domain knowledge
            _dk = getattr(self, '_domain_knowledge', None)
            normalization_prompt = _dk.get_prompt("skill_name_normalization") if _dk else ""
            if not normalization_prompt:
                # No domain prompt available, skip LLM normalization
                return func_name

            messages = [
                SystemMessage(content=normalization_prompt),
                HumanMessage(content=f"""Function name: {func_name}
Parameters: {param_str}
Function body:
```javascript
{func_body[:1000]}  // (truncated if too long)
```

Analyze this function for consistency:

1. Does the function name contain a specific type (e.g., "oak", "wooden", "iron")?
2. Does the function body use parameters to support multiple types, or does it hardcode a specific type?
3. If name contains specific type but body supports multiple types → generalize (remove type from name)
4. If name is generic but body hardcodes specific type → specialize (add type to name)
5. If name matches implementation → keep original name

Return ONLY the normalized function name (no quotes, no explanation):"""),
            ]

            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="skill_generation", function_name="skill_naming._normalize_skill_name_with_llm", skill_name=func_name)
            response = _llm_resp.content.strip()

            # Clean up the response: strip possible quotes, code-block markers, etc.
            response = response.strip('"\'`')
            response = re.sub(r'^```\w*\n?', '', response)
            response = re.sub(r'\n?```$', '', response)
            response = response.strip()

            # Verify the response is a valid function name
            if response and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', response):
                print(f"\033[36m[LLM Normalization] {func_name} → {response}\033[0m")
                return response
            else:
                print(f"\033[33m[LLM Normalization] Invalid response '{response}', using original name\033[0m")
                return func_name

        except Exception as e:
            print(f"\033[33m[LLM Normalization] Failed: {e}, falling back to rule-based method\033[0m")
            # Fallback to rule-based method
            return self._normalize_skill_name_rule_based(func_name, func_params, func_body)
