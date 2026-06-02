"""
Parameter Extraction Mixin

Babel AST / LLM / fallback extraction strategies for parameter metadata.
Split from parameters.py for maintainability.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import HumanMessage, SystemMessage

from skillnet.agents.skill_graph.utils import (
    extract_destructured_params,
    infer_param_type_from_babel,
    extract_default_value_from_babel,
    generate_param_description,
)

from ._param_constants import (
    _get_parameter_extraction_prompt,
    _get_object_schema_extraction_prompt,
)

logger = logging.getLogger(__name__)


class ParameterExtractionMixin:
    """Mixin: Babel/LLM parameter extraction methods."""

    def extract_babel(
        self,
        code: str,
        description: Optional[str] = None,
        skill_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameter info using Babel AST parsing.

        Args:
            code: Skill code
            description: Skill description (optional)
            skill_name: Skill name (optional; used to match the main function)

        Returns:
            Parameter-metadata dict; returns an empty dict on failure
        """
        try:
            from skillnet.core.dk_registry import get_domain_knowledge
            dk = get_domain_knowledge()
            if dk is None:
                raise RuntimeError("No DomainKnowledge registered; cannot parse skill code")
            lang = dk.get_skill_language_impl()
            parse_result = lang.parse(code)
            if not parse_result.success:
                self.logger.warning(
                    f"[Parameter Extraction] Parse failed: {parse_result.error}"
                )
                return None  # Extraction error → None
            parsed = parse_result.raw_ast
            if parsed is None:
                raise RuntimeError("SkillLanguage.parse() returned no raw_ast")

            # Find the main function (async function)
            functions = []
            for node in parsed.program.body:
                # Use try/except for bridge proxy access (JSPyBridge
                # raises JavaScriptError, not AttributeError)
                try:
                    if node.type != "FunctionDeclaration":
                        continue
                    try:
                        is_async = bool(node["async"])
                    except Exception:
                        is_async = False
                    if not is_async:
                        continue
                    node_name = node.id.name
                    if not node_name:
                        continue
                    try:
                        params = list(node.params) if node.params else []
                    except Exception:
                        params = []
                    functions.append({
                        "name": node_name,
                        "params": params
                    })
                except Exception:
                    continue

            if not functions:
                self.logger.warning("[Parameter Extraction] No async functions found in code")
                return None  # Extraction error → None (distinct from {} = 0 params success)

            # Choose the main function
            main_function = self._select_main_function(functions, skill_name)
            if not main_function:
                return None  # Extraction error → None

            params = main_function["params"]

            # Extract parameter info
            parameters = {}
            for param in params:
                try:
                    param_info = self._extract_param_from_babel_node(param)
                    if param_info:
                        param_name, info = param_info
                        parameters[param_name] = info
                except Exception as e:
                    param_name = getattr(param, 'name', 'unknown')
                    self.logger.warning(
                        f"[Parameter Extraction] Failed to extract parameter '{param_name}': {e}"
                    )
                    continue

            # Infer full semantics for all parameters (includes direction and transform_hint)
            if self.semantic_inference_enabled and parameters:
                func_name = main_function["name"]
                for p_name, p_info in parameters.items():
                    p_type = p_info.get("type", "unknown")

                    # Use the new full-semantic inference
                    full_semantic = self.infer_full_semantic(
                        param_name=p_name,
                        func_name=func_name,
                        code=code,
                        param_type=p_type,
                        param_info=p_info,
                    )

                    # Store the full-semantic object
                    p_info["semantic"] = full_semantic.to_dict()

                # Log the semantic-inference summary
                param_semantics = {}
                for p, info in parameters.items():
                    semantic_data = info.get("semantic", {})
                    if isinstance(semantic_data, dict):
                        qs = semantic_data.get("quantity_semantic", "none")
                        dr = semantic_data.get("direction", "config")
                        param_semantics[p] = f"{qs}/{dr}"
                    else:
                        param_semantics[p] = str(semantic_data)
                self.logger.info(f"[Semantic] {func_name} parameter semantics: {param_semantics}")

                # Validate consistency between function name and semantics
                self.validate_function_name_semantic_consistency(func_name, parameters)

            # Perform usage-based semantic analysis for nullable/unknown types
            for param_name, param_info in parameters.items():
                if param_info.get("type") in ("nullable", "unknown"):
                    inferred_type = self._infer_type_from_usage(
                        param_name, code, param_info["type"]
                    )
                    if inferred_type != param_info["type"]:
                        param_info["type"] = inferred_type
                        param_info["type_inference_source"] = "usage_analysis"

            # Diagnostic — detect when Babel parsed params but extraction yielded empty
            if not parameters and len(params) > 0:
                _entry = self.entry_parameter_name
                non_entry_params = [p for p in params if getattr(p, 'name', None) != _entry]
                if non_entry_params:
                    param_types = [getattr(p, 'type', 'unknown') for p in non_entry_params]
                    self.logger.warning(
                        f"[Parameter Extraction] ⚠️ Babel found {len(non_entry_params)} non-{_entry} params "
                        f"but extraction returned empty. Function: {main_function['name']}, "
                        f"param_types: {param_types}"
                    )

            return parameters

        except Exception as e:
            self.logger.warning(f"[Parameter Extraction] Babel parsing failed: {e}")
            return None  # Extraction error → None

    def _select_main_function(
        self,
        functions: List[Dict],
        skill_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """Choose the main function."""
        main_function = None

        # If skill_name is provided, prefer the function with the same name
        if skill_name:
            for func in functions:
                if func["name"] == skill_name:
                    main_function = func
                    self.logger.info(
                        f"[Parameter Extraction] Matched function by skill_name: {skill_name}"
                    )
                    break

        # If no same-named function was found, fall back to the heuristic
        if not main_function:
            test_prefixes = ['test', 'ci', 'mock', 'stub', 'helper', '_']

            for func in functions:
                func_name_lower = func["name"].lower()
                is_test = any(func_name_lower.startswith(prefix) for prefix in test_prefixes)
                if not is_test:
                    main_function = func
                    break

            if not main_function:
                main_function = functions[0]
                self.logger.warning(
                    f"[Parameter Extraction] All functions look like test/helper, using first one: {main_function['name']}"
                )
            elif skill_name:
                self.logger.warning(
                    f"[Parameter Extraction] Could not find function '{skill_name}', using fallback: {main_function['name']}"
                )

        return main_function

    def _extract_param_from_babel_node(self, param) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extract a single parameter's info from a Babel AST node."""
        param_type_node = getattr(param, 'type', None)

        # 1. Plain parameter: Identifier (e.g., count)
        if param_type_node == "Identifier":
            try:
                if not param.name:
                    return None
            except Exception:
                return None
            if param.name == self.entry_parameter_name:
                return None

            param_type = infer_param_type_from_babel(param)
            param_default = extract_default_value_from_babel(param)

            return param.name, {
                "type": param_type,
                "default": param_default,
                "description": generate_param_description(param.name, param)
            }

        # 2. Plain parameter with default: AssignmentPattern (e.g., count = 1)
        elif param_type_node == "AssignmentPattern":
            left = getattr(param, 'left', None)
            right = getattr(param, 'right', None)

            # 2a. Left side is Identifier: e.g., count = 1
            if left and getattr(left, 'type', None) == "Identifier":
                param_name = left.name
                if param_name == self.entry_parameter_name:
                    return None

                param_type = infer_param_type_from_babel(param)
                param_default = extract_default_value_from_babel(param)

                return param_name, {
                    "type": param_type,
                    "default": param_default,
                    "description": generate_param_description(param_name, param)
                }

            # 2b. Left side is ObjectPattern: e.g., { toolName = "axe", count = 1 } = {}
            elif left and getattr(left, 'type', None) == "ObjectPattern":
                extracted_params = extract_destructured_params(left)
                # Return the first non-bot parameter (matches _extract_param_from_babel_node's return format)
                for prop_name, prop_info in extracted_params.items():
                    if prop_name != self.entry_parameter_name:
                        return prop_name, prop_info
                return None

        # 3. Destructured parameter (no default object): ObjectPattern
        elif param_type_node == "ObjectPattern":
            extracted_params = extract_destructured_params(param)
            for prop_name, prop_info in extracted_params.items():
                if prop_name != self.entry_parameter_name:
                    return prop_name, prop_info
            return None

        # 4. Other types
        else:
            try:
                has_name = param.name and param.name != self.entry_parameter_name
            except Exception:
                has_name = False
            if has_name:
                param_type = infer_param_type_from_babel(param)
                param_default = extract_default_value_from_babel(param)

                param_info = {
                    "type": param_type,
                    "default": param_default,
                    "description": generate_param_description(param.name, param)
                }

                # For array types, if the default is a non-empty list, use it as supported_values
                if param_type == "array" and isinstance(param_default, list) and param_default:
                    param_info["supported_values"] = param_default

                return param.name, param_info

        return None

    def extract_llm(
        self,
        code: str,
        description: Optional[str] = None,
        skill_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameter info using the LLM.

        Args:
            code: Skill code
            description: Skill description (optional)
            skill_name: Skill name (optional)

        Returns:
            Parameter-metadata dict; returns an empty dict on failure
        """
        try:
            # Build the function-targeting hint
            if skill_name:
                function_hint = f"Extract parameters ONLY from the function named '{skill_name}'. Ignore all other functions (helper functions, utility functions, etc.)."
            else:
                function_hint = "Extract parameters from the main async function (the function that appears to be the primary skill function, NOT helper/utility functions)."

            system_prompt = _get_parameter_extraction_prompt().format(
                function_hint=function_hint,
                entry_parameter=self.entry_parameter_name,
            )

            human_prompt = f"""Extract parameter information from this JavaScript function:

{f"Target function: {skill_name}" if skill_name else ""}

```javascript
{code}
```

{f"Skill description: {description}" if description else ""}

Return only JSON with parameter information."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            # Change 1.1: add JSON Mode constraint (gated by configuration)
            from skillnet.config.robustness_config import RobustnessConfig

            if RobustnessConfig.ENABLE_JSON_MODE:
                try:
                    response = self.llm.invoke(
                        messages,
                        response_format={"type": "json_object"}  # Force JSON output
                    )
                except Exception as e:
                    # Fallback for backends that don't support response_format
                    self.logger.warning(f"JSON mode not supported: {e}")
                    response = self.llm.invoke(messages)
            else:
                # Baseline behavior (unchanged)
                response = self.llm.invoke(messages)

            from skillnet.utils.stats_tracker import record_llm_usage
            record_llm_usage(response, process_type="metadata_extraction", function_name="metadata.param_extraction.extract_llm", skill_name=skill_name)
            response_content = response.content.strip()

            # Change 1.2: use robust JSON extraction
            from skillnet.agents.optimizer.core.llm_invoker import extract_json_from_response

            result = extract_json_from_response(
                response_content,
                context="parameter_extraction"
            )
            parameters = result.get("parameters", {}) if result else None
            if parameters is not None:
                if parameters:
                    self.logger.info(
                        f"[Parameter Extraction] Successfully extracted {len(parameters)} parameters using LLM"
                    )
                    # Post-process: extract state_mapping and run semantic inference
                    self._post_process_llm_semantics(parameters, code, skill_name)
                return parameters
            else:
                self.logger.warning(f"[Parameter Extraction] LLM returned empty parameters, raw response ({len(response_content)} chars): {response_content[:300]}")
                return None  # Extraction error → None

        except Exception as e:
            self.logger.error(f"[Parameter Extraction] LLM extraction failed: {e}")
            return None  # Extraction error → None

    def _post_process_llm_semantics(
        self,
        parameters: Dict[str, Dict[str, Any]],
        code: str,
        skill_name: Optional[str],
    ) -> None:
        """Post-process LLM-extracted parameters: integrate state_mapping into semantic.

        Converts LLM's raw output (semantic string + optional state_mapping)
        into proper ParameterSemantic dicts by calling infer_full_semantic().
        """
        if not self.semantic_inference_enabled:
            return

        func_name = skill_name or "unknown"
        for p_name, p_info in parameters.items():
            # Move LLM-provided state_mapping to internal key for infer_full_semantic
            if "state_mapping" in p_info:
                p_info["_llm_state_mapping"] = p_info.pop("state_mapping")

            p_type = p_info.get("type", "unknown")
            full_semantic = self.infer_full_semantic(
                param_name=p_name,
                func_name=func_name,
                code=code,
                param_type=p_type,
                param_info=p_info,
            )
            p_info["semantic"] = full_semantic.to_dict()

            # Clean up internal key
            p_info.pop("_llm_state_mapping", None)

    def extract_llm_simplified(self, code: str, skill_name: str) -> dict:
        """
        Simplified LLM extraction requesting only the basic fields.
        Used as a fallback after a full extraction fails.
        """
        from skillnet.config.robustness_config import RobustnessConfig

        if not RobustnessConfig.ENABLE_SIMPLIFIED_FALLBACK:
            return None  # Feature disabled → None

        simplified_prompt = """Extract ONLY the parameter names, types, and default values from this function.
Do not include semantic, schema, or supported_values fields.

Return ONLY a JSON object in this exact format:
{{
  "parameters": {{
    "paramName": {{
      "type": "number|string|array|object|boolean",
      "default": <value>,
      "description": "brief description"
    }}
  }}
}}

Function:
```javascript
{code}
```""".format(code=code)

        try:
            from langchain_core.messages import SystemMessage
            response = self.llm.invoke([SystemMessage(content=simplified_prompt)])
            from skillnet.utils.stats_tracker import record_llm_usage
            record_llm_usage(response, process_type="metadata_extraction", function_name="metadata.param_extraction.extract_llm_simplified", skill_name=skill_name)

            from skillnet.agents.optimizer.core.llm_invoker import extract_json_from_response
            result = extract_json_from_response(response.content, context="simplified_extraction")

            if result and "parameters" in result:
                # Automatically add a default semantic
                params = result["parameters"]
                for param_name, param_info in params.items():
                    param_info.setdefault("semantic", "config")
                    param_info.setdefault("description", f"Parameter {param_name}")

                self.logger.info(f"[Parameter Extraction] Simplified fallback extracted {len(params)} parameters")
                return params
        except Exception as e:
            self.logger.warning(f"Simplified extraction failed: {e}")

        return None  # Extraction error → None

    def extract_regex_emergency(self, code: str, skill_name: str) -> dict:
        """
        Regex-based emergency extraction; the last line of defense.
        Only extracts parameter names and defaults, marked as emergency extraction.
        """
        from skillnet.config.robustness_config import RobustnessConfig

        if not RobustnessConfig.ENABLE_REGEX_EMERGENCY:
            return None  # Feature disabled → None

        params = {}

        # Extract the function signature: async function name(bot, param1 = default1, param2 = default2)
        import re
        pattern = r'async\s+function\s+\w+\s*\([^)]*\)'
        match = re.search(pattern, code)
        if not match:
            return None  # Pattern not found → None

        signature = match.group()

        # Extract parameters: param = default
        param_pattern = r'(\w+)\s*=\s*([^,)]+)'
        param_matches = re.findall(param_pattern, signature)

        for param_name, default_str in param_matches:
            if param_name == self.entry_parameter_name:  # Skip the entry parameter
                continue

            # Infer type
            default_str = default_str.strip()
            param_type = "unknown"
            default_value = default_str

            if default_str.startswith('['):
                param_type = "array"
            elif default_str.startswith('"') or default_str.startswith("'"):
                param_type = "string"
                default_value = default_str.strip('"\'')
            elif default_str.replace('-', '').replace('.', '').isdigit():
                param_type = "number"
                try:
                    default_value = int(default_str) if '.' not in default_str else float(default_str)
                except:
                    default_value = default_str
            elif default_str in ['true', 'false']:
                param_type = "boolean"
                default_value = default_str == 'true'

            params[param_name] = {
                "type": param_type,
                "default": default_value,
                "semantic": "config",
                "description": f"Parameter {param_name}",
                "extraction_method": "regex_emergency"  # Mark the extraction method
            }

        if params:
            self.logger.warning(f"[Parameter Extraction] Using regex emergency fallback for {skill_name}: {list(params.keys())}")

        return params

    def extract_object_schemas_llm(
        self,
        code: str,
        object_param_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Use the LLM to extract the internal schema for the specified object parameters.

        Args:
            code: Skill code
            object_param_names: List of object-parameter names whose schemas need extracting

        Returns:
            Dict mapping parameter name to schema
        """
        try:
            human_prompt = f"""Extract the internal schema for these object parameters: {object_param_names}

```javascript
{code}
```

Find each parameter's fields by looking for EITHER:
  - Destructuring: `const {{ field1, field2 = 3 }} = paramName;`
  - Property access: `paramName.field1`, `paramName.field2 != null ? ... : default`, `paramName?.field1`, `(paramName.x ?? 3)`, etc.

For each field, infer type from usage, default from JS defaults or null-check fallbacks, and source_hint per the system prompt rules.
Return only JSON."""

            messages = [
                SystemMessage(content=_get_object_schema_extraction_prompt()),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm.invoke(messages)
            from skillnet.utils.stats_tracker import record_llm_usage
            record_llm_usage(response, process_type="metadata_extraction", function_name="metadata.param_extraction.object_schema")
            response_content = response.content.strip()

            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                self.logger.info(
                    f"[Parameter Extraction] Successfully extracted schemas for {len(result)} object parameters"
                )
                return result
            else:
                self.logger.warning("[Parameter Extraction] Failed to parse schema response as JSON")
                return {}

        except Exception as e:
            self.logger.error(f"[Parameter Extraction] Object schema extraction failed: {e}")
            return {}
