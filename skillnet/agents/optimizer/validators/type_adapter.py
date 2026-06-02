"""
Type Adapter - type-adapter generator.

v7.3 Phase 3: detect parameter/expected-type mismatches and generate
type-conversion code.

Core components:
- TypeMismatchKind: enum of mismatch kinds.
- TypeMismatch: dataclass with mismatch details.
- TypeAdapterResult: result of type adaptation.
- TypeMismatchDetector: detects parameter type mismatches.
- TypeAdapterGenerator: generates type-conversion code.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from skillnet.agents.planning.inference.parameter_semantic import (
        ParameterSemantic,
        DirectionSemantic,
    )


class TypeMismatchKind(Enum):
    """Kinds of type mismatch."""
    SUFFIX_MISMATCH = "suffix"        # Suffix mismatch (oak_planks vs _log)
    EXPECTED_NUMBER = "number"        # Expected number, got non-number
    EXPECTED_STRING = "string"        # Expected string, got non-string
    EXPECTED_OBJECT = "object"        # Expected object, got non-object
    NULLABLE_ACCESS = "nullable"      # Property access on possibly null/undefined
    MISSING_PROPERTY = "property"     # Object missing required property


@dataclass
class TypeMismatch:
    """Type-mismatch details."""
    kind: TypeMismatchKind
    param_name: str
    expected_type: str              # Description of expected type
    actual_value: Optional[str]     # Actual value (if inferable)
    source_location: str            # Source location
    suggested_fix: Optional[str]    # Suggested conversion code
    confidence: float = 0.8


@dataclass
class TypeAdapterResult:
    """Type-adapter result."""
    has_issues: bool
    issues: List[TypeMismatch] = field(default_factory=list)
    generated_adapters: Dict[str, str] = field(default_factory=dict)  # param -> conversion code
    warnings: List[str] = field(default_factory=list)


class TypeMismatchDetector:
    """
    Detect parameter/expected-type mismatches.

    Generic JS conversion rules are always available.
    Domain-specific rules (item suffix transforms, registry lookups)
    are injected via domain_knowledge.get_type_adapter_config().
    """

    # Generic JavaScript type conversion rules (always present)
    _GENERIC_CONVERSION_RULES = {
        "string_to_number": {
            "check": lambda v: v.isdigit() if isinstance(v, str) else False,
            "adapter": "parseInt({value})",
            "description": "Convert string to number",
        },
        "number_to_string": {
            "adapter": "String({value})",
            "description": "Convert number to string",
        },
        "nullable_access": {
            "adapter": "{object}?.{property}",
            "description": "Safe nullable property access",
        },
    }

    def __init__(self, domain_knowledge=None):
        config = domain_knowledge.get_type_adapter_config() if domain_knowledge else {}
        self.conversion_rules = {
            **self._GENERIC_CONVERSION_RULES,
            **config.get("conversion_rules", {}),
        }
        self.suffix_requirements = config.get("suffix_requirements", {})

    def detect(
        self,
        param_name: str,
        param_semantic: "ParameterSemantic",
        inferred_value: str,
        skill_name: str,
    ) -> List[TypeMismatch]:
        """
        Detect type mismatches for a single parameter.

        Args:
            param_name: parameter name
            param_semantic: parameter semantic info
            inferred_value: inferred parameter value
            skill_name: skill name

        Returns:
            List[TypeMismatch]: detected mismatches.
        """
        from skillnet.agents.planning.inference.parameter_semantic import DirectionSemantic

        issues = []

        if not inferred_value:
            return issues

        # 1. Check type conversions required by direction semantics
        if param_semantic.direction == DirectionSemantic.INPUT:
            if param_semantic.transform_hint:
                # Try applying the transform; failure indicates a mismatch
                transformed = param_semantic.transform_hint.apply(inferred_value)
                if transformed is None:
                    suggested = self._suggest_transform(inferred_value, param_semantic)
                    issues.append(TypeMismatch(
                        kind=TypeMismatchKind.SUFFIX_MISMATCH,
                        param_name=param_name,
                        expected_type=f"input type matching transform hint",
                        actual_value=inferred_value,
                        source_location=f"{skill_name}.{param_name}",
                        suggested_fix=suggested,
                    ))
        # Phase 3.5 issue C: reverse check - OUTPUT direction also needs validation
        elif param_semantic.direction == DirectionSemantic.OUTPUT:
            # OUTPUT parameters should not have input-material suffixes (_log, _ore, etc.)
            input_suffixes = ["_log", "_ore", "raw_"]
            for suffix in input_suffixes:
                if inferred_value.endswith(suffix) or inferred_value.startswith(suffix):
                    # Likely a type error: OUTPUT parameter received an INPUT-typed value
                    issues.append(TypeMismatch(
                        kind=TypeMismatchKind.SUFFIX_MISMATCH,
                        param_name=param_name,
                        expected_type=f"output type (not input material)",
                        actual_value=inferred_value,
                        source_location=f"{skill_name}.{param_name}",
                        suggested_fix=None,  # Needs more complex transformation logic
                        confidence=0.6,  # Lower confidence because it may be a false positive
                    ))
                    break

        # 2. Check suffix requirements implied by the parameter name.
        # Phase 3.5 issue B: lower confidence because name-implied requirements aren't always accurate
        param_lower = param_name.lower()
        if param_lower in self.suffix_requirements:
            expected_suffix = self.suffix_requirements[param_lower]
            if not inferred_value.endswith(expected_suffix):
                suggested = self._get_suffix_conversion(inferred_value, expected_suffix)
                # Adjust confidence based on parameter-name precision.
                # Exact-match names (e.g. "logtype") have higher confidence;
                # partial-match names have lower confidence.
                confidence = 0.75 if param_lower in ["logtype", "log_type", "oretype", "ore_type"] else 0.6
                issues.append(TypeMismatch(
                    kind=TypeMismatchKind.SUFFIX_MISMATCH,
                    param_name=param_name,
                    expected_type=f"string ending with '{expected_suffix}'",
                    actual_value=inferred_value,
                    source_location=f"{skill_name}.{param_name}",
                    suggested_fix=suggested,
                    confidence=confidence,
                ))

        # 3. Check number type
        if self._looks_like_count_param(param_name):
            if not self._is_number(inferred_value):
                issues.append(TypeMismatch(
                    kind=TypeMismatchKind.EXPECTED_NUMBER,
                    param_name=param_name,
                    expected_type="number",
                    actual_value=inferred_value,
                    source_location=f"{skill_name}.{param_name}",
                    suggested_fix=f"parseInt({inferred_value})" if inferred_value.strip('"\'').isdigit() else None,
                ))

        return issues

    def detect_by_param_name_only(
        self,
        param_name: str,
        inferred_value: str,
        skill_name: str,
    ) -> List[TypeMismatch]:
        """
        v7.3 Phase 3.5: detect type mismatches using only the parameter name.

        When full ParameterSemantic info is not available, do basic type
        detection from the parameter name alone. Lightweight; confidence is
        usually lower.

        Args:
            param_name: parameter name
            inferred_value: inferred parameter value
            skill_name: skill name

        Returns:
            List[TypeMismatch]: detected mismatches.
        """
        issues = []

        if not inferred_value:
            return issues

        # 1. Check suffix requirements implied by the parameter name
        param_lower = param_name.lower()
        if param_lower in self.suffix_requirements:
            expected_suffix = self.suffix_requirements[param_lower]
            if not inferred_value.endswith(expected_suffix):
                suggested = self._get_suffix_conversion(inferred_value, expected_suffix)
                issues.append(TypeMismatch(
                    kind=TypeMismatchKind.SUFFIX_MISMATCH,
                    param_name=param_name,
                    expected_type=f"string ending with '{expected_suffix}'",
                    actual_value=inferred_value,
                    source_location=f"{skill_name}.{param_name}",
                    suggested_fix=suggested,
                    confidence=0.6,  # Lower confidence: inferred from parameter name alone
                ))

        # 2. Check number type
        if self._looks_like_count_param(param_name):
            if not self._is_number(inferred_value):
                issues.append(TypeMismatch(
                    kind=TypeMismatchKind.EXPECTED_NUMBER,
                    param_name=param_name,
                    expected_type="number",
                    actual_value=inferred_value,
                    source_location=f"{skill_name}.{param_name}",
                    suggested_fix=f"parseInt({inferred_value})" if inferred_value.strip('"\'').isdigit() else None,
                    confidence=0.7,  # Number detection is relatively reliable
                ))

        # 3. Direction inference based on parameter name.
        # If the name implies an input material, check for output-product suffixes.
        input_indicators = ["input", "source", "log_type", "ore_type", "material", "raw"]
        output_indicators = ["output", "result", "product", "target"]

        if any(ind in param_lower for ind in input_indicators):
            # Input parameters should not have output-product suffixes
            output_suffixes = ["_planks", "_ingot", "_block"]
            for suffix in output_suffixes:
                if inferred_value.endswith(suffix):
                    issues.append(TypeMismatch(
                        kind=TypeMismatchKind.SUFFIX_MISMATCH,
                        param_name=param_name,
                        expected_type="input material (not output product)",
                        actual_value=inferred_value,
                        source_location=f"{skill_name}.{param_name}",
                        suggested_fix=None,
                        confidence=0.5,  # Low-confidence inference
                    ))
                    break

        elif any(ind in param_lower for ind in output_indicators):
            # Output parameters should not have input-material suffixes
            input_suffixes = ["_log", "_ore"]
            for suffix in input_suffixes:
                if inferred_value.endswith(suffix):
                    issues.append(TypeMismatch(
                        kind=TypeMismatchKind.SUFFIX_MISMATCH,
                        param_name=param_name,
                        expected_type="output product (not input material)",
                        actual_value=inferred_value,
                        source_location=f"{skill_name}.{param_name}",
                        suggested_fix=None,
                        confidence=0.5,  # Low-confidence inference
                    ))
                    break

        return issues

    def detect_from_code(
        self,
        code: str,
        skill_graph_manager: Any,
    ) -> TypeAdapterResult:
        """
        Detect type mismatches in code.

        Analyzes function calls in the code and checks parameter-type matches.

        Args:
            code: JavaScript skill code
            skill_graph_manager: SkillGraphManager instance

        Returns:
            TypeAdapterResult: detection result.
        """
        issues = []
        adapters = {}
        warnings = []

        # Extract function calls
        function_calls = self._extract_function_calls(code)

        for call in function_calls:
            skill_name = call.get("name", "")
            args = call.get("args", {})

            # Check whether the skill is known
            if skill_graph_manager and skill_graph_manager.has_node(skill_name):
                node = skill_graph_manager.get_node(skill_name)

                # Get parameter metadata
                parameters = getattr(node, 'parameters', {}) or {}

                for param_name, param_meta in parameters.items():
                    if param_name in args:
                        arg_value = args[param_name]
                        semantic_data = param_meta.get("semantic", {})

                        if semantic_data:
                            from skillnet.agents.planning.inference.parameter_semantic import ParameterSemantic
                            semantic = ParameterSemantic.from_dict(semantic_data)
                            param_issues = self.detect(param_name, semantic, arg_value, skill_name)
                            issues.extend(param_issues)

                            # Generate adapters
                            generator = TypeAdapterGenerator()
                            for issue in param_issues:
                                adapter = generator.generate_adapter(issue, {})
                                if adapter:
                                    adapters[f"{skill_name}.{param_name}"] = adapter

        return TypeAdapterResult(
            has_issues=len(issues) > 0,
            issues=issues,
            generated_adapters=adapters,
            warnings=warnings,
        )

    def _suggest_transform(self, value: str, semantic: "ParameterSemantic") -> Optional[str]:
        """Suggest a transform based on semantics."""
        if semantic.transform_hint and semantic.transform_hint.transform_type == "suffix_replace":
            source_suffix = semantic.transform_hint.source_suffix or ""
            target_suffix = semantic.transform_hint.target_suffix or ""
            if source_suffix and value.endswith(source_suffix):
                base = value[:-len(source_suffix)]
                return f'"{base}{target_suffix}"'
        return None

    def _get_suffix_conversion(self, value: str, expected_suffix: str) -> Optional[str]:
        """Return a suffix-conversion suggestion."""
        # Try the conversion rules
        for rule_name, rule in self.conversion_rules.items():
            if "pattern" in rule and "replacement" in rule:
                match = re.match(rule["pattern"], value)
                if match:
                    result = re.sub(rule["pattern"], rule["replacement"], value)
                    if result.endswith(expected_suffix):
                        return f'"{result}"'

        # Simple suffix-replacement fallback
        suffix_replacements = [
            ("_planks", "_log"),
            ("_log", "_planks"),
            ("_ore", ""),
            ("_ingot", ""),
        ]

        for old_suffix, new_suffix in suffix_replacements:
            if value.endswith(old_suffix):
                base = value[:-len(old_suffix)] if old_suffix else value
                result = base + expected_suffix
                return f'"{result}"'

        return None

    def _looks_like_count_param(self, param_name: str) -> bool:
        """Check whether the parameter name looks like a count parameter."""
        count_keywords = ["count", "amount", "num", "quantity", "total"]
        param_lower = param_name.lower()
        return any(kw in param_lower for kw in count_keywords)

    def _is_number(self, value: str) -> bool:
        """Check whether the value is a number."""
        if not value:
            return False
        # Strip quotes
        clean_value = value.strip('"\'')
        try:
            float(clean_value)
            return True
        except ValueError:
            return False

    def _extract_function_calls(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract function calls from code.

        Simplified: extracts await skillName(...) calls.
        """
        calls = []

        # Match await functionName(args)
        pattern = r'await\s+(\w+)\s*\(([^)]*)\)'

        for match in re.finditer(pattern, code):
            func_name = match.group(1)
            args_str = match.group(2).strip()

            # Parse arguments (simplified)
            args = {}
            if args_str:
                # Try named-arg forms: key: value or key = value
                arg_pattern = r'(\w+)\s*[=:]\s*([^,]+)'
                for arg_match in re.finditer(arg_pattern, args_str):
                    arg_name = arg_match.group(1)
                    arg_value = arg_match.group(2).strip().strip('"\'')
                    args[arg_name] = arg_value

            calls.append({
                "name": func_name,
                "args": args,
            })

        return calls


class TypeAdapterGenerator:
    """
    Generate type-conversion code snippets.

    Generates conversion code based on detected type mismatches.
    """

    def generate_adapter(
        self,
        mismatch: TypeMismatch,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Generate adapter code for a type mismatch.

        Args:
            mismatch: type-mismatch details
            context: context info

        Returns:
            Optional[str]: generated adapter code, or None if not possible.
        """
        if mismatch.kind == TypeMismatchKind.SUFFIX_MISMATCH:
            return self._generate_suffix_adapter(mismatch)

        elif mismatch.kind == TypeMismatchKind.NULLABLE_ACCESS:
            return self._generate_nullable_adapter(mismatch, context)

        elif mismatch.kind == TypeMismatchKind.EXPECTED_NUMBER:
            if mismatch.actual_value:
                return f"parseInt({mismatch.actual_value})"

        elif mismatch.kind == TypeMismatchKind.EXPECTED_STRING:
            if mismatch.actual_value:
                return f"String({mismatch.actual_value})"

        return None

    def _generate_suffix_adapter(self, mismatch: TypeMismatch) -> Optional[str]:
        """Generate a suffix-conversion adapter."""
        value = mismatch.actual_value
        if not value:
            return None

        # If a suggested fix already exists, return it directly
        if mismatch.suggested_fix:
            return mismatch.suggested_fix

        # Pick a conversion based on the expected type
        expected = mismatch.expected_type

        if "_log" in expected:
            if value.endswith("_planks"):
                base = value[:-len("_planks")]
                return f'"{base}_log"'
        elif "_planks" in expected:
            if value.endswith("_log"):
                base = value[:-len("_log")]
                return f'"{base}_planks"'

        return None

    def _generate_nullable_adapter(self, mismatch: TypeMismatch, context: Dict[str, Any]) -> str:
        """Generate a nullable-access adapter."""
        obj = context.get("object", "obj")
        prop = context.get("property", "x")
        return f"{obj}?.{prop}"

    def apply_adapters_to_code(
        self,
        code: str,
        adapters: Dict[str, str],
    ) -> str:
        """
        Apply adapters to code.

        Args:
            code: original code
            adapters: mapping of parameter -> adapter code

        Returns:
            str: code with adapters applied.
        """
        result = code

        for param_key, adapter in adapters.items():
            # param_key format: "skillName.paramName"
            parts = param_key.split(".")
            if len(parts) != 2:
                continue

            skill_name, param_name = parts

            # Find and replace the parameter value in the code.
            # Simplified implementation; complex cases need AST analysis.
            pattern = rf'(await\s+{skill_name}\s*\([^)]*{param_name}\s*[=:]\s*)([^,)]+)'

            def replacer(match):
                prefix = match.group(1)
                old_value = match.group(2)
                return f"{prefix}{adapter}"

            result = re.sub(pattern, replacer, result)

        return result


# Convenience functions

def detect_type_mismatches(
    code: str,
    skill_graph_manager: Any,
    domain_knowledge=None,
) -> TypeAdapterResult:
    """
    Detect type mismatches in code.

    Convenience function for quick type checks of code.

    Args:
        code: JavaScript skill code
        skill_graph_manager: SkillGraphManager instance
        domain_knowledge: Optional DomainKnowledge for domain-specific rules

    Returns:
        TypeAdapterResult: detection result.
    """
    detector = TypeMismatchDetector(domain_knowledge=domain_knowledge)
    return detector.detect_from_code(code, skill_graph_manager)


def apply_type_adapters(
    code: str,
    result: TypeAdapterResult,
) -> str:
    """
    Apply type adapters to code.

    Convenience function for applying detected adapters to code.

    Args:
        code: original code
        result: type-adapter detection result

    Returns:
        str: code with adapters applied.
    """
    if not result.has_issues:
        return code

    generator = TypeAdapterGenerator()
    return generator.apply_adapters_to_code(code, result.generated_adapters)
