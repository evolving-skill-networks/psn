"""
Signature Compatibility Checker

v7.4 Phase 4: signature compatibility checker

Features:
1. Parse JavaScript async function signatures
2. Extract parameter names, types (inferred from JSDoc or default values), and default values
3. Compare two signatures for compatibility
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class SignatureInfo:
    """
    Function signature information.

    Represents the signature of a JavaScript async function, including
    the function name, parameter list, return type, async flag, etc.
    """
    name: str
    params: List[Tuple[str, Optional[str], Optional[str]]]  # (name, type, default)
    return_type: Optional[str] = None
    is_async: bool = True
    raw_params_str: str = ""  # Original parameter string

    @property
    def param_names(self) -> List[str]:
        """Get all parameter names."""
        return [p[0] for p in self.params]

    @property
    def required_params(self) -> List[str]:
        """Get required parameters (those without a default value)."""
        return [p[0] for p in self.params if p[2] is None]

    @property
    def optional_params(self) -> List[str]:
        """Get optional parameters (those with a default value)."""
        return [p[0] for p in self.params if p[2] is not None]


@dataclass
class SignatureCompatibilityResult:
    """
    Signature compatibility check result.
    """
    is_compatible: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    old_signature: Optional[SignatureInfo] = None
    new_signature: Optional[SignatureInfo] = None

    def __bool__(self) -> bool:
        return self.is_compatible


class SignatureCompatibilityChecker:
    """
    Signature compatibility checker.

    Checks whether a refactored function signature is compatible with the original.

    Compatibility rules:
    1. Parameter count: the new signature may add optional parameters, but cannot remove required ones
    2. Parameter order: the order of required parameters must be preserved
    3. Parameter types: types must be compatible or implicitly convertible
    4. Default values: existing default values should be preserved

    Usage:
        checker = SignatureCompatibilityChecker()

        old_sig = checker.parse_signature(old_code)
        new_sig = checker.parse_signature(new_code)

        result = checker.check_compatibility(old_sig, new_sig)
        if not result.is_compatible:
            print(f"Signature incompatible: {result.issues}")
    """

    def __init__(self, logger_instance=None):
        self.logger = logger_instance or logger

    def parse_signature(self, code: str, func_name: str = None) -> Optional[SignatureInfo]:
        """
        Parse a function signature from JavaScript code.

        Args:
            code: JavaScript code
            func_name: Function name to parse (optional; if omitted, parses the first async function)

        Returns:
            SignatureInfo: Parsed signature info, or None if parsing fails
        """
        # Build the matching pattern
        if func_name:
            pattern = rf'async\s+function\s+({re.escape(func_name)})\s*\(([^)]*)\)\s*\{{'
        else:
            pattern = r'async\s+function\s+(\w+)\s*\(([^)]*)\)\s*\{'

        match = re.search(pattern, code)
        if not match:
            self.logger.debug(f"[SignatureChecker] Function definition not found: {func_name or '(any)'}")
            return None

        name = match.group(1)
        params_str = match.group(2).strip()

        # Parse the parameter list
        params = self._parse_params(params_str)

        # Try to extract type and return-value info from JSDoc
        jsdoc_info = self._parse_jsdoc(code, match.start())

        # Merge JSDoc info
        if jsdoc_info:
            params = self._merge_jsdoc_types(params, jsdoc_info.get('params', {}))

        return SignatureInfo(
            name=name,
            params=params,
            return_type=jsdoc_info.get('return_type') if jsdoc_info else None,
            is_async=True,
            raw_params_str=params_str,
        )

    def _parse_params(
        self,
        params_str: str
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Parse the parameter string.

        Args:
            params_str: Parameter string, e.g. "bot, itemType, count = 1"

        Returns:
            List[Tuple[str, Optional[str], Optional[str]]]: Parameter list [(name, type, default), ...]
        """
        if not params_str.strip():
            return []

        params = []

        # Split parameters (handle commas inside object destructuring and default values)
        param_parts = self._split_params(params_str)

        for part in param_parts:
            part = part.strip()
            if not part:
                continue

            # Parse a single parameter
            param_info = self._parse_single_param(part)
            if param_info:
                params.append(param_info)

        return params

    def _split_params(self, params_str: str) -> List[str]:
        """
        Smartly split the parameter string (handles nested structures).

        Args:
            params_str: Parameter string

        Returns:
            List[str]: Split parameter list
        """
        params = []
        current = ""
        depth = 0  # Depth of parentheses/braces/brackets

        for char in params_str:
            if char in '({[':
                depth += 1
                current += char
            elif char in ')}]':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                params.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            params.append(current.strip())

        return params

    def _parse_single_param(
        self,
        param_str: str
    ) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """
        Parse a single parameter.

        Args:
            param_str: Parameter string, e.g. "count = 1", "{ x, y }", or "itemType"

        Returns:
            Tuple[str, Optional[str], Optional[str]]: (name, type, default)
        """
        param_str = param_str.strip()

        # Check whether it is a destructured parameter { ... } or [ ... ]
        if param_str.startswith('{') or param_str.startswith('['):
            # Destructured parameter — take the whole expression as the parameter name
            # Check whether it has a default value
            if '=' in param_str:
                # Find the outermost =
                eq_pos = self._find_default_equals(param_str)
                if eq_pos > 0:
                    name = param_str[:eq_pos].strip()
                    default = param_str[eq_pos+1:].strip()
                    return (name, 'object', default)
            return (param_str, 'object', None)

        # Check whether it has a default value
        if '=' in param_str:
            eq_pos = param_str.find('=')
            name = param_str[:eq_pos].strip()
            default = param_str[eq_pos+1:].strip()

            # Infer the type from the default value
            inferred_type = self._infer_type_from_default(default)

            return (name, inferred_type, default)

        # Ordinary parameter (no default value)
        return (param_str, None, None)

    def _find_default_equals(self, param_str: str) -> int:
        """
        Find the position of the default-value equals sign (skipping any '=' inside nested structures).

        Args:
            param_str: Parameter string

        Returns:
            int: Equals position, or -1 if not found
        """
        depth = 0
        for i, char in enumerate(param_str):
            if char in '({[':
                depth += 1
            elif char in ')}]':
                depth -= 1
            elif char == '=' and depth == 0:
                return i
        return -1

    def _infer_type_from_default(self, default: str) -> Optional[str]:
        """
        Infer the type from a default value.

        Args:
            default: Default-value string

        Returns:
            str: Inferred type, or None if it cannot be inferred
        """
        default = default.strip()

        # Number
        if re.match(r'^-?\d+\.?\d*$', default):
            return 'number'

        # Boolean
        if default in ('true', 'false'):
            return 'boolean'

        # String
        if (default.startswith('"') and default.endswith('"')) or \
           (default.startswith("'") and default.endswith("'")):
            return 'string'

        # Array
        if default.startswith('[') and default.endswith(']'):
            return 'array'

        # Object
        if default.startswith('{') and default.endswith('}'):
            return 'object'

        # null/undefined
        if default in ('null', 'undefined'):
            return 'null'

        return None

    def _parse_jsdoc(self, code: str, func_start: int) -> Optional[Dict[str, Any]]:
        """
        Parse the JSDoc comment preceding the function.

        Args:
            code: Full code
            func_start: Start position of the function definition

        Returns:
            Dict: JSDoc info, including params and return_type
        """
        # Find the JSDoc comment immediately before the function definition
        before_func = code[:func_start]

        # Match the nearest JSDoc comment block
        jsdoc_match = re.search(r'/\*\*[\s\S]*?\*/', before_func[::-1])
        if not jsdoc_match:
            return None

        jsdoc = jsdoc_match.group(0)[::-1]

        result = {
            'params': {},
            'return_type': None,
        }

        # Parse @param
        param_pattern = r'@param\s+\{([^}]+)\}\s+(\w+)'
        for match in re.finditer(param_pattern, jsdoc):
            param_type = match.group(1)
            param_name = match.group(2)
            result['params'][param_name] = param_type

        # Parse @returns
        return_pattern = r'@returns?\s+\{([^}]+)\}'
        return_match = re.search(return_pattern, jsdoc)
        if return_match:
            result['return_type'] = return_match.group(1)

        return result

    def _merge_jsdoc_types(
        self,
        params: List[Tuple[str, Optional[str], Optional[str]]],
        jsdoc_params: Dict[str, str],
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Merge JSDoc type information into the parameter list.

        Args:
            params: Original parameter list
            jsdoc_params: Type information from JSDoc

        Returns:
            List: Merged parameter list
        """
        result = []
        for name, inferred_type, default in params:
            # If JSDoc has type info, use the JSDoc version
            if name in jsdoc_params:
                result.append((name, jsdoc_params[name], default))
            else:
                result.append((name, inferred_type, default))
        return result

    def check_compatibility(
        self,
        old_signature: SignatureInfo,
        new_signature: SignatureInfo,
    ) -> SignatureCompatibilityResult:
        """
        Check the compatibility of two signatures.

        Args:
            old_signature: Original function signature
            new_signature: New function signature

        Returns:
            SignatureCompatibilityResult: Compatibility check result
        """
        issues = []
        warnings = []

        # Check 1: required parameters cannot be removed
        old_required = set(old_signature.required_params)
        new_required = set(new_signature.required_params)
        new_all = set(new_signature.param_names)

        removed_params = old_required - new_all
        if removed_params:
            issues.append(
                f"Required parameters removed: {', '.join(sorted(removed_params))}"
            )

        # Check 2: order of required parameters
        old_required_list = old_signature.required_params
        new_required_list = [p for p in new_signature.required_params if p in old_required]

        # Check whether the retained required parameters' order is consistent
        old_order = [p for p in old_required_list if p in new_all]
        new_order = [p for p in new_signature.param_names if p in old_required]

        if old_order != new_order:
            issues.append(
                f"Required parameter order changed: {old_order} -> {new_order}"
            )

        # Check 3: newly added required parameters (warning)
        new_required_added = new_required - old_required
        if new_required_added:
            warnings.append(
                f"Newly added required parameters (may affect callers): {', '.join(sorted(new_required_added))}"
            )

        # Check 4: default-value changes
        for old_param in old_signature.params:
            old_name, old_type, old_default = old_param

            # Find the corresponding parameter in the new signature
            new_param = None
            for np in new_signature.params:
                if np[0] == old_name:
                    new_param = np
                    break

            if new_param:
                new_name, new_type, new_default = new_param

                # Default value removed (parameter changed from optional to required)
                if old_default is not None and new_default is None:
                    issues.append(
                        f"Default value of parameter '{old_name}' was removed (changed from optional to required)"
                    )

                # Default value changed (warning)
                if old_default is not None and new_default is not None:
                    if old_default != new_default:
                        warnings.append(
                            f"Default value of parameter '{old_name}' changed: {old_default} -> {new_default}"
                        )

        # Check 5: type compatibility
        for old_param in old_signature.params:
            old_name, old_type, _ = old_param

            if old_type is None:
                continue

            # Find the corresponding parameter in the new signature
            new_param = None
            for np in new_signature.params:
                if np[0] == old_name:
                    new_param = np
                    break

            if new_param and new_param[1] is not None:
                new_type = new_param[1]
                if not self._types_compatible(old_type, new_type):
                    warnings.append(
                        f"Type of parameter '{old_name}' may be incompatible: {old_type} -> {new_type}"
                    )

        is_compatible = len(issues) == 0

        if not is_compatible:
            self.logger.warning(
                f"[SignatureChecker] Signature incompatible ({old_signature.name}): {issues}"
            )
        elif warnings:
            self.logger.info(
                f"[SignatureChecker] Signature compatible with warnings ({old_signature.name}): {warnings}"
            )

        return SignatureCompatibilityResult(
            is_compatible=is_compatible,
            issues=issues,
            warnings=warnings,
            old_signature=old_signature,
            new_signature=new_signature,
        )

    def _types_compatible(self, old_type: str, new_type: str) -> bool:
        """
        Check whether two types are compatible.

        Args:
            old_type: Old type
            new_type: New type

        Returns:
            bool: Whether they are compatible
        """
        if old_type == new_type:
            return True

        # Normalize type names
        old_normalized = old_type.lower().strip()
        new_normalized = new_type.lower().strip()

        if old_normalized == new_normalized:
            return True

        # Define type-compatibility rules
        compatible_pairs = [
            # number-related
            ('number', 'int'),
            ('number', 'integer'),
            ('number', 'float'),
            ('int', 'integer'),

            # string-related
            ('string', 'str'),

            # array-related
            ('array', 'list'),
            ('array', 'array<*>'),

            # object-related
            ('object', 'dict'),
            ('object', 'map'),

            # any type is compatible with everything
            ('any', '*'),
        ]

        for t1, t2 in compatible_pairs:
            if (old_normalized == t1 and new_normalized == t2) or \
               (old_normalized == t2 and new_normalized == t1):
                return True

        # any type is compatible with everything
        if 'any' in (old_normalized, new_normalized):
            return True

        return False

    def extract_signature_from_code(
        self,
        code: str,
        func_name: str = None,
    ) -> Optional[SignatureInfo]:
        """
        Convenience method: extract signature info from code.

        Args:
            code: JavaScript code
            func_name: Function name (optional)

        Returns:
            SignatureInfo: Signature info
        """
        return self.parse_signature(code, func_name)
