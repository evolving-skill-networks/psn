"""
Over-Claim Detector

Detects when a function's declared supported parameter values exceed what the code can actually handle.

Over-claim definition:
- The function declares support for multiple parameter values (e.g. material = "wooden"|"stone"|"iron")
- But the code logic can only correctly handle a subset

Typical scenario:
```javascript
async function craftPickaxe(bot, material = "wooden") {
    // material + "_planks" only works for wooden!
    // stone_planks and iron_planks don't exist!
    const planks = material + "_planks";
}
```

Correct mapping:
- wooden → oak_planks (or other planks)
- stone → cobblestone
- iron → iron_ingot
- gold → gold_ingot
- diamond → diamond
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any


@dataclass
class OverclaimResult:
    """Over-claim detection result"""
    has_overclaim: bool
    confidence: float  # 0.0 - 1.0
    skill_name: str = ""
    problematic_param: Optional[str] = None
    declared_values: Optional[List[str]] = None
    actually_supported: Optional[List[str]] = None
    evidence: str = ""

    # Detailed list of issues
    issues: List[str] = field(default_factory=list)


class OverclaimDetector:
    """
    Detect whether a function has an over-claim issue

    Over-claim: the function declares support for some parameter values, but the code cannot handle them all correctly.

    Usage:
        detector = OverclaimDetector()

        result = detector.detect(
            func_name="craftPickaxe",
            func_code=code,
            parameters={"material": {"type": "string", "default": "wooden"}}
        )

        if result.has_overclaim:
            print(f"Over-claim detected: {result.evidence}")
    """

    # Name-heuristic keywords
    MATERIAL_INDICATORS = ["material", "type", "kind", "variant", "tier"]

    def __init__(self, logger=None, concatenation_patterns=None,
                 pattern_supported_values=None, semantic_contexts=None):
        self.logger = logger
        self._concatenation_patterns = concatenation_patterns or []
        self._pattern_supported_values = pattern_supported_values or {}
        self._semantic_contexts = semantic_contexts or []

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            colors = {
                "info": "\033[36m",
                "warning": "\033[33m",
                "error": "\033[31m",
                "success": "\033[32m",
            }
            reset = "\033[0m"
            print(f"{colors.get(level, '')}{message}{reset}")

    def detect(
        self,
        func_name: str,
        func_code: str,
        parameters: Dict[str, Any],
    ) -> OverclaimResult:
        """
        Detect over-claim

        Args:
            func_name: function name
            func_code: function code
            parameters: parameter metadata {param_name: {type, default, supported_values, semantic, ...}}

        Returns:
            OverclaimResult: detection result
        """
        # 1. Check whether there are material/type-style parameters
        material_params = self._find_material_params(parameters)
        if not material_params:
            return OverclaimResult(
                has_overclaim=False,
                confidence=0.0,
                skill_name=func_name
            )

        # 2. Check the code for problematic string concatenation
        # Iterate over all material parameters and check each one
        all_issues = []
        first_problematic_param = None
        first_pattern_info = None

        for param_name in material_params:
            for pattern, name, msg in self._concatenation_patterns:
                if re.search(pattern, func_code):
                    param_info = parameters.get(param_name, {})
                    declared = param_info.get("supported_values", [])

                    issue = f"String concatenation pattern '{name}' only works for specific materials (param: {param_name})"
                    all_issues.append(issue)

                    if first_problematic_param is None:
                        first_problematic_param = param_name
                        first_pattern_info = (name, msg, declared)

        if all_issues and first_pattern_info:
            name, msg, declared = first_pattern_info
            result = OverclaimResult(
                has_overclaim=True,
                confidence=0.8,
                skill_name=func_name,
                problematic_param=first_problematic_param,
                declared_values=declared if declared else ["inferred from default"],
                actually_supported=self._infer_supported_from_pattern(name),
                evidence=f"Found '{name}' pattern in code",
                issues=all_issues  # Includes all detected issues
            )

            self._log(
                f"[OverclaimDetector] Over-claim in '{func_name}': {result.evidence}",
                "warning"
            )
            return result

        # 3. Check whether there is an explicit mapping (good pattern)
        if self._has_explicit_mapping(func_code):
            return OverclaimResult(
                has_overclaim=False,
                confidence=0.9,
                skill_name=func_name,
                evidence="Found explicit mapping pattern (good practice)"
            )

        # 4. Check whether there are conditional branches covering all declared values
        for param_name in material_params:
            param_info = parameters.get(param_name, {})
            declared = param_info.get("supported_values", [])
            if declared:
                covered = self._find_covered_values(func_code, param_name)
                uncovered = set(declared) - covered
                if uncovered:
                    return OverclaimResult(
                        has_overclaim=True,
                        confidence=0.7,  # Set to 0.7 to satisfy the check_and_log threshold
                        skill_name=func_name,
                        problematic_param=param_name,
                        declared_values=declared,
                        actually_supported=list(covered),
                        evidence=f"Values not handled in code: {uncovered}",
                        issues=[f"Missing handling for: {', '.join(uncovered)}"]
                    )

        return OverclaimResult(
            has_overclaim=False,
            confidence=0.5,
            skill_name=func_name
        )

    def _find_material_params(self, parameters: Dict[str, Any]) -> List[str]:
        """
        Find parameters that may be material types (enhanced)

        Three identification methods:
        1. Name heuristic (keywords like material, type, kind, etc.)
        2. semantic.config_context check
        3. String parameters with supported_values
        """
        result = []

        if not isinstance(parameters, dict):
            return result

        for name, info in parameters.items():
            if not isinstance(info, dict):
                continue

            # Method 1: name heuristic
            name_lower = name.lower()
            if any(ind in name_lower for ind in self.MATERIAL_INDICATORS):
                result.append(name)
                continue

            # Method 2: semantic.config_context check
            semantic = info.get("semantic", {})
            if isinstance(semantic, dict) and self._semantic_contexts and semantic.get("config_context") in self._semantic_contexts:
                result.append(name)
                continue

            # Method 3: string parameters with supported_values
            if info.get("type") == "string" and info.get("supported_values"):
                result.append(name)

        return list(set(result))  # Deduplicate

    def _has_explicit_mapping(self, code: str) -> bool:
        """Check whether there is an explicit mapping (good pattern)"""
        patterns = [
            r'MATERIAL_TO_\w+\s*=\s*\{',  # Constant mapping
            r'const\s+\w+Map\s*=\s*\{',   # xxxMap mapping
            r'const\s+\w+_TO_\w+\s*=\s*\{',  # XXX_TO_YYY mapping
            r'switch\s*\(\s*material',     # switch statement
            r'if\s*\(\s*material\s*===',   # if-else chain
            r'\[\s*material\s*\]',          # Index access mapping[material]
        ]
        return any(re.search(p, code) for p in patterns)

    def _find_covered_values(self, code: str, param_name: str) -> Set[str]:
        """Find the parameter values actually handled in code"""
        covered = set()
        # Match if (param === "value") or case "value":
        # Use re.escape to ensure special chars in the param name are escaped
        escaped_param = re.escape(param_name)
        patterns = [
            rf'{escaped_param}\s*===\s*["\'](\w+)["\']',
            rf'case\s*["\'](\w+)["\']',
            rf'\["{escaped_param}"\]\s*===\s*["\'](\w+)["\']',
            rf'{escaped_param}\s*==\s*["\'](\w+)["\']',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, code)
            covered.update(matches)
        return covered

    def _infer_supported_from_pattern(self, pattern_name: str) -> List[str]:
        """Infer actually-supported values from the problematic pattern"""
        return list(self._pattern_supported_values.get(pattern_name, []))

    def check_and_log(
        self,
        func_name: str,
        func_code: str,
        parameters: Dict[str, Any],
    ) -> Optional[OverclaimResult]:
        """
        Check and log over-claim issues

        Logs a warning if a high-confidence over-claim is detected.

        Returns:
            The result if over-claim is detected (confidence >= 0.7); otherwise None
        """
        result = self.detect(func_name, func_code, parameters)

        if result.has_overclaim and result.confidence >= 0.7:
            self._log(f"\033[33m[Overclaim] Detected in '{func_name}':\033[0m", "warning")
            self._log(f"  Parameter: {result.problematic_param}", "warning")
            self._log(f"  Declared: {result.declared_values}", "warning")
            self._log(f"  Actually supports: {result.actually_supported}", "warning")
            return result

        return None
