"""
Call Chain Impact Analyzer

v7.4 Phase 4: call-chain impact analyzer

Features:
1. Retrieve call relationships from the SkillGraph
2. Find all callers that depend on the target skill
3. Analyze the impact of modifications on those callers
"""

import logging
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillGraph, SkillNode
    from .signature_checker import SignatureInfo

logger = logging.getLogger(__name__)


@dataclass
class CallerImpact:
    """
    Impact analysis for a single caller.
    """
    caller_name: str
    impact_type: str  # "breaking", "warning", "safe"
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_breaking(self) -> bool:
        return self.impact_type == "breaking"

    @property
    def is_safe(self) -> bool:
        return self.impact_type == "safe"


@dataclass
class CallChainImpactResult:
    """
    Result of call-chain impact analysis.
    """
    is_safe: bool
    total_callers: int
    affected_callers: int
    breaking_callers: List[CallerImpact] = field(default_factory=list)
    warning_callers: List[CallerImpact] = field(default_factory=list)
    safe_callers: List[CallerImpact] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_safe


class CallChainImpactAnalyzer:
    """
    Call-chain impact analyzer.

    Analyzes the impact of modifying a skill's signature on all of its callers.

    Usage:
        analyzer = CallChainImpactAnalyzer(skill_graph)

        # Get all callers
        callers = analyzer.get_callers("mineBlock")

        # Analyze the impact of a signature change
        result = analyzer.analyze_impact(
            skill_name="mineBlock",
            old_signature=old_sig,
            new_signature=new_sig
        )

        if not result.is_safe:
            print(f"Breaking changes: {result.breaking_callers}")
    """

    def __init__(
        self,
        skill_graph=None,
        logger_instance=None,
    ):
        """
        Initialize the call-chain analyzer.

        Args:
            skill_graph: SkillGraph or SkillGraphManager instance (duck-typed)
            logger_instance: Logger instance (optional)
        """
        self.graph = skill_graph
        self.logger = logger_instance or logger

    def get_callers(self, skill_name: str) -> List[str]:
        """
        Get all callers of the skill.

        Args:
            skill_name: Skill name

        Returns:
            List[str]: List of caller names
        """
        if not self.graph or not self.graph.has_node(skill_name):
            return []

        # Use the SkillGraph API to get parents (callers)
        return self.graph.get_parents(skill_name)

    def get_all_callers_recursive(
        self,
        skill_name: str,
        max_depth: int = None,
    ) -> Set[str]:
        """
        Recursively get all direct and indirect callers of the skill.

        Args:
            skill_name: Skill name
            max_depth: Maximum recursion depth (None means no limit)

        Returns:
            Set[str]: Set of all caller names
        """
        if not self.graph or not self.graph.has_node(skill_name):
            return set()

        all_callers = set()
        visited = set()
        queue = [(skill_name, 0)]  # (node, depth)

        while queue:
            current, depth = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)

            if max_depth is not None and depth > max_depth:
                continue

            # Get direct callers
            callers = self.graph.get_parents(current)
            for caller in callers:
                if caller not in visited:
                    all_callers.add(caller)
                    queue.append((caller, depth + 1))

        return all_callers

    def get_callees(self, skill_name: str) -> List[str]:
        """
        Get all skills called by this skill (children).

        Args:
            skill_name: Skill name

        Returns:
            List[str]: List of called skill names
        """
        if not self.graph or not self.graph.has_node(skill_name):
            return []

        return self.graph.get_children(skill_name)

    def analyze_impact(
        self,
        skill_name: str,
        old_signature: "SignatureInfo",
        new_signature: "SignatureInfo",
    ) -> CallChainImpactResult:
        """
        Analyze the impact of a signature change on callers.

        Args:
            skill_name: Name of the modified skill
            old_signature: Original signature
            new_signature: New signature

        Returns:
            CallChainImpactResult: Impact analysis result
        """
        callers = self.get_callers(skill_name)
        total_callers = len(callers)

        breaking_callers = []
        warning_callers = []
        safe_callers = []
        all_issues = []
        all_warnings = []

        if not callers:
            self.logger.debug(
                f"[CallChainAnalyzer] {skill_name} has no callers"
            )
            return CallChainImpactResult(
                is_safe=True,
                total_callers=0,
                affected_callers=0,
            )

        # Analyze each caller
        for caller_name in callers:
            impact = self._analyze_caller_impact(
                caller_name=caller_name,
                target_skill=skill_name,
                old_signature=old_signature,
                new_signature=new_signature,
            )

            if impact.is_breaking:
                breaking_callers.append(impact)
                all_issues.extend(impact.issues)
            elif impact.warnings:
                warning_callers.append(impact)
                all_warnings.extend(impact.warnings)
            else:
                safe_callers.append(impact)

        is_safe = len(breaking_callers) == 0
        affected_callers = len(breaking_callers) + len(warning_callers)

        if not is_safe:
            self.logger.warning(
                f"[CallChainAnalyzer] {skill_name} signature change affects {affected_callers}/{total_callers} caller(s)"
            )
            for impact in breaking_callers:
                self.logger.warning(
                    f"[CallChainAnalyzer]   Breaking: {impact.caller_name} - {impact.issues}"
                )

        return CallChainImpactResult(
            is_safe=is_safe,
            total_callers=total_callers,
            affected_callers=affected_callers,
            breaking_callers=breaking_callers,
            warning_callers=warning_callers,
            safe_callers=safe_callers,
            issues=all_issues,
            warnings=all_warnings,
        )

    def _analyze_caller_impact(
        self,
        caller_name: str,
        target_skill: str,
        old_signature: "SignatureInfo",
        new_signature: "SignatureInfo",
    ) -> CallerImpact:
        """
        Analyze impact on a single caller.

        Args:
            caller_name: Caller name
            target_skill: Name of the called skill
            old_signature: Original signature
            new_signature: New signature

        Returns:
            CallerImpact: Impact analysis result
        """
        issues = []
        warnings = []

        # Get the caller's code
        caller_node = self.graph.get_node(caller_name)
        if not caller_node or not caller_node.code:
            # Cannot analyze; assume it may be impacted
            return CallerImpact(
                caller_name=caller_name,
                impact_type="warning",
                warnings=[f"Could not fetch code for {caller_name}; impact cannot be confirmed"]
            )

        caller_code = caller_node.code

        # Analyze how it is called
        call_analysis = self._analyze_call_in_code(
            caller_code=caller_code,
            target_skill=target_skill,
            old_signature=old_signature,
            new_signature=new_signature,
        )

        issues.extend(call_analysis.get('issues', []))
        warnings.extend(call_analysis.get('warnings', []))

        # Determine the impact type
        if issues:
            impact_type = "breaking"
        elif warnings:
            impact_type = "warning"
        else:
            impact_type = "safe"

        return CallerImpact(
            caller_name=caller_name,
            impact_type=impact_type,
            issues=issues,
            warnings=warnings,
        )

    def _analyze_call_in_code(
        self,
        caller_code: str,
        target_skill: str,
        old_signature: "SignatureInfo",
        new_signature: "SignatureInfo",
    ) -> Dict[str, List[str]]:
        """
        Analyze whether calls in the code are impacted by the signature change.

        Args:
            caller_code: Caller's code
            target_skill: Name of the called skill
            old_signature: Original signature
            new_signature: New signature

        Returns:
            Dict: {'issues': [...], 'warnings': [...]}
        """
        import re

        issues = []
        warnings = []

        # Find all calls to target_skill
        # Matches: await targetSkill(...) or targetSkill(...)
        call_pattern = rf'(?:await\s+)?{re.escape(target_skill)}\s*\(([^)]*)\)'

        for match in re.finditer(call_pattern, caller_code):
            args_str = match.group(1).strip()
            call_issues, call_warnings = self._check_call_args(
                args_str=args_str,
                old_signature=old_signature,
                new_signature=new_signature,
            )
            issues.extend(call_issues)
            warnings.extend(call_warnings)

        return {
            'issues': issues,
            'warnings': warnings,
        }

    def _check_call_args(
        self,
        args_str: str,
        old_signature: "SignatureInfo",
        new_signature: "SignatureInfo",
    ) -> Tuple[List[str], List[str]]:
        """
        Check whether the call arguments are compatible with the new signature.

        Args:
            args_str: Call-argument string
            old_signature: Original signature
            new_signature: New signature

        Returns:
            Tuple[List[str], List[str]]: (issues, warnings)
        """
        issues = []
        warnings = []

        # Parse the call arguments
        call_args = self._parse_call_args(args_str)
        num_call_args = len(call_args)

        # Number of required parameters in the new signature
        new_required_count = len(new_signature.required_params)
        old_required_count = len(old_signature.required_params)

        # Check 1: the new signature requires more required parameters
        if new_required_count > old_required_count:
            # Check whether the call provides enough arguments
            if num_call_args < new_required_count:
                issues.append(
                    f"Call provides {num_call_args} arguments but the new signature requires {new_required_count} required parameters"
                )

        # Check 2: impact of parameter-order changes
        # This is a conservative check: if the parameter order changed, calls may be affected
        old_param_names = old_signature.param_names
        new_param_names = new_signature.param_names

        # Check the first N parameters (N = the smaller of old/new required counts)
        min_required = min(old_required_count, new_required_count)
        for i in range(min_required):
            old_name = old_param_names[i] if i < len(old_param_names) else None
            new_name = new_param_names[i] if i < len(new_param_names) else None

            if old_name and new_name and old_name != new_name:
                # Parameter position changed
                if old_name in new_param_names:
                    warnings.append(
                        f"Parameter '{old_name}' position changed from {i} to {new_param_names.index(old_name)}"
                    )

        return issues, warnings

    def _parse_call_args(self, args_str: str) -> List[str]:
        """
        Parse the call-argument string.

        Args:
            args_str: Argument string

        Returns:
            List[str]: Argument list
        """
        if not args_str.strip():
            return []

        args = []
        current = ""
        depth = 0

        for char in args_str:
            if char in '({[':
                depth += 1
                current += char
            elif char in ')}]':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                if current.strip():
                    args.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            args.append(current.strip())

        return args

    def would_create_breaking_change(
        self,
        skill_name: str,
        old_signature: "SignatureInfo",
        new_signature: "SignatureInfo",
    ) -> bool:
        """
        Quick check whether the signature change would create a breaking change.

        Args:
            skill_name: Skill name
            old_signature: Original signature
            new_signature: New signature

        Returns:
            bool: Whether it would create a breaking change
        """
        result = self.analyze_impact(skill_name, old_signature, new_signature)
        return not result.is_safe

    def get_impact_summary(
        self,
        skill_name: str,
        old_signature: "SignatureInfo",
        new_signature: "SignatureInfo",
    ) -> str:
        """
        Get a summary text of the impact analysis.

        Args:
            skill_name: Skill name
            old_signature: Original signature
            new_signature: New signature

        Returns:
            str: Impact summary
        """
        result = self.analyze_impact(skill_name, old_signature, new_signature)

        if result.is_safe:
            return f"Signature change is safe; all {result.total_callers} caller(s) impacted with no breaking changes"

        summary_parts = [
            f"Signature change impact analysis:",
            f"  Total callers: {result.total_callers}",
            f"  Affected: {result.affected_callers}",
            f"  Breaking: {len(result.breaking_callers)}",
        ]

        for impact in result.breaking_callers[:3]:  # Show only the first 3
            summary_parts.append(f"    - {impact.caller_name}: {impact.issues[0] if impact.issues else 'unknown'}")

        if len(result.breaking_callers) > 3:
            summary_parts.append(f"    ... and {len(result.breaking_callers) - 3} more")

        return "\n".join(summary_parts)
