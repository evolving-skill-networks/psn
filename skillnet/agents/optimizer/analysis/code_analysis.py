"""
Code Analysis Module

Static code-analysis module:
- Try-catch detection: identifies try-catch patterns that may swallow errors.
- Infrastructure-layer error detection: identifies infrastructure-related errors.
- Code-edit metrics functions.

These analyses inform backpropagation decisions, helping determine whether
feedback should propagate to child skills.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TryCatchPattern(Enum):
    """Try-catch issue pattern."""
    EMPTY_CATCH = "empty_catch"           # Empty catch block
    LOG_ONLY = "log_only"                 # Only logs the error without handling
    LOG_AND_RETURN = "log_and_return"     # Logs the error then returns
    UNUSED_ERROR = "unused_error"         # Captured error not used


class IssueSeverity(Enum):
    """Issue severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TryCatchIssue:
    """A try-catch issue."""
    line: int
    pattern: TryCatchPattern
    description: str
    severity: IssueSeverity
    called_functions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict form."""
        return {
            "line": self.line,
            "pattern": self.pattern.value,
            "description": self.description,
            "severity": self.severity.value,
            "called_functions": self.called_functions,
        }


class CodeAnalyzer:
    """
    Static code analyzer.

    Provides:
    1. Try-catch pattern detection.
    2. Infrastructure-error detection.

    Usage:
        analyzer = CodeAnalyzer()

        # Detect try-catch issues
        issues = analyzer.detect_silent_error_swallowing(code)
    """

    # Infrastructure-related error patterns
    INFRASTRUCTURE_PATTERNS = {
        "pathfinder": [
            r"No path to",
            r"pathfind.*failed",
            r"cannot reach",
            r"Goal is not reachable",
        ],
        "networking": [
            r"ECONNREFUSED",
            r"ETIMEDOUT",
            r"connection.*closed",
            r"socket.*error",
        ],
        "timeout": [
            r"timed? ?out",
            r"timeout",
            r"blockUpdate.*expired",
        ],
        "world_loading": [
            r"chunk.*not.*loaded",
            r"world.*not.*ready",
            r"entity.*not.*found",  # May be a loading issue
        ],
    }

    def __init__(self, logger=None):
        """Initialize CodeAnalyzer."""
        self.logger = logger

    def _log(self, message: str, level: str = "info"):
        """Emit a log line."""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)

    # ==================== Try-catch detection ====================

    def detect_silent_error_swallowing(self, code: str) -> List[TryCatchIssue]:
        """
        Detect try-catch patterns in the code that may swallow errors.

        Patterns detected:
        1. catch block only prints the error without rethrowing.
        2. catch block returns directly without any error indication.
        3. Empty catch block.

        Args:
            code: JavaScript code string

        Returns:
            List of TryCatchIssue.
        """
        issues = []

        # Use regex to match try-catch blocks.
        # Note: this is a simplified pattern that cannot handle complex nesting.
        try_catch_pattern = re.compile(
            r'try\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*catch\s*\(\s*(\w+)\s*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            re.DOTALL
        )

        for match in try_catch_pattern.finditer(code):
            try_block = match.group(1)
            catch_var = match.group(2)
            catch_block = match.group(3).strip()

            # Compute the line number for the catch block
            code_before_catch = code[:match.start(3)]
            catch_line = code_before_catch.count('\n') + 1

            issue = None

            # 1. Empty catch block
            if not catch_block or catch_block.isspace():
                issue = TryCatchIssue(
                    line=catch_line,
                    pattern=TryCatchPattern.EMPTY_CATCH,
                    description="empty catch block completely swallows the error",
                    severity=IssueSeverity.HIGH,
                )

            # 2. catch block only has console.log / bot.chat and no throw
            elif (('console.log' in catch_block or 'bot.chat' in catch_block) and
                  'throw' not in catch_block and
                  'reject' not in catch_block):

                has_return = 'return' in catch_block

                # Check functions called in the try block (possibly child skills)
                called_funcs = re.findall(r'await\s+(\w+)\s*\(', try_block)

                issue = TryCatchIssue(
                    line=catch_line,
                    pattern=TryCatchPattern.LOG_AND_RETURN if has_return else TryCatchPattern.LOG_ONLY,
                    description=f"catch block only logs the error{' and returns' if has_return else ''}, does not rethrow",
                    severity=IssueSeverity.MEDIUM if has_return else IssueSeverity.LOW,
                    called_functions=called_funcs,
                )

            # 3. Captured error but not used
            elif 'return' in catch_block and catch_var not in catch_block:
                issue = TryCatchIssue(
                    line=catch_line,
                    pattern=TryCatchPattern.UNUSED_ERROR,
                    description=f"captured error '{catch_var}' is unused; returning directly",
                    severity=IssueSeverity.MEDIUM,
                )

            if issue:
                issues.append(issue)

        return issues

    # ==================== Infrastructure-layer error detection ====================

    def detect_infrastructure_error(
        self,
        error_message: str,
        execution_traces: List[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detect whether this is an infrastructure-layer error.

        Infrastructure errors should not be addressed by optimizing skill code.

        Args:
            error_message: error message
            execution_traces: execution traces (optional)

        Returns:
            If it is an infrastructure error, return a dict with type and description; otherwise None.
        """
        error_lower = error_message.lower()

        for infra_type, patterns in self.INFRASTRUCTURE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return {
                        "type": infra_type,
                        "pattern": pattern,
                        "description": f"detected infrastructure error of type {infra_type}",
                        "should_skip_optimization": True,
                        "suggestion": self._get_infrastructure_suggestion(infra_type),
                    }

        return None

    def _get_infrastructure_suggestion(self, infra_type: str) -> str:
        """Return a handling suggestion for an infrastructure error."""
        suggestions = {
            "pathfinder": "check whether the target location is reachable, or increase the pathfinding timeout",
            "networking": "check the network connection; you may need to reconnect to the server",
            "timeout": "increase the operation timeout, or check the server response",
            "world_loading": "wait for chunks to finish loading, then retry",
        }
        return suggestions.get(infra_type, "check infrastructure status")

    def is_infrastructure_error(self, error_message: str) -> bool:
        """
        Quick check for an infrastructure error.

        Args:
            error_message: error message

        Returns:
            Whether this is an infrastructure error.
        """
        return self.detect_infrastructure_error(error_message) is not None

    # ==================== Null risk detection ====================

    def analyze_null_risks(self, code: str) -> List[Dict[str, Any]]:
        """
        Statically analyze code to find points that may cause null-pointer errors.

        Patterns detected:
        1. bot.blockAt() return value used directly without a null check.
        2. bot.findBlock() return value used directly.
        3. Object property chains without null guards.
        4. mineflayer API calls (e.g. activateBlock, dig) using args that may be null.

        Args:
            code: JavaScript code string

        Returns:
            List of risk points, each containing line, pattern, code_snippet, risk_level.
        """
        risks = []
        lines = code.split('\n')

        # Define risk patterns
        risk_patterns = [
            # bot.blockAt return value may be null
            {
                "pattern": r'(bot\.blockAt\([^)]+\))\.(\w+)',
                "description": "bot.blockAt() return value accessed directly; may be null",
                "risk_level": "high",
            },
            # bot.findBlock return value may be null
            {
                "pattern": r'(bot\.findBlock\([^)]+\))\.(\w+)',
                "description": "bot.findBlock() return value accessed directly; may be null",
                "risk_level": "high",
            },
            # bot.findBlocks results' elements may be null
            {
                "pattern": r'(foundBlocks?|blocks?)\[\s*\d+\s*\]\.(\w+)',
                "description": "array element accessed directly; element may not exist",
                "risk_level": "medium",
            },
            # APIs like activateBlock/dig used with possibly-null vars
            {
                "pattern": r'bot\.(activateBlock|dig|placeBlock)\s*\(\s*(\w+)',
                "description": "mineflayer API call argument may be null",
                "risk_level": "high",
                "check_var": True,  # Need to check whether the variable has a null check
            },
            # entity.position access may fail
            {
                "pattern": r'(\w+)\.position\.(\w+)',
                "description": "object position property chain access",
                "risk_level": "medium",
            },
        ]

        # Track variables that have been checked
        checked_vars: Set[str] = set()

        # Null-check detection patterns
        null_check_patterns = [
            r'if\s*\(\s*!?\s*(\w+)\s*\)',
            r'if\s*\(\s*(\w+)\s*[!=]==?\s*(null|undefined)',
            r'(\w+)\s*\?\.',  # Optional chaining
            r'(\w+)\s*&&\s*\1\.',
        ]

        # Pre-scan: find all variables with null checks
        for i, line in enumerate(lines):
            for pattern in null_check_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        checked_vars.add(match[0])
                    else:
                        checked_vars.add(match)

        # Detect risks
        for i, line in enumerate(lines):
            for risk_def in risk_patterns:
                matches = re.finditer(risk_def["pattern"], line)
                for match in matches:
                    # If the check requires a variable and that variable already has a null check, skip
                    if risk_def.get("check_var") and len(match.groups()) >= 2:
                        var_name = match.group(2)
                        if var_name in checked_vars:
                            continue

                    # Check whether this line already contains a null check
                    line_has_check = any(
                        re.search(
                            p.replace(r'(\w+)', re.escape(match.group(1)) if match.groups() else r'(\w+)')
                             .replace(r'\1', re.escape(match.group(1)) if match.groups() else r'\1'),
                            line
                        )
                        for p in null_check_patterns
                    )
                    if line_has_check:
                        continue

                    risks.append({
                        "line": i + 1,
                        "pattern": risk_def["description"],
                        "code_snippet": line.strip()[:80],
                        "risk_level": risk_def["risk_level"],
                        "matched_text": match.group(0),
                    })

        return risks


# =============================================================================
# Code-edit metrics functions (extracted from optimizer_impl.py)
# =============================================================================




