"""
Error classifier

Contains the error classification enum and fix-target type enum.
"""

from enum import Enum


class ErrorCategory(Enum):
    """Error classification enum

    Previously had 16 members for fine-grained classification, but all
    production code paths only use UNKNOWN.  Reduced to a single member
    to remove dead complexity.  Old serialized values (e.g. from
    checkpoint JSON) are handled by the ``_missing_`` hook which maps
    any unrecognized value to UNKNOWN.
    """
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):
        """Handle deserialization of old checkpoint values gracefully."""
        return cls.UNKNOWN


class FixTargetType(Enum):
    """Fix target type"""
    CALLER_FIX = "caller_fix"           # The caller should be fixed
    CALLEE_FIX = "callee_fix"           # The callee should be fixed
    BOTH_FIX = "both_fix"               # Both need to be fixed
