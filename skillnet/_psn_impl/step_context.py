"""StepContext dataclass: shared runtime state across step() sub-methods."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class StepContext:
    """Shared runtime state across step() sub-methods."""
    planning_result: Optional[Any] = None
    parsed_result: Optional[Any] = None   # dict | str
    ai_message: Optional[Any] = None
    events: Optional[list] = None
    code: Optional[str] = None
    all_code: Optional[str] = None
    success: bool = False
    critique: str = ""
    quality_metrics: Optional[dict] = None
    optimized: bool = False
