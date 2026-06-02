"""
Gradient-related dataclasses

Storage for gradient information used in skill optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class SkillGradientItem:
    """A single gradient item with content and a usage marker."""
    content: str
    used_in_optimization: bool = False  # whether already consumed by the optimization pipeline
    optimization_id: Optional[str] = None  # associated optimization pipeline ID (if consumed)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __init__(self, content: str, used_in_optimization: bool = False,
                 optimization_id: str = None, timestamp: str = None):
        self.content = content
        self.used_in_optimization = used_in_optimization
        self.optimization_id = optimization_id
        self.timestamp = timestamp or datetime.now().isoformat()


@dataclass
class SkillGradients:
    """Gradient information for a skill (used during optimization)."""
    reflections: List[SkillGradientItem] = field(default_factory=list)  # reflections and feedback
    feedback: List[SkillGradientItem] = field(default_factory=list)  # feedback information
    optimization_suggestions: List[SkillGradientItem] = field(default_factory=list)  # optimization suggestions

    def add_feedback(self, feedback: str, optimization_id: str = None):
        """Add a feedback item."""
        self.feedback.append(SkillGradientItem(
            content=feedback,
            optimization_id=optimization_id
        ))

    def get_unused_items(self, item_type: str = "all") -> List[SkillGradientItem]:
        """Get unused gradient items."""
        items = []
        if item_type in ("all", "reflections"):
            items.extend([i for i in self.reflections if not i.used_in_optimization])
        if item_type in ("all", "feedback"):
            items.extend([i for i in self.feedback if not i.used_in_optimization])
        if item_type in ("all", "optimization_suggestions"):
            items.extend([i for i in self.optimization_suggestions if not i.used_in_optimization])
        return items

    def mark_as_used(self, items: List[SkillGradientItem], optimization_id: str):
        """Mark a gradient item as consumed."""
        for item in items:
            item.used_in_optimization = True
            item.optimization_id = optimization_id

    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            "reflections": [
                {
                    "content": item.content,
                    "used_in_optimization": item.used_in_optimization,
                    "optimization_id": item.optimization_id,
                    "timestamp": item.timestamp
                }
                for item in self.reflections
            ],
            "feedback": [
                {
                    "content": item.content,
                    "used_in_optimization": item.used_in_optimization,
                    "optimization_id": item.optimization_id,
                    "timestamp": item.timestamp
                }
                for item in self.feedback
            ],
            "optimization_suggestions": [
                {
                    "content": item.content,
                    "used_in_optimization": item.used_in_optimization,
                    "optimization_id": item.optimization_id,
                    "timestamp": item.timestamp
                }
                for item in self.optimization_suggestions
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillGradients":
        """Construct from a dict."""
        gradients = cls()
        for item_data in data.get("reflections", []):
            gradients.reflections.append(SkillGradientItem(
                content=item_data.get("content", ""),
                used_in_optimization=item_data.get("used_in_optimization", False),
                optimization_id=item_data.get("optimization_id"),
                timestamp=item_data.get("timestamp")
            ))
        for item_data in data.get("feedback", []):
            gradients.feedback.append(SkillGradientItem(
                content=item_data.get("content", ""),
                used_in_optimization=item_data.get("used_in_optimization", False),
                optimization_id=item_data.get("optimization_id"),
                timestamp=item_data.get("timestamp")
            ))
        for item_data in data.get("optimization_suggestions", []):
            gradients.optimization_suggestions.append(SkillGradientItem(
                content=item_data.get("content", ""),
                used_in_optimization=item_data.get("used_in_optimization", False),
                optimization_id=item_data.get("optimization_id"),
                timestamp=item_data.get("timestamp")
            ))
        return gradients
