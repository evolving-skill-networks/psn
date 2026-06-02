"""
Mixin modules for SkillGraphManager

Module splitting for better maintainability.
MetadataManagementMixin split into 3 mixins.
SkillExecutionMixin split into 3 mixins.
SkillDeletionAndCleanupMixin, SkillRetrievalAndDiscoveryMixin, SkillCodeUpdateMixin.
"""

from .refactor_mgmt import RefactorManagementMixin
from .skill_versioning import SkillVersioningMixin
from .metadata_mgmt import MetadataManagementMixin
from .metadata_crud import MetadataCrudMixin
from .metadata_validation import MetadataValidationMixin
from .skill_execution import SkillExecutionMixin
from .execution_lifecycle import ExecutionLifecycleMixin
from .semantics_update import SemanticsUpdateMixin
from .graph_queries import GraphQueriesMixin
from .skill_deletion import SkillDeletionAndCleanupMixin
from .skill_retrieval import SkillRetrievalAndDiscoveryMixin
from .skill_code_update import SkillCodeUpdateMixin

__all__ = [
    "RefactorManagementMixin",
    "SkillVersioningMixin",
    "MetadataManagementMixin",
    "MetadataCrudMixin",
    "MetadataValidationMixin",
    "SkillExecutionMixin",
    "ExecutionLifecycleMixin",
    "SemanticsUpdateMixin",
    "GraphQueriesMixin",
    "SkillDeletionAndCleanupMixin",
    "SkillRetrievalAndDiscoveryMixin",
    "SkillCodeUpdateMixin",
]
