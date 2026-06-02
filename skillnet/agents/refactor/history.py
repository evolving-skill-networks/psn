"""
Refactor History Tracker

Refactor history tracking module: records all refactor operations, supports auditing and analysis.

Features:
- Record detailed information for each refactor
- Persist to the file system
- Query historical records
- Statistical analysis

Usage:
    from skillnet.agents.refactor.history import RefactorHistoryTracker

    tracker = RefactorHistoryTracker(ckpt_dir="/path/to/ckpt")

    # Record a refactor
    record_id = tracker.record(result)

    # Query history
    history = tracker.get_history(skill_name="craftOakBoat")
    recent = tracker.get_recent(limit=10)

    # Statistics
    stats = tracker.get_statistics()
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
from collections import defaultdict

from .base import RefactorType, RefactorResult, RefactorOpportunity


@dataclass
class RefactorRecord:
    """
    Complete record of a single refactor

    Contains pre/post state, trigger reason, result, and related info.
    """
    record_id: str
    timestamp: str

    # Basic refactor info
    refactor_type: str                    # String value of RefactorType
    source_skill: str
    target_skill: str

    # Success/failure status
    success: bool
    error_message: Optional[str] = None

    # Code changes
    old_code: Optional[str] = None
    new_code: Optional[str] = None
    changes_made: List[str] = field(default_factory=list)

    # Context
    task: Optional[str] = None            # Task that triggered the refactor
    trigger: str = "manual"               # "manual" | "delayed" | "batch"
    session_id: Optional[str] = None      # Associated transaction session ID

    # Rollback info
    rolled_back: bool = False
    rollback_time: Optional[str] = None
    rollback_reason: Optional[str] = None

    # Related skills (used for batch refactors)
    related_skills: List[str] = field(default_factory=list)

    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: RefactorResult,
        task: Optional[str] = None,
        trigger: str = "manual",
        session_id: Optional[str] = None,
    ) -> 'RefactorRecord':
        """Create a record from a RefactorResult"""
        return cls(
            record_id=f"refactor_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now().isoformat(),
            refactor_type=result.refactor_type.value,
            source_skill=result.source_skill,
            target_skill=result.target_skill,
            success=result.success,
            error_message=result.error_message,
            old_code=result.old_code,
            new_code=result.new_code,
            changes_made=list(result.changes_made),
            task=task,
            trigger=trigger,
            session_id=session_id,
        )


@dataclass
class RefactorStatistics:
    """
    Refactor statistics

    Aggregates counts, success rates, etc. for each refactor type.
    """
    # Totals
    total_refactors: int = 0
    successful_refactors: int = 0
    failed_refactors: int = 0
    rolled_back_refactors: int = 0

    # By type
    by_type: Dict[str, int] = field(default_factory=dict)
    by_type_success: Dict[str, int] = field(default_factory=dict)
    by_type_failed: Dict[str, int] = field(default_factory=dict)

    # By trigger
    by_trigger: Dict[str, int] = field(default_factory=dict)

    # Time range
    first_refactor_time: Optional[str] = None
    last_refactor_time: Optional[str] = None

    # Affected skills
    skills_affected: int = 0
    skills_created: int = 0              # Newly created general skills

    # Success rate
    success_rate: float = 0.0

    # Most common refactor type
    most_common_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return asdict(self)


class RefactorHistoryTracker:
    """
    Refactor history tracker

    Tracks all refactor operations and provides query/statistics functionality.
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        max_memory_records: int = 1000,
        logger=None,
    ):
        """
        Initialize the history tracker

        Args:
            ckpt_dir: checkpoint directory (used for persistence)
            max_memory_records: max number of records kept in memory
            logger: logger
        """
        self.ckpt_dir = ckpt_dir
        self.max_memory_records = max_memory_records
        self.logger = logger

        # Records kept in memory
        self._records: List[RefactorRecord] = []

        # Indexes (to speed up queries)
        self._by_skill: Dict[str, List[str]] = defaultdict(list)  # skill -> record_ids
        self._by_type: Dict[str, List[str]] = defaultdict(list)   # type -> record_ids
        self._by_session: Dict[str, List[str]] = defaultdict(list)  # session -> record_ids

        # Persistence directory
        self.history_dir = None
        if ckpt_dir:
            self.history_dir = os.path.join(ckpt_dir, "refactor_history")
            os.makedirs(self.history_dir, exist_ok=True)

        # Load existing history
        self._load_history()

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)

    def record(
        self,
        result: RefactorResult,
        task: Optional[str] = None,
        trigger: str = "manual",
        session_id: Optional[str] = None,
        related_skills: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a single refactor operation

        Args:
            result: refactor result
            task: task that triggered the refactor
            trigger: trigger method ("manual", "delayed", "batch")
            session_id: associated transaction session ID
            related_skills: related skills (used for batch refactors)
            metadata: extra metadata

        Returns:
            str: record ID
        """
        record = RefactorRecord.from_result(
            result=result,
            task=task,
            trigger=trigger,
            session_id=session_id,
        )

        if related_skills:
            record.related_skills = related_skills
        if metadata:
            record.metadata = metadata

        # Add to memory
        self._records.append(record)

        # Update indexes
        self._by_skill[record.source_skill].append(record.record_id)
        if record.target_skill:
            self._by_skill[record.target_skill].append(record.record_id)
        self._by_type[record.refactor_type].append(record.record_id)
        if record.session_id:
            self._by_session[record.session_id].append(record.record_id)

        # Memory limit
        if len(self._records) > self.max_memory_records:
            self._evict_old_records()

        # Persist
        self._save_record(record)

        self._log(
            f"[RefactorHistory] Recorded refactor {record.record_id}: "
            f"{record.refactor_type} {record.source_skill} -> {record.target_skill} "
            f"({'success' if record.success else 'failure'})",
            "info"
        )

        return record.record_id

    def record_rollback(
        self,
        record_id: str,
        reason: str = "manual",
    ) -> bool:
        """
        Record a rollback operation

        Args:
            record_id: ID of the record being rolled back
            reason: rollback reason

        Returns:
            bool: success
        """
        record = self.get_record(record_id)
        if not record:
            return False

        record.rolled_back = True
        record.rollback_time = datetime.now().isoformat()
        record.rollback_reason = reason

        # Update persistence
        self._save_record(record)

        self._log(
            f"[RefactorHistory] Recorded rollback {record_id}: {reason}",
            "info"
        )

        return True

    def get_record(self, record_id: str) -> Optional[RefactorRecord]:
        """Get a single record"""
        for record in self._records:
            if record.record_id == record_id:
                return record

        # Try to load from file
        return self._load_record(record_id)

    def get_history(
        self,
        skill_name: Optional[str] = None,
        refactor_type: Optional[RefactorType] = None,
        session_id: Optional[str] = None,
        success_only: bool = False,
        include_rolled_back: bool = True,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[RefactorRecord]:
        """
        Query refactor history

        Args:
            skill_name: filter by skill name
            refactor_type: filter by refactor type
            session_id: filter by session ID
            success_only: return only successful records
            include_rolled_back: whether to include rolled-back records
            start_time: start time
            end_time: end time
            limit: max number to return

        Returns:
            List[RefactorRecord]: matching records
        """
        results = []

        for record in reversed(self._records):  # Newest first
            # Apply filters
            if skill_name and skill_name not in [record.source_skill, record.target_skill]:
                continue
            if refactor_type and record.refactor_type != refactor_type.value:
                continue
            if session_id and record.session_id != session_id:
                continue
            if success_only and not record.success:
                continue
            if not include_rolled_back and record.rolled_back:
                continue
            if start_time or end_time:
                try:
                    record_time = datetime.fromisoformat(record.timestamp)
                    if start_time and record_time < start_time:
                        continue
                    if end_time and record_time > end_time:
                        continue
                except (ValueError, TypeError):
                    # Invalid timestamp format; skip this record
                    self._log(
                        f"[RefactorHistory] Invalid timestamp: {record.timestamp}",
                        "warning"
                    )
                    continue

            results.append(record)

            if len(results) >= limit:
                break

        return results

    def get_recent(self, limit: int = 10) -> List[RefactorRecord]:
        """Get the most recent refactor records"""
        return list(reversed(self._records[-limit:]))

    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> RefactorStatistics:
        """
        Get refactor statistics

        Args:
            start_time: statistics start time
            end_time: statistics end time

        Returns:
            RefactorStatistics: statistics
        """
        stats = RefactorStatistics()
        affected_skills: Set[str] = set()
        created_skills: Set[str] = set()

        for record in self._records:
            # Time filter
            if start_time or end_time:
                try:
                    record_time = datetime.fromisoformat(record.timestamp)
                    if start_time and record_time < start_time:
                        continue
                    if end_time and record_time > end_time:
                        continue
                except (ValueError, TypeError):
                    # Invalid timestamp format; skip this record
                    continue

            stats.total_refactors += 1

            # Success/failure stats
            if record.success:
                stats.successful_refactors += 1
                stats.by_type_success[record.refactor_type] = \
                    stats.by_type_success.get(record.refactor_type, 0) + 1
            else:
                stats.failed_refactors += 1
                stats.by_type_failed[record.refactor_type] = \
                    stats.by_type_failed.get(record.refactor_type, 0) + 1

            if record.rolled_back:
                stats.rolled_back_refactors += 1

            # By type
            stats.by_type[record.refactor_type] = \
                stats.by_type.get(record.refactor_type, 0) + 1

            # By trigger
            stats.by_trigger[record.trigger] = \
                stats.by_trigger.get(record.trigger, 0) + 1

            # Time range
            if not stats.first_refactor_time or record.timestamp < stats.first_refactor_time:
                stats.first_refactor_time = record.timestamp
            if not stats.last_refactor_time or record.timestamp > stats.last_refactor_time:
                stats.last_refactor_time = record.timestamp

            # Affected skills
            affected_skills.add(record.source_skill)
            if record.target_skill:
                affected_skills.add(record.target_skill)
                # Check whether this is a newly created skill
                # Includes both new and legacy names for backward compatibility
                if record.refactor_type in [
                    'duplication',
                    'sibling', 'merge_siblings',  # New/old names
                    'extract_common_subskill', 'extract_common',  # New/old names
                ]:
                    created_skills.add(record.target_skill)

        # Compute summary values
        stats.skills_affected = len(affected_skills)
        stats.skills_created = len(created_skills)

        if stats.total_refactors > 0:
            stats.success_rate = stats.successful_refactors / stats.total_refactors

        if stats.by_type:
            stats.most_common_type = max(stats.by_type, key=stats.by_type.get)

        return stats

    def clear(self):
        """Clear historical records"""
        self._records.clear()
        self._by_skill.clear()
        self._by_type.clear()
        self._by_session.clear()

    # ==================== Persistence ====================

    def _save_record(self, record: RefactorRecord):
        """Save a single record to a file"""
        if not self.history_dir:
            return

        try:
            # Group into date directories
            date_str = record.timestamp[:10]  # YYYY-MM-DD
            date_dir = os.path.join(self.history_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)

            file_path = os.path.join(date_dir, f"{record.record_id}.json")
            with open(file_path, 'w') as f:
                json.dump(asdict(record), f, indent=2)

        except (IOError, OSError, TypeError) as e:
            self._log(f"[RefactorHistory] Failed to save record: {e}", "warning")

    def _load_record(self, record_id: str) -> Optional[RefactorRecord]:
        """Load a single record from a file"""
        if not self.history_dir:
            return None

        try:
            # Search all date directories
            for date_dir in os.listdir(self.history_dir):
                dir_path = os.path.join(self.history_dir, date_dir)
                # Skip non-directory entries (such as ordinary files)
                if not os.path.isdir(dir_path):
                    continue
                file_path = os.path.join(dir_path, f"{record_id}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        return RefactorRecord(**data)
        except (IOError, OSError, json.JSONDecodeError, TypeError) as e:
            self._log(f"[RefactorHistory] Failed to load record: {e}", "warning")

        return None

    def _load_history(self):
        """Load history records into memory"""
        if not self.history_dir or not os.path.exists(self.history_dir):
            return

        try:
            # Load by date in descending order
            date_dirs = sorted(os.listdir(self.history_dir), reverse=True)

            for date_dir in date_dirs:
                dir_path = os.path.join(self.history_dir, date_dir)
                if not os.path.isdir(dir_path):
                    continue

                for file_name in os.listdir(dir_path):
                    if not file_name.endswith('.json'):
                        continue

                    file_path = os.path.join(dir_path, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Filter out fields not recognized by RefactorRecord for backward compatibility
                            valid_fields = {
                                'record_id', 'timestamp', 'refactor_type', 'source_skill',
                                'target_skill', 'success', 'error_message', 'old_code',
                                'new_code', 'changes_made', 'task', 'trigger', 'session_id',
                                'rolled_back', 'rollback_time', 'rollback_reason',
                                'related_skills', 'metadata'
                            }
                            filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                            # Ensure required fields exist (use defaults)
                            if 'record_id' not in filtered_data or 'timestamp' not in filtered_data:
                                self._log(f"[RefactorHistory] Skipping invalid record: {file_name}", "warning")
                                continue
                            record = RefactorRecord(**filtered_data)
                            self._records.append(record)

                            # Update indexes
                            self._by_skill[record.source_skill].append(record.record_id)
                            if record.target_skill:
                                self._by_skill[record.target_skill].append(record.record_id)
                            self._by_type[record.refactor_type].append(record.record_id)
                            if record.session_id:
                                self._by_session[record.session_id].append(record.record_id)

                    except Exception as e:
                        self._log(f"[RefactorHistory] Failed to load file {file_path}: {e}", "warning")

                    # Memory limit
                    if len(self._records) >= self.max_memory_records:
                        break

                if len(self._records) >= self.max_memory_records:
                    break

            # Sort by time
            self._records.sort(key=lambda r: r.timestamp)

            self._log(
                f"[RefactorHistory] Loaded {len(self._records)} historical records",
                "info"
            )

        except Exception as e:
            self._log(f"[RefactorHistory] Failed to load history: {e}", "error")

    def _evict_old_records(self):
        """Evict old records to free memory"""
        if len(self._records) <= self.max_memory_records:
            return

        # Keep the most recent 80%
        keep_count = int(self.max_memory_records * 0.8)
        evicted = self._records[:-keep_count]
        self._records = self._records[-keep_count:]

        # Update indexes (simplified: rebuild the index)
        self._by_skill.clear()
        self._by_type.clear()
        self._by_session.clear()

        for record in self._records:
            self._by_skill[record.source_skill].append(record.record_id)
            if record.target_skill:
                self._by_skill[record.target_skill].append(record.record_id)
            self._by_type[record.refactor_type].append(record.record_id)
            if record.session_id:
                self._by_session[record.session_id].append(record.record_id)

        self._log(
            f"[RefactorHistory] Evicted {len(evicted)} old records",
            "info"
        )
