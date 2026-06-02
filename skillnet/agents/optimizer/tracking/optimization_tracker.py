"""
Optimization Tracker

Optimization-history tracker — records and analyzes skill optimization history.

extracted from optimizer_impl.py
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

from typing import Any as _Any
from skillnet.agents.optimizer.analysis.error_classifier import (
    ErrorCategory,
    FixTargetType,
)
from skillnet.agents.optimizer.feedback.types import OptimizationRecord
from .loop_manager import LoopManager


class OptimizationTracker:
    """
    Optimization-history tracker

    Features:
    1. Record each skill's optimization history
    2. Detect optimization loops (same error pattern recurring)
    3. Suggest alternative optimization strategies
    4. Distinguish caller/callee errors
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.history: Dict[str, List[OptimizationRecord]] = {}
        self.optimization_history_file = f"{ckpt_dir}/skill_graph/optimizer/optimization_history.json"
        # separated storage — detailed-log directory
        self.detailed_logs_dir = f"{ckpt_dir}/skill_graph/optimizer/detailed_logs"
        self._load_history()
        # P1: delegate to LoopManager
        self._loop_manager = LoopManager(history=self.history)

    def _load_history(self):
        """Load optimization history"""
        if os.path.exists(self.optimization_history_file):
            try:
                with open(self.optimization_history_file, 'r') as f:
                    data = json.load(f)
                    for skill_name, records in data.items():
                        self.history[skill_name] = []
                        for r in records:
                            self.history[skill_name].append(OptimizationRecord(
                                skill_name=r["skill_name"],
                                timestamp=r["timestamp"],
                                error_category=ErrorCategory(r["error_category"]),
                                error_pattern=r["error_pattern"],
                                strategy_used=r["strategy_used"],
                                fix_target=FixTargetType(r["fix_target"]),
                                successful=r["successful"],
                                version_before=r.get("version_before"),
                                version_after=r.get("version_after"),
                                # New fields
                                task=r.get("task"),
                                error_message=r.get("error_message"),
                                error_stack=r.get("error_stack"),
                                feedback_content=r.get("feedback_content"),
                                code_before=r.get("code_before"),
                                code_after=r.get("code_after"),
                                code_diff=r.get("code_diff"),
                                execution_context=r.get("execution_context"),
                                called_skills=r.get("called_skills"),
                                llm_prompt=r.get("llm_prompt"),
                                llm_response=r.get("llm_response"),
                                consistency_check_passed=r.get("consistency_check_passed"),
                                consistency_conflicts=r.get("consistency_conflicts"),
                                consistency_score=r.get("consistency_score"),
                                failure_reason=r.get("failure_reason"),
                                value_at_optimization=r.get("value_at_optimization"),
                                # separated-storage fields
                                optimization_id=r.get("optimization_id"),
                                detailed_log_path=r.get("detailed_log_path"),
                            ))
            except Exception as e:
                print(f"[OptimizationTracker] Failed to load optimization history: {e}")

    def _save_history(self):
        """
        Save optimization history (P1: separated storage — summaries in the main file)

        The summary file contains:
        - Basic info (ID, timestamp, skill name, etc.)
        - Validation result summary
        - Path to the detailed log file

        Detailed contents (code, LLM prompt/response) are stored in separate files.
        """
        try:
            os.makedirs(os.path.dirname(self.optimization_history_file), exist_ok=True)
            data = {}
            for skill_name, records in self.history.items():
                data[skill_name] = []
                for r in records:
                    # only save summary info to the main file
                    record_data = {
                        "skill_name": r.skill_name,
                        "timestamp": r.timestamp,
                        "error_category": r.error_category.value,
                        "error_pattern": r.error_pattern,
                        "strategy_used": r.strategy_used,
                        "fix_target": r.fix_target.value,
                        "successful": r.successful,
                        "version_before": r.version_before,
                        "version_after": r.version_after,
                    }

                    # separated-storage fields
                    if hasattr(r, 'optimization_id') and r.optimization_id:
                        record_data["optimization_id"] = r.optimization_id
                    if hasattr(r, 'detailed_log_path') and r.detailed_log_path:
                        record_data["detailed_log_path"] = r.detailed_log_path

                    # Add summary fields (no large text)
                    if r.task:
                        record_data["task"] = r.task
                    if r.error_message:
                        # Only save an error-message summary
                        record_data["error_message"] = r.error_message[:500] if len(r.error_message) > 500 else r.error_message
                    if r.called_skills:
                        record_data["called_skills"] = r.called_skills

                    # validation-result summary (no detailed content)
                    if r.consistency_check_passed is not None:
                        record_data["consistency_check_passed"] = r.consistency_check_passed
                    if r.consistency_score is not None:
                        record_data["consistency_score"] = r.consistency_score
                    if r.failure_reason:
                        record_data["failure_reason"] = r.failure_reason
                    if r.value_at_optimization is not None:
                        record_data["value_at_optimization"] = r.value_at_optimization

                    # line-count summary (no full code)
                    if r.code_before:
                        record_data["code_lines_before"] = len(r.code_before.split('\n'))
                    if r.code_after:
                        record_data["code_lines_after"] = len(r.code_after.split('\n'))

                    data[skill_name].append(record_data)
            with open(self.optimization_history_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[OptimizationTracker] Failed to save optimization history: {e}")

    def _save_detailed_log(self, record: 'OptimizationRecord', optimization_id: str) -> Optional[str]:
        """
        P1: save the detailed log to a separate file

        Args:
            record: optimization record
            optimization_id: unique optimization ID

        Returns:
            Relative path to the detailed log (relative to detailed_logs_dir)
        """
        try:
            # Organize by date
            date_str = record.timestamp[:10]  # YYYY-MM-DD
            daily_dir = os.path.join(self.detailed_logs_dir, date_str)
            os.makedirs(daily_dir, exist_ok=True)

            # Detailed-log contents
            detailed_data = {
                "optimization_id": optimization_id,
                "skill_name": record.skill_name,
                "timestamp": record.timestamp,
                "task": record.task,
                "error_message": record.error_message,
                "error_stack": record.error_stack,
                "feedback_content": record.feedback_content,
                "code_before": record.code_before,
                "code_after": record.code_after,
                "code_diff": record.code_diff,
                "execution_context": record.execution_context,
                "called_skills": record.called_skills,
                "llm_prompt": record.llm_prompt,  # Full prompt
                "llm_response": record.llm_response,  # Full response
                "consistency_check_passed": record.consistency_check_passed,
                "consistency_conflicts": record.consistency_conflicts,
                "consistency_score": record.consistency_score,
                "failure_reason": record.failure_reason,
                "value_at_optimization": record.value_at_optimization,
            }

            # Save the detailed log
            log_filename = f"opt_{optimization_id}.json"
            log_path = os.path.join(daily_dir, log_filename)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)

            # Return the relative path
            return f"{date_str}/{log_filename}"
        except Exception as e:
            print(f"[OptimizationTracker] Failed to save detailed log: {e}")
            return None

    def record_optimization(
        self,
        skill_name: str,
        error_category: ErrorCategory,
        error_pattern: str,
        strategy_used: str,
        fix_target: FixTargetType,
        successful: bool,
        root_cause_analysis: Optional[_Any] = None,
        version_before: Optional[int] = None,
        version_after: Optional[int] = None,
        # New parameters
        task: Optional[str] = None,
        error_message: Optional[str] = None,
        error_stack: Optional[str] = None,
        feedback_content: Optional[str] = None,
        code_before: Optional[str] = None,
        code_after: Optional[str] = None,
        code_diff: Optional[str] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        called_skills: Optional[List[str]] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[str] = None,
        consistency_check_passed: Optional[bool] = None,
        consistency_conflicts: Optional[List[str]] = None,
        consistency_score: Optional[float] = None,
        failure_reason: Optional[str] = None,
        value_at_optimization: Optional[float] = None,
    ):
        """
        Record one optimization

        Save full optimization context for reproducing issues:
        - Trigger info (task, error, feedback)
        - Code changes (before, after, diff)
        - Execution context
        - LLM interaction
        - Consistency check result
        """
        if skill_name not in self.history:
            self.history[skill_name] = []

        # generate a unique optimization ID
        optimization_id = uuid.uuid4().hex[:8]

        record = OptimizationRecord(
            skill_name=skill_name,
            timestamp=datetime.now().isoformat(),
            error_category=error_category,
            error_pattern=error_pattern,
            strategy_used=strategy_used,
            fix_target=fix_target,
            successful=successful,
            root_cause_analysis=root_cause_analysis,
            version_before=version_before,
            version_after=version_after,
            # New fields
            task=task,
            error_message=error_message,
            error_stack=error_stack,
            feedback_content=feedback_content,
            code_before=code_before,
            code_after=code_after,
            code_diff=code_diff,
            execution_context=execution_context,
            called_skills=called_skills,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            consistency_check_passed=consistency_check_passed,
            consistency_conflicts=consistency_conflicts,
            consistency_score=consistency_score,
            failure_reason=failure_reason,
            value_at_optimization=value_at_optimization,
        )

        # save the detailed log to a separate file
        detailed_log_path = self._save_detailed_log(record, optimization_id)
        record.detailed_log_path = detailed_log_path  # Store detailed-log path
        record.optimization_id = optimization_id  # Store the optimization ID

        self.history[skill_name].append(record)
        self._save_history()

