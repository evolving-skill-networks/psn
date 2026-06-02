"""
Statistics Tracker for PSN

Tracks token usage, call counts, and other statistics for various processes:
- Skill generation
- Optimization
- Refactoring
- Backpropagation
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import skillnet.utils as U


@dataclass
class LLMCallStats:
    """Statistics for a single LLM call."""
    process_type: str  # "skill_generation", "optimization", "refactor", "backprop", etc.
    function_name: str  # name of the calling function
    timestamp: str
    task: Optional[str] = None
    skill_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_name: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(
        self,
        process_type: str,
        function_name: str,
        timestamp: str = None,
        task: str = None,
        skill_name: str = None,
        input_tokens: int = None,
        output_tokens: int = None,
        total_tokens: int = None,
        model_name: str = None,
        success: bool = True,
        error_message: str = None,
        metadata: Dict[str, Any] = None,
    ):
        self.process_type = process_type
        self.function_name = function_name
        self.timestamp = timestamp or datetime.now().isoformat()
        self.task = task
        self.skill_name = skill_name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.model_name = model_name
        self.success = success
        self.error_message = error_message
        self.metadata = metadata or {}


class StatsTracker:
    """Statistics tracker."""
    
    def __init__(self, ckpt_dir: str = "ckpt"):
        self.ckpt_dir = ckpt_dir
        self.stats_dir = f"{ckpt_dir}/stats"
        U.f_mkdir(self.stats_dir)
        
        # Store statistics
        self.llm_calls: List[LLMCallStats] = []
        
        # Group statistics by process type
        self.process_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "tasks": set(),
            "skills": set(),
        })
    
    def record_llm_call(
        self,
        process_type: str,
        function_name: str,
        task: str = None,
        skill_name: str = None,
        input_tokens: int = None,
        output_tokens: int = None,
        total_tokens: int = None,
        model_name: str = None,
        success: bool = True,
        error_message: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Record one LLM call.
        
        Args:
            process_type: process type ("skill_generation", "optimization", "refactor", "backprop", etc.)
            function_name: name of the calling function
            task: associated task
            skill_name: associated skill name
            input_tokens: number of input tokens
            output_tokens: number of output tokens
            total_tokens: total number of tokens
            model_name: model name
            success: whether successful
            error_message: error message (if failed)
            metadata: other metadata
        """
        stats = LLMCallStats(
            process_type=process_type,
            function_name=function_name,
            task=task,
            skill_name=skill_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_name=model_name,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )
        
        self.llm_calls.append(stats)
        
        # Update process statistics
        process_stat = self.process_stats[process_type]
        process_stat["total_calls"] += 1
        if success:
            process_stat["successful_calls"] += 1
        else:
            process_stat["failed_calls"] += 1
        
        if input_tokens:
            process_stat["total_input_tokens"] += input_tokens
        if output_tokens:
            process_stat["total_output_tokens"] += output_tokens
        if total_tokens:
            process_stat["total_tokens"] += total_tokens
        elif input_tokens and output_tokens:
            process_stat["total_tokens"] += (input_tokens + output_tokens)
        
        if task:
            process_stat["tasks"].add(task)
        if skill_name:
            process_stat["skills"].add(skill_name)
    
    def extract_token_usage(self, response) -> Dict[str, Optional[int]]:
        """
        Extract token-usage information from an LLM response.
        
        Args:
            response: LLM response object (typically a langchain response)
        
        Returns:
            Dict with input_tokens, output_tokens, total_tokens
        """
        result = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
        
        def _extract_field(obj, key, fallback_key=None):
            """Extract a field from dict or object, with optional fallback key."""
            if isinstance(obj, dict):
                val = obj.get(key)
                if val is None and fallback_key:
                    val = obj.get(fallback_key)
                return val
            val = getattr(obj, key, None)
            if val is None and fallback_key:
                val = getattr(obj, fallback_key, None)
            return val

        # Path 1: response.response_metadata["token_usage"] or ["usage"]
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if metadata and isinstance(metadata, dict):
                usage = metadata.get("token_usage") or metadata.get("usage")
                if usage and isinstance(usage, dict):
                    result["input_tokens"] = usage.get("prompt_tokens")
                    result["output_tokens"] = usage.get("completion_tokens")
                    result["total_tokens"] = usage.get("total_tokens")

        # Path 2: response.usage_metadata (dict in 0.1.x, object in 0.2+)
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            if usage:
                result["input_tokens"] = _extract_field(usage, "input_tokens", "prompt_tokens")
                result["output_tokens"] = _extract_field(usage, "output_tokens", "completion_tokens")
                result["total_tokens"] = _extract_field(usage, "total_tokens")
        
        return result
    
    def save_stats(self):
        """Save statistics to file."""
        # Convert sets to lists (for JSON serialization)
        process_stats_serializable = {}
        for process_type, stats in self.process_stats.items():
            process_stats_serializable[process_type] = {
                "total_calls": stats["total_calls"],
                "successful_calls": stats["successful_calls"],
                "failed_calls": stats["failed_calls"],
                "total_input_tokens": stats["total_input_tokens"],
                "total_output_tokens": stats["total_output_tokens"],
                "total_tokens": stats["total_tokens"],
                "tasks": list(stats["tasks"]),
                "tasks_count": len(stats["tasks"]),
                "skills": list(stats["skills"]),
                "skills_count": len(stats["skills"]),
            }
        
        # Save detailed call records
        calls_file = f"{self.stats_dir}/llm_calls.json"
        calls_data = [asdict(call) for call in self.llm_calls]
        U.dump_json(calls_data, calls_file)
        
        # Save aggregate statistics
        summary_file = f"{self.stats_dir}/summary.json"
        summary_data = {
            "total_llm_calls": len(self.llm_calls),
            "process_stats": process_stats_serializable,
            "last_updated": datetime.now().isoformat(),
        }
        U.dump_json(summary_data, summary_file)
        
        # Save detailed statistics per process type
        for process_type, stats in self.process_stats.items():
            process_file = f"{self.stats_dir}/{process_type}_stats.json"
            process_calls = [
                asdict(call) for call in self.llm_calls
                if call.process_type == process_type
            ]
            process_data = {
                "process_type": process_type,
                "summary": process_stats_serializable[process_type],
                "calls": process_calls,
                "last_updated": datetime.now().isoformat(),
            }
            U.dump_json(process_data, process_file)
    
    def load_stats(self):
        """Load statistics from file."""
        calls_file = f"{self.stats_dir}/llm_calls.json"
        if os.path.exists(calls_file):
            calls_data = U.load_json(calls_file)
            self.llm_calls = [
                LLMCallStats(**call_data) for call_data in calls_data
            ]
            
            # Rebuild process statistics
            self.process_stats.clear()
            for call in self.llm_calls:
                process_stat = self.process_stats[call.process_type]
                process_stat["total_calls"] += 1
                if call.success:
                    process_stat["successful_calls"] += 1
                else:
                    process_stat["failed_calls"] += 1
                
                if call.input_tokens:
                    process_stat["total_input_tokens"] += call.input_tokens
                if call.output_tokens:
                    process_stat["total_output_tokens"] += call.output_tokens
                if call.total_tokens:
                    process_stat["total_tokens"] += call.total_tokens
                
                if call.task:
                    process_stat["tasks"].add(call.task)
                if call.skill_name:
                    process_stat["skills"].add(call.skill_name)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get the statistics summary."""
        process_stats_serializable = {}
        for process_type, stats in self.process_stats.items():
            process_stats_serializable[process_type] = {
                "total_calls": stats["total_calls"],
                "successful_calls": stats["successful_calls"],
                "failed_calls": stats["failed_calls"],
                "total_input_tokens": stats["total_input_tokens"],
                "total_output_tokens": stats["total_output_tokens"],
                "total_tokens": stats["total_tokens"],
                "tasks_count": len(stats["tasks"]),
                "skills_count": len(stats["skills"]),
            }
        
        return {
            "total_llm_calls": len(self.llm_calls),
            "process_stats": process_stats_serializable,
        }
    
    def print_summary(self):
        """Print the statistics summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("Statistics Summary")
        print("="*60)
        print(f"Total LLM Calls: {summary['total_llm_calls']}")
        print("\nProcess Statistics:")
        for process_type, stats in summary["process_stats"].items():
            print(f"\n  {process_type.upper()}:")
            print(f"    Total Calls: {stats['total_calls']}")
            print(f"    Successful: {stats['successful_calls']}")
            print(f"    Failed: {stats['failed_calls']}")
            print(f"    Total Tokens: {stats['total_tokens']}")
            print(f"    Input Tokens: {stats['total_input_tokens']}")
            print(f"    Output Tokens: {stats['total_output_tokens']}")
            print(f"    Tasks: {stats['tasks_count']}")
            print(f"    Skills: {stats['skills_count']}")
        print("="*60 + "\n")


# ============================================================================
# Global active-tracker registry + module-level recording helper.
#
# Motivation (token-breakdown instrumentation): most LLM call sites across
# planner / REFLECT / refactor / curriculum / critic / evolution invoke the LLM
# RAW (`llm.invoke(...)`) and never reach a StatsTracker, so their tokens are
# absent from summary.json. Threading a tracker reference through every
# subsystem's constructor would be invasive; instead the SkillGraphManager
# registers its tracker here once (set_active_tracker), and any call site can
# record into it via record_llm_usage() with a one-line, no-import-plumbing call.
#
# Invariants:
#   * No double counting: the already-tracked wrappers
#     (SkillGraphManager._invoke_llm_with_stats, LLMInvoker.invoke) do NOT call
#     this helper — only the previously-untracked raw sites do.
#   * Instrumentation must never break a run: record_llm_usage swallows all
#     exceptions and is a no-op when no tracker is registered.
# ============================================================================

_ACTIVE_TRACKER: Optional["StatsTracker"] = None


def set_active_tracker(tracker: "StatsTracker") -> None:
    """Register the run's StatsTracker as the process-global active tracker."""
    global _ACTIVE_TRACKER
    _ACTIVE_TRACKER = tracker


def get_active_tracker() -> Optional["StatsTracker"]:
    """Return the currently-registered active StatsTracker (or None)."""
    return _ACTIVE_TRACKER


def record_llm_usage(
    response,
    process_type: str,
    function_name: str,
    *,
    task: str = None,
    skill_name: str = None,
    model_name: str = None,
    success: bool = True,
    error_message: str = None,
    metadata: Dict[str, Any] = None,
) -> None:
    """Record one (previously-untracked, raw) LLM call into the active tracker.

    No-op if no tracker is registered. Never raises — instrumentation must not
    affect run behavior. `response` is the object returned by `llm.invoke(...)`;
    its token usage is extracted via StatsTracker.extract_token_usage.
    """
    tracker = _ACTIVE_TRACKER
    if tracker is None:
        return
    try:
        if success and response is not None:
            usage = tracker.extract_token_usage(response)
        else:
            usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        tracker.record_llm_call(
            process_type=process_type,
            function_name=function_name,
            task=task,
            skill_name=skill_name,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            model_name=model_name,
            success=success,
            error_message=error_message,
            metadata=metadata,
        )
    except Exception:
        # Instrumentation is best-effort; never propagate.
        pass

