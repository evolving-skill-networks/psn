"""
Skill Evolution Manager

Analyzes cross-skill failure patterns when Level 1 escalation triggers.
If a systemic issue is detected (e.g., all failures stem from underwater
environment constraints), escalates to Level 2 for Curriculum intervention.

Requires LLM for cross-skill analysis. Without LLM, returns "unable to analyze"
rather than using incomplete heuristic pattern matching.

Trigger: Level 1 escalation event from EscalationTracker
Output: EvolutionAnalysis with systemic pattern detection + proposed action
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from skillnet.utils.stats_tracker import record_llm_usage


class SystemicAction(Enum):
    """What type of systemic action is needed when a cross-skill pattern is detected."""
    NONE = "none"                          # No systemic issue — continue Level 1
    LEARN_PREREQUISITE = "learn_prerequisite"  # Need to learn a new skill first
    CHANGE_ENVIRONMENT = "change_environment"  # Current environment is unsuitable
    APPROACH_FLAW = "approach_flaw"            # Fundamental approach needs rethinking


@dataclass
class EvolutionAnalysis:
    """Result of cross-skill failure pattern analysis."""
    pattern_detected: bool = False
    common_cause: str = ""
    systemic_action: SystemicAction = SystemicAction.NONE
    proposed_capability: str = ""          # e.g., "Navigate to a plains biome"
    evidence: List[str] = field(default_factory=list)
    raw_llm_response: str = ""


# Default prompt template (domain-agnostic).
# Domains can override via dk.get_prompt("cross_skill_analysis").
_DEFAULT_CROSS_SKILL_ANALYSIS_PROMPT = """Below are recent skill execution failures from an embodied agent. \
The task "{triggering_task}" has just triggered escalation after repeated optimization failures.

Recent failures:
{failure_records}

Analyze these failures:
1. Is there a common root cause across multiple failures (not just the triggering task)?
2. If so, is this solvable by modifying individual skills, or does it require a systemic solution?
3. If systemic, what type of action is needed?

Return a JSON object with these fields:
{{
  "pattern_detected": true/false,
  "common_cause": "description of common cause (empty if no pattern)",
  "systemic_action": "none" | "learn_prerequisite" | "change_environment" | "approach_flaw",
  "proposed_capability": "description of what's needed (empty if no pattern)"
}}

Return ONLY the JSON object, no other text."""


class SkillEvolutionManager:
    """Cross-skill failure pattern analyzer using LLM.

    When the EscalationTracker triggers Level 1 for a task, this manager
    analyzes recent failures across ALL tracked tasks to determine if
    there's a systemic issue requiring Level 2 (Curriculum intervention).

    Requires LLM — without it, analysis is skipped (returns no pattern detected).
    The LLM reads raw error messages directly and identifies patterns without
    needing predefined error pattern lists.
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    def analyze_escalation(
        self,
        current_task: str,
        escalation_tracker: "EscalationTracker",
    ) -> EvolutionAnalysis:
        """Analyze cross-skill failures when Level 1 triggers.

        Args:
            current_task: The task that triggered escalation
            escalation_tracker: Tracker with per-task failure states

        Returns:
            EvolutionAnalysis with pattern detection result
        """
        if not self._llm:
            return EvolutionAnalysis(
                pattern_detected=False,
                common_cause="No LLM available for cross-skill analysis",
            )

        # Collect failure records from all tracked tasks
        failure_records = self._collect_failure_records(
            current_task, escalation_tracker
        )

        # If only 1 task has failures, no cross-skill pattern possible
        tasks_with_failures = [
            r for r in failure_records if r["iteration_count"] >= 2
        ]
        if len(tasks_with_failures) < 2:
            return EvolutionAnalysis(
                pattern_detected=False,
                common_cause="Only one task has significant failures",
            )

        return self._analyze_with_llm(current_task, failure_records)

    def _collect_failure_records(
        self,
        current_task: str,
        tracker: "EscalationTracker",
    ) -> List[Dict]:
        """Collect structured failure records from all tracked tasks."""
        records = []
        current_normalized = tracker._normalize_task(current_task)
        for task, state in tracker.iter_task_states():
            if state.iteration_count == 0:
                continue
            records.append({
                "task": task,
                "is_triggering": task == current_normalized,
                "iteration_count": state.iteration_count,
                "level": state.level,
                "v_history": state.v_history[-5:],  # Last 5
                "errors": state.failure_errors[-3:],  # Last 3
            })
        return records

    def _analyze_with_llm(
        self,
        current_task: str,
        failure_records: List[Dict],
    ) -> EvolutionAnalysis:
        """Use LLM to analyze cross-skill failure patterns."""
        # Format failure records for LLM
        formatted = []
        for r in failure_records:
            marker = " [TRIGGERING]" if r["is_triggering"] else ""
            errors_str = "\n    ".join(
                e[:200] for e in r["errors"]
            ) if r["errors"] else "(no error messages)"
            v_str = ", ".join(f"{v:.3f}" for v in r["v_history"])
            formatted.append(
                f"- Task: {r['task']}{marker}\n"
                f"  Failures: {r['iteration_count']}, V(s) history: [{v_str}]\n"
                f"  Recent errors:\n    {errors_str}"
            )

        # Get domain-specific prompt or use default
        prompt_template = _DEFAULT_CROSS_SKILL_ANALYSIS_PROMPT
        try:
            from skillnet.core.dk_registry import get_domain_knowledge
            dk = get_domain_knowledge()
            if dk:
                domain_prompt = dk.get_prompt("cross_skill_analysis")
                if domain_prompt:
                    prompt_template = domain_prompt
        except Exception:
            pass

        prompt = prompt_template.format(
            triggering_task=current_task,
            failure_records="\n".join(formatted),
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self._llm.invoke([HumanMessage(content=prompt)])
            record_llm_usage(response, process_type="evolution_sem", function_name="evolution.manager._analyze_with_llm", task=current_task)
            response_text = response.content if hasattr(response, 'content') else str(response)

            return self._parse_llm_response(response_text, failure_records)
        except Exception as e:
            print(f"\033[33m[SEM] LLM analysis failed: {e}\033[0m")
            return EvolutionAnalysis(
                pattern_detected=False,
                common_cause=f"LLM analysis failed: {e}",
            )

    def _parse_llm_response(
        self,
        response_text: str,
        failure_records: List[Dict],
    ) -> EvolutionAnalysis:
        """Parse LLM JSON response into EvolutionAnalysis."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                return EvolutionAnalysis(raw_llm_response=response_text)

            data = json.loads(json_match.group())

            action_str = data.get("systemic_action", "none")
            try:
                action = SystemicAction(action_str)
            except ValueError:
                action = SystemicAction.NONE

            evidence = [
                f"{r['task']}: {r['errors'][-1][:100] if r['errors'] else 'no error'}"
                for r in failure_records if r["iteration_count"] >= 2
            ]

            return EvolutionAnalysis(
                pattern_detected=data.get("pattern_detected", False),
                common_cause=data.get("common_cause", ""),
                systemic_action=action,
                proposed_capability=data.get("proposed_capability", ""),
                evidence=evidence,
                raw_llm_response=response_text,
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"\033[33m[SEM] Failed to parse LLM response: {e}\033[0m")
            return EvolutionAnalysis(raw_llm_response=response_text)
