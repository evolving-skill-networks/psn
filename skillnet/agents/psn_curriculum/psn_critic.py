"""PSN-specific Critic Agent - supports semantic-aware task evaluation.

PSN Critic extends the original Critic with the following features:
1. Semantic-type awareness: distinguishes DELTA from TARGET_TOTAL semantics
2. Executed-skill tracking: takes the actually executed skills into account during evaluation
3. Planned-skill tracking: provides additional context for evaluation

P1 improvement: directly use the semantic carried by TaskWithSemantic; no longer infer it.
"""

from skillnet.agents.critic import CriticAgent
from skillnet.agents.constants.task_semantics import (
    TaskSemanticType,
    TaskWithSemantic,
)
from skillnet.utils.json_utils import fix_and_parse_json
from skillnet.utils.event_utils import unpack_event
from skillnet.utils.stats_tracker import record_llm_usage
from langchain.schema import SystemMessage, HumanMessage


class PSNCriticAgent(CriticAgent):
    """PSN-specific Critic with semantic-aware evaluation.

    Key improvements:
    - Uses the PSN-specific prompt (psn_critic.txt)
    - Adds executed-skill info into the human message
    - P1: directly uses the semantic carried by TaskWithSemantic; no longer infers it
    - supports injecting the critic prompt via DomainKnowledge
    """

    _domain_knowledge = None

    def set_domain_knowledge(self, knowledge):
        """Set domain knowledge for domain-aware prompt loading."""
        self._domain_knowledge = knowledge

    def render_system_message(self):
        """Use the PSN-specific prompt, with optional enhancements"""
        from skillnet.config.robustness_config import RobustnessConfig

        # domain-aware prompt loading
        base_prompt = ""
        if self._domain_knowledge:
            base_prompt = self._domain_knowledge.get_critic_prompt_template()
        if not base_prompt:
            base_prompt = ""  # domain-owned

        if RobustnessConfig.ENABLE_ENHANCED_CRITIC_PROMPT:
            enhanced_rules = (
                "\n\nCRITICAL RULES FOR STATE CHANGES VERIFICATION:\n"
                "- If 'State Changes (during this task)' shows 'No changes detected', "
                "it means delta = 0 for ALL items. No items were gained, lost, or transformed.\n"
                "- For DELTA semantic tasks, 'No changes detected' means the task FAILED "
                "(delta 0 < any required count).\n"
                "- For TARGET semantic tasks, 'No changes detected' does NOT automatically mean failure "
                "— check whether the FINAL inventory meets the target requirement. "
                "If the target was already satisfied before the task ran, 'No changes' is expected SUCCESS.\n"
                "- NEVER infer or assume inventory changes that are not explicitly listed "
                "in the State Changes field. Only use the data provided.\n"
            )
            enhanced_rules += (
                "- Executed Skills lists skill NAMES only. A skill's name is "
                "NOT evidence that its effect occurred; judge ONLY from the "
                "observed evidence (State Changes, Nearby Blocks Changes, "
                "Placed blocks, Dimension fields).\n"
                "- For tasks that require entering another dimension (e.g. "
                "entering a nether portal), success REQUIRES a Dimension "
                "Change line showing the new dimension. A 'Dimension "
                "(unchanged during this task)' line or missing dimension "
                "evidence means the bot did NOT enter, regardless of which "
                "skills executed.\n"
            )
            base_prompt += enhanced_rules

        return SystemMessage(content=base_prompt)

    def render_human_message(self, *, events, task: TaskWithSemantic, context, chest_observation,
                              state_changes=None, executed_skills=None,
                              planned_skills=None, plan_type=None):
        """Extension: add executed-skill info and the semantic type.

        Args:
            events: list of events
            task: TaskWithSemantic object (P1: carries the semantic info)
            context: contextual info
            chest_observation: chest observation
            state_changes: state changes (optional)
            executed_skills: list of executed skill names (optional)
            planned_skills: list of planned skill names (optional, for LLM fallback)
            plan_type: plan type, "graph" or "llm" (optional)

        Returns:
            HumanMessage, or None (on error)
        """
        # extract the task string and semantic type
        task_str = task.task
        semantic_type = task.semantic_type  # P1: directly use the passed-in semantic, no re-inference

        # Call the parent method to get the base observation
        human_msg = super().render_human_message(
            events=events,
            task=task_str,  # parent expects a string
            context=context,
            chest_observation=chest_observation,
            state_changes=state_changes,
        )

        if human_msg is None:
            return None

        content = human_msg.content

        # Insert the semantic type and executed-skill info before the Task line
        task_line = f"Task: {task_str}"
        insertion_parts = []

        # Add executed-skill info
        if executed_skills:
            skills_str = ", ".join(executed_skills)
            insertion_parts.append(f"Executed Skills: {skills_str}")
        elif planned_skills:
            # No execution record but a plan exists (LLM fallback case)
            skills_str = ", ".join(planned_skills)
            insertion_parts.append(f"Planned Skills: {skills_str}")

        # Add the plan type
        if plan_type:
            insertion_parts.append(f"Plan Type: {plan_type}")

        # Add the semantic type
        insertion_parts.append(f"Task Semantic: {semantic_type.upper()}")

        if insertion_parts:
            insertion_text = "\n\n".join(insertion_parts) + "\n\n"
            content = content.replace(task_line, insertion_text + task_line)

        print(f"\033[35m****PSN Critic Agent: Semantic={semantic_type}, "
              f"Executed={executed_skills}, Planned={planned_skills}, "
              f"PlanType={plan_type}****\033[0m")

        return HumanMessage(content=content)

    def _check_deposit_success(self, task_str, events):
        """Deterministic success criterion for the full-inventory "deposit useless
        items" task, replacing the critic's improvised <=20-occupied-slots bar
        (arbitrary and unachievable: a real inventory keeps ~24 slots of
        tools/ingots). Success iff the canonical low-value/"useless" set
        (junk blocks, obsolete tools, excess building blocks) has been removed
        from the POST-execution inventory -- i.e. classify_deposit frees 0 slots.
        Tools/ingots/valuables are kept by definition, so success no longer depends
        on a fixed slot count and is aligned with the trigger's classification.

        Returns (success, critique, quality_metrics) or None when this is not a
        deposit task or the inventory cannot be read (defer to the LLM critic).

        Note: the env exposes no chest-content channel (nearbyChests is empty), so
        we verify removal-from-inventory, not arrival-in-chest. This is sound for
        the env's deposit skills, which move items via the chest container API
        rather than tossing them on the ground.
        """
        t = (task_str or "").lower()
        if "deposit" not in t or "chest" not in t:
            return None
        dk = getattr(self, "_domain_knowledge", None)
        if dk is None:
            return None
        try:
            obs = dk.extract_observation(events)
            inv = dict(obs.inventory or {})
            equip = list(obs.extra.get("equipment", [])) if obs.extra else []
        except Exception:
            return None
        if not inv:
            return None  # no post-inventory observed — let the LLM critic decide
        try:
            from skillnet.domains.minecraft.knowledge.inventory_classification import (
                classify_deposit,
            )
            res = classify_deposit(inv, equipment=equip)
        except Exception:
            return None
        quality = {
            "robustness_score": 70,
            "dependency_safety": "safe",
            "environment_awareness": "full",
        }
        if res["freed_slots"] == 0:
            return True, "All low-value items deposited; tools/ingots/valuables kept.", quality
        remaining = ", ".join(f"{k} x{v}" for k, v in sorted(res["deposit"].items()))
        return (
            False,
            f"Still holding low-value items that should be deposited into the chest "
            f"(would free {res['freed_slots']} slots): {remaining}. "
            f"Keep tools/ingots/valuables; deposit only these.",
            quality,
        )

    def check_task_success(
        self, *, events, task: TaskWithSemantic, context, chest_observation, max_retries=5,
        state_changes=None, executed_skills=None, planned_skills=None, plan_type=None
    ):
        """Extended task-success check; P1: directly uses the passed-in semantic.

        Args:
            events: list of events
            task: TaskWithSemantic object (P1: carries the semantic info)
            context: context
            chest_observation: chest observation
            max_retries: maximum number of retries
            state_changes: state changes
            executed_skills: list of executed skills
            planned_skills: list of planned skills (for LLM fallback)
            plan_type: plan type, "graph" or "llm"

        Returns:
            (success: bool, critique: str, quality_metrics: dict) tuple
        """
        # First check for onError events
        for event in events:
            event_type, event_data = unpack_event(event)
            if event_type is None:
                continue
            if event_type == "onError":
                error_msg = event_data.get('onError', 'Unknown error') if isinstance(event_data, dict) else str(event_data)
                print(f"\033[31mPSN Critic Agent: Execution error detected - {error_msg}\033[0m")
                # on error, return default quality metrics
                return False, f"Execution Error: {error_msg}", {
                    "robustness_score": 0,  # an execution error implies the code is not robust enough
                    "dependency_safety": "unsafe",
                    "environment_awareness": "none",
                }

        # Deterministic success for the full-inventory deposit task (bypasses the
        # LLM critic's arbitrary <=20-slot bar; aligned with the deposit
        # classification used by the trigger).
        deposit_verdict = self._check_deposit_success(task.task, events)
        if deposit_verdict is not None:
            success, critique, quality_metrics = deposit_verdict
            print(f"\033[35m****PSN Critic (deterministic deposit check): "
                  f"success={success}\033[0m")
            return success, critique, quality_metrics

        # Render the message (using the extended render_human_message)
        human_message = self.render_human_message(
            events=events,
            task=task,
            context=context,
            chest_observation=chest_observation,
            state_changes=state_changes,
            executed_skills=executed_skills,
            planned_skills=planned_skills,
            plan_type=plan_type,
        )

        messages = [
            self.render_system_message(),
            human_message,
        ]

        if self.mode == "manual":
            success, critique = self.human_check_task_success()
            # manual mode returns default quality metrics
            return success, critique, {
                "robustness_score": 50,
                "dependency_safety": "warning",
                "environment_awareness": "partial",
            }
        elif self.mode == "auto":
            return self.ai_check_task_success(
                messages=messages, max_retries=max_retries
            )
        else:
            raise ValueError(f"Invalid critic agent mode: {self.mode}")

    def ai_check_task_success(self, messages, max_retries=5):
        """extends the return value to include code-quality evaluation dimensions"""
        if max_retries == 0:
            print(
                "\033[31mFailed to parse Critic Agent response. Consider updating your prompt.\033[0m"
            )
            return False, "", {}

        if messages[1] is None:
            return False, "", {}

        _llm_resp = self.llm.invoke(messages)
        record_llm_usage(_llm_resp, process_type="critic", function_name="psn_critic.ai_check_task_success")
        critic = _llm_resp.content
        print(f"\033[31m****PSN Critic Agent ai message****\n{critic}\033[0m")

        try:
            response = fix_and_parse_json(critic)
            assert response["success"] in [True, False]

            if "critique" not in response:
                response["critique"] = ""

            # extract the code-quality evaluation dimensions (with defaults to remain backward-compatible)
            quality_metrics = {
                "robustness_score": response.get("robustness_score", 50),
                "dependency_safety": response.get("dependency_safety", "warning"),
                "environment_awareness": response.get("environment_awareness", "partial"),
            }

            print(f"\033[35m****PSN Critic Quality Metrics: {quality_metrics}****\033[0m")

            return response["success"], response["critique"], quality_metrics

        except Exception as e:
            print(f"\033[31mError parsing critic response: {e} Trying again!\033[0m")
            return self.ai_check_task_success(
                messages=messages,
                max_retries=max_retries - 1,
            )
