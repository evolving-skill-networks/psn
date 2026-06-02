"""
Base class for PSNCriticAgent. Use PSNCriticAgent
(psn_curriculum/psn_critic.py) instead of instantiating this directly.
"""

from skillnet.utils.json_utils import fix_and_parse_json
from skillnet.utils.llm_factory import create_chat_llm
from skillnet.utils.event_utils import unpack_event
from langchain.schema import HumanMessage, SystemMessage


class CriticAgent:
    def __init__(
        self,
        model_name="gpt-5-mini",
        temperature=0,
        request_timout=120,
        mode="auto",
        openai_api_base=None,
        openai_api_key=None,
    ):
        self.llm = create_chat_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            component="critic_agent",
        )
        assert mode in ["auto", "manual"]
        self.mode = mode
        self._domain_knowledge = None

    def set_domain_knowledge(self, knowledge):
        """Inject domain-specific knowledge for environment context."""
        self._domain_knowledge = knowledge

    def render_system_message(self):
        dk = self._domain_knowledge
        prompt = dk.get_prompt("critic") if dk else ""
        return SystemMessage(content=prompt)

    def render_human_message(self, *, events, task, context, chest_observation, state_changes=None):
        # Route observation rendering through the active DomainKnowledge.
        # ``dk.render_observation`` returns a domain-formatted Dict[str, str]
        # of section fragments; this critic concatenates a subset (in a
        # fixed order) to preserve the legacy Minecraft prompt verbatim.
        # Non-Minecraft domains (e.g., a dict-event domain) simply emit different keys
        # — missing keys default to "" and degrade gracefully.
        if self._domain_knowledge is None:
            raise RuntimeError(
                "CriticAgent has no domain_knowledge; "
                "call set_domain_knowledge(...) before render_human_message."
            )
        dk = self._domain_knowledge

        for ev in events:
            event_type, event_data = unpack_event(ev)
            if event_type is None:
                continue
            if event_type == "onError":
                err_msg = (
                    event_data.get("onError")
                    if isinstance(event_data, dict)
                    else str(event_data)
                )
                print(f"\033[31mCritic Agent: Error occurs {err_msg}\033[0m")
                return None

        # Render the observation once; concatenate the resulting per-section
        # fragments. For Minecraft we keep the legacy critic order
        # (biome / time / nearby_blocks / health / hunger / position /
        # equipment / inventory) so the prompt stays byte-identical; for
        # other domains we concat every section the domain emits, in
        # dict-insertion order, so the critic has full visibility into
        # dict-event-domain-specific signals (food / drink / energy / daylight /
        # milestones / achievements_count). ``render_observation`` itself
        # does not invoke ``get_environment_context``, so we preserve that
        # Minecraft-only enrichment by overriding the "biome" section when
        # env_context returns a non-empty string.
        sections = dk.render_observation(events=events, chest_observation="")
        is_minecraft_domain = dk.get_skill_language() == "javascript"

        observation = ""

        if is_minecraft_domain:
            obs = dk.extract_observation(events)
            status = obs.extra.get("raw_observe", {}).get("status", {}) if obs.extra else {}
            env_context = dk.get_environment_context(status)
            if env_context:
                observation += env_context.lstrip("\n") + "\n\n"
            else:
                observation += sections.get("biome", "")
            observation += sections.get("time", "")
            observation += sections.get("nearby_blocks", "")
            observation += sections.get("health", "")
            observation += sections.get("hunger", "")
            observation += sections.get("position", "")
            observation += sections.get("equipment", "")
            observation += sections.get("inventory", "")
        else:
            # Non-Minecraft critic: concat every section the domain emits,
            # skipping ones we render explicitly later (chest_observation
            # is appended further down; completed/failed_tasks are not
            # surfaced to the base critic prompt).
            for key, fragment in sections.items():
                if key in ("chests", "completed_tasks", "failed_tasks"):
                    continue
                observation += fragment

        # Add state changes section (always show for clarity, even when empty)
        if state_changes is not None:
            inv_changes = state_changes.get("inventory_changes", {})
            non_zero = {k: v for k, v in inv_changes.items()
                        if isinstance(v, dict) and v.get('delta', 0) != 0}
            if non_zero:
                changes_str = ", ".join([
                    f"{item}: {info['before']} -> {info['after']} (delta: {info['delta']:+d})"
                    for item, info in non_zero.items()
                ])
                observation += f"State Changes (during this task): {changes_str}\n\n"
            else:
                observation += f"State Changes (during this task): No changes detected\n\n"

            # Show nearby_blocks changes (helps the critic judge success/failure of find/explore tasks)
            nearby_changes = state_changes.get("nearby_blocks_changes", {})
            if nearby_changes:
                parts = []
                for block in nearby_changes.get("added", []):
                    parts.append(f"+{block}")
                for block in nearby_changes.get("removed", []):
                    parts.append(f"-{block}")
                if parts:
                    observation += f"Nearby Blocks Changes: {', '.join(parts)}\n\n"

            # Show blocks placed during execution (from onSave events).
            # returnItems() may destroy placed blocks before final observation,
            # so this is the only reliable evidence for PLACE semantic evaluation.
            placed_blocks = state_changes.get("placed_blocks", [])
            if placed_blocks:
                observation += f"Placed blocks (during this task): {', '.join(placed_blocks)}\n\n"

        observation += chest_observation

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        print(f"\033[31m****Critic Agent human message****\n{observation}\033[0m")
        return HumanMessage(content=observation)

    def human_check_task_success(self):
        confirmed = False
        success = False
        critique = ""
        while not confirmed:
            success = input("Success? (y/n)")
            success = success.lower() == "y"
            critique = input("Enter your critique:")
            print(f"Success: {success}\nCritique: {critique}")
            confirmed = input("Confirm? (y/n)") in ["y", ""]
        return success, critique

    def ai_check_task_success(self, messages, max_retries=5):
        if max_retries == 0:
            print(
                "\033[31mFailed to parse Critic Agent response. Consider updating your prompt.\033[0m"
            )
            return False, ""

        if messages[1] is None:
            return False, ""

        _llm_resp = self.llm.invoke(messages)
        from skillnet.utils.stats_tracker import record_llm_usage
        record_llm_usage(_llm_resp, process_type="critic_legacy", function_name="legacy.critic")
        critic = _llm_resp.content
        print(f"\033[31m****Critic Agent ai message****\n{critic}\033[0m")
        try:
            response = fix_and_parse_json(critic)
            assert response["success"] in [True, False]
            if "critique" not in response:
                response["critique"] = ""
            return response["success"], response["critique"]
        except Exception as e:
            print(f"\033[31mError parsing critic response: {e} Trying again!\033[0m")
            return self.ai_check_task_success(
                messages=messages,
                max_retries=max_retries - 1,
            )

    def check_task_success(
        self, *, events, task, context, chest_observation, max_retries=5,
        state_changes=None, executed_skills=None, planned_skills=None, plan_type=None
    ):
        # Note: executed_skills, planned_skills, plan_type are accepted for
        # compatibility with PSNCriticAgent but not used in the base CriticAgent
        # First check for onError events - if present, return error as critique directly
        # This ensures execution errors are properly propagated to the optimizer
        for event in events:
            event_type, event_data = unpack_event(event)
            if event_type is None:
                continue
            if event_type == "onError":
                error_msg = event_data.get('onError', 'Unknown error') if isinstance(event_data, dict) else str(event_data)
                print(f"\033[31mCritic Agent: Execution error detected - {error_msg}\033[0m")
                return False, f"Execution Error: {error_msg}"
        
        human_message = self.render_human_message(
            events=events,
            task=task,
            context=context,
            chest_observation=chest_observation,
            state_changes=state_changes,
        )

        messages = [
            self.render_system_message(),
            human_message,
        ]

        if self.mode == "manual":
            return self.human_check_task_success()
        elif self.mode == "auto":
            return self.ai_check_task_success(
                messages=messages, max_retries=max_retries
            )
        else:
            raise ValueError(f"Invalid critic agent mode: {self.mode}")
