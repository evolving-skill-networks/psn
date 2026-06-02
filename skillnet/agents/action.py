"""
BASE CLASS (v5.3): this class is the base class of ParameterizedActionAgent.
Use ParameterizedActionAgent (parameterized_action/) instead of this class directly.
"""

import os
import re
import time

import skillnet.utils as U
from skillnet.utils.llm_factory import create_chat_llm
from skillnet.utils.event_utils import unpack_event
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage


class ActionAgent:
    def __init__(
        self,
        model_name="gpt-5-mini",
        temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
        openai_api_base=None,
        openai_api_key=None,
        max_tokens=None,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}
        self.llm = create_chat_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            component="action_agent",
            max_tokens=max_tokens,
        )
        # Disable thinking for models that don't benefit from it
        # Uses ModelProfile to decide; only applied for vLLM endpoints.
        from skillnet.core.model_profile import detect_model_profile
        profile = detect_model_profile(model_name or "")
        is_vllm = bool(
            openai_api_base
            or os.getenv("ACTION_AGENT_API_BASE")
            or os.getenv("VLLM_API_BASE")
        )
        if is_vllm and not profile.enable_thinking:
            try:
                self.llm = self.llm.bind(extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                })
            except Exception:
                pass
        # Store model name for error reporting
        self.model_name = model_name
        # Domain knowledge (set via set_domain_knowledge for from_domain path)
        self._domain_knowledge = None
        # Skill language (set via set_skill_language for from_domain path)
        self._skill_language = None

    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    print(
                        f"\033[32mAction Agent removing chest {position}: {chest}\033[0m"
                    )
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    print(f"\033[32mAction Agent saving chest {position}: {chest}\033[0m")
                    self.chest_memory[position] = chest
        U.dump_json(self.chest_memory, f"{self.ckpt_dir}/action/chest_memory.json")

    def render_chest_observation(self):
        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: Empty")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: Unknown items inside")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            return f"Chests:\n{chests}\n\n"
        else:
            return f"Chests: None\n\n"

    def set_domain_knowledge(self, knowledge):
        """Inject domain-specific knowledge for prompt/primitive overrides."""
        self._domain_knowledge = knowledge
        if knowledge is not None:
            try:
                self._skill_language = knowledge.get_skill_language_impl()
            except Exception as e:
                # DKs that don't implement get_skill_language_impl yet still load,
                # but agent code that needs it will fall back to the dk_registry.
                print(f"[{type(self).__name__}] Could not cache skill language: {e}")
                self._skill_language = None
        else:
            self._skill_language = None

    def _render_spatial_neighborhood(self, spatial):
        """Format the spatial observer's per-cell block.name dump.

        Groups cells into AIR positions (foot/head marked) and NON-AIR
        positions, preserving raw (x, y, z): name lines without semantic
        labels like 'above_head' or 'foot+x'. The LLM derives placement
        candidates from this truthful spatial data instead of being given
        a curated 10-cell candidate set.
        """
        if not spatial:
            return ""
        # locate bot foot via min(x, y, z) of an offset=0 cell — the spatial
        # observer always centers on bot foot, so the cell with the median y
        # at the median x/z is the foot. Cheaper: take the first cell as a
        # reference and derive foot from observed positions.
        try:
            xs = sorted({c["x"] for c in spatial})
            ys = sorted({c["y"] for c in spatial})
            zs = sorted({c["z"] for c in spatial})
            fx = xs[len(xs) // 2]
            fy = ys[len(ys) // 2]
            fz = zs[len(zs) // 2]
        except Exception:
            return ""

        air_lines = []
        nonair_lines = []
        for c in spatial:
            x, y, z, name = c.get("x"), c.get("y"), c.get("z"), c.get("name", "?")
            if name == "air":
                tag = ""
                if (x, y, z) == (fx, fy, fz):
                    tag = " [bot foot]"
                elif (x, y, z) == (fx, fy + 1, fz):
                    tag = " [bot head]"
                air_lines.append(f"  ({x}, {y}, {z}): air{tag}")
            else:
                nonair_lines.append(f"  ({x}, {y}, {z}): {name}")

        header = (
            f"Spatial neighborhood (raw block.name for every position within "
            f"\u00b13 of bot foot ({fx}, {fy}, {fz})):"
        )
        return (
            header + "\n"
            + f"AIR positions ({len(air_lines)}):\n"
            + "\n".join(air_lines) + "\n"
            + f"NON-AIR positions ({len(nonair_lines)}):\n"
            + "\n".join(nonair_lines) + "\n"
        )

    def set_skill_language(self, language):
        """Set the skill language implementation for code operations."""
        self._skill_language = language

    def render_system_message(self, skills=[]):
        if self._domain_knowledge:
            system_template = self._domain_knowledge.get_system_prompt_template()
            programs = "\n\n".join(
                self._domain_knowledge.load_control_primitive_code() + skills
            )
        else:
            # No domain knowledge — emit minimal prompt without primitive context.
            system_template = ""
            programs = "\n\n".join(skills)
        # Domain-aware response format
        response_format = ""
        if self._domain_knowledge:
            response_format = self._domain_knowledge.get_action_response_format()
        if not response_format:
            response_format = ""  # domain-owned
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            programs=programs, response_format=response_format
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    def render_human_message(
        self, *, events, code="", task="", context="", critique=""
    ):
        chat_messages = []
        error_messages = []
        # Minecraft event tuples end with ("observe", payload); other domains
        # (dict-event domains) use dict events with no "observe" terminator. Only assert
        # the Minecraft shape when that's the active domain.
        is_minecraft_domain = (
            self._domain_knowledge is not None
            and self._domain_knowledge.get_skill_language() == "javascript"
        )
        if is_minecraft_domain:
            last_ev = events[-1]
            # Mineflayer events arrive as tuples from the JSPyBridge, but
            # after a JSON round-trip (recording, replay, pickling) they
            # become lists. Accept both shapes — both have a 2-element
            # (name, payload) structure that the rest of the loop reads
            # via unpack_event().
            assert (
                isinstance(last_ev, (tuple, list))
                and len(last_ev) >= 1
                and last_ev[0] == "observe"
            ), "Last event must be observe"
        # Minecraft-only locals; populated below if/when an "observe" event
        # is encountered. Non-Minecraft domains render the observation block
        # via dk.render_observation(...) and never touch these.
        status = {}
        spatial = []
        entities = {}
        for i, ev in enumerate(events):
            event_type, event = unpack_event(ev)
            if event_type is None:
                continue
            if event_type == "onChat":
                chat_messages.append(event["onChat"])
            elif event_type == "onError":
                error_messages.append(event["onError"])
            elif event_type == "observe":
                # The Minecraft "observe" payload nests a "status" dict;
                # other domains (e.g. a dict-event domain) don't emit observe events
                # in this shape. Gate the defensive Minecraft-shape backfill
                # on the active domain so non-Minecraft domains don't
                # accidentally mutate their payloads.
                if is_minecraft_domain:
                    # Check key fields and warn if missing (Minecraft-shape
                    # observe payloads carry "status" / "inventory" keys).
                    if "status" not in event:
                        print(f"\033[33m[ActionAgent] WARNING: 'status' field missing in observe event at index {i}\033[0m")
                        print(f"\033[33m[ActionAgent] Available keys: {list(event.keys())}\033[0m")
                        event.setdefault("status", {})
                    if "inventory" not in event:
                        print(f"\033[33m[ActionAgent] WARNING: 'inventory' field missing in observe event at index {i}\033[0m")
                        print(f"\033[33m[ActionAgent] This may indicate an issue with the mineflayer observation system\033[0m")
                        print(f"\033[33m[ActionAgent] Available keys: {list(event.keys())}\033[0m")

                    # Minecraft-only state extraction — non-Minecraft domains
                    # never reach this elif (their event types are "state",
                    # "achievement", etc.), and even if they did, the
                    # spatial-neighborhood / voxel rendering below is
                    # Minecraft-specific and wouldn't apply.
                    status = event.get("status", {})
                    spatial = event.get("spatial", [])
                    entities = status.get("entities", {})
                    assert i == len(events) - 1, "observe must be the last event"

        observation = ""

        if code:
            observation += f"Code from the last round:\n{code}\n\n"
        else:
            observation += f"Code from the last round: No code in the first round\n\n"

        if self.execution_error:
            if error_messages:
                error = "\n".join(error_messages)
                observation += f"Execution error:\n{error}\n\n"
            else:
                observation += f"Execution error: No error\n\n"

        if self.chat_log:
            if chat_messages:
                chat_log = "\n".join(chat_messages)
                observation += f"Chat log: {chat_log}\n\n"
            else:
                observation += f"Chat log: None\n\n"

        dk = getattr(self, '_domain_knowledge', None)
        if is_minecraft_domain:
            # Legacy Minecraft rendering — preserved verbatim so the prompt
            # remains byte-identical. Reads the per-section fragments from
            # dk.render_observation(...) (which already knows about voxels
            # /biome/etc.) and inserts the Minecraft-only spatial dump and
            # entities section that render_observation doesn't cover.
            sections = dk.render_observation(events=events, chest_observation="") if dk else {}
            env_context = dk.get_environment_context(status) if dk else ""
            if env_context:
                observation += env_context.lstrip("\n") + "\n\n"
            else:
                observation += sections.get("biome", "")
            observation += sections.get("time", "")
            observation += sections.get("nearby_blocks", "")

            # Phase 13.B-5: raw spatial neighborhood (every position within ±3 of
            # bot foot, with its block.name). Replaces the V3_full10 candidate-set
            # cheat with truthful per-cell data; LLM derives placement candidates
            # itself. See scripts/PHASE_11_13_PLACEMENT_SUMMARY.md.
            if spatial:
                observation += self._render_spatial_neighborhood(spatial) + "\n"

            # Action prompt has historically used the "nearest to farthest"
            # phrasing for entities — preserve it verbatim (render_observation
            # uses a shorter "Nearby entities:" form for curriculum/critic).
            if entities:
                nearby_entities = [
                    k for k, v in sorted(entities.items(), key=lambda x: x[1])
                ]
                observation += f"Nearby entities (nearest to farthest): {', '.join(nearby_entities)}\n\n"
            else:
                observation += f"Nearby entities (nearest to farthest): None\n\n"

            observation += sections.get("health", "")
            observation += sections.get("hunger", "")
            observation += sections.get("position", "")
            observation += sections.get("equipment", "")
            observation += sections.get("inventory", "")
        else:
            # Non-Minecraft (e.g., a dict-event domain): delegate the entire observation
            # block to the domain's render_observation. The dict keys differ
            # per domain (a dict-event domain has health/food/drink/energy/daylight/...),
            # so concatenate values in dict-insertion order rather than a
            # Minecraft-shaped key list.
            if dk is not None:
                sections = dk.render_observation(events=events, chest_observation="")
                for key, fragment in sections.items():
                    # Skip the in-message chest fragment — render_chest_observation()
                    # below renders the agent's persistent chest memory (different
                    # source-of-truth from anything in the event stream).
                    if key == "chests":
                        continue
                    observation += fragment

        if not (
            task == "Place and deposit useless items into a chest"
            or task.startswith("Deposit useless items into the chest at")
        ):
            observation += self.render_chest_observation()

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        if critique:
            observation += f"Critique: {critique}\n\n"
        else:
            observation += f"Critique: None\n\n"

        return HumanMessage(content=observation)

    def process_ai_message(self, message):
        assert isinstance(message, AIMessage)

        retry = 3
        error = None
        while retry > 0:
            try:
                # Resolve skill language: cache-first, registry-fallback.
                skill_lang = getattr(self, "_skill_language", None)
                if skill_lang is None:
                    from skillnet.core.dk_registry import get_domain_knowledge
                    dk = get_domain_knowledge()
                    if dk is None:
                        raise RuntimeError("No DomainKnowledge registered; cannot parse skill code")
                    skill_lang = dk.get_skill_language_impl()

                code_pattern = re.compile(r"```(?:javascript|js)(.*?)```", re.DOTALL)
                code = "\n".join(code_pattern.findall(message.content))
                parse_result = skill_lang.parse(code)
                if not parse_result.success:
                    raise ValueError(
                        f"Failed to parse skill code: {parse_result.error}"
                    )
                raw_ast = parse_result.raw_ast
                # parse_result.functions provides high-level info but we still need
                # the raw AST nodes (params objects, async flag) for downstream checks.
                functions = []
                if raw_ast is not None:
                    assert len(list(raw_ast.program.body)) > 0, "No functions found"
                    for i, node in enumerate(raw_ast.program.body):
                        if node.type != "FunctionDeclaration":
                            continue
                        # Wrap per-node access in try/except — JSPyBridge
                        # raises JavaScriptError (not AttributeError)
                        try:
                            node_type = (
                                "AsyncFunctionDeclaration"
                                if node["async"]
                                else "FunctionDeclaration"
                            )
                            node_name = node.id.name
                            if not node_name:
                                continue
                            # Prefer matching the parsed FunctionInfo by name; fall
                            # back to source-slicing if absent.
                            body_code = next(
                                (f.full_code for f in parse_result.functions if f.name == node_name),
                                None,
                            )
                            if not body_code:
                                try:
                                    body_code = code[int(node.start):int(node.end)]
                                except Exception:
                                    body_code = ""
                            functions.append(
                                {
                                    "name": node_name,
                                    "type": node_type,
                                    "body": body_code,
                                    "params": list(node["params"]),
                                }
                            )
                        except Exception:
                            continue
                else:
                    # AST-less language impls: synthesize a flat function list from
                    # parse_result.functions; params are name strings (no entry-arg
                    # name validation possible).
                    assert len(parse_result.functions) > 0, "No functions found"
                    for f in parse_result.functions:
                        functions.append({
                            "name": f.name,
                            "type": "AsyncFunctionDeclaration" if f.is_async else "FunctionDeclaration",
                            "body": f.full_code,
                            "params": list(f.params),
                        })
                # find the last function of the appropriate type for this language.
                requires_async = getattr(skill_lang, "requires_async_main", True)
                main_function = None
                for function in reversed(functions):
                    if requires_async:
                        if function["type"] == "AsyncFunctionDeclaration":
                            main_function = function
                            break
                    else:
                        main_function = function
                        break
                expected = "async function" if requires_async else "function"
                assert main_function is not None, (
                    f"No {expected} found. Your main function must be {expected}."
                )
                # Entry-arg validation: raw AST exposes param nodes with .name;
                # AST-less path exposes plain name strings.
                main_params = main_function["params"]
                if main_params and not isinstance(main_params[0], str):
                    _first_param_name = main_params[0].name
                else:
                    _first_param_name = main_params[0] if main_params else None
                assert (
                    len(main_params) == 1 and _first_param_name == "bot"
                ), f"Main function {main_function['name']} must take a single argument named 'bot'"
                program_code = "\n\n".join(function["body"] for function in functions)
                # Sync languages (Python) reject top-level `await` at compile
                # time; emit a plain function call. JS keeps await (mineflayer
                # requires it).
                if getattr(skill_lang, "requires_async_main", True):
                    exec_code = f"await {main_function['name']}(bot);"
                else:
                    exec_code = f"{main_function['name']}(bot)"
                return {
                    "program_code": program_code,
                    "program_name": main_function["name"],
                    "exec_code": exec_code,
                }
            except (ValueError, TypeError, AssertionError, KeyError) as e:
                retry -= 1
                error = e
                time.sleep(1)
        return f"Error parsing action response (before program execution): {error}"

    def summarize_chatlog(self, events):
        def filter_item(message: str):
            craft_pattern = r"I cannot make \w+ because I need: (.*)"
            craft_pattern2 = (
                r"I cannot make \w+ because there is no crafting table nearby"
            )
            mine_pattern = r"I need at least a (.*) to mine \w+!"
            if re.match(craft_pattern, message):
                return re.match(craft_pattern, message).groups()[0]
            elif re.match(craft_pattern2, message):
                return "a nearby crafting table"
            elif re.match(mine_pattern, message):
                return re.match(mine_pattern, message).groups()[0]
            else:
                return ""

        chatlog = set()
        for ev in events:
            event_type, event = unpack_event(ev)
            if event_type is None:
                continue
            if event_type == "onChat":
                item = filter_item(event["onChat"])
                if item:
                    chatlog.add(item)
        return "I also need " + ", ".join(chatlog) + "." if chatlog else ""
