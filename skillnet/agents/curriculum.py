"""
Base class for PSNCurriculumAgent. Use PSNCurriculumAgent
(psn_curriculum/agent.py) instead of instantiating this directly.
"""

from __future__ import annotations

import random
import re

import skillnet.utils as U
from skillnet.utils.json_utils import fix_and_parse_json
from skillnet.utils.llm_factory import create_chat_llm
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_chroma.vectorstores import Chroma

class CurriculumAgent:
    def __init__(
        self,
        model_name="gpt-5-mini",
        temperature=0,
        qa_model_name="gpt-5-mini",
        qa_temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        mode="auto",
        warm_up=None,
        core_inventory_items: str | None = None,
        openai_api_base=None,
        openai_api_key=None,
        qa_openai_api_base=None,
        qa_openai_api_key=None,
    ):
        self.llm = create_chat_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            component="curriculum_agent",
        )
        self.qa_llm = create_chat_llm(
            model_name=qa_model_name,
            temperature=qa_temperature,
            request_timeout=request_timout,
            openai_api_base=qa_openai_api_base,
            openai_api_key=qa_openai_api_key,
            component="curriculum_agent_qa",
        )
        assert mode in [
            "auto",
            "manual",
        ], f"mode {mode} not supported"
        self.mode = mode
        self._domain_knowledge = None
        self.ckpt_dir = ckpt_dir
        U.f_mkdir(f"{ckpt_dir}/curriculum/vectordb")
        if resume:
            print(f"\033[35mLoading Curriculum Agent from {ckpt_dir}/curriculum\033[0m")
            self.completed_tasks = U.load_json(
                f"{ckpt_dir}/curriculum/completed_tasks.json"
            )
            self.failed_tasks = U.load_json(f"{ckpt_dir}/curriculum/failed_tasks.json")
            self.qa_cache = U.load_json(f"{ckpt_dir}/curriculum/qa_cache.json")
        else:
            self.completed_tasks = []
            self.failed_tasks = []
            self.qa_cache = {}
        # vectordb for qa cache
        self.qa_cache_questions_vectordb = Chroma(
            collection_name="qa_cache_questions_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/curriculum/vectordb",
        )
        assert self.qa_cache_questions_vectordb._collection.count() == len(
            self.qa_cache
        ), (
            f"Curriculum Agent's qa cache question vectordb is not synced with qa_cache.json.\n"
            f"There are {self.qa_cache_questions_vectordb._collection.count()} questions in vectordb "
            f"but {len(self.qa_cache)} questions in qa_cache.json.\n"
            f"Did you set resume=False when initializing the agent?\n"
            f"You may need to manually delete the qa cache question vectordb directory for running from scratch.\n"
        )
        # if warm up not defined, initialize it as a dict, else, initialize all the missing value as a default value
        if not warm_up:
            warm_up = self.default_warmup
        self.warm_up = {}
        if "optional_inventory_items" in warm_up:
            assert core_inventory_items is not None
            self._core_inv_items_regex = re.compile(core_inventory_items)
            self.warm_up["optional_inventory_items"] = warm_up[
                "optional_inventory_items"
            ]
        else:
            self.warm_up["optional_inventory_items"] = 0
        for key in self.curriculum_observations:
            self.warm_up[key] = warm_up.get(key, self.default_warmup[key])
        self.warm_up["nearby_blocks"] = 0
        self.warm_up["inventory"] = 0
        self.warm_up["completed_tasks"] = 0
        self.warm_up["failed_tasks"] = 0

    @property
    def default_warmup(self):
        return {
            "context": 15,
            "biome": 10,
            "time": 15,
            "nearby_blocks": 0,
            "other_blocks": 10,
            "nearby_entities": 5,
            "health": 15,
            "hunger": 15,
            "position": 0,
            "equipment": 0,
            "inventory": 0,
            "optional_inventory_items": 7,
            "chests": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
        }

    @property
    def curriculum_observations(self):
        return [
            "context",
            "biome",
            "time",
            "nearby_blocks",
            "other_blocks",
            "nearby_entities",
            "health",
            "hunger",
            "position",
            "equipment",
            "inventory",
            "chests",
            "completed_tasks",
            "failed_tasks",
        ]

    @property
    def progress(self):
        return len(self.completed_tasks)

    def render_system_message(self):
        dk = self._domain_knowledge
        prompt = dk.get_prompt("curriculum") if dk else ""
        system_message = SystemMessage(content=prompt)
        assert isinstance(system_message, SystemMessage)
        return system_message

    def render_observation(self, *, events, chest_observation):
        """Delegate to the active DomainKnowledge.

        The legacy 88-line Minecraft-specific body lived here; it's now in
        MinecraftKnowledge.render_observation (E.1 commit). Each domain owns
        the formatting of its own event shape; this method just supplies the
        curriculum state (completed/failed tasks, progress, warm-up gates).
        """
        if self._domain_knowledge is None:
            raise RuntimeError(
                "CurriculumAgent has no domain_knowledge; "
                "call set_domain_knowledge(...) before render_observation."
            )
        warm_up_optional = (
            self.warm_up.get("optional_inventory_items", 0)
            if hasattr(self, "warm_up") and isinstance(self.warm_up, dict)
            else 0
        )
        return self._domain_knowledge.render_observation(
            events=events,
            chest_observation=chest_observation,
            completed_tasks=list(self.completed_tasks) if hasattr(self, "completed_tasks") else None,
            failed_tasks=list(self.failed_tasks) if hasattr(self, "failed_tasks") else None,
            progress=self.progress if hasattr(self, "progress") else 0,
            warm_up_optional_inventory=warm_up_optional,
        )

    def render_human_message(self, *, events, chest_observation):
        content = ""
        observation = self.render_observation(
            events=events, chest_observation=chest_observation
        )
        if self.progress >= self.warm_up["context"]:
            questions, answers = self.run_qa(
                events=events, chest_observation=chest_observation
            )
            i = 1
            for question, answer in zip(questions, answers):
                if "Answer: Unknown" in answer or "language model" in answer:
                    continue
                observation["context"] += f"Question {i}: {question}\n"
                observation["context"] += f"{answer}\n\n"
                i += 1
                if i > 5:
                    break

        for key in self.curriculum_observations:
            if self.progress >= self.warm_up[key]:
                if self.warm_up[key] != 0:
                    should_include = random.random() < 0.8
                else:
                    should_include = True
                if should_include:
                    content += observation[key]

        print(f"\033[35m****Curriculum Agent human message****\n{content}\033[0m")
        return HumanMessage(content=content)

    def propose_next_task(self, *, events, chest_observation, max_retries=5):
        from skillnet.core.dk_registry import get_domain_knowledge
        _dk = get_domain_knowledge()

        # First task — delegate to domain knowledge
        if self.progress == 0 and self.mode == "auto":
            first_task = _dk.get_initial_task() if _dk else ""
            if first_task:
                return first_task, ""

        # Read normalized observation once for the gating checks below.
        # extract_observation tolerates empty/malformed event streams.
        obs = _dk.extract_observation(events) if _dk else None

        # Emergency task (e.g., underground without pickaxe)
        try:
            if obs is not None:
                position = obs.position
                current_y = (
                    position[1] if position and len(position) >= 2 else 100
                )
                inventory = dict(obs.inventory)
            else:
                current_y = 100
                inventory = {}

            if _dk:
                emergency = _dk.get_emergency_task(current_y, inventory, [])
                if emergency:
                    return emergency, ""
        except Exception as e:
            print(f"\033[33m[Curriculum] Environment check warning: {e}\033[0m")

        # Full inventory — delegate to domain knowledge
        inventoryUsed = (
            obs.extra.get("inventoryUsed", 0) if obs is not None else 0
        )
        inv_cfg = _dk.get_inventory_config() if _dk else {}
        full_threshold = inv_cfg.get("full_threshold", 33)
        if inventoryUsed >= full_threshold:
            if _dk:
                inv = dict(obs.inventory) if obs is not None else {}
                task = _dk.get_full_inventory_task(inv, chest_observation)
                if task:
                    return task, ""
            return "Free up inventory space", ""

        messages = [
            self.render_system_message(),
            self.render_human_message(
                events=events, chest_observation=chest_observation
            ),
        ]

        if self.mode == "auto":
            return self.propose_next_ai_task(messages=messages, max_retries=max_retries)
        elif self.mode == "manual":
            return self.propose_next_manual_task()
        else:
            raise ValueError(f"Invalid curriculum agent mode: {self.mode}")

    def propose_next_ai_task(self, *, messages, max_retries=5):
        if max_retries == 0:
            raise RuntimeError("Max retries reached, failed to propose ai task.")
        _llm_resp = self.llm.invoke(messages)
        from skillnet.utils.stats_tracker import record_llm_usage
        record_llm_usage(_llm_resp, process_type="curriculum_legacy", function_name="legacy.curriculum")
        curriculum = _llm_resp.content
        print(f"\033[31m****Curriculum Agent ai message****\n{curriculum}\033[0m")
        try:
            response = self.parse_ai_message(curriculum)
            assert "next_task" in response
            context = self.get_task_context(response["next_task"])
            return response["next_task"], context
        except Exception as e:
            print(
                f"\033[35mError parsing curriculum response: {e}. Trying again!\033[0m"
            )
            return self.propose_next_ai_task(
                messages=messages,
                max_retries=max_retries - 1,
            )

    def parse_ai_message(self, message):
        task = ""
        for line in message.split("\n"):
            stripped = line.strip()
            # Remove markdown bold wrappers first: **Task:** → Task:
            if "**" in stripped:
                stripped = stripped.replace("**", "")
            # Then strip leading markdown list/quote/header markers (-, *, >, #)
            stripped = stripped.lstrip("-*>#").strip()
            if stripped.lower().startswith("task:"):
                task = stripped[5:].strip().rstrip(".").strip()
        assert task, "Task not found in Curriculum Agent response"
        return {"next_task": task}

    def propose_next_manual_task(self):
        confirmed = False
        task, context = "", ""
        while not confirmed:
            task = input("Enter task: ")
            context = input("Enter context: ")
            print(f"Task: {task}\nContext: {context}")
            confirmed = input("Confirm? (y/n)").lower() in ["y", ""]
        return task, context

    def update_exploration_progress(self, info):
        task = info["task"]
        # Preflight (contract-gate) rejects are generation/interface errors and should not be treated as "too hard".
        if info.get("error_type") == "preflight":
            print(f"\033[35m[Curriculum] Preflight rejected code for task {task}. Not counting as failed task.\033[0m")
            return
        # API/infrastructure errors are transient — don't record as task failures
        if info.get("error_type") == "api_error":
            print(f"\033[35m[Curriculum] API error for task {task}. Not counting as failed task.\033[0m")
            return
        if task.startswith("Deposit useless items into the chest at"):
            # No need to record the deposit task
            return
        if info["success"]:
            print(f"\033[35mCompleted task {task}.\033[0m")
            self.completed_tasks.append(task)
        else:
            print(
                f"\033[35mFailed to complete task {task}. Skipping to next task.\033[0m"
            )
            self.failed_tasks.append(task)

        # clean up tasks and dump to disk
        self.clean_up_tasks()

    def clean_up_tasks(self):
        updated_completed_tasks = []
        # record repeated failed tasks
        updated_failed_tasks = self.failed_tasks
        # dedup but keep order
        for task in self.completed_tasks:
            if task not in updated_completed_tasks:
                updated_completed_tasks.append(task)

        # remove completed tasks from failed tasks
        for task in updated_completed_tasks:
            while task in updated_failed_tasks:
                updated_failed_tasks.remove(task)

        self.completed_tasks = updated_completed_tasks
        self.failed_tasks = updated_failed_tasks

        # dump to json
        U.dump_json(
            self.completed_tasks, f"{self.ckpt_dir}/curriculum/completed_tasks.json"
        )
        U.dump_json(self.failed_tasks, f"{self.ckpt_dir}/curriculum/failed_tasks.json")

    def decompose_task(self, task, events):
        dk = self._domain_knowledge
        messages = [
            SystemMessage(
                content=dk.get_prompt("curriculum_task_decomposition") if dk else "",
            ),
            self.render_human_message(events=events, chest_observation=""),
            HumanMessage(content=f"Final task: {task}"),
        ]
        print(
            f"\033[31m****Curriculum Agent task decomposition****\nFinal task: {task}\033[0m"
        )
        response = self.llm(messages).content
        print(f"\033[31m****Curriculum Agent task decomposition****\n{response}\033[0m")
        return fix_and_parse_json(response)

    def run_qa(self, *, events, chest_observation):
        questions_new, _ = self.run_qa_step1_ask_questions(
            events=events, chest_observation=chest_observation
        )
        questions = []
        answers = []
        for question in questions_new:
            if self.qa_cache_questions_vectordb._collection.count() > 0:
                docs_and_scores = (
                    self.qa_cache_questions_vectordb.similarity_search_with_score(
                        question, k=1
                    )
                )
                if docs_and_scores and docs_and_scores[0][1] < 0.05:
                    question_cached = docs_and_scores[0][0].page_content
                    assert question_cached in self.qa_cache
                    answer_cached = self.qa_cache[question_cached]
                    questions.append(question_cached)
                    answers.append(answer_cached)
                    continue
            answer = self.run_qa_step2_answer_questions(question=question)
            assert question not in self.qa_cache
            self.qa_cache[question] = answer
            self.qa_cache_questions_vectordb.add_texts(
                texts=[question],
            )
            U.dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json")
            questions.append(question)
            answers.append(answer)
        assert len(questions_new) == len(questions) == len(answers)
        return questions, answers

    def get_task_context(self, task):
        # if include ore in question, gpt will try to use tool with skill touch enhancement to mine
        question = (
            f"How to {task.replace('_', ' ').replace(' ore', '').replace(' ores', '').replace('.', '').strip().lower()}"
            f" in Minecraft?"
        )
        if question in self.qa_cache:
            answer = self.qa_cache[question]
        else:
            answer = self.run_qa_step2_answer_questions(question=question)
            self.qa_cache[question] = answer
            self.qa_cache_questions_vectordb.add_texts(
                texts=[question],
            )
            U.dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json")
        context = f"Question: {question}\n{answer}"
        return context

    def render_system_message_qa_step1_ask_questions(self):
        dk = self._domain_knowledge
        prompt = dk.get_prompt("curriculum_qa_step1_ask_questions") if dk else ""
        return SystemMessage(content=prompt)

    def render_human_message_qa_step1_ask_questions(self, *, events, chest_observation):
        observation = self.render_observation(
            events=events, chest_observation=chest_observation
        )
        content = ""
        for key in self.curriculum_observations:
            content += observation[key]
        return HumanMessage(content=content)

    def run_qa_step1_ask_questions(self, *, events, chest_observation):
        # Biome is a Minecraft-specific extra field; default to "unknown" for
        # other domains so the QA step still produces a generic question set.
        obs = (
            self._domain_knowledge.extract_observation(events)
            if self._domain_knowledge is not None
            else None
        )
        biome_raw = (obs.extra.get("biome") if obs is not None else None) or "unknown"
        biome = str(biome_raw).replace("_", " ")
        questions = [
            f"What are the blocks that I can find in the {biome} in Minecraft?",
            f"What are the items that I can find in the {biome} in Minecraft?",
            f"What are the mobs that I can find in the {biome} in Minecraft?",
        ]
        concepts = [biome, biome, biome]
        messages = [
            self.render_system_message_qa_step1_ask_questions(),
            self.render_human_message_qa_step1_ask_questions(
                events=events, chest_observation=chest_observation
            ),
        ]
        _qa_resp = self.qa_llm.invoke(messages)
        from skillnet.utils.stats_tracker import record_llm_usage
        record_llm_usage(_qa_resp, process_type="curriculum_qa",
                         function_name="curriculum.run_qa_step1_ask_questions")
        qa_response = _qa_resp.content
        try:
            # Regex pattern to extract question and concept pairs
            pattern = r"Question \d+: (.+)\nConcept \d+: (.+)"
            # Extracting all question and concept pairs from the text
            pairs = re.findall(pattern, qa_response)
            # Storing each question and concept in separate lists
            questions_new = [pair[0] for pair in pairs]
            concepts_new = [pair[1] for pair in pairs]
            assert len(questions_new) == len(concepts_new)
            questions.extend(questions_new)
            concepts.extend(concepts_new)
        except Exception as e:
            print(
                f"\033[35mError parsing curriculum response for "
                f"QA step 1 ask questions: {e}.\033[0m"
            )
        return questions, concepts

    def render_system_message_qa_step2_answer_questions(self):
        dk = self._domain_knowledge
        return SystemMessage(
            content=dk.get_prompt("curriculum_qa_step2_answer_questions") if dk else ""
        )

    def render_human_message_qa_step2_answer_questions(self, question):
        content = f"Question: {question}"
        return HumanMessage(content=content)

    def run_qa_step2_answer_questions(self, question):
        messages = [
            self.render_system_message_qa_step2_answer_questions(),
            self.render_human_message_qa_step2_answer_questions(question=question),
        ]
        print(f"\033[35mCurriculum Agent Question: {question}\033[0m")
        _qa_resp = self.qa_llm.invoke(messages)
        from skillnet.utils.stats_tracker import record_llm_usage
        record_llm_usage(_qa_resp, process_type="curriculum_qa",
                         function_name="curriculum.run_qa_step2_answer_questions")
        qa_answer = _qa_resp.content
        print(f"\033[31mCurriculum Agent {qa_answer}\033[0m")
        return qa_answer
