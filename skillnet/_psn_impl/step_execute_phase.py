"""
StepExecutePhaseMixin: Phases 4-5 of step() pipeline.

Phase 4: Build code, preflight validate, execute env.step, critic evaluation.
Phase 5: Graph planner diagnostics + trajectory update + skill recording.
"""

import re

from skillnet._psn_impl.event_helpers import (
    find_last_observe,
    get_event_data_dict,
    iter_events,
)
from skillnet._psn_impl.step_context import StepContext
from skillnet.core.dk_registry import get_domain_knowledge
from skillnet.agents.skill_graph.utils.code_analysis import (
    validate_code_completeness,
    extract_self_contained_helpers,
    migrate_parent_extract_helper,
)
from skillnet.agents.skill_graph.utils.code_validation import get_control_primitives
from skillnet.agents.skill_graph.utils.async_await_validation import (
    detect_concurrency_risk,
    format_violations_message,
)
from skillnet.agents.refactor.base import analyze_code_dependencies, inject_dependencies


class StepExecutePhaseMixin:
    """Phases 4-5: Environment execution and diagnostics."""

    def _step_execute(self, ctx: StepContext):
        """Phase 4: build code, validate, execute env.step, critic evaluation."""
        ctx.code = ctx.parsed_result["program_code"] + "\n" + ctx.parsed_result["exec_code"]

        # Register-at-birth: extract self-contained helper functions defined
        # inline in the generated code into standalone experimental skill nodes,
        # and migrate the code to call them. This makes a helper that ran but was
        # not its own node (e.g. an inline counter carrying a bug) first-class: a
        # graph node (candidate-eligible for the optimizer) AND wrapped (emits
        # skillStart, so it appears in the actual-execution sub-graph). Closure-
        # capturing helpers are left inline (cannot stand alone). Runs BEFORE
        # skill_names is computed so the extracted helpers flow into wrapping and
        # the candidate set with no other change. Mutates ctx.code in place.
        self._register_self_contained_helpers(ctx)

        # Extract the skill-name list: only wrap skills actually needed in the
        # call chain (not all skills in the graph), to avoid evaluating large
        # amounts of useless wrapper code.
        directly_called = self.skill_manager.extract_called_skills(ctx.code)
        skill_names = self.skill_manager.expand_with_dependencies(directly_called)

        # Also extract newly-defined function names from code (top-level only,
        # via the active SkillLanguage's parser). Filter to async functions to
        # match the previous extract_toplevel_function_names(include_async=True,
        # include_regular=False) contract. Even if a new skill isn't in the
        # graph yet, a wrapper will be created for it.
        dk = get_domain_knowledge()
        if dk is None:
            raise RuntimeError(
                "No DomainKnowledge registered; cannot parse skill code"
            )
        skill_lang = dk.get_skill_language_impl()
        parse_result = skill_lang.parse(ctx.code)
        code_toplevel_funcs = [
            f.name for f in parse_result.functions if f.is_async
        ]
        new_skill_names = [name for name in code_toplevel_funcs if name not in skill_names]
        if new_skill_names:
            skill_names.extend(new_skill_names)
            print(f"\033[36m[Skill Wrapper] Found {len(new_skill_names)} new skills in the current code: {', '.join(new_skill_names)}\033[0m")

        # Pre-register all newly-defined skills before env.step() so
        # extract_called_skills() inside _auto_record_skill_executions()
        # returns them correctly (otherwise new skills are marked "not executed").
        skill_defs = self.skill_manager.extract_all_function_definitions(ctx.code)
        name_mapping = {}  # original name -> final name (used to update code)
        for skill_def in skill_defs:
            final_name, updated_code = self.skill_manager.pre_register_skill(
                name=skill_def["name"],
                code=skill_def["code"],
                task=self.task
            )
            if final_name != skill_def["name"]:
                name_mapping[skill_def["name"]] = final_name

        # If any renaming occurred, update the code and skill_names list
        if name_mapping:
            for old_name, new_name in name_mapping.items():
                ctx.code = re.sub(
                    rf'\b{re.escape(old_name)}\s*\(',
                    f"{new_name}(",
                    ctx.code
                )
                if old_name in skill_names:
                    skill_names.remove(old_name)
                    if new_name not in skill_names:
                        skill_names.append(new_name)
            print(f"\033[36m[Skill Pre-Register] Code updated; rename mapping: {name_mapping}\033[0m")

        # Debug: record which skills will be wrapped
        if skill_names and len(skill_names) <= 20:
            print(f"\033[36m[Skill Wrapper] Will create event wrappers for the following {len(skill_names)} skills: {', '.join(skill_names)}\033[0m")
        else:
            print(f"\033[36m[Skill Wrapper] Will create event wrappers for {len(skill_names) if skill_names else 0} skills\033[0m")

        # Code-completeness validation (static layer): detect undefined function
        # calls before env.step() to surface missing-dependency issues early.
        graph_nodes = set(self.skill_manager.get_all_skill_names(include_task_specific=True))
        control_primitives = set(get_control_primitives())
        is_valid, code_issues = validate_code_completeness(ctx.code, graph_nodes, control_primitives)
        if not is_valid:
            print(f"\033[33m[Code Validation] Code-completeness validation found issues:\033[0m")
            for issue in code_issues:
                print(f"\033[33m  - {issue}\033[0m")
            # Lenient: warn but continue, letting runtime errors provide more detail.

        # PSN main-flow dependency injection: detect and inject missing deps
        # (e.g. GoalNear / GoalXZ from mineflayer-pathfinder). Solves TypeError
        # when LLM-generated code uses undeclared dependencies.
        missing_deps = analyze_code_dependencies(ctx.code)
        if missing_deps:
            print(f"\033[36m[Dependency Injection] Injecting missing dependencies: {missing_deps}\033[0m")
            ctx.code = inject_dependencies(ctx.code, missing_deps)

        # Strict concurrency-risk validation: unawaited async calls inside
        # sync callbacks generate unbounded concurrent work that crashes
        # mineflayer's Node.js event loop. Block here before env.step().
        concurrency_violations = detect_concurrency_risk(
            ctx.code + "\n\n" + self.skill_manager.programs
        )
        if concurrency_violations:
            self._preflight_rejects_this_task += 1
            error_msg = format_violations_message(concurrency_violations)
            print(f"\033[31m[Concurrency Validation] Rejecting code without env.step "
                  f"(rejects this task: {self._preflight_rejects_this_task})\033[0m")
            print(f"\033[31m[Concurrency Validation] {len(concurrency_violations)} "
                  f"crash_risk violation(s)\033[0m")
            for v in concurrency_violations:
                print(f"\033[31m  - L{v.line}: {v.kind} — {v.name}\033[0m")
            ctx.events = self._build_synthetic_error_events(error_msg)
            ctx.critique = error_msg
            ctx.success = False
            return

        ctx.events = self.env.step(
            ctx.code,
            programs=self.skill_manager.programs,
            skill_names=skill_names,
        )

        self.recorder.record(ctx.events, self.task)
        # Record step in trajectory. Handle ai_message=None (graph planner case).
        ai_message_content = None
        if ctx.ai_message is not None:
            ai_message_content = ctx.ai_message.content
        elif ctx.planning_result is not None:
            ai_message_content = ctx.planning_result.plan or "Graph-based planning"
        else:
            ai_message_content = "Unknown planning method"

        self.trajectory_recorder.record_step(
            action=ctx.code,
            events=ctx.events,
            conversation=self.conversations[-1] if self.conversations else None,
            ai_message=ai_message_content,
        )
        # Safely update chest memory - check if events is not empty
        observe_data = find_last_observe(ctx.events)
        if observe_data is not None:
            nearby_chests = observe_data.get("nearbyChests", {})
            if isinstance(nearby_chests, dict):
                self.action_agent.update_chest_memory(nearby_chests)
            elif nearby_chests:
                print(f"\033[33m[Warning] nearbyChests has wrong type: {type(nearby_chests)}\033[0m")

        # State changes for critic (used to verify mining/collection tasks),
        # measured against task_initial_events at task start.
        task_state_changes = self._calculate_task_state_changes(
            pre_events=self.task_initial_events if hasattr(self, 'task_initial_events') else None,
            post_events=ctx.events
        )

        executed_skills = self._get_executed_skills(ctx.events)

        # Planned skill sequence (used for semantic inference in LLM fallback).
        planned_skills = None
        plan_type = None
        if ctx.planning_result is not None:
            plan_type = ctx.planning_result.plan_type
            if ctx.planning_result.skill_sequence:
                planned_skills = [sc.skill_name for sc in ctx.planning_result.skill_sequence]

        # Critic assesses task completion. executed_skills/planned_skills
        # power the semantics-aware evaluation in PSNCriticAgent.
        ctx.success, ctx.critique, ctx.quality_metrics = self.critic_agent.check_task_success(
            events=ctx.events,
            task=self._task_semantic,
            context=self.context,
            chest_observation=self.action_agent.render_chest_observation(),
            max_retries=5,
            state_changes=task_state_changes,
            executed_skills=executed_skills,
            planned_skills=planned_skills,
            plan_type=plan_type,
        )

        if ctx.quality_metrics:
            print(f"\033[35m[Critic Quality] robustness={ctx.quality_metrics.get('robustness_score', 'N/A')}, "
                  f"dependency_safety={ctx.quality_metrics.get('dependency_safety', 'N/A')}, "
                  f"environment_awareness={ctx.quality_metrics.get('environment_awareness', 'N/A')}\033[0m")

    def _register_self_contained_helpers(self, ctx: StepContext):
        """Register-at-birth. See call site in _step_execute for rationale.

        For each extractable self-contained helper defined inline in ctx.code:
          1. migrate ctx.code (remove the inline def + rewrite calls to the
             async/bot contract) -- all-or-nothing, verified by the JS parser;
          2. register the helper's standalone code as an experimental node
             (lightweight pre_register_skill).
        Skips closure-capturing helpers, existing nodes, and any helper whose
        parent migration does not re-parse cleanly (the parent is left untouched
        for that helper, so this never breaks the running code).
        """
        try:
            helpers = extract_self_contained_helpers(ctx.code)
        except Exception as e:
            print(f"\033[33m[Register-at-birth] helper detection failed: {e}\033[0m")
            return
        extractable = [h for h in helpers if h.get("extractable") and h.get("standalone_code")]
        if not extractable:
            return

        dk = get_domain_knowledge()
        skill_lang = dk.get_skill_language_impl() if dk else None
        registered = []

        for h in extractable:
            name = h["name"]
            # Do not clobber an existing graph node / already-registered helper.
            if self.skill_manager.has_node(name):
                continue
            # Migrate the parent: remove inline def + rewrite call sites.
            migrated, ok = migrate_parent_extract_helper(
                ctx.code, name, h["needs_bot_injection"]
            )
            if not ok:
                continue
            # All-or-nothing: the migrated code must still parse, else leave inline.
            if skill_lang is not None:
                try:
                    res = skill_lang.validate_syntax(migrated)
                    if not getattr(res, "valid", False):
                        continue
                except Exception:
                    continue
            # Register the helper as an experimental node. Pass task="" (not the
            # current task): the helper is a generic utility, and letting the
            # parent task drive its intent-based effect extraction would taint it
            # with the parent's target effect (e.g. a pure name-resolver extracted
            # under "Craft 1 wooden pickaxe" would be advertised as producing a
            # wooden_pickaxe, so the matcher recommends it as the crafter and the
            # agent crafts nothing). The helper's effects must come from its own
            # code only.
            try:
                final_name, _code = self.skill_manager.pre_register_skill(
                    name=name, code=h["standalone_code"], task=""
                )
            except Exception as e:
                print(f"\033[33m[Register-at-birth] pre_register failed for '{name}': {e}\033[0m")
                continue
            # If pre_register renamed (e.g. primitive-name conflict), rewrite the
            # migrated code's calls to the final name so they still resolve.
            if final_name and final_name != name:
                migrated = re.sub(
                    rf'\b{re.escape(name)}\s*\(', f"{final_name}(", migrated
                )
            ctx.code = migrated
            registered.append(final_name or name)

        if registered:
            print(
                f"\033[36m[Register-at-birth] extracted self-contained "
                f"helper(s) -> {', '.join(registered)}\033[0m"
            )

    def _step_diagnose_and_record(self, ctx: StepContext):
        """Phase 5: Graph planner failure diagnostics + trajectory update + skill recording."""
        # For Graph Planner failures, emit detailed diagnostics
        if (not ctx.success and
            ctx.planning_result is not None and
            ctx.planning_result.plan_type == "graph"):

            print(f"\033[31m{'='*80}\033[0m")
            print(f"\033[31m[Graph Planner execution-failure diagnostics]\033[0m")
            print(f"\033[31m{'='*80}\033[0m")

            # Basic info
            print(f"\033[33m  Task: {self.task}\033[0m")
            print(f"\033[33m  Context: {self.context[:200]}...\033[0m")

            # Graph Planner info
            if ctx.planning_result.skill_sequence:
                skill_names = [sc.skill_name for sc in ctx.planning_result.skill_sequence]
                print(f"\033[33m  Skills used: {', '.join(skill_names)}\033[0m")
            if ctx.planning_result.metadata:
                if "target_effects" in ctx.planning_result.metadata:
                    print(f"\033[33m  Target effects: {ctx.planning_result.metadata['target_effects']}\033[0m")

            # Generated code
            print(f"\033[33m  Generated code:\033[0m")
            print(f"\033[36m{ctx.code[:500]}...\033[0m" if len(ctx.code) > 500 else f"\033[36m{ctx.code}\033[0m")

            # Extract execution-error info
            error_messages = []
            skill_errors = {}

            for event_type, event_data in iter_events(ctx.events):
                if event_type == "error":
                    error_msg = str(event_data.get("error", "")) if isinstance(event_data, dict) else str(event_data)
                    if error_msg:
                        error_messages.append(error_msg)
                elif event_type == "skillError":
                    skill_name = event_data.get("skillName") if isinstance(event_data, dict) else None
                    if skill_name:
                        error_msg = event_data.get("error", "")
                        error_stack = event_data.get("stack", "")
                        skill_errors[skill_name] = {
                            "error": error_msg,
                            "stack": error_stack,
                            "timestamp": event_data.get("timestamp"),
                        }
                elif event_type == "onError":
                    error_msg = event_data.get("onError", "") if isinstance(event_data, dict) else str(event_data)
                    if error_msg:
                        error_messages.append(error_msg)

            # Print error info
            if error_messages:
                print(f"\033[31m  Execution error:\033[0m")
                for i, err in enumerate(error_messages, 1):
                    print(f"\033[31m  {i}. {err}\033[0m")

            # Print skill-execution errors
            if skill_errors:
                print(f"\033[31m  Skill execution errors:\033[0m")
                for skill_name, error_info in skill_errors.items():
                    print(f"\033[31m  - {skill_name}:\033[0m")
                    print(f"\033[31m    Error: {error_info['error']}\033[0m")
                    if error_info.get('stack'):
                        print(f"\033[31m    Stack:\n{error_info['stack'][:500]}...\033[0m" if len(error_info['stack']) > 500 else f"\033[31m    Stack:\n{error_info['stack']}\033[0m")

            # If no specific error found, print critique
            if not error_messages and not skill_errors and ctx.critique:
                print(f"\033[33m  Critic feedback: {ctx.critique}\033[0m")

            # Print current state
            diag_observe = find_last_observe(ctx.events)
            if diag_observe is not None:
                inventory = diag_observe.get("inventory", {})
                position = diag_observe.get("status", {}).get("position", {})
                print(f"\033[33m  Post-execution state:\033[0m")
                print(f"\033[33m  - Inventory: {inventory}\033[0m")
                print(f"\033[33m  - Position: {position}\033[0m")

            print(f"\033[31m{'='*80}\033[0m")

        # Update trajectory with success and critique
        if self.trajectory_recorder.current_trajectory:
            last_step = self.trajectory_recorder.current_trajectory[-1]
            last_step["success"] = ctx.success
            last_step["critique"] = ctx.critique

        # Auto-record skill executions (always uses SkillGraphManager)
        self._auto_record_skill_executions(
            code=ctx.code,
            parsed_result=ctx.parsed_result,
            events=ctx.events,
            success=ctx.success,
            critique=ctx.critique,
        )
