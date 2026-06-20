"""
Skill Graph Management System for PSN

This module implements a graph-based skill management system that extends
the original SkillManager with:
- Graph structure for skill dependencies
- Extended skill attributes (preconditions, effects, versions, etc.)
- Skill optimization capabilities
"""

import os
import copy
import re
import json
import shutil
import threading
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple, Iterator, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Import all dataclasses from models/ (sole source after modular refactor)
from skillnet.agents.skill_graph.models import (
    FailureCategory,
    PreconditionValueFunctionConfig,
    EffectValueFunctionConfig,
    SkillPrecondition,
    SkillEffect,
    ActualEffect,
    SkillVersion,
    SkillExecutionTrace,
    SkillStatistics,
    SkillGradientItem,
    SkillGradients,
    SkillNode,
    SkillGraph,
    should_skip_covered_skill,
    CoverageType,
    # Unified code update interface types
    CodeSource,
    UpdateResult,
    ConfirmResult,
)

# Import utility functions from utils/
from skillnet.agents.skill_graph.utils import (
    _strip_comments_and_strings,
    validate_code_brackets,
    validate_code_syntax,
    check_line_length,
    get_control_primitives,
    is_control_primitive_name,
    resolve_primitive_name_conflict,
    _find_validated_string_vars,
    # parameter parser
    extract_destructured_params,
    infer_type_from_node,
    extract_value_from_node,
    infer_param_type_from_babel,
    extract_default_value_from_babel,
    extract_value_from_babel_node,
    generate_param_description,
    # code validator
    classify_effect_importance,
    validate_naming_effect_consistency,
    # code analyzer
    camel_to_snake,
    sanitize_python_to_js,
    extract_function_calls,
    serialize_effects,
    filter_preconditions,
    is_task_specific_skill,
    is_general_skill,
    # execution analyzer
    calculate_correlation,
    categorize_failure,
    parse_js_value,
    parse_js_arguments,
)

# Refactor detection module
from skillnet.agents.refactor.detection import (
    RefactorPrescreener,
    RefactorRelationshipAnalyzer,
)

# Metadata extraction module
from skillnet.agents.skill_graph.metadata import (
    PreconditionExtractor,
    EffectExtractor,
    ParameterExtractor,
)

# Over-claim detection
from skillnet.agents.optimizer.validators import (
    OverclaimDetector,
    OverclaimResult,
)

from skillnet.agents.skill_graph.execution import (
    calculate_state_changes,
)

# Import LLM factory
from skillnet.utils.llm_factory import create_chat_llm

# Import LangChain components
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage

# Import PSN utilities
import skillnet.utils as U
from skillnet.utils.log_utils import LoggerManager
from skillnet.utils.stats_tracker import StatsTracker

# Import Refactor module adapter
from skillnet.agents.refactor import (
    RefactorExecutor,
    RefactorType,
    RefactorResult,
)

# Extracted Mixin modules
# MetadataManagementMixin split into 3 mixins
# SkillExecutionMixin split into 3 mixins
from skillnet.agents.skill_graph._impl.mixins import (
    RefactorManagementMixin,
    SkillVersioningMixin,
    MetadataManagementMixin,
    MetadataCrudMixin,
    MetadataValidationMixin,
    SkillExecutionMixin,
    ExecutionLifecycleMixin,
    SemanticsUpdateMixin,
    GraphQueriesMixin,
    SkillDeletionAndCleanupMixin,
    SkillRetrievalAndDiscoveryMixin,
    SkillCodeUpdateMixin,
)


# =============================================================================
# Mixin Classes for SkillGraphManager
# Group SkillGraphManager methods by functionality to improve readability and maintainability
# =============================================================================

class SkillAdditionMixin:
    """Skill Addition Mixin - New skill creation, naming, and initialization.

    These methods logically belong to the skill addition group but remain in
    SkillGraphManager because add_new_skill involves 635 lines and 31+ method calls,
    making it unsafe to extract.

    Methods (implemented in SkillGraphManager):
        add_new_skill: Add new skill (~635 lines, core method)
        _try_normalize_skill_name: Try to normalize skill name
        _is_task_specific_skill: Check if skill is task-specific
        _is_general_skill: Check if skill is general
        _save_task_specific_skill: Save task-specific skill
        generate_skill_description: Generate skill description
        _extract_function_signature: Extract function signature
        _check_and_apply_semantic_rename: Check and apply semantic rename
        _validate_function_name_semantic_consistency: Validate function name semantic consistency

    Warning:
        add_new_skill is highly coupled with 32 _save_to_checkpoint() calls.
        Not recommended for extraction to standalone module.
    """
    pass


# Execution Mixins (split into 3 files)
# SkillExecutionMixin (skill_execution.py): record_execution
# ExecutionLifecycleMixin (execution_lifecycle.py): _trigger_delayed_refactor,
# _handle_refactored_skill_failure, _rollback_skill_refactor, on_task_completed
# SemanticsUpdateMixin (semantics_update.py): _update_semantics_from_executions,
# _update_effects_from_executions, _analyze_parameterized_effects,
# _calculate_correlation, _classify_effect_importance, _parse_js_args_to_dict,
# _parse_call_args_from_exec_code, _parse_js_arguments, _parse_js_value,
# _categorize_failure, _calculate_precondition_value, _calculate_effect_value,
# _verify_precondition_in_code, _update_preconditions_from_executions


# SkillVersioningMixin - Imported from skillnet.agents.skill_graph._impl.mixins.skill_versioning
# See that module for implementation of:
# create_new_version, rollback_skill, rollback_skill_and_subgraph,
# rollback_graph, get_skill_versions, get_graph_versions,
# get_current_skill_version, get_current_graph_version


# RefactorManagementMixin - Imported from skillnet.agents.skill_graph._impl.mixins.refactor_mgmt
# See that module for implementation of:
# _prescreen_refactor_candidates, _prescreen_refactor_candidates_detailed,
# _detect_refactor_relationships, _validate_refactor_direction,
# _execute_refactor, _handle_covered_skills


# Metadata Mixins (split into 3 files)
# MetadataManagementMixin (metadata_mgmt.py): _update_skill_metadata, _analyze_interface_impact,
# _extract_preconditions, _extract_effects, _filter_preconditions, _code_has_precondition_check,
# _validate_naming_effect_consistency
# MetadataCrudMixin (metadata_crud.py): add/remove/get preconditions and effects,
# remove_effect_by_item, _filter_out_item_from_effects, _serialize_effects, _log_effects_change
# MetadataValidationMixin (metadata_validation.py): _validate_and_clean_effects_on_load,
# _repair_missing_primary_effects, _fix_function_name_mismatch_on_load,
# _validate_effects_against_code, _validate_or_effect_conditions, _code_has_effect_implementation


class SkillGraphManager(
    SkillAdditionMixin,
    SkillCodeUpdateMixin,
    SkillExecutionMixin,
    ExecutionLifecycleMixin,
    SemanticsUpdateMixin,
    SkillVersioningMixin,
    RefactorManagementMixin,
    MetadataManagementMixin,
    MetadataCrudMixin,
    MetadataValidationMixin,
    SkillDeletionAndCleanupMixin,
    SkillRetrievalAndDiscoveryMixin,
    GraphQueriesMixin,
):
    """
    Skill Graph Manager - Graph-based skill management.

    Compatible with original SkillManager interface while providing graph management features.
    """
    
    def __init__(
        self,
        model_name="gpt-5-mini",
        temperature=0,
        retrieval_top_k=8,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        auto_extract_preconditions=True,  # whether to auto-extract preconditions
        auto_extract_effects=True,  # whether to auto-extract effects
        parameter_extraction_mode="babel_then_llm",  # parameter extraction mode: "babel_then_llm" or "llm_only"
        merge_mode="llm",  # Merge mode: "llm" (LLM-driven smart merge) or "simple" (string-matching merge)
        value_function_lambda=1.0,  # Value function λ parameter
        value_function_alpha=1.0,  # Value function α parameter (Bayesian prior)
        value_function_beta=1.0,  # Value function β parameter (Bayesian prior)
        # Semantic inference configuration
        semantic_inference_enabled=True,  # whether to enable parameter semantic inference
        semantic_llm_fallback=True,  # whether to enable LLM-assisted inference (when disabled, rules only)
        semantic_default="delta",  # default when inference fails: "delta" or "target_total"
        # Semantic rename configuration
        auto_semantic_rename=True,  # whether to auto-fix semantic naming inconsistencies (e.g. craftXxx + target_total -> ensureXxx)
        # Refactor switch
        enable_refactor=True,  # whether to enable refactor (delayed refactor and checkpoint migration)
        # Skill cap for experiments (B2)
        max_skills=None,  # None = unlimited; int = cap graph size
        # Multi-backend LLM support
        openai_api_base=None,
        openai_api_key=None,
    ):
        self.llm = create_chat_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            component="skill_manager",
        )
        
        # Initialize statistics tracker
        self.stats_tracker = StatsTracker(ckpt_dir=ckpt_dir)
        if resume:
            self.stats_tracker.load_stats()
        # Register as the process-global active tracker so previously-untracked
        # raw LLM call sites (planner / REFLECT / refactor / curriculum / critic /
        # evolution) can record into it via stats_tracker.record_llm_usage().
        from skillnet.utils.stats_tracker import set_active_tracker
        set_active_tracker(self.stats_tracker)
        
        # Create directories
        U.f_mkdir(f"{ckpt_dir}/skill_graph/code")
        U.f_mkdir(f"{ckpt_dir}/skill_graph/description")
        U.f_mkdir(f"{ckpt_dir}/skill_graph/preconditions")
        U.f_mkdir(f"{ckpt_dir}/skill_graph/effects")
        U.f_mkdir(f"{ckpt_dir}/skill_graph/metadata")
        U.f_mkdir(f"{ckpt_dir}/skill_graph/vectordb")
        U.f_mkdir(f"{ckpt_dir}/skill_graph/graph")
        
        # Initialize the graph structure
        self.graph = SkillGraph()
        
        # Control primitives (base functions) — populated by set_domain_knowledge()
        # once the domain is injected; readers see [] until then.
        self.control_primitives = []
        
        # Configuration
        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        self.auto_extract_preconditions = auto_extract_preconditions
        self.auto_extract_effects = auto_extract_effects
        self.merge_mode = merge_mode  # "llm" or "simple"
        self.parameter_extraction_mode = parameter_extraction_mode  # "babel_then_llm" or "llm_only"
        
        # Value function configuration parameters
        self.value_function_lambda = value_function_lambda
        self.value_function_alpha = value_function_alpha
        self.value_function_beta = value_function_beta
        
        # Semantic inference configuration
        self.semantic_inference_enabled = semantic_inference_enabled
        self.semantic_llm_fallback = semantic_llm_fallback
        self.semantic_default = semantic_default
        
        # Semantic rename configuration
        self.auto_semantic_rename = auto_semantic_rename

        # Refactor switch
        self.enable_refactor = enable_refactor

        # Skill cap (B2: experiment infrastructure)
        self.max_skills = max_skills

        # Recursion guard flag (prevents infinite recursion when update_skill_code triggers callbacks)
        self._is_updating_code = False

        # Domain knowledge (injected by PSNAgent.from_domain)
        self._domain_knowledge = None

        # Skill-language cache (populated by set_domain_knowledge / set_skill_language)
        self._skill_language = None

        # [P0 Fix] Concurrency lock - prevents conflicts between record_execution and add_new_skill temporary nodes
        # Use RLock so the same thread can re-enter (safe for recursive calls)
        self._graph_update_lock = threading.RLock()

        # List of count-parameter names (used to identify params needing semantic inference)
        self.QUANTITY_PARAM_NAMES = ["count", "amount", "num", "quantity", "number", "total", "target"]
        
        # Initialize the logger
        self.logger = LoggerManager.get_logger("SkillGraphManager", self.ckpt_dir)
        
        # Vector database (for retrieval)
        self.vectordb = Chroma(
            collection_name="skill_graph_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/skill_graph/vectordb",
        )
        
        # Restore from checkpoint
        if resume:
            self._load_from_checkpoint()
        else:
            # Verify the vector database is empty
            assert self.vectordb._collection.count() == 0, (
                "Skill Graph Manager's vectordb is not empty. "
                "Set resume=False or delete the vectordb directory."
            )
        
        # Interface-change callback (set by SkillGraphOptimizer)
        # When skill code changes, this callback runs to perform interface checks and update callers
        self._on_code_changed_callback = None

        # Refactor executor (lazy-init to avoid circular dependencies)
        self.__refactor_executor = None
        self.__refactor_history_tracker = None

        # Refactor detection components (lazy-init)
        self.__refactor_prescreener = None
        self.__refactor_analyzer = None

        # Metadata extraction components (lazy-init)
        self.__precondition_extractor = None
        self.__effect_extractor = None
        self.__parameter_extractor = None

        # v3.H+ step-scoped refactor modification tracking.
        # Populated by update_skill_code when source starts with "refactor:".
        # Reset by PSN at start of each step (_psn_impl/step_execution.step()).
        # Consulted by psn.py:1002 add_new_skill caller AND graph_manager_impl
        # Layer 3 wrapper protection to prevent action_agent's original inline
        # code from overwriting a refactor's modifications mid-step (covers
        # behavioral Case A and extract_common-callers edge cases where
        # is_covered stays False but code was changed by refactor).
        self._refactor_modified_skills_this_step: Set[str] = set()

    def reset_step_refactor_tracking(self) -> None:
        """v3.H+: clear the per-step refactor-modification tracking set.
        Called by PSN at the start of each iteration step."""
        self._refactor_modified_skills_this_step.clear()

    def was_modified_by_refactor_this_step(self, skill_name: str) -> bool:
        """v3.H+: whether the given skill was modified by any refactor type
        during the current step (i.e., update_skill_code was called with a
        source field starting with 'refactor:')."""
        return skill_name in self._refactor_modified_skills_this_step

    def set_domain_knowledge(self, knowledge):
        """Inject domain knowledge (called by PSNAgent.from_domain)."""
        self._domain_knowledge = knowledge
        # Reload control primitives from the domain so any domain-specific
        # set replaces the defaults loaded during __init__.
        if knowledge:
            try:
                self.control_primitives = knowledge.load_control_primitive_code()
            except Exception:
                pass  # Keep existing primitives if domain load fails
        # Cache skill-language implementation so hot paths read once and dispatch
        # through the Protocol; registry-fallback covers module-level helpers.
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

    def set_skill_language(self, language):
        """Inject skill language implementation (called by PSNAgent.from_domain)."""
        self._skill_language = language

    def _get_file_extension(self) -> str:
        """Get file extension for skill code files from skill language."""
        if hasattr(self, '_skill_language') and self._skill_language:
            return self._skill_language.file_extension
        return ".js"

    def set_on_code_changed_callback(self, callback: Optional[Callable]) -> None:
        """
        Set the code-change callback.

        Invoked whenever skill code changes (refactor, version update, etc.).
        Typically set by SkillGraphOptimizer to trigger interface checks and caller updates.

        Args:
            callback: callback function with signature:
                callback(skill_name: str, old_code: str, new_code: str, change_source: str) -> Dict
        """
        self._on_code_changed_callback = callback
        print(f"\033[36m[SkillGraphManager] Code-change callback registered\033[0m")

    @property
    def _refactor_executor(self) -> "RefactorExecutor":
        """
        Lazily-initialized Refactor executor.

        Lazy initialization avoids the circular-dependency issue of constructing RefactorExecutor in __init__.
        RefactorExecutor needs a fully-initialized SkillGraphManager instance.
        """
        if self.__refactor_executor is None:
            self.__refactor_executor = RefactorExecutor(
                skill_graph_manager=self,
                llm=self.llm,
                logger=self.logger,
                on_code_changed_callback=self._notify_code_changed,
            )
        return self.__refactor_executor

    @property
    def _refactor_tracker(self):
        """Lazily-initialized RefactorHistoryTracker, used to persist refactor history."""
        if self.__refactor_history_tracker is None:
            from skillnet.agents.refactor.history import RefactorHistoryTracker
            self.__refactor_history_tracker = RefactorHistoryTracker(ckpt_dir=self.ckpt_dir)
        return self.__refactor_history_tracker

    @property
    def _refactor_prescreener(self) -> "RefactorPrescreener":
        """
        lazily-initialized Refactor prescreener.
        """
        if self.__refactor_prescreener is None:
            self.__refactor_prescreener = RefactorPrescreener(
                graph=self,  # pass SkillGraphManager (duck typing)
                vectordb=self.vectordb,
                llm=self.llm,
                custom_logger=self.logger,
            )
        return self.__refactor_prescreener

    @property
    def _refactor_analyzer(self) -> "RefactorRelationshipAnalyzer":
        """
        lazily-initialized Refactor relationship analyzer.
        """
        if self.__refactor_analyzer is None:
            self.__refactor_analyzer = RefactorRelationshipAnalyzer(
                graph=self,  # pass SkillGraphManager (duck typing)
                llm=self.llm,
                prescreener=self._refactor_prescreener,
                custom_logger=self.logger,
            )
        return self.__refactor_analyzer

    @property
    def _precondition_extractor(self) -> "PreconditionExtractor":
        """
        lazily-initialized Precondition extractor.
        """
        if self.__precondition_extractor is None:
            self.__precondition_extractor = PreconditionExtractor(
                llm=self.llm,
                merge_mode=self.merge_mode,
                custom_logger=self.logger,
            )
        return self.__precondition_extractor

    @property
    def _effect_extractor(self) -> "EffectExtractor":
        """
        lazily-initialized Effect extractor.
        """
        if self.__effect_extractor is None:
            self.__effect_extractor = EffectExtractor(
                llm=self.llm,
                merge_mode=self.merge_mode,
                custom_logger=self.logger,
            )
        return self.__effect_extractor

    @property
    def _parameter_extractor(self) -> "ParameterExtractor":
        """
        lazily-initialized Parameter extractor.
        """
        if self.__parameter_extractor is None:
            _dk = getattr(self, '_domain_knowledge', None)
            _entry = _dk.get_entry_parameter_name() if _dk else "bot"
            # Plan v3-rev Fix 2B: model-aware object schema enrichment strategy
            from skillnet.core.model_profile import detect_model_profile
            _model_name = getattr(self.llm, "model_name", None) or ""
            _profile = detect_model_profile(_model_name)
            self.__parameter_extractor = ParameterExtractor(
                llm=self.llm,
                extraction_mode=self.parameter_extraction_mode,
                semantic_inference_enabled=self.semantic_inference_enabled,
                semantic_llm_fallback=self.semantic_llm_fallback,
                semantic_default=self.semantic_default,
                quantity_param_names=self.QUANTITY_PARAM_NAMES,
                custom_logger=self.logger,
                entry_parameter_name=_entry,
                object_schema_strategy=_profile.object_schema_strategy,
            )
        return self.__parameter_extractor

    # _notify_code_changed - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    def _load_from_checkpoint(self) -> None:
        """Load from checkpoint."""
        print(f"\033[33mLoading Skill Graph Manager from {self.ckpt_dir}/skill_graph\033[0m")
        
        # Load the graph structure
        graph_file = f"{self.ckpt_dir}/skill_graph/graph/graph.json"
        if os.path.exists(graph_file):
            graph_data = U.load_json(graph_file)
            self.graph = SkillGraph.from_dict(graph_data)
            print(f"Loaded {len(self.graph.nodes)} skills from graph")

            # [Migration] If refactor is disabled, clear existing coverage marks
            # This ensures previously-set is_covered/covered_by does not block skill retrieval
            if not self.enable_refactor:
                cleared_count = 0
                for node in self.graph.nodes.values():
                    if node.is_covered:
                        node.is_covered = False
                        node.covered_by = None
                        cleared_count += 1
                if cleared_count > 0:
                    print(f"\033[33m[Checkpoint Migration] Cleared coverage from {cleared_count} skills (refactor disabled)\033[0m")
                    self._save_to_checkpoint()

            # [Migration] Identify and mark graph planner wrappers that were missed
            # These are skill compositions previously missed by the _is_general_skill() check
            wrapper_count = 0
            for node in self.graph.nodes.values():
                if not getattr(node, 'is_task_specific', False) and self._is_skill_composition(node.code or ""):
                    # Likely a missed graph planner wrapper; confirm via heuristic check
                    # Pass node so that refactor-extracted skills are not misclassified
                    if self._is_task_specific_skill(node.name, node.code or "", node):
                        node.is_task_specific = True
                        wrapper_count += 1
            if wrapper_count > 0:
                print(f"\033[33m[Checkpoint Migration] Marked {wrapper_count} skill compositions as task-specific\033[0m")
                self._save_to_checkpoint()

            # Auto-validate and clean up incorrect dependency relations
            # This fixes incorrect edges introduced by code generation errors
            cleanup_result = self.validate_and_clean_graph(verbose=False)
            if cleanup_result["removed_edges"]:
                print(f"\033[33m[Graph Cleanup] Auto-removed {len(cleanup_result['removed_edges'])} incorrect dependency edges\033[0m")
                for edge in cleanup_result["removed_edges"]:
                    print(f"\033[33m  - removed: {edge['parent']} -> {edge['child']}\033[0m")
                # Save the cleaned graph
                self._save_to_checkpoint()

            # Validate and fix incorrect effects caused by LLM hallucination
            # This prevents Graph Planner from incorrectly matching skills
            effects_cleanup = self._validate_and_clean_effects_on_load()
            if effects_cleanup["cleaned_skills"]:
                print(f"\033[33m[Effect Cleanup] Fixed incorrect effects on {len(effects_cleanup['cleaned_skills'])} skill(s)\033[0m")
                for info in effects_cleanup["cleaned_skills"]:
                    print(f"\033[33m  - {info['skill']}: {info['before']} → {info['after']} effects\033[0m")
                # Save the repaired graph
                self._save_to_checkpoint()

            # Repair primary effects missing on wrapper skills
            # Wrapper skills (e.g. craftWoodenPickaxe) sometimes have primary effects mis-inferred by LLM effect inference,
            # causing EffectMatcher to fail to match. Here we infer and supplement primary effect from the function name.
            primary_repair = self._repair_missing_primary_effects()
            if primary_repair["repaired_skills"]:
                print(f"\033[33m[Effect Repair] Supplemented primary effect for {len(primary_repair['repaired_skills'])} skill(s)\033[0m")
                for info in primary_repair["repaired_skills"]:
                    print(f"\033[33m  - {info['skill']}: added primary effect → {info['item']}\033[0m")
                self._save_to_checkpoint()

            # Validate and fix mismatches between function name and registration name
            # This prevents Graph Planner from emitting wrong calls and Skill Wrapper from losing track
            func_name_cleanup = self._fix_function_name_mismatch_on_load()
            if func_name_cleanup["fixed_skills"]:
                print(f"\033[33m[Function Name Fix] Fixed function-name mismatch on {len(func_name_cleanup['fixed_skills'])} skill(s)\033[0m")
                for info in func_name_cleanup["fixed_skills"]:
                    print(f"\033[33m  - {info['skill']}: '{info['old_func_name']}' → '{info['new_func_name']}'\033[0m")
                # Save the repaired graph
                self._save_to_checkpoint()

            # [Cascading-failure defense] Detect and quarantine syntactically broken skill code files.
            # If left in skill_graph/code/, mineflayer will splice them into
            # the ${programs} bundle, where top-level await/return will fail every eval.
            # Broken skills are moved to skill_graph/quarantined/ and removed from the graph,
            # so the task loop can regenerate them normally.
            corrupt_cleanup = self._quarantine_corrupt_skill_code_on_load()
            if corrupt_cleanup["quarantined_skills"]:
                print(f"\033[31m[Skill Quarantine] Detected and quarantined {len(corrupt_cleanup['quarantined_skills'])} syntactically broken skill(s):\033[0m")
                for info in corrupt_cleanup["quarantined_skills"]:
                    print(f"\033[31m  - {info['skill']}: {info['error'][:120]}\033[0m")
                self._save_to_checkpoint()
        else:
            print("No graph file found, starting with empty graph")

        # If vectordb is empty but graph has nodes, auto-rebuild vectordb from the graph
        vectordb_count = self.vectordb._collection.count()
        graph_count = len(self.graph.nodes)
        if vectordb_count == 0 and graph_count > 0:
            print(f"\033[33m[VectorDB] Detected empty vectordb with {graph_count} graph nodes; rebuilding...\033[0m")
            rebuilt_count = 0
            for skill_name, node in self.graph.nodes.items():
                # Skip task-specific skills (should not live in vectordb)
                if getattr(node, 'is_task_specific', False):
                    continue
                if node.description:
                    # === P0 fix: enrich metadata to support vector-layer filtering ===
                    self.vectordb.add_texts(
                        texts=[node.description],
                        metadatas=[{
                            "name": skill_name,
                            "is_deprecated": getattr(node, 'is_deprecated', False),
                            "is_task_specific": getattr(node, 'is_task_specific', False),
                        }],
                        ids=[skill_name]
                    )
                    # === end of fix ===
                    rebuilt_count += 1
            print(f"\033[32m[VectorDB] Rebuilt {rebuilt_count} skill(s) into vectordb from graph\033[0m")

        # Verify vector database synchronization
        # Note: task-specific skills and nodes without a description are not added to vectordb
        vectordb_final_count = self.vectordb._collection.count()
        expected_skill_names = set(
            name for name, node in self.graph.nodes.items()
            if not getattr(node, 'is_task_specific', False) and node.description
        )
        expected_vectordb_count = len(expected_skill_names)

        # Sync vectordb with graph (handle stale and missing entries)
        if vectordb_final_count != expected_vectordb_count:
            print(f"\033[33m[VectorDB] vectordb has {vectordb_final_count} entries; graph has {expected_vectordb_count} valid skill(s)\033[0m")
            print(f"\033[33m[VectorDB] Syncing...\033[0m")

            try:
                # Get all skill names currently in vectordb
                all_docs = self.vectordb._collection.get()
                vectordb_skill_names = set(all_docs.get('ids', []))

                # Step 1: delete stale entries (in vectordb but not in expected)
                stale_entries = vectordb_skill_names - expected_skill_names
                if stale_entries:
                    print(f"\033[33m[VectorDB] Found {len(stale_entries)} stale entries: {list(stale_entries)[:10]}{'...' if len(stale_entries) > 10 else ''}\033[0m")
                    self.vectordb._collection.delete(ids=list(stale_entries))
                    print(f"\033[32m[VectorDB] Deleted {len(stale_entries)} stale entries\033[0m")

                # Step 2: add missing entries (in expected but not in vectordb)
                missing_entries = expected_skill_names - vectordb_skill_names
                if missing_entries:
                    print(f"\033[33m[VectorDB] Found {len(missing_entries)} missing entries\033[0m")
                    for skill_name in missing_entries:
                        node = self.graph.nodes.get(skill_name)
                        if node and node.description:
                            self.vectordb.add_texts(
                                texts=[node.description],
                                metadatas=[{
                                    "name": skill_name,
                                    "is_deprecated": getattr(node, 'is_deprecated', False),
                                    "is_task_specific": getattr(node, 'is_task_specific', False),
                                }],
                                ids=[skill_name]
                            )
                    print(f"\033[32m[VectorDB] Added {len(missing_entries)} missing entries\033[0m")

                # Update counts
                vectordb_final_count = self.vectordb._collection.count()
            except Exception as e:
                print(f"\033[31m[VectorDB] Error during sync: {e}\033[0m")
                print(f"\033[33m[VectorDB] Suggest manually deleting the vectordb directory and re-running\033[0m")

        assert vectordb_final_count == expected_vectordb_count, (
            f"Skill Graph Manager's vectordb is not synced with graph.\n"
            f"There are {vectordb_final_count} skills in vectordb "
            f"but expected {expected_vectordb_count} (total {len(self.graph.nodes)} nodes, "
            f"excluding task-specific and description-less nodes).\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )
    
    # =====================================================
    # Skill Name/Behavior Validation Methods
    # =====================================================
    
    def _detect_hidden_behaviors_from_params(
        self, 
        skill_name: str, 
        parameters: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Detect hidden behaviors from parameter names.
        
        Some parameter names imply specific behaviors, for example:
        - fuelName, fuelType → smelt
        - craftingTable → craft
        - recipe → craft
        
        Args:
            skill_name: skill name
            parameters: parameter metadata for the skill
        
        Returns:
            List[str]: list of detected hidden behaviors
        """
        BEHAVIOR_INDICATOR_PARAMS = {
            "fuelname": "smelt",
            "fueltype": "smelt",
            "fuel": "smelt",
            "craftingtable": "craft",
            "recipe": "craft",
            "tabletype": "craft",
        }
        
        hidden_behaviors = []
        skill_name_lower = skill_name.lower()
        
        if not parameters:
            return []
        
        for param_name in parameters.keys():
            param_lower = param_name.lower()
            for indicator, behavior in BEHAVIOR_INDICATOR_PARAMS.items():
                if indicator in param_lower:
                    # Check whether the skill name already reflects this behavior
                    if behavior not in skill_name_lower:
                        if behavior not in hidden_behaviors:
                            hidden_behaviors.append(behavior)
        
        return hidden_behaviors
    
    def _validate_skill_name_matches_behavior(
        self, 
        skill_name: str, 
        code: str,
        parameters: Dict[str, Dict[str, Any]] = None
    ) -> Tuple[bool, str, List[str]]:
        """
        Verify whether the skill name matches the code behavior.
        
        Detect whether behaviors in the code exceed those implied by the name.
        For example: mineCopperOre but the code contains a smelt call.
        
        Args:
            skill_name: skill name
            code: skill code
            parameters: skill parameters (optional)
        
        Returns:
            Tuple[bool, str, List[str]]: 
                - whether they match
                - reason for the mismatch
                - list of detected hidden behaviors
        """
        # Define behavior keywords
        BEHAVIOR_KEYWORDS = {
            "mine": ["mineblock", "findblock", "dig", "break"],
            "smelt": ["smeltitem", "furnace", "fuel", "smelt"],
            "craft": ["craftitem", "recipe", "craft"],
            "equip": ["equipitem", "equip"],
            "place": ["placeitem", "place"],
        }
        
        code_lower = code.lower()
        skill_name_lower = skill_name.lower()
        
        # Detect behaviors in code
        detected_behaviors = []
        for behavior, keywords in BEHAVIOR_KEYWORDS.items():
            for keyword in keywords:
                if keyword in code_lower:
                    detected_behaviors.append(behavior)
                    break
        
        # Detect hidden behaviors from parameters
        if parameters:
            param_behaviors = self._detect_hidden_behaviors_from_params(skill_name, parameters)
            for behavior in param_behaviors:
                if behavior not in detected_behaviors:
                    detected_behaviors.append(behavior)
        
        # Check whether the skill name reflects these behaviors
        hidden_behaviors = []
        for behavior in detected_behaviors:
            if behavior not in skill_name_lower:
                hidden_behaviors.append(behavior)
        
        if hidden_behaviors:
            reason = f"Code contains [{', '.join(hidden_behaviors)}] but name '{skill_name}' doesn't reflect this"
            return False, reason, hidden_behaviors
        
        return True, "", []
    
    def _warn_misleading_skill_name(
        self,
        skill_name: str,
        code: str,
        parameters: Dict[str, Dict[str, Any]] = None
    ) -> None:
        """
        Emit a warning if the skill name may be misleading.
        
        Args:
            skill_name: skill name
            code: skill code
            parameters: skill parameters (optional)
        """
        is_valid, reason, hidden_behaviors = self._validate_skill_name_matches_behavior(
            skill_name, code, parameters
        )
        
        if not is_valid:
            print(f"\033[33m[Skill Naming Warning] {reason}\033[0m")
            print(f"\033[33m  Consider renaming '{skill_name}' to a name that includes {hidden_behaviors}\033[0m")
            print(f"\033[33m  e.g.: '{skill_name}And{'And'.join([b.capitalize() for b in hidden_behaviors])}'\033[0m")
    
    def _quarantine_corrupt_skill_code_on_load(self) -> Dict[str, Any]:
        """Detect skills whose persisted .js file has invalid JS syntax and
        quarantine them.

        Why: mineflayer/index.js bundles every file in skill_graph/code/ into a
        single ${programs} string and feeds it to eval(). One file with a
        top-level await/return/yield (e.g. a malformed Phase-2 optimizer
        output) makes EVERY iteration's eval throw SyntaxError — the bot
        appears frozen forever. See ckpt_psnv11_qwen3_fp8_r2r1's
        mineIronOreTask.js for a real instance.

        Action: move the corrupt file to skill_graph/quarantined/ and remove
        the node from the in-memory graph so subsequent saves clean it up.
        Task loop will see the missing skill and let the LLM regenerate.
        """
        ext = self._get_file_extension()
        code_dir = f"{self.ckpt_dir}/skill_graph/code"
        quarantine_dir = f"{self.ckpt_dir}/skill_graph/quarantined"

        quarantined: List[Dict[str, str]] = []
        if not os.path.isdir(code_dir):
            return {"quarantined_skills": quarantined}

        for fname in list(os.listdir(code_dir)):
            if not fname.endswith(ext):
                continue
            path = f"{code_dir}/{fname}"
            try:
                with open(path, "r") as f:
                    code = f.read()
            except OSError:
                continue
            if not code.strip():
                continue

            ok, err = validate_code_syntax(code)
            if ok:
                continue

            skill_name = fname[: -len(ext)]
            U.f_mkdir(quarantine_dir)
            dest = f"{quarantine_dir}/{fname}"
            try:
                shutil.move(path, dest)
            except Exception as move_err:
                print(f"\033[31m[Skill Quarantine] failed to move {path}: {move_err}\033[0m")
                continue

            # Drop from in-memory graph so it isn't re-saved.
            try:
                if self.graph.has_node(skill_name):
                    self.graph.remove_node(skill_name)
            except Exception as graph_err:
                print(f"\033[31m[Skill Quarantine] graph removal failed for "
                      f"'{skill_name}': {graph_err}\033[0m")

            # Best-effort cleanup of sibling files (metadata/desc/etc).
            for sibling_dir, sibling_ext in [
                ("description", ".txt"), ("metadata", ".json"),
                ("preconditions", ".json"), ("effects", ".json"),
            ]:
                sibling = f"{self.ckpt_dir}/skill_graph/{sibling_dir}/{skill_name}{sibling_ext}"
                if os.path.exists(sibling):
                    try:
                        sibling_dest = f"{quarantine_dir}/{skill_name}{sibling_ext}"
                        shutil.move(sibling, sibling_dest)
                    except Exception:
                        pass

            quarantined.append({"skill": skill_name, "error": err.replace("\n", " ")})

        return {"quarantined_skills": quarantined}

    def _save_to_checkpoint(self) -> None:
        """Save to checkpoint."""
        # Save the graph structure
        graph_file = f"{self.ckpt_dir}/skill_graph/graph/graph.json"
        U.dump_json(self.graph.to_dict(), graph_file)
        
        # Save each skill's code and description (for compatibility)
        for name, node in self.graph.nodes.items():
            # Skip task-specific skills (they are saved to apply_skills_for_task/)
            if getattr(node, 'is_task_specific', False):
                continue

            # Skip experimental skills (consistent with graph.to_dict())
            # experimental skills exist only in memory; persisted only after successful validation
            if getattr(node, 'is_experimental', False):
                continue

            # Secondary check: even if the flag is unset, judge by function name and code
            # Pass node so refactor-extracted skills are not misclassified as task-specific
            if self._is_task_specific_skill(name, node.code, node):
                print(f"\033[33m[Save] Skipping task-specific skill (secondary check): {name}\033[0m")
                continue
            
            # Log version save
            if node.versions:
                print(f"\033[36m[Save Checkpoint] Saving {len(node.versions)} versions for '{name}':\033[0m")
                for v in node.versions:
                    print(f"\033[36m  - Version {v.version}: {v.created_at} - {v.change_log[:50]}\033[0m")
            code_file = f"{self.ckpt_dir}/skill_graph/code/{name}{self._get_file_extension()}"
            desc_file = f"{self.ckpt_dir}/skill_graph/description/{name}.txt"
            U.dump_text(node.code, code_file)
            U.dump_text(node.description, desc_file)
            
            # Save preconditions
            preconditions_file = f"{self.ckpt_dir}/skill_graph/preconditions/{name}.json"
            preconditions_data = [
                {
                    "description": p.description,
                    "code": p.code,
                    "state_representation": p.state_representation if p.state_representation else {}
                }
                for p in node.preconditions
            ]
            U.dump_json(preconditions_data, preconditions_file)
            
            # Save effects
            effects_file = f"{self.ckpt_dir}/skill_graph/effects/{name}.json"
            effects_data = [
                {
                    "description": s.description,
                    "code": s.code,
                    "state_representation": s.state_representation if s.state_representation else {},
                    "is_primary": getattr(s, 'is_primary', False)
                }
                for s in node.expected_effects
            ]
            U.dump_json(effects_data, effects_file)
            
            # Save full skill metadata (graph properties, statistics, etc.)
            metadata_file = f"{self.ckpt_dir}/skill_graph/metadata/{name}.json"
            metadata = {
                # Basic info
                "name": node.name,
                "description": node.description,
                
                # Graph structural attributes
                "graph_info": {
                    "parents": node.parents,  # parent skills that call this skill
                    "children": node.children,  # child skills called by this skill
                    "in_degree": node.in_degree,  # in-degree (number of times called)
                    "out_degree": node.out_degree,  # out-degree (number of other skills called)
                },
                
                # Statistics
                "statistics": {
                    "total_executions": node.statistics.total_executions,  # total executions
                    "successful_executions": node.statistics.successful_executions,  # successful executions
                    "failed_executions": node.statistics.failed_executions,  # failed executions
                    "success_rate": node.statistics.success_rate,  # success rate
                    "execution_traces_count": len(node.statistics.execution_traces),  # number of execution traces
                },
                
                # Version info (save complete information including code and description)
                "versions": [
                    {
                        "version": v.version,
                        "created_at": v.created_at,
                        "change_log": v.change_log,
                        "code": v.code,  # include code
                        "description": v.description,  # include description
                        "preconditions": [
                            {
                                "description": p.description,
                                "code": p.code,
                                "state_representation": p.state_representation if p.state_representation else {}
                            }
                            for p in v.preconditions
                        ],
                        "effects": [
                            {
                                "description": e.description,
                                "code": e.code,
                                "state_representation": e.state_representation if e.state_representation else {}
                            }
                            for e in v.effects
                        ],
                        "parameters": v.parameters,
                        "value_function": v.value_function,
                        "statistics_snapshot": v.statistics_snapshot,
                        "update_source": v.update_source,
                        "update_reason": v.update_reason,
                        "optimization_id": v.optimization_id,
                        "used_feedbacks": v.used_feedbacks,
                        "backpropagation_info": v.backpropagation_info,
                        "affected_subgraph": v.affected_subgraph,
                        "sliding_window_size": v.sliding_window_size,
                        "execution_window": v.execution_window,
                    }
                    for v in node.versions
                ],
                "current_version": node.versions[-1].version if node.versions else "1.0.0",
                
                # Parameter info
                "parameters": node.parameters,
                
                # Optimization control
                "stop_gradient": node.stop_gradient,
                
                # Preconditions and state changes (brief)
                "preconditions_count": len(node.preconditions),
                "effects_count": len(node.expected_effects),
                
                # Gradient info (brief statistics)
                "gradients": {
                    "feedback_count": len(node.gradients.feedback),
                    "reflections_count": len(node.gradients.reflections),
                    "optimization_suggestions_count": len(node.gradients.optimization_suggestions),
                    "unused_feedback_count": len([f for f in node.gradients.feedback if not f.used_in_optimization]),
                },
                
                # Deprecation info
                "deprecation_info": {
                    "is_deprecated": node.is_deprecated,
                    "deprecation_reason": node.deprecation_reason,
                    "deprecated_for_task": node.deprecated_for_task,
                },
                
                # Refactor/coverage info (Part 7 fix: ensure refactor state is persisted)
                # [P1-1 fix] Add coverage_type field
                "refactor_info": {
                    "is_covered": node.is_covered,
                    "covered_by": node.covered_by,
                    "coverage_type": node.coverage_type.value if node.coverage_type else None,
                    "is_general_skill": node.is_general_skill,
                },
                
                # Value function info
                "value_function": node.value_function,
                "value_function_params": {
                    "lambda": node.value_function_lambda,
                    "alpha": node.value_function_alpha,
                    "beta": node.value_function_beta,
                },
            }
            U.dump_json(metadata, metadata_file)
        
        # Save statistics
        self.stats_tracker.save_stats()
    
    @property
    def programs(self) -> str:
        """
        Get all programs (compatible with the legacy interface).

        Returns:
            str: concatenation of all skill code and control primitives
        """
        programs = ""
        for skill_name, node in self.graph.nodes.items():
            # remove top-level Vec3 and mcData declarations from skill code
            # because globalDepsCode already declares these variables via var
            # In-function declarations are kept (no conflict)
            cleaned_code = self._remove_toplevel_global_deps(node.code)
            programs += f"{cleaned_code}\n\n"
        for primitives in self.control_primitives:
            programs += f"{primitives}\n\n"
        return programs

    # _remove_toplevel_global_deps - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    @property
    def skills(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the skills dictionary (compatible with the legacy interface).

        Returns:
            Dict[str, Dict]: mapping from skill name to skill info
        """
        return {
            name: {
                "code": node.code,
                "description": node.description,
            }
            for name, node in self.graph.nodes.items()
        }

    # =========================================================================
    # RefactorManagementMixin method group - refactor management
    # Methods moved to skillnet.agents.skill_graph._impl.mixins.refactor_mgmt
    # =========================================================================

    def _try_normalize_skill_name(self, func_name: str, func_code: str) -> Optional[str]:
        """
        Try to normalize the function name: if the function name contains a specific type but the parameter supports multiple types, normalize it.
        
        Improvements: check hardcoded values and parameter-type matching.
        
        Args:
            func_name: original function name
            func_code: function code
        
        Returns:
            Normalized function name, or None if normalization is not needed.
        """
        import re
        
        # Extract function signature
        func_match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', func_code)
        if not func_match:
            return None
        
        actual_func_name = func_match.group(1)
        if actual_func_name != func_name:
            return None  # function name mismatch, skip
        
        func_params_str = func_match.group(2)
        
        # Extract parameter names
        func_params = [p.strip().split('=')[0].strip() for p in func_params_str.split(',') if p.strip() and p.strip() != 'bot']
        param_names_str = ' '.join(func_params)
        
        # Define type-specific keywords (grouped by category)
        log_types = {'birch', 'oak', 'spruce', 'jungle', 'acacia', 'dark_oak', 'mangrove', 'cherry'}
        tool_types = {'wooden', 'iron', 'gold', 'diamond', 'netherite', 'stone', 'leather', 'chainmail'}
        specific_types = log_types | tool_types
        
        # Check whether function name contains a specific type
        func_name_lower = func_name.lower()
        found_types = [t for t in specific_types if t in func_name_lower]
        
        # Improvement: even if function name does not contain a specific type, also check for hardcoded specific-type values in code
        # Example: craftPickaxe has a generic name, but code hardcodes "wooden_pickaxe"
        # In this case, do not normalize (function name is already generic but implementation is specific)
        if not found_types:
            # Check whether the code has hardcoded specific-type values
            # If function name is generic (e.g. craftPickaxe) but code hardcodes a specific type (e.g. "wooden_pickaxe"),
            # and there is no parameter to change that hardcoded value, do not normalize
            hardcoded_type_patterns = [
                r'["\'](' + '|'.join(specific_types) + r')_[\w]+["\']',  # e.g. "wooden_pickaxe", "oak_log"
                r'itemsByName\[["\'](' + '|'.join(specific_types) + r')_[\w]+["\']',
                r'blocksByName\[["\'](' + '|'.join(specific_types) + r')_[\w]+["\']',
            ]
            
            hardcoded_in_code = []
            for pattern in hardcoded_type_patterns:
                matches = re.findall(pattern, func_code, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        hardcoded_in_code.extend([m.lower() for m in match if m])
                    elif match:
                        hardcoded_in_code.append(match.lower())
            
            if hardcoded_in_code:
                # Extract type prefixes from hardcoded values
                hardcoded_type_prefixes = set()
                for value in hardcoded_in_code:
                    for specific_type in specific_types:
                        if value.startswith(specific_type.lower() + '_'):
                            hardcoded_type_prefixes.add(specific_type.lower())
                            break
                
                # Check whether any parameter is used to change these hardcoded types
                param_used_to_change_type = False
                for param_name in func_params:
                    if not param_name or param_name == 'bot':
                        continue
                    param_name_lower = param_name.lower()
                    
                    # Check whether the parameter is used to change hardcoded types
                    type_change_patterns = [
                        rf'\b{re.escape(param_name)}\s*\+\s*["\']_[\w]+',
                        rf'["\'_][\w]*\s*\+\s*{re.escape(param_name)}',
                        rf'itemsByName\[{re.escape(param_name)}',
                        rf'blocksByName\[{re.escape(param_name)}',
                        rf'mcData\.itemsByName\[{re.escape(param_name)}',
                        rf'mcData\.blocksByName\[{re.escape(param_name)}',
                        rf'\b(pickaxe|log|plank|item|block|tool|material)Type\b.*{re.escape(param_name)}',
                    ]
                    
                    for pattern in type_change_patterns:
                        if re.search(pattern, func_code, re.IGNORECASE):
                            param_used_to_change_type = True
                            break
                    
                    if param_used_to_change_type:
                        break
                
                # If hardcoded values are detected but no parameter modifies them, do not normalize
                # Because the function name is already generic but the implementation is specific, normalizing would over-generalize
                if not param_used_to_change_type:
                    print(f"\033[33m[Name Normalization] Function '{func_name}' has a generic name, but code hardcodes specific types {hardcoded_type_prefixes} with no parameter to alter them; skipping normalization (prevents over-generalization)\033[0m")
                    return None
            
            # Function name lacks specific types and code has no hardcoded specific types; no normalization needed
            return None
        
        # Determine the type category in the function name (log type vs tool type vs other)
        found_log_type = any(t in log_types for t in found_types)
        found_tool_type = any(t in tool_types for t in found_types)
        
        # Check for hardcoded values in the code (related to types in the function name)
        # Example: craftWoodenPickaxe hardcodes "wooden_pickaxe"
        hardcoded_patterns = [
            r'["\'](' + '|'.join(found_types) + r')_[\w]+["\']',  # e.g. "wooden_pickaxe", "oak_log"
            r'itemsByName\[["\'](' + '|'.join(found_types) + r')_[\w]+["\']',  # itemsByName["wooden_pickaxe"]
            r'blocksByName\[["\'](' + '|'.join(found_types) + r')_[\w]+["\']',  # blocksByName["oak_log"]
        ]
        
        hardcoded_values = []
        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, func_code, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    hardcoded_values.extend([m.lower() for m in match if m])
                elif match:
                    hardcoded_values.append(match.lower())
        
        # If hardcoded values are detected, check whether parameters change them
        if hardcoded_values:
            # Extract type prefixes from hardcoded values (e.g. "wooden" from "wooden_pickaxe")
            hardcoded_type_prefixes = set()
            for value in hardcoded_values:
                for found_type in found_types:
                    if value.startswith(found_type.lower() + '_'):
                        hardcoded_type_prefixes.add(found_type.lower())
                        break
            
            # Check whether parameters are used to change these hardcoded types
            # Need to check whether parameters are used in code to replace hardcoded values
            param_used_to_change_type = False
            for param_name in func_params:
                if not param_name or param_name == 'bot':
                    continue
                param_name_lower = param_name.lower()
                
                # Check whether the parameter is used to change hardcoded types
                # E.g. pickaxeType parameter is used to replace "wooden_pickaxe"
                type_change_patterns = [
                    # Parameter is used directly to build a type name
                    rf'\b{re.escape(param_name)}\s*\+\s*["\']_[\w]+',  # pickaxeType + "_pickaxe"
                    rf'["\'_][\w]*\s*\+\s*{re.escape(param_name)}',  # "wooden_" + pickaxeType
                    # Parameter is used for array/object access (may contain a type)
                    rf'itemsByName\[{re.escape(param_name)}',  # itemsByName[pickaxeType]
                    rf'blocksByName\[{re.escape(param_name)}',  # blocksByName[logType]
                    rf'mcData\.itemsByName\[{re.escape(param_name)}',  # mcData.itemsByName[pickaxeType]
                    rf'mcData\.blocksByName\[{re.escape(param_name)}',  # mcData.blocksByName[logType]
                    # Parameter name itself implies a type (e.g. pickaxeType, logType)
                    rf'\b(pickaxe|log|plank|item|block|tool|material)Type\b.*{re.escape(param_name)}',
                ]
                
                for pattern in type_change_patterns:
                    if re.search(pattern, func_code, re.IGNORECASE):
                        param_used_to_change_type = True
                        break
                
                if param_used_to_change_type:
                    break
            
            # If hardcoded values are detected but no parameter changes them, do not normalize
            if not param_used_to_change_type:
                print(f"\033[33m[Name Normalization] Function '{func_name}' contains hardcoded type values {hardcoded_type_prefixes} but parameters do not change them; skipping normalization\033[0m")
                return None
        
        # Check for type-related parameters in the parameter list
        type_param_patterns = [
            r'\b(woodType|plankType|logType|itemType|blockType|materialType|type|material|pickaxeType|toolType)\b',
            r'\b(wood|plank|log|item|block|material|pickaxe|tool)\w*Type\b',
            r'\b(allowed\w*Types?|allowed\w*Logs?|allowed\w*Planks?)\b',  # supports allowedLogTypes, etc.
        ]
        
        has_type_param = False
        matching_params = []
        for pattern in type_param_patterns:
            # Check parameter names
            for param_name in func_params:
                if re.search(pattern, param_name, re.IGNORECASE):
                    has_type_param = True
                    matching_params.append(param_name)
            # Check usage in code
            if re.search(pattern, func_code, re.IGNORECASE):
                has_type_param = True
        
        if not has_type_param:
            return None  # no type parameter; keep original name
        
        # Key improvement: if code has type keywords but matching_params is empty (parameter name does not match the pattern),
        # check all parameters to see whether any is used to change the hardcoded type
        # E.g. craftWoodenPickaxe has logTypes in a code comment, but parameter preferredLog does not match the logType pattern
        # In this case, check whether preferredLog is used to change the hardcoded pickaxe type
        if has_type_param and not matching_params:
            # Check all parameters for ones that change the hardcoded type
            # If function name contains a tool type, but parameter is log/plank type (e.g. preferredLog), they do not match
            if found_tool_type:
                # Check for log/plank-related parameters
                log_plank_param_patterns = [
                    r'\b(log|plank|wood)Type\b',
                    r'\bpreferredLog\b',
                    r'\bpreferredPlank\b',
                    r'\ballowedLogs?\b',
                ]
                has_log_plank_param = False
                for pattern in log_plank_param_patterns:
                    for param_name in func_params:
                        if not param_name or param_name == 'bot':
                            continue
                        if re.search(pattern, param_name, re.IGNORECASE):
                            has_log_plank_param = True
                            break
                    if has_log_plank_param:
                        break
                
                # If log/plank parameters are detected, check whether they change the hardcoded pickaxe type
                if has_log_plank_param:
                    # Check whether the parameter changes the hardcoded pickaxe type
                    param_used_to_change_pickaxe_type = False
                    for param_name in func_params:
                        if not param_name or param_name == 'bot':
                            continue
                        # Check whether the parameter builds the pickaxe type name
                        pickaxe_type_change_patterns = [
                            rf'\b{re.escape(param_name)}\s*\+\s*["\']_pickaxe',  # param + "_pickaxe"
                            rf'["\']wooden_["\']\s*\+\s*{re.escape(param_name)}',  # "wooden_" + param
                            rf'itemsByName\[{re.escape(param_name)}\s*\+\s*["\']_pickaxe',  # itemsByName[param + "_pickaxe"]
                            rf'mcData\.itemsByName\[{re.escape(param_name)}\s*\+\s*["\']_pickaxe',
                        ]
                        for pattern in pickaxe_type_change_patterns:
                            if re.search(pattern, func_code, re.IGNORECASE):
                                param_used_to_change_pickaxe_type = True
                                break
                        if param_used_to_change_pickaxe_type:
                            break
                    
                    # If the parameter does not change the pickaxe type, do not normalize
                    if not param_used_to_change_pickaxe_type:
                        print(f"\033[33m[Name Normalization] Function '{func_name}' contains tool type '{found_types}' but parameter is log/plank type and does not change pickaxe type; type mismatch, skipping normalization\033[0m")
                        return None
        
        # Key improvement: verify that parameter types match the types in the function name
        # E.g. "wooden" in craftWoodenPickaxe is a pickaxe type
        # preferredLog is a log-type parameter, mismatched, should not normalize
        if found_tool_type and matching_params:
            # Check whether parameters relate to tool types (e.g. pickaxeType, toolType)
            tool_param_patterns = [
                r'\b(pickaxe|tool|sword|axe|shovel|hoe|helmet|chestplate|leggings|boots)Type\b',
            ]
            has_tool_type_param = False
            for pattern in tool_param_patterns:
                for param_name in matching_params:
                    if re.search(pattern, param_name, re.IGNORECASE):
                        has_tool_type_param = True
                        break
                if has_tool_type_param:
                    break
            
            # If function name has a tool type but parameter is a log type (e.g. preferredLog), they do not match
            if not has_tool_type_param:
                # Check for log-related parameters (e.g. preferredLog, logType)
                log_param_patterns = [
                    r'\b(log|plank|wood)Type\b',
                    r'\bpreferredLog\b',
                ]
                has_log_param = False
                for pattern in log_param_patterns:
                    for param_name in matching_params:
                        if re.search(pattern, param_name, re.IGNORECASE):
                            has_log_param = True
                            break
                    if has_log_param:
                        break
                
                if has_log_param:
                    print(f"\033[33m[Name Normalization] Function '{func_name}' contains tool type '{found_types}' but parameter is a log type; type mismatch, skipping normalization\033[0m")
                    return None
        
        if found_log_type and matching_params:
            # Check whether parameters are related to log types
            log_param_patterns = [
                r'\b(log|plank|wood)Type\b',
                r'\bpreferredLog\b',
                r'\ballowedLogs?\b',
            ]
            has_log_type_param = False
            for pattern in log_param_patterns:
                for param_name in matching_params:
                    if re.search(pattern, param_name, re.IGNORECASE):
                        has_log_type_param = True
                        break
                if has_log_type_param:
                    break
            
            # If function name contains a log type but parameter is a tool type, they do not match
            if not has_log_type_param:
                tool_param_patterns = [
                    r'\b(pickaxe|tool|sword|axe)Type\b',
                ]
                has_tool_param = False
                for pattern in tool_param_patterns:
                    for param_name in matching_params:
                        if re.search(pattern, param_name, re.IGNORECASE):
                            has_tool_param = True
                            break
                    if has_tool_param:
                        break
                
                if has_tool_param:
                    print(f"\033[33m[Name Normalization] Function '{func_name}' contains log type '{found_types}' but parameter is a tool type; type mismatch, skipping normalization\033[0m")
                    return None
        
        # Generate the generic name: strip specific-type keywords
        normalized_name = func_name
        for specific_type in found_types:
            # Strip type keywords (preserving camelCase)
            patterns = [
                rf'{specific_type}(\w+)',  # birchLog -> Log
                rf'(\w+){specific_type}',  # craftBirch -> craft
                rf'{specific_type}',        # birch -> remove
            ]
            
            for pattern in patterns:
                if re.search(pattern, normalized_name, re.IGNORECASE):
                    if '(' in pattern:
                        normalized_name = re.sub(pattern, r'\1', normalized_name, flags=re.IGNORECASE)
                    else:
                        normalized_name = re.sub(pattern, '', normalized_name, flags=re.IGNORECASE)
                    
                    # If replacement leaves an empty string, use a generic suffix
                    if not normalized_name or normalized_name in ['_', '-', '']:
                        if func_name_lower.startswith('craft'):
                            normalized_name = 'craftPlanks' if 'plank' in func_name_lower else 'craftItem'
                        elif func_name_lower.startswith('mine'):
                            normalized_name = 'mineLogs' if 'log' in func_name_lower else 'mineBlock'
                        elif func_name_lower.startswith('place'):
                            normalized_name = 'placeBlock'
                        else:
                            normalized_name = 'genericAction'
                    break
        
        # Clean up the name
        if not normalized_name or normalized_name is None:
            return None
        
        normalized_name = re.sub(r'[_-]+', '', normalized_name)
        
        if not normalized_name or normalized_name is None:
            return None
        
        # Ensure the first character is lowercase
        if normalized_name:
            normalized_name = normalized_name[0].lower() + normalized_name[1:] if len(normalized_name) > 1 else normalized_name.lower()
        
        # If the normalized name is too short or invalid, keep the original
        if not normalized_name or len(normalized_name) < 3 or normalized_name == func_name.lower():
            return None
        
        return normalized_name
    
    def _is_task_specific_skill(self, func_name: str, func_code: str, node: "SkillNode" = None) -> bool:
        """
        Check whether this is a task-specific skill.

        Delegates to the code_analysis module, with an added refactor check:
        if the skill was extracted via refactor (update_source starts with "refactor:"),
        it is never task-specific because those are general sub-skills.

        Args:
            func_name: function name
            func_code: function code
            node: optional; if provided, check update_source

        Returns:
            bool: whether it is a task-specific skill
        """
        # Check update_source - skills extracted via refactor are not task-specific
        if node:
            update_source = getattr(node, 'update_source', '') or ''
            if update_source.startswith("refactor:"):
                return False
            # Already-validated skills should not be intercepted by the secondary check
            # Validated means the skill ran successfully in execution and must be persisted to disk
            # Otherwise the bot-only-param heuristic would mis-classify skills like craftWoodenPickaxe(bot)
            if getattr(node, 'is_verified', False):
                return False

        return is_task_specific_skill(func_name, func_code)
    
    def is_task_specific_skill_in_storage(self, skill_name: str) -> bool:
        """
        Check whether a skill is stored in the apply_skills_for_task directory as task-specific.
        
        This method checks whether the skill has been saved to the task-specific directory,
        rather than judging from code traits whether it is task-specific.
        
        Args:
            skill_name: skill name
        
        Returns:
            bool: whether the skill is in the task-specific directory
        """
        import os
        task_specific_path = os.path.join(
            self.ckpt_dir, "skill_graph", "apply_skills_for_task", f"{skill_name}{self._get_file_extension()}"
        )
        return os.path.exists(task_specific_path)
    
    # get_skill_code - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    # update_skill_code - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    # _update_task_specific_skill - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    def _is_skill_composition(self, func_code: str) -> bool:
        """
        Check whether the code is just a wrapper that calls existing skills (skill composition).
        
        Traits of skill composition:
        1. The function body is mostly other skill calls
        2. No complex logic (loops, conditionals, etc.)
        3. The called skills already exist in the skill graph
        
        Args:
            func_code: function code
        
        Returns:
            bool: whether it is a skill composition
        """
        import re
        
        # Extract the function body
        func_match = re.search(r'async\s+function\s+\w+\s*\([^)]*\)\s*\{(.*)\}', func_code, re.DOTALL)
        if not func_match:
            return False
        
        func_body = func_match.group(1).strip()
        
        # If the function body is too long, it is probably not a simple composition
        if len(func_body) > 2000:
            return False
        
        # Check whether it mainly consists of await calls
        await_calls = re.findall(r'await\s+(\w+)\s*\(', func_body)
        
        if not await_calls:
            return False
        
        # Check whether the called skills exist in the skill graph
        existing_skills_called = 0
        for called_func in await_calls:
            if called_func in self.graph.nodes:
                existing_skills_called += 1
        
        # If most await calls go to existing skills, treat as a skill composition
        if existing_skills_called > 0 and existing_skills_called >= len(await_calls) * 0.5:
            return True
        
        # Check whether the function body is just simple calls (no complex logic)
        # Count statements
        statements = [s.strip() for s in func_body.split(';') if s.strip()]
        
        # Few statements and mostly await calls means it is a skill composition
        if len(statements) <= 5 and len(await_calls) >= 1:
            # Check for complex logic (for/while loops, nested if, etc.)
            complex_patterns = [
                r'\bfor\s*\(',
                r'\bwhile\s*\(',
                r'\bswitch\s*\(',
            ]
            for pattern in complex_patterns:
                if re.search(pattern, func_body):
                    return False  # has complex logic; not a simple composition
            
            return True
        
        return False
    
    def _save_task_specific_skill(self, skill_name: str, skill_code: str, task: str, info: Dict[str, Any]) -> None:
        """
        Save a task-specific skill to the apply_skills_for_task directory.
        
        These skills are generated by the graph planner reusing existing skills; reuse value is low,
        so they are not added to the skill graph - only saved to apply_skills_for_task for recording and traceback.
        
        Args:
            skill_name: skill name
            skill_code: skill code
            task: task description
            info: skill information
        """
        import os
        import re
        
        # Syntax validation (warning mode - allow saving but mark the issue)
        is_syntax_valid, syntax_error = validate_code_syntax(skill_code)
        if not is_syntax_valid:
            print(f"\033[33m[Task-Specific] Warning: Skill '{skill_name}' has syntax issues: {syntax_error}\033[0m")
            info["has_syntax_error"] = True
            info["syntax_error_message"] = syntax_error
        
        # Check line lengths
        long_lines = check_line_length(skill_code)
        if long_lines:
            print(f"\033[33m[Task-Specific] Warning: Skill '{skill_name}' has overlong lines:\033[0m")
            for warning in long_lines[:3]:  # show at most 3 lines
                print(f"\033[33m[Task-Specific]   {warning}\033[0m")
        
        # Create the apply_skills_for_task directory
        task_skills_dir = f"{self.ckpt_dir}/skill_graph/apply_skills_for_task"
        os.makedirs(task_skills_dir, exist_ok=True)
        
        # Save the code
        code_file = f"{task_skills_dir}/{skill_name}{self._get_file_extension()}"
        U.dump_text(skill_code, code_file)
        
        # Extract called skills
        await_calls = re.findall(r'await\s+(\w+)\s*\(', skill_code)
        called_skills = [call for call in await_calls if call in self.graph.nodes]
        
        # Save metadata
        metadata_file = f"{task_skills_dir}/{skill_name}.json"
        metadata = {
            "name": skill_name,
            "task": task,
            "description": info.get("description", ""),
            "is_deprecated": True,
            "deprecation_reason": "Graph planner generated task-specific skill composition, low reuse value, for record and backtracking only",
            "deprecated_for_task": task,
            "from_graph_planner": True,
            "called_skills": called_skills,  # record which existing skills are called
            "is_task_specific": self._is_task_specific_skill(skill_name, skill_code),
            "is_skill_composition": self._is_skill_composition(skill_code),
            "has_syntax_error": info.get("has_syntax_error", False),  # flag for syntax issues
            "syntax_error_message": info.get("syntax_error_message", ""),
            "created_at": datetime.now().isoformat(),
        }
        U.dump_json(metadata, metadata_file)
        
        print(f"\033[33m[Skill Graph] Task-specific skill '{skill_name}' saved to {task_skills_dir}\033[0m")
        if called_skills:
            print(f"\033[33m[Skill Graph]   called existing skills: {', '.join(called_skills)}\033[0m")

    # =========================================================================
    # SkillAdditionMixin method group - skill addition
    # See SkillAdditionMixin class doc for details
    # Warning: add_new_skill is a core method (~635 lines), highly coupled
    # =========================================================================

    def pre_register_skill(
        self, name: str, code: str, task: str = None
    ) -> Tuple[str, str]:
        """
        Pre-register the skill before execution to support correct Skill Tracking.

        Core method for Eager Skill Registration: before env.step() runs the code,
        pre-register newly-defined skills into the graph to ensure:
        1. extract_called_skills() returns these skills correctly
        2. record_execution() records execution info correctly
        3. early-return bugs are not triggered

        Differences from add_new_skill():
        - pre_register_skill: lightweight, only normalizes the name and creates a temporary node
        - add_new_skill: full version, including LLM calls, metadata extraction, persistence, etc.

        Args:
            name: original function name
            code: full function code
            task: current task (optional)

        Returns:
            Tuple[str, str]: (final name, updated code)
                - If renamed, the function name in the returned code is updated
                - The caller should use the returned code to run env.step()
        """
        original_name = name

        # 1. Name normalization (must be exactly aligned with add_new_skill)
        # 1.1 Resolve naming conflict with control primitives
        name, code, was_renamed = resolve_primitive_name_conflict(name, code)
        if was_renamed:
            print(f"\033[36m[Pre-Register] Original name '{original_name}' conflicts with control primitive; renamed to '{name}'\033[0m")

        # 1.2 Semantic naming normalization
        if self.auto_semantic_rename:
            old_name = name
            name, code, was_semantic_renamed = self._check_and_apply_semantic_rename(name, code)
            if was_semantic_renamed:
                print(f"\033[36m[Pre-Register] Semantic rename: '{old_name}' → '{name}'\033[0m")

        # 1.3 Type-parameter normalization
        normalized = self._try_normalize_skill_name(name, code)
        if normalized and normalized != name:
            # Update the function name in code
            code = re.sub(
                rf'async\s+function\s+{re.escape(name)}\s*\(',
                f"async function {normalized}(",
                code
            )
            code = re.sub(
                rf'\b{re.escape(name)}\s*\(',
                f"{normalized}(",
                code
            )
            print(f"\033[36m[Pre-Register] Type normalization: '{name}' → '{normalized}'\033[0m")
            name = normalized

        # 2. If the node already exists, use the unified interface to update the code
        if self.graph.has_node(name):
            node = self.graph.get_node(name)

            # Use the unified interface to update code
            result = node.update_code(code, CodeSource.LLM)

            if result.need_verification:
                print(f"\033[36m[Pre-Register] '{name}' is a stable skill; code pending validation\033[0m")
            elif result.redirect_to:
                print(f"\033[33m[Pre-Register] ⚠️ Wrapper protection: suggest modifying {result.redirect_to}\033[0m")
            elif result.reason == "wrapper_architecture_violation":
                print(f"\033[33m[Pre-Register] ⚠️ Wrapper architecture violation: {result.message}\033[0m")
            elif result.success:
                if result.reason == "code_unchanged":
                    print(f"\033[36m[Pre-Register] '{name}' code unchanged\033[0m")
                else:
                    print(f"\033[36m[Pre-Register] '{name}' code updated\033[0m")
                    # Re-derive call-graph edges from the updated code; otherwise the
                    # persisted children stay stale and credit assignment blames the
                    # wrong sub-skill.
                    self._update_graph_from_code(name)

            # Return code to use for execution (may be pending code)
            return name, node.get_execution_code()

        # 3. Create a lightweight temporary node
        node = SkillNode(
            name=name,
            code=code,
            description=f"Pending skill for task: {task}" if task else f"Pending skill: {name}",
        )

        # Set flags
        node.is_experimental = True
        node.created_for_task = task
        node.experimental_task = task

        # Detect composition wrappers: name equals the task's normalized name
        # Composition wrappers are task-specific (encode specific counts and items) and should not be returned by EffectMatcher
        if task:
            from skillnet.agents.planning.code_generator import normalize_task_name
            normalized_task = normalize_task_name(task)
            if name == normalized_task:
                node.is_task_specific = True
                print(f"\033[33m[Pre-Register] Marking '{name}' as task-specific (composition wrapper)\033[0m")

        # Set value-function parameters (inherited from self)
        node.value_function_lambda = self.value_function_lambda
        node.value_function_alpha = self.value_function_alpha
        node.value_function_beta = self.value_function_beta

        # 4. Add to graph (do not save to checkpoint, do not add to vectordb)
        self.graph.add_node(node)  # P0 fix: pass node only; name is taken from node.name
        self._update_graph_from_code(name)  # P1 fix: extract call relations

        # Early effects extraction - prefer code inference (LLM), fall back to rule-based extraction (provisional)
        if task and len(node.expected_effects) == 0:
            extracted_from_code = False
            try:
                extracted_effects, warnings = self._extract_effects(
                    code, node.description or f"Skill: {name}",
                    task=task, skill_name=name,
                )
                if extracted_effects:
                    node.expected_effects.extend(extracted_effects)
                    extracted_from_code = True
                    print(f"\033[36m[Pre-Register] Inferred {len(extracted_effects)} effects from code\033[0m")
            except Exception:
                pass

            if not extracted_from_code:
                try:
                    from skillnet.agents.planning.effect_matcher import EffectMatcher
                    from skillnet.agents.skill_graph.models.precondition import SkillEffect
                    task_effects = EffectMatcher._extract_target_effects_rules(task)
                    for effect_dict in task_effects:
                        node.expected_effects.append(SkillEffect(
                            description=f"{effect_dict.get('operation', 'add').capitalize()}s "
                                       f"{effect_dict.get('count', 1)} {effect_dict.get('item', 'unknown')}",
                            code="",  # provisional
                            state_representation=effect_dict, is_primary=True,
                        ))
                    if node.expected_effects:
                        print(f"\033[33m[Pre-Register] Rule-based fallback extracted {len(node.expected_effects)} provisional effects\033[0m")
                except Exception:
                    pass

        print(f"\033[32m[Pre-Register] Pre-registered skill '{name}' in graph (experimental=True)\033[0m")
        if name != original_name:
            print(f"\033[32m[Pre-Register]   original name: '{original_name}' → final name: '{name}'\033[0m")

        return name, code

    def add_new_skill(self, info: Dict[str, Any]) -> Optional[str]:
        """
        Add a new skill (compatible with the legacy interface; also updates the graph).

        Args:
            info: dict containing task, program_name, program_code, etc.

        Returns:
            Optional[str]: skill name on success, None on failure or skip
        """
        if info.get("task", "").startswith("Deposit useless items into the chest at"):
            # No need to reuse the deposit skill
            return
        
        program_name = info["program_name"]
        program_code = info["program_code"]
        task = info.get("task", "")

        # B2: Skill cap enforcement — skip new skills when at capacity
        if (self.max_skills is not None
                and not self.graph.has_node(program_name)
                and self.skill_count >= self.max_skills):
            print(f"\033[33m[Skill Graph] Skill cap reached ({self.skill_count}/{self.max_skills}), "
                  f"skipping new skill '{program_name}'\033[0m")
            return None

        # ===== Check and resolve naming conflicts with control primitives =====
        program_name, program_code, was_renamed = resolve_primitive_name_conflict(program_name, program_code)
        if was_renamed:
            info["program_name"] = program_name
            info["program_code"] = program_code
            info["was_renamed_from_primitive_conflict"] = True
        
        # ===== Check and resolve semantic naming inconsistencies =====
        if self.auto_semantic_rename:
            old_name = program_name
            program_name, program_code, was_semantic_renamed = self._check_and_apply_semantic_rename(
                program_name, program_code
            )
            if was_semantic_renamed:
                info["program_name"] = program_name
                info["program_code"] = program_code
                info["was_renamed_for_semantic_consistency"] = True
                info["original_name_before_semantic_rename"] = old_name
        
        # Validate that brackets are balanced
        is_bracket_valid, bracket_error, bracket_details = validate_code_brackets(program_code)
        if not is_bracket_valid:
            print(f"\033[31m[Skill Graph] REJECTED: Skill '{program_name}' unbalanced brackets: {bracket_error}\033[0m")
            print(f"\033[31m[Skill Graph]   bracket details: parentheses({bracket_details['parentheses']['open']}/{bracket_details['parentheses']['close']}), "
                  f"square brackets({bracket_details['brackets']['open']}/{bracket_details['brackets']['close']}), "
                  f"curly braces({bracket_details['braces']['open']}/{bracket_details['braces']['close']})\033[0m")
            print(f"\033[31m[Skill Graph]   Reason: unbalanced brackets cause the ${{programs}} bundle to fail to parse, breaking all skill execution. Skill not added.\033[0m")
            return None

        # Full syntax validation (Babel) - reject any code Babel cannot parse
        # Critical: skillnet/domains/minecraft/action_space/env/mineflayer/index.js concatenates all skill_graph/code/*.js into
        # ${programs} and feeds eval(); top-level await/return/yield/break in any one file is
        # "script-context illegal" syntax that makes every iteration's eval fail (cascading).
        # Even if Babel is unavailable (no JS bridge), validate_code_syntax degrades to
        # bracket check - intercepted above; just preserve reject semantics here.
        is_syntax_valid, syntax_error = validate_code_syntax(program_code)
        if not is_syntax_valid:
            print(f"\033[31m[Skill Graph] REJECTED: Skill '{program_name}' syntax error: {syntax_error[:200]}\033[0m")
            print(f"\033[31m[Skill Graph]   Reason: invalid syntax saved into skill_graph/code enters the ${{programs}} bundle, "
                  f"causing every subsequent eval(fullCode) to throw SyntaxError due to top-level await/return/yield - "
                  f"the bot stalls permanently. Skill not added; the caller (action agent) can regenerate via LLM retry.\033[0m")
            return None

        # Line-length warning (non-blocking)
        long_lines = check_line_length(program_code)
        if long_lines:
            print(f"\033[33m[Skill Graph] Warning: Skill '{program_name}' has overlong lines:\033[0m")
            for warning in long_lines[:3]:
                print(f"\033[33m[Skill Graph]   {warning}\033[0m")
        
        # Detect and fix function-name conflicts (a local function defined with the same name as an existing skill)
        # This is a common LLM code-gen issue: the LLM redefines a local function with the same name as an existing skill,
        # so the local function shadows the external skill and calls use the local version instead
        fixed_code, conflicts, was_fixed = self._detect_and_fix_function_conflicts(program_code, program_name)
        if was_fixed:
            program_code = fixed_code
            info["program_code"] = fixed_code
            info["had_function_conflicts"] = True
            info["conflicting_functions"] = conflicts
            print(f"\033[32m[Skill Graph] Auto-fixed function-name conflict; the code will now call the external skill correctly\033[0m")
        
        # Note: no longer check for functional duplication here
        # Unified handling now happens in the failure-handling flow in psn.py; always save the new skill
        # Let the backpropagation mechanism automatically locate the problem (new skill / sub-skill / both)
        
        # Check whether this skill came from the graph planner
        # Code emitted by the graph planner is usually a wrapper that calls existing skills with low reuse value
        is_from_graph_planner = info.get("from_graph_planner", False) or info.get("planner_mode") == "graph"

        # Check whether this is a skill extracted via refactor (those should not be classified as task-specific)
        update_source = info.get("update_source", "") or ""
        if update_source.startswith("refactor:"):
            # refactor-extracted skills are general sub-skills, not task-specific
            is_task_specific = False
        else:
            existing_node = self.graph.get_node(program_name) if self.graph.has_node(program_name) else None
            is_task_specific = self._is_task_specific_skill(program_name, program_code, node=existing_node)

        is_skill_composition = self._is_skill_composition(program_code)  # New: check whether it is just a wrapper calling existing skills
        
        # Flag variable: identifies a task-specific skill (added to memory but not persisted)
        task_specific_flag = False
        
        # ===== Task-specific skills from any source are saved to apply_skills_for_task =====
        # Fix: previously only graph-planner-sourced skills were checked for task-specific, so task-specific skills from other sources were wrongly added to the skill graph
        if is_task_specific:
            print(f"\033[33m[Skill Graph] Task-specific skill '{program_name}' will be saved to apply_skills_for_task\033[0m")
            print(f"\033[33m[Skill Graph]   traits: is_task_specific={is_task_specific}, from_graph_planner={is_from_graph_planner}\033[0m")
            self._save_task_specific_skill(program_name, program_code, task, info)
            task_specific_flag = True  # mark as task-specific; continue adding to in-memory graph
        
        # Graph-planner-generated skill compositions are handled separately
        if is_from_graph_planner and not task_specific_flag:
            # Graph-planner-generated skill compositions are all task-specific
            # They are just wrappers calling existing skills and should not participate in retrieve or refactor
            if is_skill_composition:
                # Save to the apply_skills_for_task folder
                print(f"\033[33m[Skill Graph] Skill '{program_name}' is a graph-planner-generated skill composition; "
                      f"will be saved to the apply_skills_for_task folder\033[0m")
                self._save_task_specific_skill(program_name, program_code, task, info)
                task_specific_flag = True  # mark as task-specific; continue adding to in-memory graph
        
        # Attempt normalization: if the function name contains a specific type but the parameter supports multiple types, it should be normalized
        # If already normalized in ParameterizedActionAgent, skip redundant normalization
        already_normalized = info.get("already_normalized", False)
        if already_normalized:
            print(f"\033[36m[Name Normalization] Function name '{program_name}' already normalized in ParameterizedActionAgent; skipping SkillGraphManager normalize\033[0m")
            normalized_name = None  # skip normalize
        else:
            normalized_name = self._try_normalize_skill_name(program_name, program_code)
        if normalized_name and normalized_name != program_name:
            # Check whether the function body calls a function with the same name as the normalized result (avoid recursive calls)
            # Note: re is already imported at module level, don't re-import here
            func_match = re.search(
                rf'async\s+function\s+{re.escape(program_name)}\s*\([^)]*\)\s*\{{(.*?)\}}',
                program_code,
                re.DOTALL
            )
            
            if func_match:
                func_body = func_match.group(1)
                # Check whether the function body calls the normalized name (excluding the function definition itself)
                calls_normalized = re.search(rf'\b{re.escape(normalized_name)}\s*\(', func_body)
                
                if calls_normalized:
                    # If the function body calls the normalized name, skip normalization to avoid recursion
                    print(f"\033[33m[Name Normalization] Warning: function '{program_name}' body calls '{normalized_name}'; skipping normalization to avoid recursion\033[0m")
                    normalized_name = program_name
            
            if normalized_name != program_name:
                print(f"\033[36m[Name Normalization] Function name normalized from '{program_name}' to '{normalized_name}' (parameterized function supports multiple types)\033[0m")
                # Update function name and code
                program_code = re.sub(
                    rf'async\s+function\s+{re.escape(program_name)}\s*\(',
                    f"async function {normalized_name}(",
                    program_code
                )
                # Update function calls (only replace calls to the original name; do not replace calls to the normalized name)
                program_code = re.sub(
                    rf'\b{re.escape(program_name)}\s*\(',
                    f"{normalized_name}(",
                    program_code
                )
                program_name = normalized_name
                info["program_name"] = normalized_name
                info["program_code"] = program_code
        
        # Debug log: record the call
        import traceback
        import sys
        caller_info = traceback.extract_stack()[-2] if len(traceback.extract_stack()) > 1 else None
        caller_file = caller_info.filename.split('/')[-1] if caller_info else "unknown"
        caller_line = caller_info.lineno if caller_info else 0
        print(f"\033[35m[DEBUG] add_new_skill called for '{program_name}' from {caller_file}:{caller_line}\033[0m")
        
        # Generate description
        # Get task info (if available)
        task = info.get("task", None)
        skill_description = self.generate_skill_description(program_name, program_code, task=task)
        print(
            f"\033[33mSkill Graph Manager generated description for {program_name}:\n{skill_description}\033[0m"
        )
        
        # Verify whether the skill name matches code behavior (Part 1 & Part 9)
        # Emit a warning if the name may be misleading (e.g. mineCopperOre but code contains smelt logic)
        self._warn_misleading_skill_name(program_name, program_code)
        
        # Check whether it already exists
        if self.graph.has_node(program_name):
            node = self.graph.get_node(program_name)
            latest_version = node.get_latest_version()
            current_code = node.code

            # Handle pending code (if any)
            # This handles the _pending_code scenario set via pre_register_skill
            if node.has_pending_code():
                # For add_new_skill calls, we assume the success path
                # (failure cases are handled separately in psn.py)
                success = info.get("success", True)
                result = node.confirm_pending(success=success)
                if result.adopted:
                    print(f"\033[32m[add_new_skill] Pending code adopted: {program_name}\033[0m")
                    # Update current_code to the adopted code
                    current_code = node.code
                elif result.message:
                    print(f"\033[33m[add_new_skill] {result.message}\033[0m")

            # [Fix] Sync task-specific flag onto existing nodes
            # Fixes: temporary nodes created by record_execution lack the is_task_specific flag,
            # which causes incorrect vectordb operations and retrieval filtering downstream
            if task_specific_flag and not getattr(node, 'is_task_specific', False):
                node.is_task_specific = True
                print(f"\033[33m[Task-Specific] Syncing task-specific flag onto existing node '{program_name}'\033[0m")

            # Phase 3: sync experimental-related fields onto existing nodes
            # Ensures temporary nodes created by record_execution have complete experimental info
            if not hasattr(node, 'created_for_task') or node.created_for_task is None:
                node.created_for_task = task
            if not hasattr(node, 'experimental_task') or node.experimental_task is None:
                node.experimental_task = task

            # Debug log: record node state
            print(f"\033[35m[DEBUG] Skill '{program_name}' already exists. Current state:\033[0m")
            print(f"\033[35m[DEBUG]   - preconditions count: {len(node.preconditions)}\033[0m")
            print(f"\033[35m[DEBUG]   - expected_effects count: {len(node.expected_effects)}\033[0m")
            print(f"\033[35m[DEBUG]   - parameters: {len(node.parameters) if node.parameters else 0}\033[0m")
            print(f"\033[35m[DEBUG]   - versions count: {len(node.versions)}\033[0m")
            print(f"\033[35m[DEBUG]   - is_deprecated: {node.is_deprecated}\033[0m")
            print(f"\033[35m[DEBUG]   - total_executions: {node.statistics.total_executions}\033[0m")
            
            # Check whether this is a temporary node (created by record_execution, awaiting formal save)
            is_temporary_node = (
                node.statistics.total_executions > 0 and  # already has execution records
                len(node.versions) == 1 and  # only one initial version
                node.versions[0].change_log == "Initial version"  # it is the initial version
            )
            
            if is_temporary_node:
                if task_specific_flag:
                    print(f"\033[36m[Skill Update] Detected temporary node '{program_name}' ({node.statistics.total_executions} execution records); updating in-memory state (task-specific, not persisted to graph.json)\033[0m")
                else:
                    print(f"\033[36m[Skill Update] Detected temporary node '{program_name}' ({node.statistics.total_executions} execution records); will be promoted to a formal node\033[0m")
            
            # Phase 9: deprecated skills are now deleted; no reactivation case
            # But still need to handle skills with consecutive_failures > 0 that have not reached the deprecation threshold
            if node.consecutive_failures > 0:
                print(f"\033[36m[Skill Recovery] Detected skill '{program_name}' with failure history:\033[0m")
                print(f"\033[36m[Skill Recovery]   consecutive failures: {node.consecutive_failures}\033[0m")

                # Reset failure counter (calling add_new_skill means new code was generated)
                recovery_result = node.mark_execution_success(task=task)

                if recovery_result.get("consecutive_failures_reset"):
                    print(f"\033[32m[Skill Recovery] Reset consecutive failure count; skill '{program_name}' restored to normal state\033[0m")

                # Check whether to trigger a delayed refactor
                if recovery_result.get("should_trigger_refactor"):
                    print(f"\033[36m[Delayed Refactor] Skill '{program_name}' validated successfully; triggering delayed refactor\033[0m")
                    self._trigger_delayed_refactor(program_name)

            # Improvements 2 and 3: detect code changes (pass parameter metadata for semantic-similarity checks)
            old_param_metadata = node.parameters if node.parameters else None
            # Temporarily extract the parameter metadata of the new code (for comparison)
            new_param_metadata = None
            try:
                new_param_metadata = self._extract_parameters(program_code, skill_description, program_name)
            except Exception as e:
                print(f"\033[33mWarning: Failed to extract new parameters for comparison: {e}\033[0m")

            # [Phase 7.0b] Over-claim detection (uses the already-extracted parameter metadata)
            if new_param_metadata:
                try:
                    detector = OverclaimDetector()
                    overclaim_result = detector.check_and_log(
                        program_name,
                        program_code,
                        new_param_metadata
                    )
                    if overclaim_result:
                        # Record into info for downstream analysis
                        if not hasattr(self, '_overclaim_warnings'):
                            self._overclaim_warnings = []
                        self._overclaim_warnings.append({
                            'skill_name': program_name,
                            'result': overclaim_result
                        })
                except Exception as e:
                    print(f"\033[33m[add_new_skill] Over-claim detection failed: {e}\033[0m")

            has_changes, change_description = self._detect_code_changes(
                current_code, 
                program_code,
                old_param_metadata=old_param_metadata,
                new_param_metadata=new_param_metadata
            )
            
            if has_changes:
                print(f"\033[33mSkill {program_name} already exists. Code changes detected: {change_description}\033[0m")

                # Check whether change_description contains a recursive-call warning
                recursive_keywords = ["recursive call", "recursive", "calls itself", "unintended recursive", "same function name"]
                has_recursive_warning = any(keyword.lower() in change_description.lower() for keyword in recursive_keywords)

                if has_recursive_warning:
                    print(f"\033[31m[Version Creation] Warning: recursive call detected; refusing to create new version\033[0m")
                    print(f"\033[31m[Version Creation] Change description: {change_description}\033[0m")
                    print(f"\033[31m[Version Creation] Suggestion: fix the recursive call in the code and retry\033[0m")
                    return  # do not create a new version

                # ──────────────────────────────────────────────────────────────
                # Layer 3 fix (Phase E validation): wrapper protection at the
                # add_new_skill entry, BEFORE any version creation. This blocks
                # both `create_new_version` AND its fallback path
                # (graph_manager_impl.py:2188-2205) from overwriting a wrapper
                # with inline code.
                #
                # Bug: action_agent's regen path goes through psn.py:1002 →
                # add_new_skill → create_new_version (default update_source=
                # "manual" which bypassed the existing wrapper guard in
                # update_skill_code). Even after fixing the source label,
                # there's a fallback path that bypasses again. The cleanest
                # surgical fix: enforce wrapper protection at the top of the
                # has_changes branch, before either write path runs.
                #
                # Behavior on block: print warning, return without overwriting.
                # The wrapper code on disk is preserved. Layers 1+2 should
                # ensure the action_agent never gets to this code path in the
                # first place — this is defense in depth for edge cases (e.g.,
                # multi-hop covered_by chains, optimizer failures, etc).
                # ──────────────────────────────────────────────────────────────
                if hasattr(node, '_should_protect_wrapper') and node._should_protect_wrapper(program_code):
                    covered_by = getattr(node, 'covered_by', 'unknown')
                    print(f"\033[33m[Layer3 Wrapper Protection] {program_name} is a wrapper of "
                          f"{covered_by}; rejecting inline regen attempt\033[0m")
                    print(f"\033[33m[Layer3 Wrapper Protection] Suggestion: modify {covered_by} "
                          f"or extend the general skill rather than overwriting the wrapper\033[0m")
                    self.logger.warning(
                        f"[Layer3] Wrapper protection blocked add_new_skill rewrite of "
                        f"'{program_name}' (covered_by={covered_by}); preserving wrapper code"
                    )
                    return  # do not write — preserve wrapper

                # v3.H+ Layer 3 extension: also block when this skill was just
                # modified by ANY refactor this step (behavioral / extract_common
                # cases where is_covered stays False but the refactor changed
                # the code). action_agent's pre-refactor inline code would
                # overwrite the refactor's edit. Tracking set is populated by
                # update_skill_code when source starts with "refactor", reset
                # by PSN at step start.
                if self.was_modified_by_refactor_this_step(program_name):
                    print(f"\033[33m[Layer3+ Refactor Protection] {program_name} was modified by "
                          f"refactor in this step (not a wrapper, but action_agent inline "
                          f"would undo the refactor edit); rejecting overwrite\033[0m")
                    self.logger.warning(
                        f"[Layer3+] Refactor protection blocked add_new_skill rewrite of "
                        f"'{program_name}'; the skill was modified by refactor this step"
                    )
                    return  # do not write — preserve refactor edit

                # Improvement 4: unify on create_new_version (auto-updates metadata)
                if latest_version:
                    # Increment version number
                    version_parts = latest_version.version.split(".")
                    version_parts[-1] = str(int(version_parts[-1]) + 1)
                    new_version = ".".join(version_parts)
                else:
                    new_version = "1.0.1"

                # Use create_new_version (auto-updates metadata)
                # Layer 3: pass update_source="action_agent" so the version
                # history is correctly labeled (was defaulting to "manual"
                # which was misleading and bypassed wrapper guards).
                success = self.create_new_version(
                    skill_name=program_name,
                    new_code=program_code,
                    new_description=skill_description,
                    change_log=change_description or "Auto-generated new version",
                    auto_update_metadata=True,  # auto-update metadata
                    auto_update_callers=False,  # do not auto-update callers (not needed inside add_new_skill)
                    update_source="action_agent",
                    update_reason=info.get("update_source") or change_description or "action_agent regenerated skill code",
                )
                
                if success:
                    print(f"\033[32m[Version Created] Created version {new_version} for '{program_name}' with metadata updated\033[0m")
                else:
                    # If create_new_version fails (e.g. version conflict), fall back to the original method
                    print(f"\033[33mWarning: create_new_version failed, using fallback method\033[0m")

                    # [Function-name consistency check] the fallback path must also ensure function-name consistency
                    validated_code = program_code
                    func_name_match = re.search(r'async\s+function\s+(\w+)\s*\(', validated_code)
                    if func_name_match:
                        actual_func_name = func_name_match.group(1)
                        if actual_func_name != program_name:
                            print(f"\033[33m[Fallback] Auto-correcting function name: '{actual_func_name}' → '{program_name}'\033[0m")
                            # Step 1: fix the function definition
                            validated_code = re.sub(
                                rf'(async\s+function\s+){re.escape(actual_func_name)}(\s*\()',
                                rf'\g<1>{program_name}\g<2>',
                                validated_code,
                                count=1
                            )
                            # Step 2: replace all variable references
                            validated_code = re.sub(
                                rf'\b{re.escape(actual_func_name)}\b',
                                program_name,
                                validated_code
                            )

                    new_version_obj = SkillVersion(
                        version=new_version,
                        code=validated_code,
                        description=skill_description,
                        change_log=change_description or "Auto-generated new version"
                    )
                    node.add_version(new_version_obj)

                    # Improvement 4: manually update metadata (if create_new_version failed)
                    self._update_skill_metadata(program_name, validated_code, skill_description, update_source="add_new_skill")

                    # Update node code and description
                    node.code = validated_code
                    node.description = skill_description
                    
                    print(f"\033[32m[Version Created] Created version {new_version} for '{program_name}':\033[0m")
                    print(f"\033[32m  - Change log: {change_description or 'Auto-generated new version'}\033[0m")
                    print(f"\033[32m  - Total versions: {len(node.versions)}\033[0m")
            else:
                # If there is no significant change, check whether a temporary node needs updating
                if is_temporary_node:
                    print(f"\033[36m[Skill Update] Temporary node '{program_name}' code unchanged, but updating description and metadata\033[0m")

                    # [Function-name consistency check] temporary-node updates also need verification
                    validated_code = program_code
                    func_name_match = re.search(r'async\s+function\s+(\w+)\s*\(', validated_code)
                    if func_name_match:
                        actual_func_name = func_name_match.group(1)
                        if actual_func_name != program_name:
                            print(f"\033[33m[Temp Node] Auto-correcting function name: '{actual_func_name}' → '{program_name}'\033[0m")
                            validated_code = re.sub(
                                rf'(async\s+function\s+){re.escape(actual_func_name)}(\s*\()',
                                rf'\g<1>{program_name}\g<2>',
                                validated_code,
                                count=1
                            )
                            validated_code = re.sub(
                                rf'\b{re.escape(actual_func_name)}\b',
                                program_name,
                                validated_code
                            )

                    # Update description (potentially more accurate)
                    node.description = skill_description
                    # Update code (to keep things consistent)
                    node.code = validated_code
                    # Update the latest version's code and description
                    if node.versions:
                        node.versions[-1].code = validated_code
                        node.versions[-1].description = skill_description
                    # Update metadata (if not yet set)
                    if not node.parameters or len(node.parameters) == 0:
                        try:
                            extracted_parameters = self._extract_parameters(validated_code, skill_description, program_name)
                            if extracted_parameters:
                                node.parameters = extracted_parameters
                                print(f"\033[32m[Skill Update] Updated temporary node parameter info\033[0m")
                        except Exception as e:
                            print(f"\033[33m[Skill Update] Warning: parameter extraction failed: {e}\033[0m")
                else:
                    print(f"\033[33mSkill {program_name} already exists. No significant changes detected, skipping version creation.\033[0m")
                    # Even without significant changes, still update the description (it may be improved)
                    node.description = skill_description
        else:
            # Create a new node
            print(f"\033[35m[DEBUG] Creating new node for '{program_name}'\033[0m")
            node = SkillNode(
                name=program_name,
                code=program_code,
                description=skill_description,
            )
            # Set value-function parameters
            node.value_function_lambda = self.value_function_lambda
            node.value_function_alpha = self.value_function_alpha
            node.value_function_beta = self.value_function_beta

            # Phase 3: immediately mark as experimental on creation
            # All newly created skills are experimental until the first successful validation
            node.is_experimental = True
            node.created_for_task = task
            node.experimental_task = task
            print(f"\033[33m[Experimental Skill] New skill '{program_name}' marked experimental (awaiting validation)\033[0m")

            # Preserve backward compatibility: if there is a first-failure reason, record it
            if info.get("first_failure_reason"):
                node.first_failure_reason = info.get("first_failure_reason")
                print(f"\033[33m[Experimental Skill] First-failure reason: {node.first_failure_reason[:100]}...\033[0m")

            # Set the task-specific flag
            node.is_task_specific = task_specific_flag
            if task_specific_flag:
                print(f"\033[33m[Task-Specific] Marking '{program_name}' as task-specific skill (in memory only)\033[0m")

            self.graph.add_node(node)
            print(f"\033[35m[DEBUG] New node created. Initial state:\033[0m")
            print(f"\033[35m[DEBUG]   - preconditions count: {len(node.preconditions)}\033[0m")
            print(f"\033[35m[DEBUG]   - expected_effects count: {len(node.expected_effects)}\033[0m")
            print(f"\033[35m[DEBUG]   - is_task_specific: {node.is_task_specific}\033[0m")
        
        # Update vector database
        if self.graph.has_node(program_name):
            node = self.graph.get_node(program_name)
            if getattr(node, 'is_task_specific', False):
                # Task-specific skill: delete from vectordb (record_execution may have added it)
                try:
                    # Idempotent delete: check existence first
                    existing = self.vectordb._collection.get(ids=[program_name])
                    if existing and existing.get("ids"):
                        self.vectordb._collection.delete(ids=[program_name])
                        print(f"\033[33m[Task-Specific] Removed '{program_name}' from vectordb (not retrievable)\033[0m")
                except:
                    pass
                # Do not add to vectordb
            else:
                # Normal skill: delete old, add new (idempotent delete)
                try:
                    existing = self.vectordb._collection.get(ids=[program_name])
                    if existing and existing.get("ids"):
                        self.vectordb._collection.delete(ids=[program_name])
                except:
                    pass

                # === P0 fix: enrich metadata ===
                self.vectordb.add_texts(
                    texts=[skill_description],
                    ids=[program_name],
                    metadatas=[{
                        "name": program_name,
                        "is_deprecated": getattr(node, 'is_deprecated', False),
                        "is_task_specific": getattr(node, 'is_task_specific', False),
                    }],
                )
                # === end of fix ===

        # [P5 fix] remove redundant checkpoint save
        # Reason: preconditions/effects auto-extraction happens below;
        # the complete data is uniformly saved at line 2643 once extraction finishes

        # Auto-extract preconditions and state changes (if enabled)
        if self.auto_extract_preconditions or self.auto_extract_effects:
            if self.graph.has_node(program_name):
                node = self.graph.get_node(program_name)
                
                # Debug log: record node state before extraction
                print(f"\033[35m[DEBUG] Before extraction check for '{program_name}':\033[0m")
                print(f"\033[35m[DEBUG]   - auto_extract_preconditions: {self.auto_extract_preconditions}\033[0m")
                print(f"\033[35m[DEBUG]   - auto_extract_effects: {self.auto_extract_effects}\033[0m")
                print(f"\033[35m[DEBUG]   - node.preconditions count: {len(node.preconditions)}\033[0m")
                print(f"\033[35m[DEBUG]   - node.expected_effects count: {len(node.expected_effects)}\033[0m")
                print(f"\033[35m[DEBUG]   - Will extract preconditions: {self.auto_extract_preconditions and len(node.preconditions) == 0}\033[0m")
                print(f"\033[35m[DEBUG]   - Will extract effects: {self.auto_extract_effects and len(node.expected_effects) == 0}\033[0m")
                
                # Get task and context (if available)
                task = info.get("task", None)
                context = info.get("context", None)
                
                # If context is not in info, try extracting from conversations (backward compatible)
                if not context and info.get("conversations"):
                    # context is usually in the human message
                    conversations = info.get("conversations", [])
                    if conversations:
                        # Find a conversation containing context
                        for conv in conversations:
                            if len(conv) >= 2 and "Context:" in str(conv[1]):
                                context_match = re.search(r'Context:\s*(.+?)(?:\n\n|\nTask:|$)', str(conv[1]), re.DOTALL)
                                if context_match:
                                    extracted_context = context_match.group(1).strip()
                                    if extracted_context.lower() != "none":
                                        context = extracted_context
                                    break
                
                if self.auto_extract_preconditions and len(node.preconditions) == 0:
                    print(f"\033[33mAuto-extracting preconditions for {program_name}...\033[0m")
                    # Recheck node state (in case the node was modified during extraction)
                    node_check = self.graph.get_node(program_name)
                    if node_check != node:
                        print(f"\033[31m[DEBUG] WARNING: Node reference changed during extraction!\033[0m")
                        print(f"\033[31m[DEBUG]   - Original node id: {id(node)}\033[0m")
                        print(f"\033[31m[DEBUG]   - New node id: {id(node_check)}\033[0m")
                        node = node_check
                    print(f"\033[35m[DEBUG] Extracting preconditions. Node state:\033[0m")
                    print(f"\033[35m[DEBUG]   - preconditions before: {len(node.preconditions)}\033[0m")
                    extracted_preconditions, warnings = self._extract_preconditions(
                        program_code, skill_description, task=task, context=context
                    )
                    node.preconditions.extend(extracted_preconditions)
                    print(f"\033[35m[DEBUG]   - preconditions after: {len(node.preconditions)}\033[0m")
                    if extracted_preconditions:
                        print(f"\033[32mExtracted {len(extracted_preconditions)} preconditions\033[0m")
                    if warnings:
                        print(f"\033[33mPrecondition extraction warnings:\033[0m")
                        for warning in warnings:
                            print(f"\033[33m  - {warning}\033[0m")
                    
                    # Sync preconditions onto the latest version object
                    if node.versions and extracted_preconditions:
                        latest_version = node.versions[-1]
                        # Use deepcopy to avoid reference issues
                        latest_version.preconditions.extend([copy.deepcopy(p) for p in extracted_preconditions])
                        print(f"\033[36m[Version Sync] Synced {len(extracted_preconditions)} preconditions onto version {latest_version.version}\033[0m")
                else:
                    print(f"\033[35m[DEBUG] Skipping preconditions extraction (enabled={self.auto_extract_preconditions}, count={len(node.preconditions)})\033[0m")
                
                if self.auto_extract_effects and len(node.expected_effects) == 0:
                    print(f"\033[33mAuto-extracting effects for {program_name}...\033[0m")
                    # Recheck node state (in case the node was modified during extraction)
                    node_check = self.graph.get_node(program_name)
                    if node_check != node:
                        print(f"\033[31m[DEBUG] WARNING: Node reference changed during extraction!\033[0m")
                        print(f"\033[31m[DEBUG]   - Original node id: {id(node)}\033[0m")
                        print(f"\033[31m[DEBUG]   - New node id: {id(node_check)}\033[0m")
                        node = node_check
                    print(f"\033[35m[DEBUG] Extracting effects. Node state:\033[0m")
                    print(f"\033[35m[DEBUG]   - expected_effects before: {len(node.expected_effects)}\033[0m")
                    # Get execution history for runtime validation
                    exec_traces = node.statistics.execution_traces if hasattr(node, 'statistics') and node.statistics else None
                    extracted_effects, warnings = self._extract_effects(
                        program_code, skill_description, task=task, context=context,
                        skill_name=program_name,  # pass skill name to infer primary effect
                        execution_traces=exec_traces
                    )
                    node.expected_effects.extend(extracted_effects)
                    print(f"\033[35m[DEBUG]   - expected_effects after: {len(node.expected_effects)}\033[0m")
                    if extracted_effects:
                        print(f"\033[32mExtracted {len(extracted_effects)} effects\033[0m")
                    if warnings:
                        print(f"\033[33mEffect extraction warnings:\033[0m")
                        for warning in warnings:
                            print(f"\033[33m  - {warning}\033[0m")
                    
                    # Validate consistency between naming and effects
                    consistency_warnings = self._validate_naming_effect_consistency(
                        program_name, extracted_effects
                    )
                    if consistency_warnings:
                        print(f"\033[33m[Naming-Effect Consistency] Detected inconsistency between naming and effects:\033[0m")
                        for warning in consistency_warnings:
                            print(f"\033[33m  {warning}\033[0m")
                    
                    # Sync effects onto the latest version object
                    if node.versions and extracted_effects:
                        latest_version = node.versions[-1]
                        # Use deepcopy to avoid reference issues
                        latest_version.effects.extend([copy.deepcopy(e) for e in extracted_effects])
                        print(f"\033[36m[Version Sync] Synced {len(extracted_effects)} effects onto version {latest_version.version}\033[0m")
                else:
                    print(f"\033[35m[DEBUG] Skipping effects extraction (enabled={self.auto_extract_effects}, count={len(node.expected_effects)})\033[0m")
                
                # Save updated data
                self._save_to_checkpoint()
                print(f"\033[35m[DEBUG] After extraction, final state for '{program_name}':\033[0m")
                final_node = self.graph.get_node(program_name)
                print(f"\033[35m[DEBUG]   - preconditions count: {len(final_node.preconditions)}\033[0m")
                print(f"\033[35m[DEBUG]   - expected_effects count: {len(final_node.expected_effects)}\033[0m")
        
        # Auto-extract parameter info
        if self.graph.has_node(program_name):
            node = self.graph.get_node(program_name)
            # If parameter info is empty, extract it
            if not node.parameters:
                print(f"\033[33mAuto-extracting parameters for {program_name}...\033[0m")
                extracted_parameters = self._extract_parameters(program_code, skill_description, program_name)
                if extracted_parameters:
                    self.set_skill_parameters(program_name, extracted_parameters)
                    print(f"\033[32mExtracted {len(extracted_parameters)} parameters: {', '.join(extracted_parameters.keys())}\033[0m")

                    # [Phase 7.0b] Over-claim detection
                    try:
                        detector = OverclaimDetector()
                        overclaim_result = detector.check_and_log(
                            program_name,
                            program_code,
                            extracted_parameters
                        )
                        if overclaim_result:
                            if not hasattr(self, '_overclaim_warnings'):
                                self._overclaim_warnings = []
                            self._overclaim_warnings.append({
                                'skill_name': program_name,
                                'result': overclaim_result
                            })
                    except Exception as e:
                        print(f"\033[33m[add_new_skill] Over-claim detection failed: {e}\033[0m")

                    # Save updated data
                    self._save_to_checkpoint()
                else:
                    print(f"\033[33mNo parameters extracted for {program_name}\033[0m")

        # New refactor-detection flow (if the skill was added to the graph)
        # [Fix 7] experimental skills skip refactor detection entirely
        # Reason: failing code should not be the basis for refactor; avoids creating problematic general skills
        # The refactor will be triggered after successful validation via _trigger_delayed_refactor
        skip_refactor = info.get("skip_refactor", False)

        if not self.enable_refactor:
            # Honor the global refactor toggle (--no-refactor); the delayed-refactor
            # path checks this too, but this immediate flow must as well.
            skip_refactor = True
            self.logger.info(f"\033[33m[Refactor] Skipping refactor detection (refactor disabled)\033[0m")
        elif skip_refactor:
            self.logger.info(f"\033[33m[Refactor] Skipping refactor detection (user-specified)\033[0m")
        elif not self.graph.has_node(program_name):
            # The node should already be added; skip otherwise
            skip_refactor = True
        else:
            # [Bug 9 Fix] Check the node's actual state instead of the info dict
            # info.get("is_experimental") returns False for the first-time-successful new skill (not set)
            # But node.is_experimental was correctly set to True at line 2472
            node = self.graph.get_node(program_name)
            if getattr(node, 'is_experimental', False):
                self.logger.info(f"\033[33m[Refactor] Skipping refactor detection (experimental skill; will trigger after success)\033[0m")
                skip_refactor = True
            elif getattr(node, 'is_task_specific', False) or self._is_task_specific_skill(program_name, program_code):
                # task-specific skills are only useful within the current task; no need to be covered by other skills
                self.logger.info(f"\033[33m[Refactor] Skipping refactor detection (task-specific skill)\033[0m")
                skip_refactor = True
                # Also set the attribute so downstream checks recognize it
                node.is_task_specific = True

        if not skip_refactor and self.graph.has_node(program_name):
            node = self.graph.get_node(program_name)

            # Step a: prescreen candidates
            self.logger.info(f"\033[36m[Refactor] Starting refactor detection flow...\033[0m")
            candidates = self._prescreen_refactor_candidates(
                program_name,
                skill_description,
                top_k=5
            )
            
            if candidates:
                self.logger.info(f"\033[36m[Refactor] Prescreened candidate skills: {', '.join([name for name, _ in candidates])}\033[0m")
                
                # Step b: LLM detection and classification
                refactor_info = self._detect_refactor_relationships(
                    new_skill_name=program_name,
                    new_skill_code=program_code,
                    new_skill_description=skill_description,
                    new_skill_params=node.parameters,
                    new_skill_preconditions=node.preconditions,
                    new_skill_effects=node.expected_effects,
                    candidate_skills=candidates
                )
                
                # Step c: execute refactor
                if refactor_info:
                    self.logger.info(f"\033[36m[Refactor] Executing refactor action...\033[0m")
                    success = self._execute_refactor(
                        program_name,
                        program_code,
                        refactor_info
                    )
                    if success:
                        self.logger.info(f"\033[32m[Refactor] Refactor executed successfully\033[0m")
                    else:
                        self.logger.warning(f"\033[33m[Refactor] Refactor execution failed\033[0m")
            else:
                self.logger.info(f"\033[36m[Refactor] No candidate skills found; skipping refactor detection\033[0m")
        
        # Check whether the new skill covers existing skills (bidirectional check) - delegated to _handle_covered_skills
        covered_skills = info.get("covered_existing_skills", [])
        self._handle_covered_skills(program_name, program_code, covered_skills)

        # Auto-update the graph structure (extract call relations from code)
        if self.graph.has_node(program_name):
            self._update_graph_from_code(program_name)
            # Save the updated graph structure
            self._save_to_checkpoint()
        
        print(f"\033[33mSkill {program_name} added to graph. "
              f"Graph now has {len(self.graph.nodes)} skills.\033[0m")

        # Return the final (possibly renamed) skill name so callers can detect
        # success. Without this the success path fell through to an implicit None,
        # indistinguishable from the reject paths' explicit `return None`; callers
        # that branch on the return (HelperExtractor.extract_and_register) then
        # treated every successful registration as a failure and never migrated
        # the parent to call the newly-registered helper, leaving it a dangling
        # duplicate. info["program_name"] is mutated in place on rename, so this
        # matches what the node was registered under.
        return info.get("program_name", program_name)

    def _invoke_llm_with_stats(
        self,
        messages,
        process_type: str,
        function_name: str,
        task: str = None,
        skill_name: str = None,
        metadata: Dict[str, Any] = None,
        max_retries: int = 3,  # Part 16: added retry parameter
        initial_delay: float = 1.0,  # initial delay (seconds)
    ):
        """
        Call the LLM and record statistics (with retry mechanism).
        
        Args:
            messages: list of LLM messages
            process_type: process type
            function_name: function name
            task: associated task
            skill_name: associated skill name
            metadata: other metadata
            max_retries: maximum retry count (added in Part 16)
            initial_delay: initial retry delay (seconds)
        
        Returns:
            LLM response object
        """
        import time
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                token_usage = self.stats_tracker.extract_token_usage(response)
                
                # Record statistics
                self.stats_tracker.record_llm_call(
                    process_type=process_type,
                    function_name=function_name,
                    task=task,
                    skill_name=skill_name,
                    input_tokens=token_usage.get("input_tokens"),
                    output_tokens=token_usage.get("output_tokens"),
                    total_tokens=token_usage.get("total_tokens"),
                    model_name=self.llm.model_name,
                    success=True,
                    metadata=metadata,
                )
                
                return response
                
            except Exception as e:
                last_exception = e
                
                # Part 16: retry logic
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)  # exponential backoff
                    print(f"\033[33m[LLM Retry] {function_name} call failed; retrying in {delay:.1f}s ({attempt+1}/{max_retries}): {e}\033[0m")
                    time.sleep(delay)
                else:
                    # Final attempt also failed
                    print(f"\033[31m[LLM Retry] {function_name} call failed; reached max retries ({max_retries}): {e}\033[0m")
                    
                    # Record the failed call
                    self.stats_tracker.record_llm_call(
                        process_type=process_type,
                        function_name=function_name,
                        task=task,
                        skill_name=skill_name,
                        model_name=self.llm.model_name,
                        success=False,
                        error_message=str(e),
                        metadata=metadata,
                    )
        
        # All retries failed; raise the last exception
        raise last_exception
    
    def _extract_function_signature(self, program_code: str, program_name: str) -> str:
        """
        Extract the function signature (parameter list) from code.
        
        Args:
            program_code: program code
            program_name: function name
        
        Returns:
            str: function signature string, e.g. "(bot, count = 1, allowedLogTypes = [...])"
                 If extraction fails, returns the default "(bot)"
        """
        try:
            # Resolve skill language: cache-first, registry-fallback
            skill_lang = getattr(self, "_skill_language", None)
            if skill_lang is None:
                from skillnet.core.dk_registry import get_domain_knowledge
                dk = get_domain_knowledge()
                if dk is None:
                    raise RuntimeError(
                        "No DomainKnowledge registered; cannot parse skill code"
                    )
                skill_lang = dk.get_skill_language_impl()

            parse_result = skill_lang.parse(program_code)
            if not parse_result.success:
                raise ValueError(f"parse failed: {parse_result.error}")

            parsed = parse_result.raw_ast
            if parsed is None:
                raise ValueError("SkillLanguage.parse returned no raw_ast")

            # Access parsed AST — use try/except because JSPyBridge
            # raises JavaScriptError (not AttributeError) on property failures
            try:
                body = parsed.program.body
            except Exception:
                raise ValueError("Failed to access parsed AST structure")

            # Find the main function (last async function)
            for node in body:
                try:
                    if node.type != "FunctionDeclaration":
                        continue
                    try:
                        is_async = bool(node["async"])
                    except Exception:
                        is_async = False
                    if not is_async:
                        continue
                    node_name = node.id.name
                    if node_name != program_name:
                        continue
                except Exception:
                    continue

                # Found a matching function; extract parameters
                try:
                    params = list(node.params) if node.params else []
                except Exception:
                    params = []

                # Generate the parameter signature string
                param_strings = []
                for param in params:
                    try:
                        param_name = None
                        default_value_node = None

                        param_type = param.type
                        if param_type == "Identifier":
                            param_name = param.name
                        elif param_type == "AssignmentPattern":
                            try:
                                if param.left:
                                    param_name = param.left.name
                            except Exception:
                                pass
                            try:
                                if param.right:
                                    default_value_node = param.right
                            except Exception:
                                pass

                        if param_name:
                            param_str = param_name
                            if default_value_node is not None:
                                try:
                                    default_code = skill_lang.regenerate(default_value_node)
                                    param_str += f" = {default_code}"
                                except Exception as e:
                                    print(f"\033[33m[Function Signature Extraction] Failed to generate default value code: {e}\033[0m")
                            param_strings.append(param_str)
                    except Exception:
                        continue

                if param_strings:
                    return f"({', '.join(param_strings)})"
                break

            # If not found or extraction failed, raise to trigger the fallback
            raise ValueError(f"Function {program_name} not found or has no parameters")

        except Exception as e:
            print(f"\033[33m[Function Signature Extraction] Skill-language extraction failed for {program_name}: {e}\033[0m")
            print(f"\033[36m[Function Signature Extraction] Falling back to LLM extraction...\033[0m")
            
            # ========== Fallback: extract via LLM ==========
            return self._extract_function_signature_llm(program_code, program_name)
    
    def generate_skill_description(self, program_name: str, program_code: str, task: str = None) -> str:
        """
        Generate skill description (compatible with legacy interface).
        
        Args:
            program_name: program name
            program_code: program code
            task: associated task (for statistics)
        
        Returns:
            str: skill description, including the complete function signature (all parameters)
        """
        messages = [
            SystemMessage(content=self._domain_knowledge.get_prompt("skill") if self._domain_knowledge else ""),
            HumanMessage(
                content=program_code
                + "\n\n"
                + f"The main function is `{program_name}`."
            ),
        ]
        
        response = self._invoke_llm_with_stats(
            messages=messages,
            process_type="skill_generation",
            function_name="generate_skill_description",
            task=task,
            skill_name=program_name,
        )
        
        # Extract function signature (includes all parameters)
        function_signature = self._extract_function_signature(program_code, program_name)
        
        skill_description = f"    // {response.content}"
        return f"async function {program_name}{function_signature} {{\n{skill_description}\n}}"
    
    def _extract_function_signature_llm(self, program_code: str, program_name: str) -> str:
        """
        Use the LLM to extract a function signature string (fallback method).
        
        Args:
            program_code: program code
            program_name: function name
        
        Returns:
            str: function signature string, e.g. "(bot, count = 1, allowedLogTypes = [...])"
                 If extraction fails, returns the default "(bot)"
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage
            import json
            import re
            
            print(f"\033[36m[Function Signature Extraction] Using LLM to extract signature for {program_name}\033[0m")
            
            system_prompt = """You are a code analysis expert. Extract the function signature (parameter list) from JavaScript function code.

Return ONLY the parameter list in the format: (param1, param2 = defaultValue, param3 = [...])
- Include the 'bot' parameter as the first parameter
- Include default values exactly as they appear in the code
- For arrays, include the full array literal with brackets
- For strings, include quotes
- For numbers and booleans, include them as-is
- Return ONLY the parameter list, nothing else

Example outputs:
- (bot)
- (bot, count = 1)
- (bot, count = 1, allowedTypes = ["oak_log", "birch_log"])
- (bot, itemType = "diamond", count = 5)
- (bot, enabled = true, timeout = 1000)"""

            human_prompt = f"""Extract the function signature for the function named '{program_name}' from this JavaScript code:

```javascript
{program_code}
```

Return ONLY the parameter list in the format: (param1, param2 = defaultValue, ...)
Do not include the function name or function body. Make sure to include 'bot' as the first parameter."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self._invoke_llm_with_stats(
                messages=messages,
                process_type="function_signature_extraction",
                function_name="_extract_function_signature_llm",
                skill_name=program_name,
            )
            
            response_content = response.content.strip()
            
            # Try extracting parameter list
            # Match format: (param1, param2 = value, ...) or param1, param2 = value, ...
            signature_match = re.search(r'\(([^)]*)\)', response_content)
            if signature_match:
                params_str = signature_match.group(1).strip()
                if params_str:
                    # Validate parameter list format
                    # Check that 'bot' is the first parameter
                    params = [p.strip() for p in params_str.split(',')]
                    if params and params[0].split('=')[0].strip() == 'bot':
                        print(f"\033[32m[Function Signature Extraction] ✓ Successfully extracted signature using LLM: ({params_str})\033[0m")
                        return f"({params_str})"
                    else:
                        print(f"\033[33m[Function Signature Extraction] LLM response missing 'bot' parameter, trying regex fallback\033[0m")
            
            # If the format is incorrect, try using the response content directly
            if response_content.startswith('(') and response_content.endswith(')'):
                # Validate that bot is included
                if 'bot' in response_content.lower():
                    print(f"\033[32m[Function Signature Extraction] ✓ Successfully extracted signature using LLM: {response_content}\033[0m")
                    return response_content
                else:
                    print(f"\033[33m[Function Signature Extraction] LLM response format valid but missing 'bot', trying regex fallback\033[0m")
            
            # If everything fails, try extracting from code (simple regex fallback)
            print(f"\033[33m[Function Signature Extraction] LLM response format invalid, trying regex fallback\033[0m")
            print(f"\033[33m[Function Signature Extraction] LLM raw response: {response_content[:200]}...\033[0m")
            return self._extract_function_signature_regex(program_code, program_name)
            
        except Exception as e:
            print(f"\033[33m[Function Signature Extraction] LLM extraction failed: {e}\033[0m")
            import traceback
            print(f"\033[33m[Function Signature Extraction] LLM extraction traceback: {traceback.format_exc()}\033[0m")
            # Last-resort fallback: regex
            return self._extract_function_signature_regex(program_code, program_name)
    
    def _extract_function_signature_regex(self, program_code: str, program_name: str) -> str:
        """
        Extract the function signature using a regex (last-resort fallback).
        
        Args:
            program_code: program code
            program_name: function name
        
        Returns:
            str: function signature string; returns "(bot)" on failure
        """
        import re
        
        print(f"\033[36m[Function Signature Extraction] Using regex fallback for {program_name}\033[0m")
        
        # Match async function functionName(...params)
        pattern = rf'async\s+function\s+{re.escape(program_name)}\s*\(([^)]*)\)'
        match = re.search(pattern, program_code, re.MULTILINE | re.DOTALL)
        
        if match:
            params_str = match.group(1).strip()
            if params_str:
                print(f"\033[32m[Function Signature Extraction] ✓ Successfully extracted signature using regex: ({params_str})\033[0m")
                return f"({params_str})"
        
        print(f"\033[33m[Function Signature Extraction] ✗ All extraction methods failed for {program_name}, using default (bot)\033[0m")
        return "(bot)"
    
    # find_skill_by_task_intent - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)
    
    # retrieve_skills - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)

    # ensure_skill_in_vectordb - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)

    # set_skill_parameters - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)
    
    def _extract_parameters(self, program_code: str, skill_description: str = None, skill_name: str = None) -> Dict[str, Dict[str, Any]]:
        """
        delegate to ParameterExtractor.

        Extract parameter info (main method; selects an approach by configuration).

        Args:
            program_code: skill code
            skill_description: skill description (optional)
            skill_name: skill name (optional; used to match the main function)

        Returns:
            Parameter metadata dict
        """
        return self._parameter_extractor.extract(
            code=program_code,
            description=skill_description,
            skill_name=skill_name,
        )
    
    def _determine_delta_prefix(self, func_name: str) -> str:
        """
        Determine the prefix to use during reverse renaming (ensure -> craft/mine).
        
        Decide between 'mine' and 'craft' based on function-name traits:
        - Contains keywords like ore/log/block/sand/gravel/stone/dirt/coal -> mine
        - Otherwise -> craft (default)
        
        Args:
            func_name: function name (e.g. ensureLogs, ensurePlanks)
        
        Returns:
            Recommended prefix: "mine" or "craft"
        """
        mine_keywords = ["ore", "log", "block", "sand", "gravel", "stone", "dirt", "coal", "cobble", "wood"]
        func_name_lower = func_name.lower()
        
        for keyword in mine_keywords:
            if keyword in func_name_lower:
                return "mine"
        
        return "craft"  # default to craft
    
    def _validate_function_name_semantic_consistency(
        self,
        func_name: str,
        parameters: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Validate consistency between function name and parameter semantics.
        
        Rules checked:
        1. Function name starts with ensure* but parameter semantics are delta → warning (suggest renaming to craft/mine)
        2. Function name starts with craft/mine/collect but parameter semantics are target_total → warning (suggest renaming to ensure)
        
        Args:
            func_name: function name
            parameters: parameter metadata dict
        
        Returns:
            list of warnings; each contains {"param_name", "semantic", "expected", "suggestion", "suggested_name"}
        """
        warnings = []
        
        # Mapping from function-name prefix to expected semantics
        ensure_prefixes = ["ensure"]
        delta_prefixes = ["craft", "mine", "collect", "get", "gather", "harvest", "produce"]
        
        func_name_lower = func_name.lower()
        
        # Check each count-parameter
        quantity_param_names = ["count", "amount", "num", "quantity", "number", "total", "target"]
        
        for param_name, param_info in parameters.items():
            if param_name.lower() not in quantity_param_names:
                continue
            
            semantic = param_info.get("semantic")
            if not semantic or semantic == "config":
                continue
            
            # Check for ensure* function names with delta semantics -> suggest craft/mine
            is_ensure_name = any(func_name_lower.startswith(prefix) for prefix in ensure_prefixes)
            if is_ensure_name and semantic == "delta":
                # Use _determine_delta_prefix to choose between craft and mine
                new_prefix = self._determine_delta_prefix(func_name)
                # Extract the part after ensure (preserving case)
                suffix = func_name[6:]  # strip "ensure" (6 chars)
                suggested_name = f"{new_prefix}{suffix}"
                warnings.append({
                    "param_name": param_name,
                    "semantic": semantic,
                    "expected": "target_total",
                    "suggested_name": suggested_name,
                    "suggestion": f"Function name '{func_name}' starts with 'ensure'; suggest parameter '{param_name}' use target_total semantics, "
                                 f"or rename the function to '{suggested_name}'"
                })
            
            # Check for craft/mine* function names with target_total semantics -> suggest ensure
            is_delta_name = any(func_name_lower.startswith(prefix) for prefix in delta_prefixes)
            if is_delta_name and semantic == "target_total":
                # Find the matching prefix
                matched_prefix = next((p for p in delta_prefixes if func_name_lower.startswith(p)), "")
                # Extract the part after the prefix (preserving case)
                suffix = func_name[len(matched_prefix):]
                suggested_name = f"ensure{suffix}"
                warnings.append({
                    "param_name": param_name,
                    "semantic": semantic,
                    "expected": "delta",
                    "suggested_name": suggested_name,
                    "suggestion": f"Function name '{func_name}' starts with '{matched_prefix}' (typically denotes delta), "
                                 f"but parameter '{param_name}' uses target_total semantics. "
                                 f"Suggest renaming to '{suggested_name}' to avoid confusion"
                })
        
        # Emit warning logs
        for warning in warnings:
            print(f"\033[33m[Semantic Naming] {func_name}: function name vs. semantics mismatch\033[0m")
            print(f"\033[33m[Semantic Naming]   parameter: {warning['param_name']}, semantics: {warning['semantic']}, expected: {warning['expected']}\033[0m")
            print(f"\033[33m[Semantic Naming]   suggestion: {warning['suggestion']}\033[0m")
        
        return warnings
    
    def _apply_semantic_rename(
        self,
        old_name: str,
        suggested_name: str,
        code: str
    ) -> Tuple[str, str, bool]:
        """
        Apply a semantic rename.
        
        Handles edge cases:
        1. Check whether the new name already exists; append a numeric suffix if so
        2. Check recursion risk (if the code already calls the suggested name)
        3. Update the function definition in code
        4. Re-check for primitive name conflicts
        
        Args:
            old_name: original function name
            suggested_name: suggested new function name
            code: function code
        
        Returns:
            (final_name, new_code, was_renamed):
            - final_name: final function name (may be suggested_name or a suffixed variant)
            - new_code: updated code
            - was_renamed: whether the rename succeeded
        """
        import re
        
        new_name = suggested_name
        
        # 1. Check whether the new name already exists in the skill graph
        if self.graph.has_node(new_name):
            # Append a numeric suffix
            suffix = 2
            while self.graph.has_node(f"{suggested_name}{suffix}"):
                suffix += 1
            new_name = f"{suggested_name}{suffix}"
            print(f"\033[33m[Semantic Rename] '{suggested_name}' already exists; using '{new_name}'\033[0m")
        
        # 2. Check recursion risk
        # If the code already calls the new name (excluding typeof checks), renaming would cause recursion
        body_without_typeof = re.sub(r'typeof\s+\w+\s*===\s*["\']function["\']', '', code, flags=re.IGNORECASE)
        call_pattern = rf'\b{re.escape(new_name)}\s*\('
        if re.search(call_pattern, body_without_typeof):
            print(f"\033[33m[Semantic Rename] Skipping: '{old_name}' code already calls '{new_name}'; renaming would cause recursion\033[0m")
            return old_name, code, False
        
        # 3. Update the function definition in code
        new_code = re.sub(
            rf'(async\s+function\s+){re.escape(old_name)}(\s*\()',
            rf'\g<1>{new_name}\g<2>',
            code
        )
        
        # 4. Re-check for primitive conflict
        final_name, final_code, was_primitive_renamed = resolve_primitive_name_conflict(new_name, new_code)
        if was_primitive_renamed:
            print(f"\033[33m[Semantic Rename] After semantic rename, '{new_name}' conflicts with a primitive; changed to '{final_name}'\033[0m")
        else:
            final_name = new_name
            final_code = new_code
        
        print(f"\033[32m[Semantic Rename] '{old_name}' → '{final_name}'\033[0m")
        return final_name, final_code, True
    
    def _check_and_apply_semantic_rename(
        self,
        program_name: str,
        program_code: str
    ) -> Tuple[str, str, bool]:
        """
        Check semantic consistency and apply rename.
        
        Entry method that integrates parameter extraction and semantic renaming.
        
        Flow:
        1. Check whether auto_semantic_rename is enabled in config
        2. Eagerly extract parameters and semantics
        3. Validate consistency between function name and semantics
        4. If there are warnings containing suggested_name, apply the rename
        
        Args:
            program_name: original function name
            program_code: function code
        
        Returns:
            (new_name, new_code, was_renamed):
            - new_name: final function name
            - new_code: updated code
            - was_renamed: whether a rename occurred
        """
        # Check configuration
        if not self.auto_semantic_rename:
            return program_name, program_code, False
        
        # Eagerly extract parameters and semantics
        try:
            parameters = self._extract_parameters(program_code, None, program_name)
            if not parameters:
                return program_name, program_code, False
        except Exception as e:
            print(f"\033[33m[Semantic Rename] Parameter extraction failed; skipping semantic rename: {e}\033[0m")
            return program_name, program_code, False
        
        # Validate function-name vs. semantics consistency
        warnings = self._validate_function_name_semantic_consistency(program_name, parameters)
        
        if not warnings:
            return program_name, program_code, False
        
        # Check whether suggested_name is present
        for warning in warnings:
            suggested_name = warning.get("suggested_name")
            if suggested_name:
                # Apply the rename
                new_name, new_code, was_renamed = self._apply_semantic_rename(
                    program_name, suggested_name, program_code
                )
                if was_renamed:
                    return new_name, new_code, True
        
        # No suggested_name or rename failed
        return program_name, program_code, False
    
    # ========== Graph query / structure methods ==========
    # Moved to mixins/graph_queries.py: GraphQueriesMixin
    # Methods: add_edge, remove_edge, get_node, get_subgraph, has_node,
    # get_all_skill_names, get_parents, get_children, add_skill_node,
    # remove_skill_node, has_edge, would_create_cycle, iter_skills,
    # skill_count (property), save, update_skill_dependencies,
    # is_task_specific, expand_with_dependencies, extract_effects

    # Phase 9 methods (delete_skill, _backup_skill_before_delete, _delete_from_graph,
    # _delete_from_vectordb, _delete_from_disk, _cleanup_coverage_relations)
    # - Moved to mixins/skill_deletion.py (SkillDeletionAndCleanupMixin)

    # extract_called_skills - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)
    # _expand_skills_with_call_chain - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)
    
    def _calculate_state_changes(
        self,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        environment_events: List[Any] = None,
    ) -> ActualEffect:
        """
        delegate to execution.calculate_state_changes.

        Compute state changes between pre and post execution.

        Args:
            pre_state: state before execution
            post_state: state after execution
            environment_events: list of Environment events

        Returns:
            ActualEffect: computed actual effect
        """
        return calculate_state_changes(pre_state, post_state, environment_events)

    # =========================================================================
    # SkillExecutionMixin method group - skill execution recording
    # Moved to skillnet.agents.skill_graph._impl.mixins.skill_execution
    # Methods: record_execution, _trigger_delayed_refactor, _handle_refactored_skill_failure,
    # _rollback_skill_refactor, on_task_completed, _update_semantics_from_executions,
    # _update_effects_from_executions, _analyze_parameterized_effects,
    # _calculate_correlation, _classify_effect_importance, _parse_js_args_to_dict,
    # _parse_call_args_from_exec_code, _parse_js_arguments, _parse_js_value,
    # _categorize_failure, _calculate_precondition_value, _calculate_effect_value,
    # _verify_precondition_in_code, _update_preconditions_from_executions
    # =========================================================================

    # MetadataManagementMixin method group - metadata management
    # Moved to skillnet.agents.skill_graph._impl.mixins.metadata_mgmt
    # Methods: add_precondition, remove_precondition, add_effect, remove_effect,
    # remove_effect_by_item, _filter_out_item_from_effects, _serialize_effects,
    # _log_effects_change, get_preconditions, get_effects, get_actual_effects
    # =========================================================================

    # =========================================================================
    # SkillVersioningMixin method group - skill versioning
    # Moved to skillnet.agents.skill_graph._impl.mixins.skill_versioning
    # Methods: create_new_version, rollback_skill_and_subgraph, rollback_skill,
    # rollback_graph, get_skill_versions, get_graph_versions,
    # get_current_skill_version, get_current_graph_version
    # =========================================================================

    # create_new_version - Moved to mixins/skill_versioning.py
    # rollback_skill_and_subgraph - Moved to mixins/skill_versioning.py

    # Metadata methods - Moved to mixins/
    # metadata_mgmt.py: _update_skill_metadata, _analyze_interface_impact,
    # _extract_preconditions, _filter_preconditions, _code_has_precondition_check,
    # _extract_effects, _validate_naming_effect_consistency
    # metadata_validation.py: _validate_and_clean_effects_on_load, _repair_missing_primary_effects,
    # _fix_function_name_mismatch_on_load, _validate_effects_against_code,
    # _validate_or_effect_conditions, _code_has_effect_implementation
    # metadata_crud.py: add_precondition, remove_precondition, add_effect, remove_effect,
    # get_preconditions, get_effects, get_actual_effects,
    # remove_effect_by_item, _filter_out_item_from_effects, _serialize_effects, _log_effects_change
    
    # _check_parameter_semantic_similarity - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    # _check_functional_equivalence - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    # _detect_function_name_conflicts - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    # _fix_function_name_conflicts - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    # _detect_and_fix_function_conflicts - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)
    
    # _detect_code_changes - Moved to mixins/skill_code_update.py (SkillCodeUpdateMixin)

    # _handle_covered_skills - Moved to mixins/refactor_mgmt.py

    # _extract_function_calls - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)
    # extract_all_function_definitions - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)
    
    # _update_graph_from_code - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)
    
    # _extract_all_skill_calls_deep - Moved to mixins/skill_retrieval.py (SkillRetrievalAndDiscoveryMixin)

    # validate_and_clean_graph - Moved to mixins/skill_deletion.py (SkillDeletionAndCleanupMixin)

