"""
Metadata Management Mixin for SkillGraphManager

Core metadata orchestration and extraction delegate methods.

Extracted from graph_manager_impl.py for better modularity.
Split into 3 mixins for further modularity:
  - MetadataCrudMixin (metadata_crud.py): CRUD operations + getters
  - MetadataValidationMixin (metadata_validation.py): Validation + load-time repair
  - MetadataManagementMixin (this file): Core orchestration + extraction delegates
"""

import copy
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from skillnet.agents.skill_graph.models import (
    SkillPrecondition,
    SkillEffect,
)
from skillnet.agents.skill_graph.models.execution import SkillExecutionTrace
from skillnet.agents.skill_graph.utils import (
    filter_preconditions,
    code_has_precondition_check,
    code_has_effect_implementation,
    validate_naming_effect_consistency,
)
import skillnet.utils as U

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class MetadataManagementMixin:
    """Metadata Management Mixin - Core orchestration and extraction delegates.

    Methods:
        _update_skill_metadata: Update skill metadata after code changes
        _analyze_interface_impact: Analyze interface impact on callers
        _extract_preconditions: Extract preconditions (delegates to extractor)
        _filter_preconditions: Filter preconditions (delegates to utils)
        _code_has_precondition_check: Check precondition in code (delegates to utils)
        _extract_effects: Extract effects (delegates to extractor)
        _validate_naming_effect_consistency: Validate naming consistency (delegates to utils)

    Attributes (from SkillGraphManager):
        graph: SkillGraph instance
        ckpt_dir: Checkpoint directory
        logger: Logger instance
        _precondition_extractor: PreconditionExtractor instance
        _effect_extractor: EffectExtractor instance
    """

    # =========================================================================
    # Metadata Update - Core method for updating skill metadata after code changes
    # =========================================================================

    def _update_skill_metadata(
        self: "SkillGraphManager",
        skill_name: str,
        new_code: str,
        new_description: str,
        update_source: str = "",
    ) -> bool:
        """
        Update skill metadata (parameters, preconditions, effects, etc.)

        Args:
            skill_name: Skill name
            new_code: New code
            new_description: New description
            update_source: Update source (e.g. "refactor:parametric", "optimizer")

        Returns:
            True if metadata update succeeded, False if blocked (e.g. all effects invalidated by refactor)
        """
        if skill_name not in self.graph.nodes:
            return True

        node = self.graph.get_node(skill_name)
        old_parameters = node.parameters.copy()

        # 1. Re-extract parameters
        print(f"\033[36m[Metadata Update] Re-extracting parameters for {skill_name}...\033[0m")
        extracted_parameters = self._extract_parameters(new_code, new_description, skill_name)

        if extracted_parameters:
            # Semantic inheritance: if new parameter lacks semantic info, inherit from old
            for param_name, new_param in extracted_parameters.items():
                if param_name in old_parameters:
                    old_param = old_parameters[param_name]
                    old_semantic = old_param.get("semantic")
                    new_semantic = new_param.get("semantic")

                    # If new has no semantic but old does, inherit
                    if not new_semantic and old_semantic:
                        new_param["semantic"] = old_semantic
                        print(f"\033[36m[Semantic] {skill_name}.{param_name}: Inheriting old semantic {old_semantic}\033[0m")
                    # If semantic changed, log it
                    elif new_semantic and old_semantic and new_semantic != old_semantic:
                        print(f"\033[33m[Semantic] {skill_name}.{param_name}: Semantic changed {old_semantic} -> {new_semantic}\033[0m")

            node.parameters = extracted_parameters
            self.set_skill_parameters(skill_name, extracted_parameters)

            # Detect parameter changes
            if old_parameters != extracted_parameters:
                print(f"\033[33m[Metadata Update] Parameter changes detected:\033[0m")
                old_param_names = set(old_parameters.keys())
                new_param_names = set(extracted_parameters.keys())

                added = new_param_names - old_param_names
                removed = old_param_names - new_param_names
                changed = old_param_names & new_param_names

                if added:
                    print(f"  Added parameters: {', '.join(added)}")
                if removed:
                    print(f"  Removed parameters: {', '.join(removed)}")
                if changed:
                    # Check if parameter type or default changed
                    for param_name in changed:
                        old_param = old_parameters[param_name]
                        new_param = extracted_parameters[param_name]
                        if old_param != new_param:
                            print(f"  Modified parameter: {param_name}")
                            if old_param.get("type") != new_param.get("type"):
                                print(f"    Type: {old_param.get('type')} -> {new_param.get('type')}")
                            if old_param.get("default") != new_param.get("default"):
                                print(f"    Default: {old_param.get('default')} -> {new_param.get('default')}")
                            if old_param.get("semantic") != new_param.get("semantic"):
                                print(f"    Semantic: {old_param.get('semantic')} -> {new_param.get('semantic')}")
            else:
                print(f"\033[32m[Metadata Update] Parameters unchanged\033[0m")
        else:
            print(f"\033[33m[Metadata Update] Could not extract parameters, keeping original\033[0m")

        # 2. Validate existing effects still match new code
        strategy_changed = False  # Track strategy change (for description update)
        if hasattr(node, 'expected_effects') and node.expected_effects:
            original_effects = copy.deepcopy(node.expected_effects)  # deepcopy for potential rollback
            original_effect_count = len(node.expected_effects)
            print(f"\033[36m[Effect Validation] Validating {original_effect_count} effects for {skill_name}...\033[0m")
            validated_effects, warnings = self._validate_effects_against_code(
                node.expected_effects, new_code
            )

            removed_count = original_effect_count - len(validated_effects)
            if removed_count > 0:
                # Detect strategy change: effects removed >= 50% indicates major change
                removal_ratio = removed_count / original_effect_count
                if removal_ratio >= 0.5 or len(validated_effects) == 0:
                    strategy_changed = True
                    print(f"\033[35m[Strategy Change] Detected potential major strategy change (effects removal ratio: {removal_ratio:.0%})\033[0m")
                print(f"\033[33m[Effect Update] {removed_count} effects inconsistent with new code, removed:\033[0m")
                # Find and print removed effects
                validated_set = {e.description for e in validated_effects}
                for effect in node.expected_effects:
                    if effect.description not in validated_set:
                        print(f"  - {effect.description}")

                # Update effects
                node.expected_effects = validated_effects

                # Save to disk
                effects_file = f"{self.ckpt_dir}/skill_graph/effects/{skill_name}.json"
                effects_data = self._serialize_effects(validated_effects)
                U.dump_json(effects_data, effects_file)

                # Block or warn if all effects invalidated
                if len(validated_effects) == 0:
                    if update_source.startswith("refactor"):
                        # Refactoring caused all effects to be lost → semantically incompatible, BLOCK
                        print(f"\033[31m[Effect Guard] All effects for {skill_name} invalidated by {update_source} — BLOCKING\033[0m")
                        # Rollback effects (both in-memory and on-disk)
                        node.expected_effects = [copy.deepcopy(e) for e in original_effects]
                        effects_data = self._serialize_effects(node.expected_effects)
                        U.dump_json(effects_data, effects_file)
                        return False  # Block signal
                    else:
                        # Non-refactor updates (optimizer etc.) → keep existing warning behavior
                        print(f"\033[31m[Effect Warning] All effects for {skill_name} invalidated, manual review needed\033[0m")
            else:
                print(f"\033[32m[Effect Validation] All effects validated\033[0m")

            # Print warnings if any
            for warning in warnings:
                print(f"\033[33m[Effect Warning] {warning}\033[0m")

        # 3. Validate existing preconditions still match new code
        # Note: Preconditions use conservative strategy - warn but don't remove
        if hasattr(node, 'preconditions') and node.preconditions:
            print(f"\033[36m[Precondition Validation] Validating {len(node.preconditions)} preconditions for {skill_name}...\033[0m")
            _, precondition_warnings = self._precondition_extractor.validate_against_code(
                node.preconditions, new_code
            )

            if precondition_warnings:
                print(f"\033[33m[Precondition Warning] {len(precondition_warnings)} preconditions may be inconsistent with new code:\033[0m")
                for warning in precondition_warnings:
                    print(f"  - {warning}")
                print(f"\033[33m[Precondition Note] Conservative handling: keeping preconditions, manual review recommended\033[0m")
            else:
                print(f"\033[32m[Precondition Validation] All preconditions validated\033[0m")

        # 4. When strategy changes significantly, update description
        # Trigger: effects removed >= 50% or all invalidated
        if strategy_changed:
            print(f"\033[36m[Description Update] Strategy change detected, regenerating description for {skill_name}...\033[0m")
            try:
                old_description = node.description if hasattr(node, 'description') else ""

                # Regenerate description
                new_generated_description = self.generate_skill_description(skill_name, new_code)

                if new_generated_description and new_generated_description != old_description:
                    # Update node description
                    node.description = new_generated_description

                    # Save to disk
                    desc_file = f"{self.ckpt_dir}/skill_graph/description/{skill_name}.txt"
                    with open(desc_file, "w") as f:
                        f.write(new_generated_description)

                    # Update vector database (idempotent deletion)
                    try:
                        existing = self.vectordb._collection.get(ids=[skill_name])
                        if existing and existing.get("ids"):
                            self.vectordb._collection.delete(ids=[skill_name])
                        self.vectordb.add_texts(
                            texts=[new_generated_description],
                            ids=[skill_name],
                            metadatas=[{"name": skill_name}]
                        )
                        print(f"\033[32m[Description Update] Updated description and vector database\033[0m")
                    except Exception as ve:
                        print(f"\033[33m[Description Warning] Vector database update failed: {ve}\033[0m")

                    # Print old/new description comparison (simplified)
                    old_first_line = old_description.split('\n')[1] if old_description and '\n' in old_description else old_description[:50]
                    new_first_line = new_generated_description.split('\n')[1] if '\n' in new_generated_description else new_generated_description[:50]
                    print(f"\033[36m  Old description: {old_first_line}...\033[0m")
                    print(f"\033[36m  New description: {new_first_line}...\033[0m")
                else:
                    print(f"\033[32m[Description Update] Description unchanged, no update needed\033[0m")
            except Exception as e:
                print(f"\033[33m[Description Warning] Description update failed: {e}\033[0m")

        return True  # Metadata update succeeded

    def _analyze_interface_impact(
        self: "SkillGraphManager",
        old_code: str,
        new_code: str,
    ) -> Dict[str, Any]:
        """
        Analyze interface impact (impact on callers).

        Args:
            old_code: Old code
            new_code: New code

        Returns:
            Dict[str, Any]: Interface impact analysis
        """
        # Extract function signatures
        old_signature_match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', old_code)
        new_signature_match = re.search(r'async\s+function\s+(\w+)\s*\(([^)]*)\)', new_code)

        old_signature = old_signature_match.group(0) if old_signature_match else ""
        new_signature = new_signature_match.group(0) if new_signature_match else ""

        # Extract parameter lists (for detailed comparison)
        old_params_str = old_signature_match.group(2) if old_signature_match else ""
        new_params_str = new_signature_match.group(2) if new_signature_match else ""

        old_params = [p.strip().split('=')[0].strip() for p in old_params_str.split(',') if p.strip() and p.strip() != 'bot']
        new_params = [p.strip().split('=')[0].strip() for p in new_params_str.split(',') if p.strip() and p.strip() != 'bot']

        has_interface_change = old_signature != new_signature

        return {
            "has_interface_change": has_interface_change,
            "old_signature": old_signature,
            "new_signature": new_signature,
            "old_params": old_params,
            "new_params": new_params,
            "impact_level": "high" if has_interface_change else "low",
        }

    # =========================================================================
    # Extraction Methods - Delegate to metadata module
    # =========================================================================

    def _extract_preconditions(
        self: "SkillGraphManager",
        code: str,
        description: str,
        task: str = None,
        context: str = None
    ) -> Tuple[List[SkillPrecondition], List[str]]:
        """
        Delegate to PreconditionExtractor

        Mixed extraction of preconditions: from code and task intent, validate consistency.
        """
        return self._precondition_extractor.extract(
            code=code,
            description=description,
            task=task,
            context=context
        )

    def _extract_effects(
        self: "SkillGraphManager",
        code: str,
        description: str,
        task: str = None,
        context: str = None,
        skill_name: str = None,
        execution_traces: Optional[List[SkillExecutionTrace]] = None
    ) -> Tuple[List[SkillEffect], List[str]]:
        """
        Delegate to EffectExtractor

        Mixed extraction of effects: from code and task intent, validate consistency.

        Args:
            code: Skill code
            description: Skill description
            task: Task description (optional)
            context: Context info (optional)
            skill_name: Skill name for inference
            execution_traces: Execution history for runtime validation (optional)
        """
        return self._effect_extractor.extract(
            code=code,
            description=description,
            task=task,
            context=context,
            skill_name=skill_name,
            execution_traces=execution_traces
        )

    def _validate_naming_effect_consistency(
        self: "SkillGraphManager",
        skill_name: str,
        effects: List[SkillEffect]
    ) -> List[str]:
        """Delegate to code_verifier module (v4.0 refactor)"""
        return validate_naming_effect_consistency(skill_name, effects)
