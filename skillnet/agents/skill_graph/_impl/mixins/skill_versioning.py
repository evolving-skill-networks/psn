"""
Skill Versioning Mixin for SkillGraphManager

This module contains methods for version creation, rollback, and history management.

Extracted from graph_manager_impl.py for better modularity.
"""

import copy
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from skillnet.agents.skill_graph.models import (
    SkillVersion,
)
from skillnet.agents.skill_graph.utils import (
    validate_code_brackets,
    validate_code_syntax,
)

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class SkillVersioningMixin:
    """Skill Versioning Mixin - Version creation, rollback, and history management.

    Methods:
        create_new_version: Create new version (~240 lines)
        rollback_skill: Rollback single skill
        rollback_skill_and_subgraph: Rollback skill and its subgraph

    Note:
        Version management involves complex cascading updates and dependency handling.

    Attributes (from SkillGraphManager):
        graph: SkillGraph instance
        vectordb: ChromaDB instance
        ckpt_dir: Checkpoint directory
        _save_to_checkpoint: Method to save to checkpoint
        generate_skill_description: Method to generate skill description
        _update_skill_metadata: Method to update skill metadata
        _analyze_interface_impact: Method to analyze interface impact
        _notify_code_changed: Method to notify code changed
    """

    def create_new_version(
        self: "SkillGraphManager",
        skill_name: str,
        new_code: str,
        new_description: str = None,
        change_log: str = "",
        auto_update_metadata: bool = True,
        auto_update_callers: bool = True,
        update_source: str = "manual",
        update_reason: str = "",
        optimization_id: str = None,
        used_feedbacks: List[Dict[str, Any]] = None,
        backpropagation_info: Dict[str, Any] = None,
    ) -> bool:
        """
        Create a new version of a skill.

        Args:
            skill_name: Skill name
            new_code: New version code
            new_description: New version description (if None, will be regenerated)
            change_log: Change description
            auto_update_metadata: Whether to auto-update metadata (parameters, preconditions, effects)
            auto_update_callers: Whether to auto-update callers (if interface change detected)
            update_source: Update source ("optimizer", "refactor", "manual", "backpropagation", "unknown")
            update_reason: Update reason (detailed explanation)
            optimization_id: Associated optimization process ID (if from optimizer)
            used_feedbacks: List of used feedbacks
            backpropagation_info: Backpropagation information

        Returns:
            bool: Whether version was successfully created
        """
        if skill_name not in self.graph.nodes:
            print(f"\033[31mError: Skill '{skill_name}' not found.\033[0m")
            return False

        # Validate bracket balance in new code
        is_bracket_valid, bracket_error, bracket_details = validate_code_brackets(new_code)
        if not is_bracket_valid:
            print(f"\033[31m[Version Create] Error: Skill '{skill_name}' has unbalanced brackets: {bracket_error}\033[0m")
            print(f"\033[31m[Version Create]   Details: parentheses({bracket_details['parentheses']['open']}/{bracket_details['parentheses']['close']}), "
                  f"brackets({bracket_details['brackets']['open']}/{bracket_details['brackets']['close']}), "
                  f"braces({bracket_details['braces']['open']}/{bracket_details['braces']['close']})\033[0m")
            print(f"\033[31m[Version Create] Rejected: Optimized code must have no syntax errors\033[0m")
            return False

        # Full syntax validation (using Babel)
        is_syntax_valid, syntax_error = validate_code_syntax(new_code)
        if not is_syntax_valid:
            print(f"\033[31m[Version Create] Error: Skill '{skill_name}' syntax validation failed - {syntax_error}\033[0m")
            print(f"\033[31m[Version Create] Rejected: Optimized code must have no syntax errors\033[0m")
            return False

        # [Function name consistency validation]
        func_name_match = re.search(r'async\s+function\s+(\w+)\s*\(', new_code)
        if func_name_match:
            actual_func_name = func_name_match.group(1)
            if actual_func_name != skill_name:
                print(f"\033[33m[Version Create] Warning: Function name '{actual_func_name}' "
                      f"doesn't match skill name '{skill_name}'\033[0m")
                print(f"\033[33m[Version Create] Auto-correcting: '{actual_func_name}' → '{skill_name}'\033[0m")

                # Step 1: Fix function definition name
                new_code = re.sub(
                    rf'(async\s+function\s+){re.escape(actual_func_name)}(\s*\()',
                    rf'\g<1>{skill_name}\g<2>',
                    new_code,
                    count=1
                )

                # Step 2: Replace all variable references to old function name
                new_code = re.sub(
                    rf'\b{re.escape(actual_func_name)}\b',
                    skill_name,
                    new_code
                )

                # Re-validate syntax after fix
                is_syntax_valid_after_fix, syntax_error_after_fix = validate_code_syntax(new_code)
                if not is_syntax_valid_after_fix:
                    print(f"\033[31m[Version Create] Error: Syntax validation failed after name fix - {syntax_error_after_fix}\033[0m")
                    return False

        node = self.graph.get_node(skill_name)
        old_code = node.code

        # Generate new description if needed
        if new_description is None:
            new_description = self.generate_skill_description(skill_name, new_code)

        # Determine new version number
        latest_version = node.get_latest_version()
        if latest_version:
            version_parts = latest_version.version.split(".")
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            new_version = ".".join(version_parts)
        else:
            new_version = "1.0.1"

        # 1. Auto-update metadata if enabled
        if auto_update_metadata:
            metadata_ok = self._update_skill_metadata(
                skill_name, new_code, new_description,
                update_source=update_source,
            )
            if not metadata_ok:
                self.logger.warning(
                    f"[Version] Metadata validation blocked version creation for '{skill_name}' (source: {update_source})"
                )
                return None  # Block version creation

        # 2. Detect interface changes
        interface_impact = None
        if auto_update_callers:
            interface_impact = self._analyze_interface_impact(old_code, new_code)

        # Create version snapshot
        preconditions_snapshot = [copy.deepcopy(p) for p in node.preconditions]
        effects_snapshot = [copy.deepcopy(e) for e in node.expected_effects]
        parameters_snapshot = copy.deepcopy(node.parameters)
        value_function_snapshot = node.value_function
        statistics_snapshot = {
            "total_executions": node.statistics.total_executions,
            "successful_executions": node.statistics.successful_executions,
            "failed_executions": node.statistics.failed_executions,
            "success_rate": node.statistics.success_rate,
            "value_function_lambda": node.value_function_lambda,
            "value_function_alpha": node.value_function_alpha,
            "value_function_beta": node.value_function_beta,
        }

        # Record affected subgraph (for rollback)
        previous_version = latest_version.version if latest_version else None
        directly_affected = self.graph.get_parents(skill_name)
        indirectly_affected = []

        # Recursively get indirectly affected skills
        visited = set()
        def get_indirect_affected(skill: str):
            if skill in visited:
                return
            visited.add(skill)
            parents = self.graph.get_parents(skill)
            for parent in parents:
                if parent not in directly_affected and parent != skill_name:
                    indirectly_affected.append(parent)
                    get_indirect_affected(parent)

        for affected_skill in directly_affected:
            get_indirect_affected(affected_skill)

        # Create graph snapshot
        graph_snapshot = {
            "node_versions": {n: self.graph.get_node(n).get_latest_version().version if self.graph.get_node(n).get_latest_version() else "1.0.0"
                             for n in self.graph.nodes},
            "edges": {p: list(children) if isinstance(children, set) else children for p, children in self.graph.edges.items()},
        }

        affected_subgraph = {
            "directly_affected": directly_affected,
            "indirectly_affected": indirectly_affected,
            "graph_snapshot": graph_snapshot,
            "previous_version": previous_version,
        }

        # Create new version with complete snapshot
        version_obj = SkillVersion(
            version=new_version,
            code=new_code,
            description=new_description,
            change_log=change_log or f"{update_source} version update to {new_version}",
            preconditions=preconditions_snapshot,
            effects=effects_snapshot,
            parameters=parameters_snapshot,
            value_function=value_function_snapshot,
            statistics_snapshot=statistics_snapshot,
            update_source=update_source,
            update_reason=update_reason or change_log or f"{update_source} version update to {new_version}",
            optimization_id=optimization_id,
            used_feedbacks=used_feedbacks or [],
            backpropagation_info=backpropagation_info,
            affected_subgraph=affected_subgraph,
        )

        # Check for version number conflict
        existing_versions = [v.version for v in node.versions]
        if new_version in existing_versions:
            print(f"\033[33m[WARNING] Version {new_version} already exists for '{skill_name}'! Existing versions: {existing_versions}\033[0m")
            print(f"\033[33m[WARNING] Skipping version creation to avoid conflict.\033[0m")
            return False

        node.add_version(version_obj)

        # Log version creation
        print(f"\033[32m[Version Created] Created version {new_version} for '{skill_name}':\033[0m")
        print(f"\033[32m  - Update source: {update_source}\033[0m")
        print(f"\033[32m  - Change log: {change_log or f'{update_source} version update to {new_version}'}\033[0m")
        print(f"\033[32m  - Total versions: {len(node.versions)}\033[0m")
        if optimization_id:
            print(f"\033[32m  - Optimization ID: {optimization_id}\033[0m")
        if used_feedbacks:
            print(f"\033[32m  - Used {len(used_feedbacks)} feedbacks\033[0m")
        if backpropagation_info:
            print(f"\033[32m  - Backpropagation from: {backpropagation_info.get('source_skill', 'unknown')}\033[0m")

        # Update node code and description
        node.code = new_code
        node.description = new_description

        # Update vector database (idempotent deletion)
        try:
            existing = self.vectordb._collection.get(ids=[skill_name])
            if existing and existing.get("ids"):
                self.vectordb._collection.delete(ids=[skill_name])
        except:
            pass

        self.vectordb.add_texts(
            texts=[new_description],
            ids=[skill_name],
            metadatas=[{"name": skill_name}],
        )

        # 3. If interface change detected, trigger caller update via callback
        if auto_update_callers and interface_impact and interface_impact.get("has_interface_change"):
            print(f"\033[33m[Metadata Update] Interface change detected: {skill_name}\033[0m")
            print(f"  Old signature: {interface_impact.get('old_signature', 'N/A')}")
            print(f"  New signature: {interface_impact.get('new_signature', 'N/A')}")

            self._notify_code_changed(
                skill_name=skill_name,
                old_code=old_code,
                new_code=new_code,
                change_source="version_update",
            )

        # Save to checkpoint
        self._save_to_checkpoint()

        print(f"\033[32mCreated new version {new_version} for '{skill_name}'\033[0m")
        return True

    def rollback_skill_and_subgraph(
        self: "SkillGraphManager",
        skill_name: str,
        target_version: str,
        rollback_subgraph: bool = True,
    ) -> bool:
        """
        Rollback skill and its affected subgraph to specified version.

        Args:
            skill_name: Skill name
            target_version: Target version number
            rollback_subgraph: Whether to also rollback affected subgraph

        Returns:
            bool: Whether rollback was successful
        """
        if skill_name not in self.graph.nodes:
            print(f"\033[31m[Rollback] Error: Skill '{skill_name}' not found\033[0m")
            return False

        node = self.graph.get_node(skill_name)
        current_version = node.get_latest_version()

        if not current_version:
            print(f"\033[31m[Rollback] Error: Skill '{skill_name}' has no version info\033[0m")
            return False

        # Get affected subgraph info
        affected_subgraph = current_version.affected_subgraph
        if not affected_subgraph:
            print(f"\033[33m[Rollback] Warning: No affected subgraph info found, only rolling back {skill_name}\033[0m")
            rollback_subgraph = False

        # 1. Rollback main skill
        success = node.rollback_to_version(target_version)
        if not success:
            print(f"\033[31m[Rollback] Failed to rollback main skill: {skill_name}\033[0m")
            return False

        print(f"\033[32m[Rollback] Successfully rolled back {skill_name} to version {target_version}\033[0m")

        # 2. Rollback affected subgraph if enabled
        if rollback_subgraph and affected_subgraph:
            directly_affected = affected_subgraph.get("directly_affected", [])
            graph_snapshot = affected_subgraph.get("graph_snapshot", {})
            node_versions = graph_snapshot.get("node_versions", {})

            # Rollback directly affected skills
            for affected_skill in directly_affected:
                if affected_skill in self.graph.nodes:
                    snapshot_version = node_versions.get(affected_skill)
                    if snapshot_version:
                        affected_node = self.graph.get_node(affected_skill)
                        if affected_node:
                            affected_node.rollback_to_version(snapshot_version)
                            print(f"\033[32m[Rollback] Rolled back affected skill: {affected_skill} to version {snapshot_version}\033[0m")

            # Rollback indirectly affected skills
            indirectly_affected = affected_subgraph.get("indirectly_affected", [])
            for affected_skill in indirectly_affected:
                if affected_skill in self.graph.nodes:
                    snapshot_version = node_versions.get(affected_skill)
                    if snapshot_version:
                        affected_node = self.graph.get_node(affected_skill)
                        if affected_node:
                            affected_node.rollback_to_version(snapshot_version)
                            print(f"\033[32m[Rollback] Rolled back indirectly affected skill: {affected_skill} to version {snapshot_version}\033[0m")

        # Save to checkpoint
        self._save_to_checkpoint()

        return True

    def rollback_skill(self: "SkillGraphManager", skill_name: str, target_version: str) -> bool:
        """
        Rollback skill to specified version (public interface).

        Args:
            skill_name: Skill name
            target_version: Target version number (e.g. "1.0.0")

        Returns:
            bool: Whether rollback was successful
        """
        if skill_name not in self.graph.nodes:
            print(f"\033[31m[Rollback] Error: Skill '{skill_name}' not found\033[0m")
            return False

        node = self.graph.get_node(skill_name)
        success = node.rollback_to_version(target_version)

        if success:
            # Update vector database (idempotent deletion)
            try:
                existing = self.vectordb._collection.get(ids=[skill_name])
                if existing and existing.get("ids"):
                    self.vectordb._collection.delete(ids=[skill_name])
            except:
                pass

            self.vectordb.add_texts(
                texts=[node.description],
                ids=[skill_name],
                metadatas=[{"name": skill_name}],
            )

            # Save to checkpoint
            self._save_to_checkpoint()

        return success
