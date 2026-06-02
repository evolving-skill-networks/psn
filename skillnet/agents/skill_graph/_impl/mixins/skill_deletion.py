"""
SkillDeletionAndCleanupMixin for SkillGraphManager

Skill deletion (5-layer cleanup + backup) and graph validation.
Extracted from graph_manager_impl.py for better maintainability.
"""

import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

import skillnet.utils as U

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class SkillDeletionAndCleanupMixin:
    """Skill deletion (5-layer cleanup + backup) and graph validation."""

    # ========== Phase 9: delete_skill() full deletion method ==========

    def delete_skill(self, skill_name: str, reason: str = "") -> bool:
        """
        Fully delete a skill (5-layer cleanup + backup).

        Phase 9 implementation: deprecated skills are truly removed, not just flagged.

        Args:
            skill_name: name of the skill to delete
            reason: reason for deletion (used in logs and backup records)

        Returns:
            bool: whether deletion succeeded
        """
        node = self.get_node(skill_name)
        if not node:
            self.logger.warning(f"[Delete] Skill {skill_name} does not exist")
            return False

        # Dependency protection: check for validated callers
        # If a verified skill depends on this one, refuse to delete to avoid breaking deps
        verified_callers = []
        for caller in self.get_parents(skill_name):
            caller_node = self.get_node(caller)
            if caller_node and getattr(caller_node, 'is_verified', False):
                verified_callers.append(caller)

        if verified_callers:
            self.logger.warning(
                f"[Delete] ⚠️ Refusing to delete {skill_name}: called by verified skills {verified_callers}."
                f"Deletion would break those skills' dependencies."
            )
            return False

        self.logger.info(f"[Delete] Starting deletion of {skill_name}, Reason: {reason}")

        # Back up first (to prevent accidental deletion)
        self._backup_skill_before_delete(skill_name, reason)

        # Layer 1: in-memory graph cleanup
        self._delete_from_graph(skill_name)

        # Layer 2: vector-database cleanup
        self._delete_from_vectordb(skill_name)

        # Layer 3: disk-file cleanup
        self._delete_from_disk(skill_name)

        # Layer 4: graph JSON snapshot update
        self._save_to_checkpoint()

        # Layer 5: relationship cleanup
        self._cleanup_coverage_relations(skill_name)

        self.logger.info(f"[Delete] Finished deleting {skill_name}")
        return True

    def _backup_skill_before_delete(self, skill_name: str, reason: str = "") -> None:
        """Back up the skill to the deleted/ directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f"{self.ckpt_dir}/skill_graph/deleted/{skill_name}_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)

        # Copy all files to the backup directory
        subdirs_and_exts = [
            ("code", self._get_file_extension()),
            ("description", ".txt"),
            ("preconditions", ".json"),
            ("effects", ".json"),
            ("metadata", ".json"),
        ]

        for subdir, ext in subdirs_and_exts:
            src = f"{self.ckpt_dir}/skill_graph/{subdir}/{skill_name}{ext}"
            if os.path.exists(src):
                dst = f"{backup_dir}/{skill_name}{ext}"
                shutil.copy2(src, dst)
                self.logger.debug(f"[Delete] Backed up file {src} -> {dst}")

        # Record deletion info
        deletion_info = {
            "skill_name": skill_name,
            "deleted_at": datetime.now().isoformat(),
            "reason": reason,
            "backup_dir": backup_dir,
        }
        U.dump_json(deletion_info, f"{backup_dir}/deletion_info.json")

        self.logger.info(f"[Delete] Backed up {skill_name} to {backup_dir}")

    def _delete_from_graph(self, skill_name: str) -> None:
        """Layer 1: remove the node and all its edges from the in-memory graph"""
        if self.graph.has_node(skill_name):
            self.graph.remove_node(skill_name)
            self.logger.debug(f"[Delete] Removed {skill_name} from in-memory graph")

    def _delete_from_vectordb(self, skill_name: str) -> None:
        """Layer 2: remove the record from the vector database (idempotent)"""
        try:
            # Check existence first to avoid warnings from deleting a missing ID
            existing = self.vectordb._collection.get(ids=[skill_name])
            if existing and existing.get("ids"):
                self.vectordb._collection.delete(ids=[skill_name])
                self.logger.debug(f"[Delete] Removed {skill_name} from vector database")
            else:
                self.logger.debug(f"[Delete] {skill_name} is not in the vector database; skipping")
        except Exception as e:
            self.logger.warning(f"[Delete] Vector-database delete failed: {e}")

    def _delete_from_disk(self, skill_name: str) -> None:
        """Layer 3: delete every related file on disk"""
        files_to_remove = [
            f"{self.ckpt_dir}/skill_graph/code/{skill_name}{self._get_file_extension()}",
            f"{self.ckpt_dir}/skill_graph/description/{skill_name}.txt",
            f"{self.ckpt_dir}/skill_graph/preconditions/{skill_name}.json",
            f"{self.ckpt_dir}/skill_graph/effects/{skill_name}.json",
            f"{self.ckpt_dir}/skill_graph/metadata/{skill_name}.json",
        ]

        for filepath in files_to_remove:
            if os.path.exists(filepath):
                os.remove(filepath)
                self.logger.debug(f"[Delete] Removed file {filepath}")

    def _cleanup_coverage_relations(self, skill_name: str) -> None:
        """Layer 5: clean up coverage relationships"""
        for other_name, other_node in self.graph.nodes.items():
            if getattr(other_node, 'covered_by', None) == skill_name:
                other_node.is_covered = False
                other_node.covered_by = None
                self.logger.debug(f"[Delete] Cleared coverage relation of {other_name}")

    # ========== End Phase 9 ==========

    def validate_and_clean_graph(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Validate and clean up incorrect dependencies in the skill graph.

        For each skill, check whether the children are actually called in its code.
        If a child is not called in the code, remove that edge.

        Used to repair invalid dependencies caused by code-generation errors.
        For example: if craft_1_stone_pickaxe's children includes mineCoal,
        but craft_1_stone_pickaxe's code does not call mineCoal,
        that edge will be removed.

        Args:
            verbose: whether to print detailed info

        Returns:
            Dict[str, Any]: cleanup-result statistics
            {
                "checked_skills": number of skills checked,
                "removed_edges": list of removed edges [{"parent": str, "child": str}, ...],
                "valid_edges": number of valid edges
            }
        """
        result = {
            "checked_skills": 0,
            "removed_edges": [],
            "valid_edges": 0
        }

        for skill_name in list(self.graph.nodes.keys()):
            node = self.graph.get_node(skill_name)
            if not node or not node.code:
                continue

            result["checked_skills"] += 1

            # Extract the functions actually called in the code
            called_functions = set(self._extract_function_calls(node.code))

            # Check whether each child is called in the code
            children_to_remove = []
            for child_name in node.children:
                if child_name not in called_functions:
                    # This child is not called in the code; it's an invalid dependency
                    children_to_remove.append(child_name)
                    result["removed_edges"].append({
                        "parent": skill_name,
                        "child": child_name
                    })
                    if verbose:
                        print(f"\033[33m[Graph Cleanup] Removing invalid edge: {skill_name} -> {child_name} "
                              f"(skill is not called in the code)\033[0m")
                else:
                    result["valid_edges"] += 1

            # Remove invalid edges
            for child_name in children_to_remove:
                self.graph.remove_edge(skill_name, child_name)

        if verbose:
            print(f"\033[32m[Graph Cleanup] Done: checked {result['checked_skills']} skills, "
                  f"removed {len(result['removed_edges'])} invalid edges, "
                  f"kept {result['valid_edges']} valid edges\033[0m")

        return result
