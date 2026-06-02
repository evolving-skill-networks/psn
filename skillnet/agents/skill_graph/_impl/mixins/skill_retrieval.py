"""
Skill retrieval and discovery mixin for SkillGraphManager.

Provides methods for finding, retrieving, and analyzing skills:
- Task-intent based skill lookup (deduplication)
- Vector DB similarity search with filtering
- Code analysis: extracting function calls, definitions, and call chains
- Graph structure updates from code dependencies
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from skillnet.agents.skill_graph.utils import extract_function_calls
from skillnet.agents.skill_graph.models import should_skip_covered_skill

if TYPE_CHECKING:
    from ..graph_manager_impl import SkillGraphManager


class SkillRetrievalAndDiscoveryMixin:

    def find_skill_by_task_intent(
        self,
        task: str,
        context: str = None,
        threshold: float = 0.15
    ) -> Optional[str]:
        """
        Find an existing skill with the same functionality based on task intent.

        Used to avoid creating duplicate skills: when a new skill fails, check whether an existing
        skill can accomplish the same task; if so, do not save the new skill.

        Args:
            task: task description (e.g. "Craft 1 wooden_pickaxe")
            context: task context
            threshold: similarity threshold (lower scores indicate more similar)

        Returns:
            Optional[str]: name of the matching existing skill, or None if not found
        """
        import re

        if not task:
            return None

        # Extract target effects from the task
        # E.g. "Craft 1 wooden_pickaxe" → extract "wooden_pickaxe"
        # "Mine 3 iron_ore" → extract "iron_ore"
        target_item = None
        target_count = 1
        task_type = None

        # Parse the task pattern
        craft_match = re.match(r'[Cc]raft\s+(\d+)\s+(\w+)', task)
        mine_match = re.match(r'[Mm]ine\s+(\d+)\s+(\w+)', task)
        obtain_match = re.match(r'[Oo]btain\s+(\d+)\s+(\w+)', task)

        if craft_match:
            target_count = int(craft_match.group(1))
            target_item = craft_match.group(2)
            task_type = "craft"
        elif mine_match:
            target_count = int(mine_match.group(1))
            target_item = mine_match.group(2)
            task_type = "mine"
        elif obtain_match:
            target_count = int(obtain_match.group(1))
            target_item = obtain_match.group(2)
            task_type = "obtain"

        if not target_item:
            return None

        print(f"\033[36m[Find Skill] Looking for an existing skill that can accomplish '{task}'...\033[0m")
        print(f"\033[36m[Find Skill] Target: {task_type} {target_count} {target_item}\033[0m")

        # Method 1: match via effects
        for skill_name, node in self.graph.nodes.items():
            # Skip deprecated skills (set by the graph planner on task-specific
            # low-reuse compositions; see SkillNode.is_deprecated).
            if node.is_deprecated:
                continue
            # Skip task-specific skills
            if getattr(node, 'is_task_specific', False):
                continue

            # Check the skill's expected_effects
            for effect in node.expected_effects:
                effect_desc = getattr(effect, 'description', str(effect)).lower()

                # Check whether the effect matches the target
                if target_item.lower() in effect_desc:
                    if task_type == "craft" and ("craft" in skill_name.lower() or "add" in effect_desc):
                        print(f"\033[32m[Find Skill] Matched via effects: {skill_name}\033[0m")
                        return skill_name
                    elif task_type == "mine" and ("mine" in skill_name.lower() or "add" in effect_desc):
                        print(f"\033[32m[Find Skill] Matched via effects: {skill_name}\033[0m")
                        return skill_name
                    elif task_type == "obtain" and "add" in effect_desc:
                        print(f"\033[32m[Find Skill] Matched via effects: {skill_name}\033[0m")
                        return skill_name

        # Method 2: match by name similarity
        expected_names = []
        if task_type == "craft":
            # craftWoodenPickaxe, craftPickaxe
            camel_case_item = ''.join(word.capitalize() for word in target_item.split('_'))
            expected_names = [
                f"craft{camel_case_item}",
                f"craft{target_item.replace('_', '')}",
            ]
        elif task_type == "mine":
            camel_case_item = ''.join(word.capitalize() for word in target_item.split('_'))
            expected_names = [
                f"mine{camel_case_item}",
                f"mine{target_item.replace('_', '')}",
            ]

        for expected_name in expected_names:
            for skill_name in self.graph.nodes:
                if skill_name.lower() == expected_name.lower():
                    node = self.graph.get_node(skill_name)
                    if not node.is_deprecated and not getattr(node, 'is_task_specific', False):
                        print(f"\033[32m[Find Skill] Matched by name: {skill_name}\033[0m")
                        return skill_name

        # Method 3: semantic search via the vector database
        if self.vectordb._collection.count() > 0:
            try:
                docs_and_scores = self.vectordb.similarity_search_with_score(task, k=3)
                for doc, score in docs_and_scores:
                    skill_name = doc.metadata.get('name')
                    if score <= threshold and skill_name in self.graph.nodes:
                        node = self.graph.get_node(skill_name)
                        if not node.is_deprecated:
                            # Verify it is actually the same functionality
                            if target_item.lower().replace('_', '') in skill_name.lower():
                                print(f"\033[32m[Find Skill] Matched via semantic search: {skill_name} (score={score:.4f})\033[0m")
                                return skill_name
            except Exception as e:
                print(f"\033[33m[Find Skill] Semantic search failed: {e}\033[0m")

        print(f"\033[33m[Find Skill] No functionally equivalent existing skill found\033[0m")
        return None

    def retrieve_skills(self, query: str, return_metadata: bool = False) -> List[str]:
        """
        Retrieve related skills (compatible with the legacy interface).

        Args:
            query: query string
            return_metadata: whether to return parameter metadata (for parameterized skills)

        Returns:
            List[str]: list of matching skill code (up to retrieval_top_k valid results)
            If return_metadata=True, returns a (skills, metadata) tuple
        """
        # Retrieve more candidates to ensure enough valid results after filtering
        # Assume up to 50% of skills may be filtered (deprecated + covered), so retrieve retrieval_top_k * 2
        max_retrieve = min(self.vectordb._collection.count(), self.retrieval_top_k * 2)
        if max_retrieve == 0:
            return [] if not return_metadata else ([], {})

        self.logger.info(f"\033[33mSkill Graph Manager retrieving for {max_retrieve} skills (target: {self.retrieval_top_k} valid results)\033[0m")
        self.logger.info(f"\033[36m[Skill Retrieval] Query: {query[:300]}...\033[0m")
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=max_retrieve)
        retrieved_names = [doc.metadata['name'] for doc, _ in docs_and_scores]
        self.logger.info(
            f"\033[33mSkill Graph Manager retrieved skills: "
            f"{', '.join(retrieved_names)}\033[0m"
        )
        self.logger.info(f"\033[36m[Skill Retrieval] Retrieved skills and similarity scores:\033[0m")
        for doc, score in docs_and_scores:
            self.logger.info(f"\033[36m  - {doc.metadata['name']}: {score:.4f}\033[0m")

        skills = []
        metadata = {}

        for doc, _ in docs_and_scores:
            # If enough valid results have been collected, stop early
            if len(skills) >= self.retrieval_top_k:
                break

            skill_name = doc.metadata["name"]

            # === P0 fix: prefer vectordb metadata filtering (avoid graph-query overhead) ===
            # This handles legacy skills present in vectordb but missing from the graph
            if doc.metadata.get("is_deprecated", False):
                self.logger.warning(f"\033[33m[Skill Retrieval]   Skipping deprecated skill (metadata): {skill_name}\033[0m")
                continue
            if doc.metadata.get("is_task_specific", False):
                self.logger.warning(f"\033[33m[Skill Retrieval]   Skipping task-specific skill (metadata): {skill_name}\033[0m")
                continue
            # === end of fix ===

            if skill_name in self.graph.nodes:
                node = self.graph.nodes[skill_name]
                # Exclude covered skills (except extract_common_subskill type, which are base functionality)
                if should_skip_covered_skill(node):
                    self.logger.warning(f"\033[33m[Skill Retrieval]   Skipping covered skill: {skill_name} (covered by {node.covered_by})\033[0m")
                    # ──────────────────────────────────────────────────────────
                    # Layer 2 fix (Phase C validation): when filtering out a
                    # wrapper, surface its covered_by general skill into the
                    # result. Without this, vector-similarity-based retrieval
                    # may rank the (generic-described) general skill below
                    # top-K and the LLM/planner never sees it. This bridge
                    # ensures every filtered wrapper has its general skill
                    # represented for downstream consumers.
                    # ──────────────────────────────────────────────────────────
                    covered_by_name = getattr(node, 'covered_by', None)
                    if covered_by_name and covered_by_name in self.graph.nodes:
                        cover_node = self.graph.nodes[covered_by_name]
                        # Skip if cover node itself shouldn't be surfaced
                        # (e.g., deprecated, task-specific, or already in
                        # skills/metadata from a prior iteration)
                        already_added = (
                            (cover_node.code in skills) if cover_node.code else False
                        ) or covered_by_name in metadata
                        if (not already_added
                                and not getattr(cover_node, 'is_deprecated', False)
                                and not getattr(cover_node, 'is_task_specific', False)):
                            self.logger.info(
                                f"\033[36m[Skill Retrieval]   Layer2: surfacing covered_by "
                                f"'{covered_by_name}' (general skill of '{skill_name}')\033[0m"
                            )
                            skills.append(cover_node.code)
                            if return_metadata and cover_node.parameters:
                                metadata[covered_by_name] = {
                                    "parameters": cover_node.parameters,
                                    "description": getattr(cover_node, 'description', ''),
                                    "contract": {"is_contract_valid": True, "violations": []},
                                }
                    continue
                # Exclude deprecated skills
                if node.is_deprecated:
                    self.logger.warning(f"\033[33m[Skill Retrieval]   Skipping deprecated skill: {skill_name} (reason: {node.deprecation_reason})\033[0m")
                    continue
                # Exclude task-specific skills (hardcoded parameters, no reuse value)
                if getattr(node, 'is_task_specific', False):
                    self.logger.warning(f"\033[33m[Skill Retrieval]   Skipping task-specific skill: {skill_name} (hardcoded parameters, no reuse value)\033[0m")
                    continue
                skills.append(node.code)

                # Collect parameter metadata
                if return_metadata:
                    if node.parameters:
                        metadata[skill_name] = {
                            "parameters": node.parameters,
                            "description": getattr(node, 'description', ''),  # include the overall skill description
                            "contract": {
                                "is_contract_valid": True,
                                "violations": [],
                            }
                        }
                        self.logger.info(f"\033[36m[Skill Retrieval]   {skill_name} has parameter metadata: {list(node.parameters.keys())}\033[0m")
                    else:
                        self.logger.warning(f"\033[33m[Skill Retrieval]   Warning: {skill_name} lacks parameter metadata (parameters is empty)\033[0m")
            else:
                self.logger.warning(f"\033[33m[Skill Retrieval]   Warning: {skill_name} is in vectordb but not in graph\033[0m")

        # Record the number of valid results actually returned
        if len(skills) < self.retrieval_top_k:
            self.logger.warning(
                f"\033[33m[Skill Retrieval]   Warning: only {len(skills)} valid results after filtering; fewer than target {self.retrieval_top_k}\033[0m"
            )
        else:
            self.logger.info(
                f"\033[36m[Skill Retrieval]   Returning {len(skills)} valid results successfully\033[0m"
            )

        if return_metadata:
            self.logger.info(f"\033[36m[Skill Retrieval] Returned metadata includes skills: {', '.join(metadata.keys()) if metadata else 'None'}\033[0m")

        if return_metadata:
            return skills, metadata
        return skills

    def ensure_skill_in_vectordb(self, skill_name: str) -> bool:
        """
        Ensure the skill is in vectordb; add it if missing.
        Used to fix cases where pre_register_skill leaves the skill out of vectordb.

        pre_register_skill only adds to the graph, not to vectordb. When the skill fails,
        since the skill is already in the graph, record_execution and add_new_skill both skip the vectordb add,
        making the skill unretrievable via retrieve_skills.

        Args:
            skill_name: skill name

        Returns:
            bool: True=newly added, False=already exists or skipped
        """
        if skill_name not in self.graph.nodes:
            return False

        node = self.graph.nodes[skill_name]

        # Task-specific skills are not added to vectordb (consistent with add_new_skill)
        if getattr(node, 'is_task_specific', False):
            self.logger.info(f"\033[33m[VectorDB Sync] Skipping task-specific skill '{skill_name}'\033[0m")
            return False

        # Validate description
        if not node.description or len(node.description.strip()) < 20:
            self.logger.warning(f"\033[33m[VectorDB Sync] Skipping '{skill_name}' - no valid description\033[0m")
            return False

        try:
            existing = self.vectordb._collection.get(ids=[skill_name])
            if existing and existing.get("ids"):
                return False  # already exists

            # === P0 fix: enrich metadata ===
            self.vectordb.add_texts(
                texts=[node.description],
                ids=[skill_name],
                metadatas=[{
                    "name": skill_name,
                    "is_deprecated": getattr(node, 'is_deprecated', False),
                    "is_task_specific": getattr(node, 'is_task_specific', False),
                }],
            )
            # === end of fix ===
            self.logger.info(f"\033[32m[VectorDB Sync] Added '{skill_name}' to vector database\033[0m")
            return True
        except Exception as e:
            self.logger.warning(f"\033[33m[VectorDB Sync] Failed to add '{skill_name}': {e}\033[0m")
            return False

    def set_skill_parameters(self, skill_name: str, parameters: Dict[str, Dict[str, Any]]) -> bool:
        """
        Set the parameter metadata for a skill.

        Args:
            skill_name: skill name
            parameters: parameter metadata dict
                Format: {
                    "paramName": {
                        "type": "number|string|boolean",
                        "default": default_value,
                        "description": "parameter description"
                    }
                }

        Returns:
            bool: whether setting succeeded
        """
        if skill_name not in self.graph.nodes:
            print(f"\033[31mError: Skill '{skill_name}' not found.\033[0m")
            return False

        node = self.graph.nodes[skill_name]
        node.parameters = parameters

        # Save to checkpoint
        self._save_to_checkpoint()

        print(f"\033[32mSet parameters for '{skill_name}': {list(parameters.keys())}\033[0m")
        return True

    def extract_called_skills(self, code: str) -> List[str]:
        """
        Extract called skills from code (returns only skills present in the graph).

        Args:
            code: JavaScript code

        Returns:
            List[str]: list of called skill names (only those present in the graph)
        """
        called_functions = self._extract_function_calls(code)
        # Only return skills present in the graph
        return [func_name for func_name in called_functions if func_name in self.graph.nodes]

    def _extract_function_calls(self, code: str) -> List[str]:
        """Delegate to code_analysis module (v4.0 refactor)."""
        return extract_function_calls(code)

    def extract_all_function_definitions(self, code: str) -> List[Dict[str, str]]:
        """
        Extract all async function definitions from code.

        Used by Eager Skill Registration: extract all function definitions before running code,
        so they can be pre-registered into the skill graph for correct Skill Tracking.

        Delegates to the code_analysis module (v8.0 Skill Tracking fix).

        Args:
            code: JavaScript code

        Returns:
            List[Dict[str, str]]: list of function definitions; each element contains name, code, params
        """
        from skillnet.agents.skill_graph.utils import extract_all_function_definitions as _extract_all_func_defs
        return _extract_all_func_defs(code)

    def _expand_skills_with_call_chain(
        self,
        skills: List[str],
        max_depth: int = 5
    ) -> List[str]:
        """
        [Fix 4] Expand the skill list to include the full call chain.

        Recursively expand each skill's children (called skills) to the given depth.
        Ensures the optimization scope includes all related skills.

        Args:
            skills: initial skill list
            max_depth: maximum recursion depth (to prevent infinite loops)

        Returns:
            List[str]: expanded skill list (order preserved, deduplicated)
        """
        result = []
        visited = set()

        def expand_skill(skill_name: str, depth: int):
            if depth > max_depth or skill_name in visited:
                return

            visited.add(skill_name)

            # Only add skills present in the graph
            if skill_name in self.graph.nodes:
                if skill_name not in result:
                    result.append(skill_name)

                # Recursively expand children
                node = self.graph.get_node(skill_name)
                if node and node.children:
                    for child_name in node.children:
                        expand_skill(child_name, depth + 1)

        # Expand each initial skill
        for skill in skills:
            expand_skill(skill, 0)

        return result

    def _update_graph_from_code(self, skill_name: str) -> None:
        """
        Automatically extract call relationships from code and update the graph structure.

        Args:
            skill_name: skill name
        """
        if skill_name not in self.graph.nodes:
            return

        node = self.graph.get_node(skill_name)
        code = node.code

        # [Fix 2] Use deep extraction, including calls inside helper functions
        called_functions = self._extract_all_skill_calls_deep(code)

        # Get the set of control-primitive names (for identification, no special handling)
        control_primitives_names = set()
        for prim in self.control_primitives:
            func_match = re.search(r'async\s+function\s+(\w+)', prim)
            if func_match:
                control_primitives_names.add(func_match.group(1))

        # Save the current children set (for later comparison)
        old_children_set = set(node.children.copy())

        # Remove all old children edges (including control primitives; we re-add them in order later)
        old_children = node.children.copy()
        for child_name in old_children:
            self.graph.remove_edge(skill_name, child_name)

        # Re-add children relationships in call order (only those present in the graph)
        # Ensures the children list is ordered by call order
        for func_name in called_functions:
            if func_name in self.graph.nodes and func_name != skill_name:
                # Re-add edges (even if previously present, they are added in the new order)
                self.graph.add_edge(skill_name, func_name)
                # Only print on new addition (when previously absent)
                if func_name not in old_children_set:
                    print(f"\033[32mAuto-detected edge: {skill_name} -> {func_name}\033[0m")

    def _extract_all_skill_calls_deep(self, code: str) -> List[str]:
        """
        [Fix 2] Deep extraction of all calls in code to skills in the graph.

        Unlike _extract_function_calls, this method:
        1. Extracts all functions defined in the code (including helpers)
        2. Extracts the called functions from each one
        3. Identifies which calls go to skills already in the graph

        Solves the problem of calls inside helper functions not being tracked.
        E.g.: ensurePickaxe calls craftCraftingTable through setupCraftingTable;
        this method recognizes that call relationship correctly.

        Args:
            code: JavaScript code

        Returns:
            List[str]: list of all called skill names (in call order, deduplicated)
        """
        all_called_skills = []
        seen = set()

        # Step 1: extract all function names defined in code
        defined_functions = set()
        func_pattern = re.compile(r'(?:async\s+)?function\s+(\w+)\s*\(')
        for match in func_pattern.finditer(code):
            defined_functions.add(match.group(1))

        # Step 2: extract all function calls in code
        # This includes all calls in the main function and helpers
        all_calls = self._extract_function_calls(code)

        # Step 3: keep only calls to skills already in the graph
        # Note: exclude helper functions defined in code (they are not skills in the graph)
        for func_name in all_calls:
            # Skip helper functions defined in code (unless they are also in the graph)
            is_local_helper = func_name in defined_functions and func_name not in self.graph.nodes
            if is_local_helper:
                continue

            # Only add skills present in the graph
            if func_name in self.graph.nodes:
                if func_name not in seen:
                    all_called_skills.append(func_name)
                    seen.add(func_name)

        return all_called_skills
