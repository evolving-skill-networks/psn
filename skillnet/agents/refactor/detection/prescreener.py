"""
Refactor Prescreener - prescreener for refactor candidates.

Methods migrated from graph_manager_impl.py:
- _prescreen_refactor_candidates
- _prescreen_refactor_candidates_detailed
- _calculate_name_similarity
- _find_name_similar_skills
- _is_generalization_of

v5.0 architectural reorganization.
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from skillnet.core.dk_registry import get_domain_knowledge
from skillnet.utils.stats_tracker import record_llm_usage

if TYPE_CHECKING:
    from skillnet.agents.skill_graph.models import SkillPrecondition, SkillEffect

logger = logging.getLogger(__name__)


# ============================================================================
# Pure functions
# ============================================================================

def calculate_name_similarity(name_a: str, name_b: str) -> float:
    """
    Compute the similarity of two function names.

    Uses the longest common subsequence (LCS) algorithm.

    Args:
        name_a: First name
        name_b: Second name

    Returns:
        float: Similarity score (0.0 - 1.0)
    """
    if not name_a or not name_b:
        return 0.0

    # Lowercase for comparison
    a = name_a.lower()
    b = name_b.lower()

    # LCS algorithm
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    # Similarity = LCS length / longest-name length
    return lcs_length / max(m, n)


def is_generalization_of(general_name: str, specific_name: str) -> bool:
    """
    Check whether general_name is a generalized version of specific_name.

    For example:
    - mineLogs is a generalization of mineOakLogs
    - smeltRaw is a generalization of smeltRawIron
    - craftPickaxe is a generalization of craftWoodenPickaxe

    Args:
        general_name: Possible generalized name
        specific_name: Possible specialized name

    Returns:
        bool: Whether general_name is a generalization of specific_name
    """
    # Extract the action part (e.g. mine, smelt, craft)
    general_lower = general_name.lower()
    specific_lower = specific_name.lower()

    # Common type suffixes/prefixes — populated from domain knowledge
    dk = get_domain_knowledge()
    if dk:
        type_keywords = dk.get_type_keywords()
        type_words = type_keywords.get('specific_types', [])
    else:
        type_words = []

    # Check whether specific_name contains general_name + a type word
    # For example: mineOakLogs contains mineLogs
    for type_word in type_words:
        # Pattern 1: specificName = generalName + TypeWord (e.g. smeltRawIron = smeltRaw + Iron)
        if specific_lower.startswith(general_lower):
            suffix = specific_lower[len(general_lower):]
            if suffix.lower().startswith(type_word):
                return True

        # Pattern 2: specificName = action + TypeWord + Object, generalName = action + Object
        # e.g. mineOakLogs vs mineLogs
        if type_word in specific_lower and type_word not in general_lower:
            # Try comparing after removing the type word
            cleaned = re.sub(rf'{type_word}', '', specific_lower, flags=re.IGNORECASE)
            if cleaned.lower() == general_lower:
                return True

    return False


# ============================================================================
# RefactorPrescreener class
# ============================================================================

class RefactorPrescreener:
    """
    Refactor candidate prescreener.

    Responsibilities:
    1. Use embedding similarity + function-name similarity to screen the top-K candidates
    2. Use the LLM to prescreen candidates based on metadata
    """

    def __init__(
        self,
        graph,  # SkillGraph
        vectordb,  # VectorDB
        llm,  # LangChain LLM
        custom_logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the prescreener.

        Args:
            graph: SkillGraph or SkillGraphManager instance (duck-typed)
            vectordb: VectorDB instance (used for embedding search)
            llm: LangChain LLM instance
            custom_logger: Optional custom logger
        """
        self.graph = graph
        self.vectordb = vectordb
        self.llm = llm
        self.logger = custom_logger or logger

    def prescreen_candidates(
        self,
        new_skill_name: str,
        new_skill_description: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Prescreen candidates: filter the top-K candidate skills via embedding
        similarity + function-name similarity.

        Args:
            new_skill_name: Name of the new skill
            new_skill_description: Description of the new skill
            top_k: Number of candidates to return (default 5)

        Returns:
            List[Tuple[str, float]]: [(skill_name, similarity_score), ...] candidate list
        """
        from skillnet.agents.skill_graph.models.graph import should_skip_covered_skill

        candidates_dict = {}  # Use a dict for de-duplication {skill_name: similarity_score}

        # Method 1: vector-database similarity search
        max_retrieve = min(self.vectordb._collection.count(), top_k * 4)
        if max_retrieve > 0:
            docs_and_scores = self.vectordb.similarity_search_with_score(
                new_skill_description,
                k=max_retrieve
            )

            for doc, score in docs_and_scores:
                skill_name = doc.metadata['name']
                # Exclude self
                if skill_name == new_skill_name:
                    continue
                # Exclude already-covered skills (except for extract_common_subskill types)
                if self.graph.has_node(skill_name):
                    node = self.graph.get_node(skill_name)
                    if should_skip_covered_skill(node):
                        continue
                    # Exclude deprecated skills
                    if node.is_deprecated:
                        self.logger.debug(
                            f"[Refactor Prescreen] Skipping deprecated skill: {skill_name} "
                            f"(reason: {node.deprecation_reason})"
                        )
                        continue
                    # Exclude task-specific skills
                    if getattr(node, 'is_task_specific', False):
                        continue
                # Similarity score (lower distance = higher similarity, so use 1-score)
                similarity = 1.0 - score if score <= 1.0 else 0.0
                candidates_dict[skill_name] = similarity

        # Method 2: candidate filtering based on function name
        name_similar_skills = self._find_name_similar_skills(new_skill_name, threshold=0.4)
        for skill_name in name_similar_skills:
            # Exclude self
            if skill_name == new_skill_name:
                continue
            # Exclude already-covered/deprecated/task-specific skills
            if self.graph.has_node(skill_name):
                node = self.graph.get_node(skill_name)
                if (should_skip_covered_skill(node) or
                    node.is_deprecated or
                    getattr(node, 'is_task_specific', False)):
                    continue

            # If not yet a candidate, add it; if already present, take the higher score
            name_similarity = calculate_name_similarity(new_skill_name, skill_name)
            if skill_name not in candidates_dict:
                candidates_dict[skill_name] = name_similarity
                self.logger.info(
                    f"[Refactor Prescreen] Added candidate via name similarity: {skill_name} "
                    f"(similarity: {name_similarity:.2f})"
                )
            else:
                # Take the higher score across the two methods
                candidates_dict[skill_name] = max(candidates_dict[skill_name], name_similarity)

        # Convert to a list and sort
        candidates = [(name, score) for name, score in candidates_dict.items()]
        candidates.sort(key=lambda x: x[1], reverse=True)
        result = candidates[:top_k]

        # Log the actual number of candidates returned
        if len(result) < top_k:
            self.logger.debug(
                f"[Refactor Prescreen] After filtering only {len(result)} valid candidates remained, "
                f"fewer than the target {top_k}"
            )

        return result

    def prescreen_candidates_detailed(
        self,
        new_skill_name: str,
        new_skill_description: str,
        new_skill_params: Dict[str, Dict[str, Any]],
        new_skill_preconditions: List["SkillPrecondition"],
        new_skill_effects: List["SkillEffect"],
        new_skill_children: List[str],
        candidate_skills: List[Tuple[str, float]]
    ) -> List[str]:
        """
        Use the new skill's metadata (name, parameters, description, skeleton,
        preconditions, effects) to compare against candidates one by one and
        determine whether refactoring relationships may exist.

        Args:
            new_skill_name: Name of the new skill
            new_skill_description: Description of the new skill
            new_skill_params: Parameter metadata of the new skill
            new_skill_preconditions: Preconditions of the new skill
            new_skill_effects: Effects of the new skill
            new_skill_children: List of child skills called by the new skill (skeleton)
            candidate_skills: List of candidate skills [(name, similarity), ...]

        Returns:
            List[str]: List of candidate skill names that may be related
        """
        if not candidate_skills:
            return []

        try:
            from langchain.schema import HumanMessage, SystemMessage

            # Collect metadata for candidate skills (excluding full code)
            candidate_info = []
            for skill_name, similarity in candidate_skills:
                if not self.graph.has_node(skill_name):
                    continue
                node = self.graph.get_node(skill_name)
                candidate_info.append({
                    "name": skill_name,
                    "description": node.description,
                    "parameters": node.parameters,
                    "preconditions": [{"description": p.description} for p in node.preconditions],
                    "effects": [{"description": e.description} for e in node.expected_effects],
                    "children": node.children,  # skeleton: called child skills
                    "similarity": similarity
                })

            if not candidate_info:
                return []

            # Prepare metadata for the new skill
            new_skill_info = {
                "name": new_skill_name,
                "description": new_skill_description,
                "parameters": new_skill_params,
                "preconditions": [{"description": p.description} for p in new_skill_preconditions],
                "effects": [{"description": e.description} for e in new_skill_effects],
                "children": new_skill_children  # skeleton
            }

            # First-step detection: use metadata only
            system_prompt = self._get_prescreen_system_prompt()

            human_prompt = f"""NEW SKILL METADATA:
{json.dumps(new_skill_info, indent=2, ensure_ascii=False)}

CANDIDATE EXISTING SKILLS METADATA (sorted by similarity):
{json.dumps(candidate_info, indent=2, ensure_ascii=False)}

Based on metadata only (name, description, parameters, preconditions, effects, skeleton/children), identify which candidate skills MIGHT have refactoring relationships with the new skill. Return only JSON."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            self.logger.info("[Refactor Prescreen] Step 1: prescreening candidate skills based on metadata...")
            _llm_resp = self.llm.invoke(messages)
            record_llm_usage(_llm_resp, process_type="refactor", function_name="refactor.prescreener.prescreen_candidates_detailed", skill_name=new_skill_name)
            response = _llm_resp.content

            # Backstop: always preserve the top-K most similar candidates
            # (by embedding similarity) regardless of what the LLM filters out.
            #
            # Why: in r34 the prescreen LLM dropped placeCraftingTableNearby
            # from craftFurnace's candidates, calling it "a subroutine, not a
            # full crafting skill". This left the detection LLM unable to even
            # consider the correct refactor target. The LLM filter is too
            # aggressive at filtering out helper/utility skills, which are
            # PRECISELY the candidates that matter for behavioral refactor.
            #
            # Top-K by embedding similarity gets a guaranteed slot.
            BACKSTOP_TOP_K = 3
            top_k_names = [
                name for name, _ in candidate_skills[:BACKSTOP_TOP_K]
                if self.graph.has_node(name)
            ]

            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                potential_candidates = result.get("potential_refactor_candidates", [])
                reasoning = result.get("reasoning", "")

                # Apply backstop: union LLM result with top-K
                llm_set = set(potential_candidates) if potential_candidates else set()
                added_back = [n for n in top_k_names if n not in llm_set]
                final_candidates = list(potential_candidates) + added_back

                if added_back:
                    self.logger.info(
                        f"[Refactor Prescreen] ⚠ Backstop added {len(added_back)} top-K "
                        f"candidates dropped by LLM: {added_back}"
                    )

                if final_candidates:
                    self.logger.info(
                        f"[Refactor Prescreen] Prescreen result: {', '.join(final_candidates)}"
                    )
                    if reasoning:
                        self.logger.info(f"[Refactor Prescreen] LLM reasoning: {reasoning}")
                    return final_candidates
                else:
                    self.logger.info(
                        "[Refactor Prescreen] No potential refactor relationships found (including backstop)"
                    )
                    return []
            else:
                self.logger.warning(
                    "[Refactor Prescreen] Unable to parse LLM response; using all candidate skills"
                )
                return [name for name, _ in candidate_skills]

        except Exception as e:
            self.logger.warning(f"[Refactor Prescreen] Prescreen failed: {e}; using all candidate skills")
            import traceback
            self.logger.warning(f"[Refactor Prescreen] Traceback: {traceback.format_exc()}")
            return [name for name, _ in candidate_skills]

    def _find_name_similar_skills(
        self,
        new_skill_name: str,
        threshold: float = 0.5
    ) -> List[str]:
        """
        Find possible refactor candidates based on function name.

        For example:
        - smeltRawIron -> smeltRaw (remove type suffix)
        - mineOakLogs -> mineLogs (remove type prefix)
        - craftWoodenPickaxe -> craftPickaxe (remove material prefix)

        Args:
            new_skill_name: Name of the new skill
            threshold: Similarity threshold

        Returns:
            List[str]: List of similar skill names
        """
        candidates = []

        for skill_name in self.graph.get_all_skill_names(include_task_specific=True):
            if skill_name == new_skill_name:
                continue

            # Skip task-specific skills
            node = self.graph.get_node(skill_name)
            if node and getattr(node, 'is_task_specific', False):
                continue

            # Method 1: name similarity
            similarity = calculate_name_similarity(new_skill_name, skill_name)
            if similarity >= threshold:
                candidates.append(skill_name)
                continue

            # Method 2: check whether one is a "generalized version" of the other
            if is_generalization_of(skill_name, new_skill_name):
                candidates.append(skill_name)
            elif is_generalization_of(new_skill_name, skill_name):
                candidates.append(skill_name)

        return candidates

    def _get_prescreen_system_prompt(self) -> str:
        """Get the prescreening system prompt."""
        return """You are a code analysis expert specializing in skill refactoring. Your task is to quickly screen if a new skill MIGHT have refactoring relationships with existing skills based on metadata only (name, description, parameters, preconditions, effects, and skeleton/children).

This is a PRELIMINARY screening step. You should identify skills that are POTENTIALLY related, not make final decisions.

CRITICAL EXCLUSION RULES - Skills should be EXCLUDED if they are candidates for PARAMETRIC coverage but:
1. Operate on COMPLETELY DIFFERENT item/block types (e.g., mineSand operates on "sand", mineLogs operates on "logs" - these are different categories)
2. Require DIFFERENT tools (e.g., mineSand uses shovel, mineLogs uses axe - different tool categories)
3. Produce COMPLETELY DIFFERENT products (e.g., sand vs logs - different material categories)

IMPORTANT DISTINCTION:
- **Parametric Coverage**: If a general skill exists (e.g., mineLogs with parameter logType), and a new specialized skill (e.g., mineOakLogs) appears to be just a specialization with a fixed parameter value, it should be included as a candidate for refactoring ONLY if they operate on the SAME category of items (e.g., both mine logs, just different log types). Example: mineSand vs mineLogs should be EXCLUDED because they operate on different material categories.

- **Behavioral/Subgraph Coverage**: A new skill (e.g., craftCraftingTable) that contains logic duplicating an existing basic skill (e.g., mineLogs) should ALWAYS be considered for refactoring, even if their final products are different. The new skill should call the existing basic skill instead of reimplementing the logic. Example: craftCraftingTable internally mines logs - it should call mineLogs even though its final product is crafting_table (not logs).

To determine item categories, look at:
- The effects: what items/blocks are added to inventory or placed?
- The description: what specific items/blocks does the skill operate on?
- Minecraft knowledge: logs (oak_log, birch_log) are wood materials, sand is a different material category

Examples of skills that should be EXCLUDED from refactoring:
- mineSand (mines sand) vs mineLogs (mines logs) - different material categories
- mineGravel (mines gravel) vs mineLogs (mines logs) - different material categories
- mineOre (mines ores) vs mineLogs (mines logs) - different material categories

Examples of skills that CAN be refactored:
- mineOakLogs (mines oak_log) vs mineLogs (mines logs with logType parameter) - same category (logs), different types
- craftOakPlanks (crafts oak_planks) vs craftPlanks (crafts planks with logType parameter) - same category (planks), different types

Return ONLY a JSON object:
{
  "potential_refactor_candidates": ["skill_name1", "skill_name2", ...],  // List of candidate skill names that MIGHT have refactoring relationships
  "reasoning": "brief explanation of why these candidates were selected"
}

If no potential relationships exist, return {"potential_refactor_candidates": []}."""
