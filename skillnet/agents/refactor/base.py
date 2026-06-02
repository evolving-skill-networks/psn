"""
Refactor Base — RefactorDetector + Re-exports

Refactor opportunity detector (RefactorDetector) and public symbols re-exported from sub-modules.

Split from the original base.py (1,908 LOC) into:
- models.py: Data structures (RefactorType, RefactorOpportunity, RefactorResult)
- applier.py: Refactor execution (SkillRefactor)
- code_utils.py: JS dependency analysis tools (constants + pure functions)
- base.py: Detection logic (RefactorDetector) + re-exports
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING

from skillnet.utils.stats_tracker import record_llm_usage

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillNode

# Re-exports: Import from sub-modules to maintain `from .base import X` compatibility
from .models import RefactorType, RefactorOpportunity, RefactorResult  # noqa: F401
from .applier import SkillRefactor  # noqa: F401
from .code_utils import (  # noqa: F401
    GLOBAL_PROVIDED_DEPENDENCIES,
    INJECTABLE_DEPENDENCIES,
    STANDARD_DEPENDENCIES,
    analyze_code_dependencies,
    inject_dependencies,
    ensure_bot_parameter,
)


class RefactorDetector:
    """
    Refactor opportunity detector

    Detects refactor opportunities between skills:
    - Parametric coverage
    - Behavioral inclusion
    - Functional duplication
    - Common sub-skill extraction
    """

    # Functional scope difference threshold
    FUNCTIONAL_SCOPE_THRESHOLD = 0.3

    # Parameterization difference threshold
    PARAMETERIZATION_THRESHOLD = 0.3

    def __init__(
        self,
        skill_graph_manager=None,
        llm=None,
        logger=None,
        domain_knowledge=None,
    ):
        """
        Initialize RefactorDetector

        Args:
            skill_graph_manager: skill graph manager
            llm: LLM instance
            logger: logger
            domain_knowledge: DomainKnowledge instance (optional)
        """
        self.skill_graph_manager = skill_graph_manager
        self.llm = llm
        self.logger = logger
        self.domain_knowledge = domain_knowledge

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)

    # ========== Unified Detection Architecture ==========
    # Default parameters
    DEFAULT_EMBEDDING_TOP_K = 8  # Initial embedding candidate count (kept small)
    DEFAULT_MAX_CANDIDATES_PER_TYPE = 8  # Max candidates per type

    def detect_opportunities(
        self,
        skill_name: str,
        refactor_types: Optional[List[RefactorType]] = None,
        skip_covered_wrappers: bool = True,
        use_llm_confirmation: bool = True,
        top_k: int = None,
    ) -> List[RefactorOpportunity]:
        """
        Unified refactor opportunity detection (new architecture)

        Detection flow:
        1. Embedding Top-K: get initial candidate set (efficient)
        2. Type-specific expansion: each type expands the candidate set with its own method
        3. LLM confirmation: confirm real refactor opportunities within the candidate set

        Args:
            skill_name: name of the skill to check
            refactor_types: list of types to detect (None means all 5)
            skip_covered_wrappers: whether to skip already-covered wrapper skills
            use_llm_confirmation: whether to use LLM confirmation (recommended on)
            top_k: initial embedding candidate count (default 10)

        Returns:
            List[RefactorOpportunity]: detected refactor opportunities
        """
        if not self.skill_graph_manager:
            return []

        source_node = self.skill_graph_manager.get_node(skill_name)
        if not source_node:
            return []

        # Check if already a covered wrapper
        if skip_covered_wrappers and self._is_covered_wrapper(source_node):
            self._log(
                f"[RefactorDetector] Skipping '{skill_name}': already a covered wrapper",
                "info"
            )
            return []

        top_k = top_k or self.DEFAULT_EMBEDDING_TOP_K

        # Determine which types to detect
        if refactor_types is None:
            refactor_types = [
                RefactorType.PARAMETRIC,
                RefactorType.MERGE_SIBLINGS,
                RefactorType.EXTRACT_COMMON,
                RefactorType.BEHAVIORAL,
                RefactorType.DUPLICATION,
            ]

        self._log(
            f"[RefactorDetector] Detecting refactor opportunities for '{skill_name}' "
            f"(types={[t.value for t in refactor_types]}, top_k={top_k})",
            "info"
        )

        # === 1. Embedding Top-K initial candidate set ===
        initial_candidates = self._find_embedding_similar_skills(
            skill_name, max_candidates=top_k
        )
        self._log(
            f"[RefactorDetector] Embedding initial candidates: {len(initial_candidates)}",
            "info"
        )

        # === 2. Type-specific expansion + heuristic detection ===
        all_opportunities = []
        for refactor_type in refactor_types:
            type_opps = self._detect_for_type(
                skill_name,
                refactor_type,
                initial_candidates,
            )
            all_opportunities.extend(type_opps)

        self._log(
            f"[RefactorDetector] Heuristic detection found {len(all_opportunities)} candidate opportunities",
            "info"
        )

        # === 3. LLM confirmation ===
        if use_llm_confirmation and all_opportunities and self.llm:
            confirmed = self._confirm_opportunities_with_llm(all_opportunities)
            self._log(
                f"[RefactorDetector] After LLM confirmation: {len(confirmed)}/{len(all_opportunities)} opportunities",
                "info"
            )
            return confirmed

        return all_opportunities

    def _detect_for_type(
        self,
        skill_name: str,
        refactor_type: RefactorType,
        initial_candidates: List[str],
    ) -> List[RefactorOpportunity]:
        """
        Detect refactor opportunities for a specific type

        Args:
            skill_name: source skill name
            refactor_type: refactor type
            initial_candidates: initial embedding candidate set

        Returns:
            List[RefactorOpportunity]: detected opportunities
        """
        # Expand the candidate set
        enlarged = self._enlarge_candidates(skill_name, initial_candidates, refactor_type)

        if not enlarged:
            return []

        # Dispatch to the corresponding detector by type
        try:
            if refactor_type == RefactorType.MERGE_SIBLINGS:
                return self._detect_merge_siblings(skill_name, enlarged)

            elif refactor_type == RefactorType.DUPLICATION:
                return self._detect_duplication(skill_name, enlarged)

            elif refactor_type == RefactorType.EXTRACT_COMMON:
                return self._detect_extract_common(skill_name, enlarged)

            elif refactor_type == RefactorType.BEHAVIORAL:
                return self._detect_behavioral(skill_name, enlarged)

            elif refactor_type == RefactorType.PARAMETRIC:
                return self._detect_parametric(skill_name, enlarged)

        except Exception as e:
            self._log(
                f"[RefactorDetector] {refactor_type.value} detection failed: {e}",
                "warning"
            )

        return []

    def _enlarge_candidates(
        self,
        skill_name: str,
        initial_candidates: List[str],
        refactor_type: RefactorType,
    ) -> List[str]:
        """
        Type-specific candidate set expansion

        Different types need different expansion strategies because embeddings can miss certain relations.

        Args:
            skill_name: source skill name
            initial_candidates: initial candidate set
            refactor_type: refactor type

        Returns:
            List[str]: expanded candidate set
        """
        enlarged = set(initial_candidates)
        max_per_type = self.DEFAULT_MAX_CANDIDATES_PER_TYPE

        all_skills = self.skill_graph_manager.get_all_skill_names(include_task_specific=True)

        if refactor_type == RefactorType.MERGE_SIBLINGS:
            # Sibling relation: name pattern matching is key (embeddings can miss it)
            related = self._find_related_skills(skill_name, all_skills, max_related=max_per_type)
            enlarged |= set(related)

        elif refactor_type == RefactorType.BEHAVIORAL:
            # Behavioral inclusion: requires graph-structure analysis (call chain)
            graph_related = self._find_graph_related_skills(skill_name, max_per_type)
            enlarged |= set(graph_related)

        elif refactor_type == RefactorType.PARAMETRIC:
            # Parametric: find general versions (name pattern + parameter analysis)
            general_versions = self._find_potential_general_skills(skill_name, all_skills, max_per_type)
            enlarged |= set(general_versions)

        elif refactor_type in (RefactorType.DUPLICATION, RefactorType.EXTRACT_COMMON):
            # Code duplication/extraction: code heuristic filtering
            code_similar = self._filter_by_code_heuristics(
                skill_name, list(enlarged), max_per_type * 2
            )
            enlarged = set(code_similar)

        # Remove self
        enlarged.discard(skill_name)

        return list(enlarged)[:max_per_type * 2]  # Cap total count

    def _find_graph_related_skills(
        self,
        skill_name: str,
        max_results: int = 8,
    ) -> List[str]:
        """
        Find graph-related skills (used for BEHAVIORAL detection)

        Analyzes call chains and dependencies to find skills that may contain or be contained.

        Args:
            skill_name: source skill name
            max_results: max number to return

        Returns:
            List[str]: list of graph-related skills
        """
        if not self.skill_graph_manager:
            return []

        related = set()
        mgr = self.skill_graph_manager

        # 1. Skills called directly (child skills)
        node = mgr.get_node(skill_name)
        if node:
            dependencies = getattr(node, 'dependencies', []) or []
            related.update(dependencies[:max_results // 2])

        # 2. Skills that call this one (parent skills)
        for name, n in mgr.iter_skills(include_task_specific=True):
            if name == skill_name:
                continue
            deps = getattr(n, 'dependencies', []) or []
            if skill_name in deps:
                related.add(name)
                if len(related) >= max_results:
                    break

        # 3. Skills with similar effects
        if node and node.expected_effects:
            source_effects = set(str(e) for e in node.expected_effects)
            for name, n in mgr.iter_skills(include_task_specific=True):
                if name == skill_name or name in related:
                    continue
                if n.expected_effects:
                    target_effects = set(str(e) for e in n.expected_effects)
                    if source_effects & target_effects:  # Has intersection
                        related.add(name)
                        if len(related) >= max_results:
                            break

        return list(related)[:max_results]

    def _find_potential_general_skills(
        self,
        skill_name: str,
        all_skills: List[str],
        max_results: int = 8,
    ) -> List[str]:
        """
        Find potential general-version skills (used for PARAMETRIC detection)

        Args:
            skill_name: source skill name (potentially a specialized version)
            all_skills: all skill names
            max_results: max number to return

        Returns:
            List[str]: list of potential general skills
        """
        import re

        results = []

        # Extract parts of the skill name
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', skill_name)
        if not parts:
            return []

        # Verb (typically the first word)
        verb = parts[0].lower()

        # Variant words (e.g. Oak, Birch, Iron) — domain-injectable with fallback
        _FALLBACK_VARIANT_SUFFIXES = []
        dk = getattr(self, 'domain_knowledge', None)
        variant_patterns = (
            dk.get_variant_suffixes() if dk and dk.get_variant_suffixes()
            else _FALLBACK_VARIANT_SUFFIXES
        )

        # Check whether this is a specialized version
        skill_lower = skill_name.lower()
        is_specialized = any(
            re.search(p, skill_lower) for p in variant_patterns
        )

        if is_specialized:
            # Look for a general version without variant words
            for other_name in all_skills:
                if other_name == skill_name:
                    continue

                other_lower = other_name.lower()

                # Check whether it starts with the same verb
                other_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', other_name)
                if not other_parts or other_parts[0].lower() != verb:
                    continue

                # Check that it has no variant word (possibly the general version)
                has_variant = any(
                    re.search(p, other_lower) for p in variant_patterns
                )

                if not has_variant:
                    # Resource domain compatibility check:
                    # Source skill's non-verb, non-variant words define its resource domain.
                    # Candidate must share at least one resource word.
                    source_non_verb = [p.lower() for p in parts[1:]]
                    source_resource = set()
                    for w in source_non_verb:
                        is_variant_word = any(re.search(p, w) for p in variant_patterns)
                        if not is_variant_word:
                            source_resource.add(w)

                    candidate_resource = set(p.lower() for p in other_parts[1:])

                    if source_resource and candidate_resource:
                        if not (source_resource & candidate_resource):
                            continue  # Resource domain mismatch, skip

                    results.append(other_name)

                    if len(results) >= max_results:
                        break

        return results

    # === Type-specific detection methods ===

    def _detect_merge_siblings(
        self,
        skill_name: str,
        candidates: List[str],
    ) -> List[RefactorOpportunity]:
        """Detect MERGE_SIBLINGS opportunities"""
        from .sibling import detect_sibling_opportunities

        skills_to_check = [skill_name] + candidates
        opps = detect_sibling_opportunities(
            self.skill_graph_manager,
            skill_names=skills_to_check,
            min_group_size=2,
            logger=self.logger,
            domain_knowledge=self.domain_knowledge,
        )

        # Only return ones related to the source skill
        return [
            opp for opp in opps
            if opp.source_skill == skill_name or skill_name in (opp.covered_skills or [])
        ]

    def _detect_duplication(
        self,
        skill_name: str,
        candidates: List[str],
    ) -> List[RefactorOpportunity]:
        """Detect DUPLICATION opportunities"""
        from .duplication import detect_duplication_opportunities

        skills_to_check = [skill_name] + candidates
        opps = detect_duplication_opportunities(
            self.skill_graph_manager,
            skill_names=skills_to_check,
            min_similarity=0.7,
            logger=self.logger,
        )

        return [
            opp for opp in opps
            if opp.source_skill == skill_name or opp.target_skill == skill_name
        ]

    def _detect_extract_common(
        self,
        skill_name: str,
        candidates: List[str],
    ) -> List[RefactorOpportunity]:
        """Detect EXTRACT_COMMON opportunities"""
        from .subskill_extraction import detect_extraction_opportunities

        skills_to_check = [skill_name] + candidates
        opps = detect_extraction_opportunities(
            self.skill_graph_manager,
            skill_names=skills_to_check,
            min_common_lines=3,
            min_similarity=0.7,
            logger=self.logger,
        )

        return [
            opp for opp in opps
            if opp.source_skill == skill_name or skill_name in (opp.covered_skills or [])
        ]

    def _detect_behavioral(
        self,
        skill_name: str,
        candidates: List[str],
    ) -> List[RefactorOpportunity]:
        """
        Detect BEHAVIORAL opportunities

        BEHAVIORAL: skill A contains skill B's logic but does not call B
        """
        opportunities = []

        source_node = self.skill_graph_manager.get_node(skill_name)
        if not source_node or not source_node.code:
            return opportunities

        source_code = source_node.code.lower()
        source_deps = set(getattr(source_node, 'dependencies', []) or [])

        for candidate in candidates:
            target_node = self.skill_graph_manager.get_node(candidate)
            if not target_node or not target_node.code:
                continue

            # Skip if already called
            if candidate in source_deps:
                continue

            # Check if it contains target's key logic
            target_code = target_node.code.lower()

            # Simple heuristic: check whether key code snippets are similar
            # Extract target's core logic lines
            target_lines = [
                line.strip() for line in target_code.split('\n')
                if line.strip() and not line.strip().startswith('//')
            ]

            if len(target_lines) < 3:
                continue

            # Check whether source contains most of target's logic
            matching_lines = 0
            for line in target_lines[1:-1]:  # Skip function declaration and trailing line
                if len(line) > 10 and line in source_code:
                    matching_lines += 1

            coverage = matching_lines / max(len(target_lines) - 2, 1)

            if coverage >= 0.5:  # 50%+ logic match
                opportunities.append(RefactorOpportunity(
                    refactor_type=RefactorType.BEHAVIORAL,
                    source_skill=skill_name,
                    target_skill=candidate,
                    reason=f"'{skill_name}' contains approximately {coverage*100:.0f}% of '{candidate}'s logic but does not call it",
                    confidence=min(coverage, 0.9),
                ))

        return opportunities

    def _detect_parametric(
        self,
        skill_name: str,
        candidates: List[str],
    ) -> List[RefactorOpportunity]:
        """
        Detect PARAMETRIC opportunities

        PARAMETRIC: a general skill already exists; the specialized version should become a wrapper
        """
        opportunities = []

        source_node = self.skill_graph_manager.get_node(skill_name)
        if not source_node:
            return opportunities

        source_params = len(source_node.parameters) if source_node.parameters else 0

        for candidate in candidates:
            target_node = self.skill_graph_manager.get_node(candidate)
            if not target_node:
                continue

            target_params = len(target_node.parameters) if target_node.parameters else 0

            # General version should have more parameters
            if target_params > source_params:
                # Check whether effects can be covered (handle the None case)
                source_effects = (
                    set(str(e) for e in source_node.expected_effects)
                    if source_node.expected_effects else set()
                )
                target_effects = (
                    set(str(e) for e in target_node.expected_effects)
                    if target_node.expected_effects else set()
                )

                # Resource domain compatibility check
                if not self._are_resource_compatible(skill_name, candidate):
                    continue

                # Source's effects should be a subset of target's
                if source_effects and source_effects.issubset(target_effects):
                    opportunities.append(RefactorOpportunity(
                        refactor_type=RefactorType.PARAMETRIC,
                        source_skill=skill_name,
                        target_skill=candidate,
                        reason=f"'{candidate}' has {target_params} parameters and can cover '{skill_name}' ({source_params} parameters)",
                        confidence=0.7,
                        covered_skills=[skill_name],
                    ))

        return opportunities

    def _are_resource_compatible(self, source_name: str, target_name: str) -> bool:
        """Check if two skills operate on the same resource domain.

        Extracts non-verb words from camelCase skill names and checks for overlap.
        E.g. ensureOakLogs vs ensureCobblestone → {"oak", "logs"} & {"cobblestone"} = ∅ → False
        E.g. ensureOakLogs vs ensureLogs → {"oak", "logs"} & {"logs"} = {"logs"} → True
        """
        import re
        def extract_resource(name: str) -> set:
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            verbs = {'ensure', 'craft', 'mine', 'collect', 'get', 'make',
                     'smelt', 'harvest', 'place', 'find', 'obtain'}
            return set(p.lower() for p in parts if p.lower() not in verbs)
        src = extract_resource(source_name)
        tgt = extract_resource(target_name)
        if not src or not tgt:
            return False
        return bool(src & tgt)

    def _find_embedding_similar_skills(
        self,
        skill_name: str,
        max_candidates: int = 40,
        max_distance: float = 1.5,
    ) -> List[str]:
        """
        Use embedding similarity to find semantically similar skills

        Args:
            skill_name: current skill name
            max_candidates: max number to return
            max_distance: max distance threshold (smaller is more similar)

        Returns:
            List[str]: list of semantically similar skills
        """
        if not self.skill_graph_manager:
            return []

        # Check whether vectordb is available
        vectordb = getattr(self.skill_graph_manager, 'vectordb', None)
        if not vectordb:
            return []

        try:
            # Get source skill's description for the query
            source_node = self.skill_graph_manager.get_node(skill_name)
            if not source_node:
                return []

            # Use description or code as the query (make sure code is not None)
            query = source_node.description or (source_node.code[:500] if source_node.code else "")
            if not query:
                return []

            # Execute similarity search
            # Try to get the collection size; fall back to default on failure
            try:
                collection_count = vectordb._collection.count()
            except (AttributeError, NotImplementedError):
                # vectordb implementations may differ; use default max candidates
                collection_count = max_candidates

            k = min(collection_count, max_candidates)
            if k == 0:
                return []

            docs_and_scores = vectordb.similarity_search_with_score(query, k=k)

            similar_skills = []
            for doc, score in docs_and_scores:
                name = doc.metadata.get('name')
                # Skip invalid names (None or empty string)
                if not name:
                    continue
                # Skip self
                if name == skill_name:
                    continue
                # Filter out too-distant results
                if score > max_distance:
                    continue
                similar_skills.append(name)

            self._log(
                f"[RefactorDetector] Embedding similarity search: {skill_name} -> "
                f"found {len(similar_skills)} similar skills",
                "info"
            )

            return similar_skills

        except Exception as e:
            self._log(f"[RefactorDetector] Embedding search failed: {e}", "warning")
            return []

    def _filter_by_code_heuristics(
        self,
        skill_name: str,
        candidates: List[str],
        max_candidates: int = 20,
    ) -> List[str]:
        """
        Filter candidate skills using code heuristics

        Filter conditions:
        1. Code-length ratio > 0.3
        2. API keyword overlap > 0.2

        Args:
            skill_name: source skill name
            candidates: candidate skill list
            max_candidates: max number to return

        Returns:
            List[str]: filtered skill list
        """
        if not self.skill_graph_manager:
            return []

        source_node = self.skill_graph_manager.get_node(skill_name)
        if not source_node or not source_node.code:
            return []

        source_code = source_node.code
        source_len = len(source_code)

        # Extract source code keywords (function calls, APIs, etc.)
        import re
        source_keywords = set(re.findall(
            r'\b(?:bot|await|async|function|const|let|var)\b|\.\w+\(',
            source_code.lower()
        ))

        similar_skills = []

        for candidate in candidates:
            if candidate == skill_name:
                continue

            node = self.skill_graph_manager.get_node(candidate)
            if not node or not node.code:
                continue

            candidate_code = node.code
            candidate_len = len(candidate_code)

            # Filter 1: skip if code length difference is too large
            if max(source_len, candidate_len) > 0:
                len_ratio = min(source_len, candidate_len) / max(source_len, candidate_len)
                if len_ratio < 0.3:  # Skip if length difference exceeds 70%
                    continue

            # Filter 2: keyword overlap check
            candidate_keywords = set(re.findall(
                r'\b(?:bot|await|async|function|const|let|var)\b|\.\w+\(',
                candidate_code.lower()
            ))
            if source_keywords and candidate_keywords:
                keyword_overlap = len(source_keywords & candidate_keywords) / max(
                    len(source_keywords), len(candidate_keywords)
                )
                if keyword_overlap < 0.2:  # Skip if keyword overlap is too low
                    continue

            similar_skills.append(candidate)

            if len(similar_skills) >= max_candidates:
                break

        return similar_skills

    def _find_related_skills(
        self,
        skill_name: str,
        candidates: List[str],
        max_related: int = 10,
    ) -> List[str]:
        """
        Find skills related to a given skill (used to narrow the check scope)

        Relation criteria:
        1. Same name prefix (e.g. craftOakBoat, craftBirchBoat)
        2. Names contain the same keywords

        Args:
            skill_name: current skill name
            candidates: candidate skill list
            max_related: max number to return

        Returns:
            List[str]: related skill list
        """
        import re

        related = []

        # Extract parts of the name
        # For example "craftOakBoat" -> ["craft", "Oak", "Boat"]
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', skill_name)
        parts_lower = [p.lower() for p in parts]

        # Extract prefix (first verb)
        prefix = parts_lower[0] if parts_lower else ""

        for candidate in candidates:
            if candidate == skill_name:
                continue

            # Check prefix match
            candidate_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', candidate)
            candidate_parts_lower = [p.lower() for p in candidate_parts]
            candidate_prefix = candidate_parts_lower[0] if candidate_parts_lower else ""

            if prefix and prefix == candidate_prefix:
                related.append(candidate)
                continue

            # Check keyword overlap (at least two in common)
            common_parts = set(parts_lower) & set(candidate_parts_lower)
            if len(common_parts) >= 2:
                related.append(candidate)
                continue

        # Cap the count
        return related[:max_related]

    def _confirm_opportunities_with_llm(
        self,
        opportunities: List[RefactorOpportunity],
    ) -> List[RefactorOpportunity]:
        """
        Use the LLM to confirm detected refactor opportunities

        For each candidate, have the LLM analyze the relevant skills' code
        and confirm whether the refactor relation of that type really exists.

        Args:
            opportunities: list of candidate refactor opportunities

        Returns:
            List[RefactorOpportunity]: confirmed refactor opportunities
        """
        if not self.llm or not opportunities:
            return opportunities

        confirmed = []

        for opp in opportunities:
            try:
                is_confirmed = self._confirm_single_opportunity(opp)
                if is_confirmed:
                    confirmed.append(opp)
                else:
                    self._log(
                        f"[RefactorDetector] LLM rejected: {opp.refactor_type.value} "
                        f"({opp.source_skill} -> {opp.target_skill})",
                        "info"
                    )
            except Exception as e:
                # Keep the candidate when LLM confirmation fails (prefer false positives over false negatives)
                self._log(f"[RefactorDetector] LLM confirmation failed: {e}", "warning")
                confirmed.append(opp)

        return confirmed

    def _confirm_single_opportunity(
        self,
        opportunity: RefactorOpportunity,
    ) -> bool:
        """
        Use the LLM to confirm a single refactor opportunity

        Args:
            opportunity: refactor opportunity to confirm

        Returns:
            bool: whether the refactor opportunity is confirmed
        """
        if not self.llm or not self.skill_graph_manager:
            return True  # Keep by default if confirmation isn't possible

        # Get the relevant skills' code
        source_node = self.skill_graph_manager.get_node(opportunity.source_skill)
        target_node = self.skill_graph_manager.get_node(opportunity.target_skill)

        if not source_node:
            return False

        source_code = source_node.code or ""
        target_code = target_node.code if target_node else ""

        # If there are covered_skills, fetch their code too
        covered_codes = {}
        for skill_name in (opportunity.covered_skills or []):
            node = self.skill_graph_manager.get_node(skill_name)
            if node and node.code:
                covered_codes[skill_name] = node.code

        # Build the confirmation prompt
        prompt = self._build_confirmation_prompt(
            opportunity.refactor_type,
            opportunity.source_skill,
            source_code,
            opportunity.target_skill,
            target_code,
            covered_codes,
            opportunity.reason,
        )

        try:
            from langchain.schema import HumanMessage

            response = self.llm.invoke([HumanMessage(content=prompt)])
            record_llm_usage(response, process_type="refactor", function_name="refactor.base._confirm_single_opportunity", skill_name=opportunity.source_skill)
            response_text = response.content.strip().lower()

            # Parse the response
            # Expect the LLM to return a response starting with "CONFIRMED" or "REJECTED"
            if response_text.startswith("confirmed"):
                return True
            elif response_text.startswith("rejected"):
                return False
            else:
                # When parsing fails, try to infer intent from the response
                if "yes" in response_text or "true" in response_text or "confirm" in response_text:
                    return True
                elif "no" in response_text or "false" in response_text or "reject" in response_text:
                    return False
                else:
                    # Truly unparseable; keep conservatively
                    self._log(
                        f"[RefactorDetector] LLM response could not be parsed: {response_text[:100]}",
                        "warning"
                    )
                    return True

        except Exception as e:
            self._log(f"[RefactorDetector] LLM call failed: {e}", "warning")
            return True  # Keep on failure

    def _build_confirmation_prompt(
        self,
        refactor_type: RefactorType,
        source_skill: str,
        source_code: str,
        target_skill: str,
        target_code: str,
        covered_codes: Dict[str, str],
        reason: str,
    ) -> str:
        """
        Build the LLM confirmation prompt

        Args:
            refactor_type: refactor type
            source_skill: source skill name
            source_code: source skill code
            target_skill: target skill name
            target_code: target skill code
            covered_codes: code of the covered skills
            reason: heuristic detection reason

        Returns:
            str: LLM confirmation prompt
        """
        type_descriptions = {
            RefactorType.PARAMETRIC: (
                "PARAMETRIC (parameterized coverage): a general skill already exists; the specialized version should become a wrapper that calls the general version.\n"
                "Precondition: a general skill (e.g. craftPickaxe(type)) already exists\n"
                "Refactor action: the specialized version (e.g. craftWoodenPickaxe) becomes a call to the general version\n"
                "Example: craftPickaxe(type) already exists → craftWoodenPickaxe is changed to call it"
            ),
            RefactorType.MERGE_SIBLINGS: (
                "MERGE_SIBLINGS (merge siblings): multiple sibling skills with no existing general version; a general skill needs to be created.\n"
                "Precondition: multiple sibling skills (e.g. craftOakBoat, craftBirchBoat)\n"
                "Refactor action: create a general skill (craftBoat(type)); all siblings become wrappers\n"
                "Example: craftOakBoat, craftBirchBoat → create craftBoat(type)"
            ),
            RefactorType.EXTRACT_COMMON: (
                "EXTRACT_COMMON (extract common code): multiple skills share a code snippet; extract the common code into a standalone sub-skill.\n"
                "Precondition: multiple skills share a code snippet (e.g. all contain furnace-setup logic)\n"
                "Refactor action: extract the common code into a standalone sub-skill\n"
                "Example: multiple skills contain furnace-setup logic → extract setupFurnace"
            ),
            RefactorType.BEHAVIORAL: (
                "BEHAVIORAL (behavioral inclusion): skill A contains skill B's logic but does not call B; modify A to call B.\n"
                "Precondition: skill A contains skill B's logic but does not call B (B is a required step of A)\n"
                "Refactor action: modify A to call B\n"
                "Example: ensureWoodenPickaxe contains ensureCraftingTable logic → change it to call ensureCraftingTable"
            ),
            RefactorType.DUPLICATION: (
                "DUPLICATION (functional duplication): two skills are functionally identical; keep the one with higher maturity.\n"
                "Precondition: two skills are functionally identical (true duplicates)\n"
                "Refactor action: keep the one with higher maturity; mark the other as deprecated\n"
                "Example: two functionally identical skills → delete the newer one"
            ),
            # NB: do not re-add entries for the SIBLING / EXTRACT_COMMON_SUBSKILL /
            # FUNCTIONAL_SUPERSET aliases — they share enum identity with their
            # canonical counterparts above (RefactorType.SIBLING is
            # RefactorType.MERGE_SIBLINGS), so any extra dict entry would silently
            # overwrite the canonical full description with a short alias line.
        }

        type_desc = type_descriptions.get(
            refactor_type,
            f"Refactor type: {refactor_type.value}"
        )

        # Build the code section
        code_section = f"""
## Source Skill: {source_skill}
```javascript
{source_code[:1500]}
```
"""
        if target_code:
            code_section += f"""
## Target Skill: {target_skill}
```javascript
{target_code[:1500]}
```
"""
        if covered_codes:
            code_section += "\n## Other Related Skills:\n"
            for name, code in list(covered_codes.items())[:3]:  # Show at most 3
                code_section += f"""
### {name}
```javascript
{code[:800]}
```
"""

        prompt = f"""You are a code refactoring expert. Please analyze whether the following skills truly have the specified type of refactor relation.

## Refactor Type Description
{type_desc}

## Heuristic Detection Reason
{reason}

{code_section}

## Task
Carefully analyze the code above and decide whether these skills truly have a {refactor_type.value} refactor relation.

Criteria:
1. Whether the code structure is truly similar or has an inclusion relation
2. Whether the functional semantics belong to the same category
3. Whether this refactor is worthwhile (consider actual benefits)

Please reply in one of the two formats below only:
- CONFIRMED: [brief reason]
- REJECTED: [brief reason]
"""
        return prompt

    def _is_covered_wrapper(self, node: 'SkillNode') -> bool:
        """
        Check whether a skill is a covered wrapper

        A covered wrapper is a skill marked by refactoring with
        is_covered=True and coverage_type.is_wrapper=True.

        These skills have already been refactored; there's no need to detect more refactor opportunities for them.

        Args:
            node: SkillNode instance

        Returns:
            bool: whether it is a covered wrapper
        """
        if not getattr(node, 'is_covered', False):
            return False

        # Check coverage_type
        coverage_type = getattr(node, 'coverage_type', None)
        if coverage_type is not None:
            # Use CoverageType's is_wrapper attribute
            return getattr(coverage_type, 'is_wrapper', False)

        return False
