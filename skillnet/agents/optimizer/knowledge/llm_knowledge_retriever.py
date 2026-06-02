"""
LLM Knowledge Retriever - Domain-agnostic knowledge retrieval framework.

LLM-driven query generation for knowledge retrieval
Removed fast-path detectors
Extracted domain-specific content (prompt, fallback queries)

Design:
- LLM analyzes feedback + chat_log + code to generate knowledge queries
- Queries are searched against KnowledgeIndex
- Domain-specific prompt and fallback function are injected via constructor
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Callable, Any

from .index import KnowledgeIndex, KnowledgeQuery, get_knowledge_index
from skillnet.utils.stats_tracker import record_llm_usage

logger = logging.getLogger(__name__)

# Module-level diagnostic counters for knowledge retrieval
_kr_stats = {
    "total_calls": 0,
    "llm_attempts": 0,
    "llm_successes": 0,
    "fallback_used": 0,
    "empty_results": 0,
}


def get_kr_stats():
    """Return a copy of the knowledge retrieval stats."""
    return dict(_kr_stats)


@dataclass
class RetrievalResult:
    """Retrieval result."""
    knowledge_text: str
    queries_used: List[KnowledgeQuery]
    source: str  # "llm" or "fallback"


# Type alias for fallback query functions
FallbackQueriesFn = Callable[[str, str, str], List[KnowledgeQuery]]


class LLMKnowledgeRetriever:
    """
    Domain-agnostic LLM-driven knowledge retrieval.

    Workflow:
    1. LLM analyzes feedback + chat_log + code using domain-specific prompt
    2. LLM generates knowledge retrieval queries
    3. Queries are searched against KnowledgeIndex
    4. Returns formatted knowledge text for analysis

    Domain-specific content is injected via constructor:
    - query_prompt: LLM prompt template with {feedback_content}, {chat_log}, {skill_code} placeholders
    - fallback_queries_fn: Function returning keyword-based queries when LLM unavailable
    """

    def __init__(
        self,
        llm: Any = None,
        knowledge_index: Optional[KnowledgeIndex] = None,
        use_llm: bool = True,
        query_prompt: str = "",
        fallback_queries_fn: Optional[FallbackQueriesFn] = None,
    ):
        """
        Initialize retriever.

        Args:
            llm: LLM instance (must have invoke method)
            knowledge_index: Knowledge index (defaults to global singleton)
            use_llm: Whether to use LLM for query generation
            query_prompt: Domain-specific prompt template for LLM query generation.
                Must contain {feedback_content}, {chat_log}, {skill_code} placeholders.
                Empty string disables LLM query generation.
            fallback_queries_fn: Domain-specific function for keyword-based fallback.
                Signature: (feedback: str, chat_log: str, skill_code: str) -> List[KnowledgeQuery]
                None means no fallback queries.
        """
        self.llm = llm
        self.knowledge_index = knowledge_index or get_knowledge_index()
        self.use_llm = use_llm
        self.query_prompt = query_prompt
        self.fallback_queries_fn = fallback_queries_fn

    def retrieve(
        self,
        feedback_content: str,
        chat_log: str = "",
        skill_code: str = "",
        kr_logger=None,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge.

        Args:
            feedback_content: Execution feedback/error message
            chat_log: Game chat log during execution
            skill_code: Failed skill code
            kr_logger: Optional diagnostic logger

        Returns:
            RetrievalResult with formatted knowledge text
        """
        kr = kr_logger
        queries = []
        source = "fallback"

        # Generate queries via LLM (requires prompt template)
        if self.use_llm and self.llm and chat_log and self.query_prompt:
            try:
                llm_queries = self._generate_queries_with_llm(
                    feedback_content, chat_log, skill_code
                )
                if llm_queries:
                    queries.extend(llm_queries)
                    source = "llm"
                    logger.info(f"[LLMKnowledgeRetriever] LLM generated {len(llm_queries)} queries")
                    if kr:
                        kr.info(f"[LLMQuery] generated {len(llm_queries)} queries: "
                                f"{[q.query_type for q in llm_queries]}")
            except KeyboardInterrupt:
                logger.warning("[LLMKnowledgeRetriever] LLM query generation interrupted, skipping")
                if kr:
                    kr.warning("[LLMQuery] interrupted by KeyboardInterrupt")
            except Exception as e:
                logger.warning(f"[LLMKnowledgeRetriever] LLM query generation failed: {e}")
                if kr:
                    kr.warning(f"[LLMQuery] failed: {e}")

        # Always merge keyword fallback queries with
        # LLM queries instead of using fallback only when LLM returned nothing.
        #
        # Motivation: LLMs sometimes anchor hard on exception stacks and
        # generate queries that completely miss the target resource's keywords
        # (observed in r0 ensureCoal: LLM produced api_behavior/primitive/
        # tool_tier queries with keywords ['GoalGetToBlock', 'bot.dig',
        # 'pickaxe', ...] — none mentioning 'coal' at all). The keyword
        # fallback has domain-specific rules that fire on content keywords
        # ("coal", "ore", etc.) and add resource-typed queries with specific
        # resource names. Merging both sources provides defensive coverage
        # without forcing specific knowledge injection — the retrieval's
        # scoring/filtering still operates normally, and LLM's own query
        # choices are fully preserved.
        if self.fallback_queries_fn:
            try:
                keyword_queries = self.fallback_queries_fn(
                    feedback_content, chat_log, skill_code
                )
            except Exception as e:
                logger.warning(f"[LLMKnowledgeRetriever] fallback_queries_fn failed: {e}")
                keyword_queries = []

            if not queries:
                # LLM returned no queries — use fallback alone (legacy path)
                queries = keyword_queries
                source = "fallback"
                if kr:
                    kr.info(f"[Fallback] generated {len(queries)} keyword queries (LLM empty)")
            elif keyword_queries:
                # Merge: add fallback queries not already covered by LLM queries.
                # Dedupe by (query_type, sorted keyword tuple) signature.
                seen = {(q.query_type, tuple(sorted(q.keywords))) for q in queries}
                added = 0
                for kq in keyword_queries:
                    sig = (kq.query_type, tuple(sorted(kq.keywords)))
                    if sig not in seen:
                        queries.append(kq)
                        seen.add(sig)
                        added += 1
                if added > 0:
                    source = "llm+fallback"
                    if kr:
                        kr.info(
                            f"[HybridMerge] LLM={len(queries) - added} + "
                            f"fallback-added={added} → merged total={len(queries)}"
                        )

        # Execute search
        all_knowledge = []
        for query in queries:
            results = self.knowledge_index.search(query, max_results=2)
            all_knowledge.extend(results)
            if kr:
                kr.info(f"[IndexSearch] query_type={query.query_type} "
                        f"keywords={query.keywords[:5]} results={len(results)}")
                for r in results:
                    kr.info(f"  -> {str(r)[:200]}...")

        # Deduplicate
        unique_knowledge = list(dict.fromkeys(all_knowledge))

        # Format
        if unique_knowledge:
            knowledge_text = "## Retrieved Knowledge\n\n" + "\n\n".join(unique_knowledge)
        else:
            knowledge_text = ""

        if kr:
            kr.info(f"[Result] source={source} queries={len(queries)} "
                    f"unique_results={len(unique_knowledge)} chars={len(knowledge_text)}")

        # Update module-level stats
        _kr_stats["total_calls"] += 1
        # "llm+fallback" counts as both an LLM success
        # (LLM returned non-empty) AND as fallback being used (defensive merge).
        if source == "llm" or source == "llm+fallback":
            _kr_stats["llm_successes"] += 1
        if source == "fallback" or source == "llm+fallback":
            _kr_stats["fallback_used"] += 1
        if not unique_knowledge:
            _kr_stats["empty_results"] += 1
        if self.use_llm and self.llm and chat_log and self.query_prompt:
            _kr_stats["llm_attempts"] += 1

        return RetrievalResult(
            knowledge_text=knowledge_text,
            queries_used=queries,
            source=source
        )

    def _generate_queries_with_llm(
        self,
        feedback_content: str,
        chat_log: str,
        skill_code: str,
    ) -> List[KnowledgeQuery]:
        """Generate knowledge queries using LLM."""
        from langchain.schema import HumanMessage
        import time

        prompt = self.query_prompt.format(
            feedback_content=feedback_content[:1000],
            chat_log=chat_log[:1500],
            skill_code=skill_code[:2000] if skill_code else "(not provided)"
        )

        for attempt in range(3):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                record_llm_usage(response, process_type="knowledge_retrieval", function_name="llm_knowledge_retriever._generate_queries_with_llm")
                response_text = response.content if hasattr(response, 'content') else str(response)

                if not response_text or len(response_text.strip()) == 0:
                    logger.warning(f"[LLMKnowledgeRetriever] Empty response from LLM (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"[LLMKnowledgeRetriever] 3 attempts all returned empty responses, using fallback")
                        return []

                queries = self._parse_llm_response(response_text)
                if queries or attempt == 2:
                    if not queries and attempt == 2:
                        logger.warning(f"[LLMKnowledgeRetriever] Failed to parse valid queries after 3 attempts")
                    return queries

                logger.warning(f"[LLMKnowledgeRetriever] Failed to parse response (attempt {attempt + 1}/3)")
                if attempt < 2:
                    time.sleep(2)

            except Exception as e:
                logger.warning(f"[LLMKnowledgeRetriever] LLM invocation exception (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

        return []

    def _parse_llm_response(self, response: str) -> List[KnowledgeQuery]:
        """Parse LLM JSON response."""
        extractors = [
            self._extract_json_array,
            self._extract_from_markdown_code_block,
            self._extract_json_objects,
        ]

        for extractor in extractors:
            try:
                result = extractor(response)
                if result:
                    return result
            except Exception:
                continue

        logger.warning(f"[LLMKnowledgeRetriever] Failed to parse LLM response")
        return []

    def _extract_json_array(self, text: str) -> List[KnowledgeQuery]:
        """Extract JSON array from text."""
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            data = json.loads(json_str)
            return [
                KnowledgeQuery(
                    query_type=item.get("query_type", ""),
                    keywords=item.get("keywords", []),
                    context=item.get("context", ""),
                )
                for item in data
                if isinstance(item, dict)
            ]
        return []

    def _extract_from_markdown_code_block(self, text: str) -> List[KnowledgeQuery]:
        """Extract JSON from markdown code block."""
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return self._extract_json_array(match.group(1))
        return []

    def _extract_json_objects(self, text: str) -> List[KnowledgeQuery]:
        """Extract individual JSON objects from text."""
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, text)
        queries = []
        for match in matches:
            try:
                item = json.loads(match)
                if isinstance(item, dict) and "query_type" in item:
                    queries.append(KnowledgeQuery(
                        query_type=item.get("query_type", ""),
                        keywords=item.get("keywords", []),
                        context=item.get("context", ""),
                    ))
            except json.JSONDecodeError:
                continue
        return queries


# Convenience function
def retrieve_knowledge_for_analysis(
    feedback_content: str,
    chat_log: str = "",
    skill_code: str = "",
    llm: Any = None,
    kr_logger=None,
    query_prompt: str = "",
    fallback_queries_fn: Optional[FallbackQueriesFn] = None,
) -> str:
    """
    Convenience function: retrieve knowledge for analysis.

    Args:
        feedback_content: Feedback content
        chat_log: Chat log
        skill_code: Skill code
        llm: LLM instance (optional)
        kr_logger: Optional diagnostic logger
        query_prompt: Domain-specific LLM prompt template
        fallback_queries_fn: Domain-specific keyword fallback function

    Returns:
        Formatted knowledge text
    """
    retriever = LLMKnowledgeRetriever(
        llm=llm, use_llm=bool(llm),
        query_prompt=query_prompt,
        fallback_queries_fn=fallback_queries_fn,
    )
    result = retriever.retrieve(feedback_content, chat_log, skill_code, kr_logger=kr_logger)
    return result.knowledge_text
