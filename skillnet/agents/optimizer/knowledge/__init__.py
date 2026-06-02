"""
Knowledge - Domain-agnostic knowledge framework

Provides knowledge base types and retrieval.
Domain-specific knowledge data is loaded via registered builders.

All symbols are available from their respective submodules:
- .index: KnowledgeIndex, register_knowledge_builder
- .llm_knowledge_retriever: retrieve_knowledge_for_analysis
- .physical_constraints: ConstraintViolation, ConstraintType

Base types (canonical source: skillnet.core.knowledge_base):
- KnowledgeDomain, KnowledgeCategory, KnowledgeItem, RetrievedKnowledge
"""
