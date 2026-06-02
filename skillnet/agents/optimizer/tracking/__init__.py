"""
Optimizer tracking — transactions, snapshots/rollback, bloat tracking,
optimization history.

Transaction hierarchy:
- SessionTransaction: session-level (contains multiple subgraph transactions)
- SubgraphTransaction: subgraph-level (contains multiple skill transactions)
- SkillTransaction: single-skill transaction

Usage::

    from skillnet.agents.optimizer.tracking import TransactionManager

    txn_mgr = TransactionManager(skill_graph_manager=manager)
    session = txn_mgr.begin_session("optimization", primary_skill="craftIronSword")
    subgraph_txn = txn_mgr.begin_subgraph_transaction(session, "craftIronSword")
"""

from .transaction import (
    TransactionType,
    TransactionStatus,
    SkillTransaction,
    SubgraphTransaction,
    SessionTransaction,
    TransactionManager,
)

from .bloat_tracker import CodeBloatTracker
from .loop_manager import LoopManager
from .optimization_tracker import OptimizationTracker

__all__ = [
    "TransactionType",
    "TransactionStatus",
    "SkillTransaction",
    "SubgraphTransaction",
    "SessionTransaction",
    "TransactionManager",
    "CodeBloatTracker",
    "LoopManager",
    "OptimizationTracker",
]
