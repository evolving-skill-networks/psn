"""
Transaction Manager

Transaction management module that supports nested transactions and rollback
for optimization and refactor operations.

Transaction hierarchy:
- SessionTransaction: session-level transaction (contains multiple subgraph transactions)
- SubgraphTransaction: subgraph-level transaction (contains multiple skill transactions)
- SkillTransaction: transaction for an individual skill

Usage:

    # Create the transaction manager
    txn_mgr = TransactionManager(skill_graph_manager=manager)

    # Begin a session-level transaction
    session = txn_mgr.begin_session("optimization", primary_skill="craftIronSword")

    # Begin a subgraph transaction
    subgraph_txn = txn_mgr.begin_subgraph_transaction(
        session, primary_skill="craftIronSword", task="craft iron sword"
    )

    # Begin a single skill transaction
    skill_txn = txn_mgr.begin_skill_transaction(subgraph_txn, "craftIronSword")

    try:
        # Perform optimization operations...
        txn_mgr.commit_skill(skill_txn)
    except Exception:
        txn_mgr.rollback_skill(skill_txn)

    # Commit or roll back the subgraph transaction
    txn_mgr.commit_subgraph(subgraph_txn)
    # or txn_mgr.rollback_subgraph(subgraph_txn)

    # End the session
    txn_mgr.end_session(session)
"""

import uuid
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
from datetime import datetime
from enum import Enum

from skillnet.agents.skill_graph.models.node import SkillNode

if TYPE_CHECKING:
    from skillnet.agents.skill_graph import SkillPrecondition, SkillEffect


class TransactionType(Enum):
    """Transaction type."""
    OPTIMIZATION = "optimization"
    REFACTOR = "refactor"
    MIXED = "mixed"  # Both optimization and refactor


class TransactionStatus(Enum):
    """Transaction status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    PARTIAL_ROLLBACK = "partial_rollback"


@dataclass
class SkillSnapshot:
    """
    Snapshot of a single skill.

    Saves the complete state of the skill at the start of the transaction, used for rollback.
    """
    skill_name: str
    code: str
    version: str
    description: str

    # Preconditions and effects (serialized lists)
    preconditions: List[Dict[str, Any]] = field(default_factory=list)
    expected_effects: List[Dict[str, Any]] = field(default_factory=list)

    # Parameters
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Coverage status
    is_covered: bool = False
    covered_by: Optional[str] = None
    coverage_type: Optional[str] = None  # CoverageType enum value as string

    # Dependency relationships
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)

    # Metadata
    snapshot_time: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_node(cls, node: 'SkillNode') -> 'SkillSnapshot':
        """Create a snapshot from a SkillNode."""
        # Serialize preconditions
        preconditions = []
        for pc in node.preconditions:
            preconditions.append({
                "description": pc.description,
                "check_code": pc.check_code,
                "error_message": pc.error_message,
            })

        # Serialize effects
        effects = []
        for eff in node.expected_effects:
            effects.append({
                "description": eff.description,
                "check_code": eff.check_code,
                "is_verified": eff.is_verified,
            })

        return cls(
            skill_name=node.name,
            code=node.code,
            version=node.version,
            description=node.description,
            preconditions=preconditions,
            expected_effects=effects,
            parameters=dict(node.parameters) if node.parameters else {},
            is_covered=node.is_covered,
            covered_by=node.covered_by,
            coverage_type=node.coverage_type.value if node.coverage_type else None,
            dependencies=list(node.dependencies),
            dependents=list(node.dependents),
        )


@dataclass
class SkillTransaction:
    """
    Transaction for a single skill.

    Tracks modifications of one skill and supports rollback to the state before the transaction started.
    """
    transaction_id: str
    skill_name: str
    snapshot_before: SkillSnapshot
    transaction_type: TransactionType = TransactionType.OPTIMIZATION

    # Status
    status: TransactionStatus = TransactionStatus.PENDING
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Modification record
    changes: List[str] = field(default_factory=list)
    snapshot_after: Optional[SkillSnapshot] = None

    # Error message
    error_message: Optional[str] = None


@dataclass
class SubgraphTransaction:
    """
    Subgraph-level transaction.

    Contains multiple skill transactions; supports:
    - Commit all
    - Roll back all
    - Partial rollback (only rolls back specified skills)
    """
    transaction_id: str
    primary_skill: str  # The primary skill that triggered the transaction
    transaction_type: TransactionType = TransactionType.OPTIMIZATION
    task: Optional[str] = None

    # Nested skill transactions
    skill_transactions: Dict[str, SkillTransaction] = field(default_factory=dict)

    # Status
    status: TransactionStatus = TransactionStatus.PENDING
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Dependency relationships (used for correct rollback order)
    dependency_order: List[str] = field(default_factory=list)

    # Partial-rollback record
    rolled_back_skills: List[str] = field(default_factory=list)
    committed_skills: List[str] = field(default_factory=list)

    # Error message
    error_message: Optional[str] = None


@dataclass
class SessionTransaction:
    """
    Session-level transaction.

    A complete optimization/refactor session, which may contain multiple subgraph transactions.
    """
    session_id: str
    session_type: TransactionType
    primary_skill: Optional[str] = None
    task: Optional[str] = None

    # Nested subgraph transactions
    subgraph_transactions: Dict[str, SubgraphTransaction] = field(default_factory=dict)

    # Status
    status: TransactionStatus = TransactionStatus.PENDING
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Statistics
    total_skills_modified: int = 0
    total_skills_rolled_back: int = 0


class TransactionManager:
    """
    Transaction manager.

    Manages transactions for optimization and refactor operations; supports:
    1. Nested transactions (session -> subgraph -> skill)
    2. Snapshots and rollback
    3. Partial rollback
    4. Transaction log persistence
    """

    def __init__(
        self,
        skill_graph_manager=None,
        ckpt_dir: Optional[str] = None,
        logger=None,
    ):
        """
        Initialize the transaction manager.

        Args:
            skill_graph_manager: Skill graph manager
            ckpt_dir: Checkpoint directory (used for persisting transaction logs)
            logger: Logger
        """
        self.skill_graph_manager = skill_graph_manager
        self.ckpt_dir = ckpt_dir
        self.logger = logger

        # Active sessions
        self.active_sessions: Dict[str, SessionTransaction] = {}

        # Transaction log directory
        self.txn_log_dir = None
        if ckpt_dir:
            self.txn_log_dir = os.path.join(ckpt_dir, "transactions")
            os.makedirs(self.txn_log_dir, exist_ok=True)

    def _log(self, message: str, level: str = "info"):
        """Emit a log message."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())[:8]

    # ==================== Session-level transactions ====================

    def begin_session(
        self,
        session_type: str,
        primary_skill: Optional[str] = None,
        task: Optional[str] = None,
    ) -> SessionTransaction:
        """
        Begin a session-level transaction.

        Args:
            session_type: "optimization" or "refactor"
            primary_skill: Primary skill
            task: Associated task

        Returns:
            SessionTransaction: Session transaction
        """
        txn_type = TransactionType(session_type) if session_type in ["optimization", "refactor"] else TransactionType.MIXED

        session = SessionTransaction(
            session_id=f"session_{self._generate_id()}",
            session_type=txn_type,
            primary_skill=primary_skill,
            task=task,
            status=TransactionStatus.IN_PROGRESS,
        )

        self.active_sessions[session.session_id] = session

        self._log(
            f"[Transaction] Begin session {session.session_id} "
            f"({session_type}, primary={primary_skill})",
            "info"
        )

        return session

    def end_session(
        self,
        session: SessionTransaction,
        commit: bool = True
    ) -> bool:
        """
        End the session.

        Args:
            session: Session transaction
            commit: True to commit, False to roll back

        Returns:
            bool: Whether the operation succeeded
        """
        try:
            if commit:
                # Commit all subgraph transactions
                for subgraph_txn in session.subgraph_transactions.values():
                    if subgraph_txn.status == TransactionStatus.IN_PROGRESS:
                        self.commit_subgraph(subgraph_txn)
                session.status = TransactionStatus.COMMITTED
            else:
                # Roll back all subgraph transactions
                for subgraph_txn in session.subgraph_transactions.values():
                    if subgraph_txn.status != TransactionStatus.ROLLED_BACK:
                        self.rollback_subgraph(subgraph_txn)
                session.status = TransactionStatus.ROLLED_BACK

            session.completed_at = datetime.now().isoformat()

            # Compute statistics
            for subgraph_txn in session.subgraph_transactions.values():
                session.total_skills_modified += len(subgraph_txn.committed_skills)
                session.total_skills_rolled_back += len(subgraph_txn.rolled_back_skills)

            # Save the transaction log
            self._save_session_log(session)

            # Cleanup
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]

            self._log(
                f"[Transaction] End session {session.session_id} "
                f"(status={session.status.value}, "
                f"modified={session.total_skills_modified}, "
                f"rolled_back={session.total_skills_rolled_back})",
                "info"
            )

            return True

        except Exception as e:
            self._log(f"[Transaction] Failed to end session: {e}", "error")
            session.status = TransactionStatus.FAILED
            session.error_message = str(e)
            return False

    # ==================== Subgraph-level transactions ====================

    def begin_subgraph_transaction(
        self,
        session: SessionTransaction,
        primary_skill: str,
        task: Optional[str] = None,
        transaction_type: Optional[str] = None,
    ) -> SubgraphTransaction:
        """
        Begin a subgraph-level transaction.

        Args:
            session: Parent session transaction
            primary_skill: The primary skill that triggered the transaction
            task: Associated task
            transaction_type: Transaction type (defaults to inheriting the session type)

        Returns:
            SubgraphTransaction: Subgraph transaction
        """
        txn_type = (
            TransactionType(transaction_type)
            if transaction_type
            else session.session_type
        )

        subgraph_txn = SubgraphTransaction(
            transaction_id=f"subgraph_{self._generate_id()}",
            primary_skill=primary_skill,
            transaction_type=txn_type,
            task=task or session.task,
            status=TransactionStatus.IN_PROGRESS,
        )

        # Add to session
        session.subgraph_transactions[subgraph_txn.transaction_id] = subgraph_txn

        self._log(
            f"[Transaction] Begin subgraph transaction {subgraph_txn.transaction_id} "
            f"(primary={primary_skill})",
            "info"
        )

        return subgraph_txn

    def commit_subgraph(self, subgraph_txn: SubgraphTransaction) -> bool:
        """
        Commit the subgraph transaction.

        Args:
            subgraph_txn: Subgraph transaction

        Returns:
            bool: Whether the operation succeeded
        """
        try:
            # Commit all uncommitted skill transactions
            for skill_name, skill_txn in subgraph_txn.skill_transactions.items():
                if skill_txn.status == TransactionStatus.IN_PROGRESS:
                    self.commit_skill(skill_txn)

                if skill_txn.status == TransactionStatus.COMMITTED:
                    subgraph_txn.committed_skills.append(skill_name)

            subgraph_txn.status = TransactionStatus.COMMITTED
            subgraph_txn.completed_at = datetime.now().isoformat()

            self._log(
                f"[Transaction] Commit subgraph transaction {subgraph_txn.transaction_id} "
                f"({len(subgraph_txn.committed_skills)} skills committed)",
                "info"
            )

            return True

        except Exception as e:
            self._log(f"[Transaction] Failed to commit subgraph transaction: {e}", "error")
            subgraph_txn.status = TransactionStatus.FAILED
            subgraph_txn.error_message = str(e)
            return False

    def rollback_subgraph(self, subgraph_txn: SubgraphTransaction) -> bool:
        """
        Roll back the subgraph transaction.

        Roll back all skills in reverse dependency order.

        Args:
            subgraph_txn: Subgraph transaction

        Returns:
            bool: Whether the operation succeeded
        """
        try:
            # Roll back in reverse order
            rollback_order = list(reversed(subgraph_txn.dependency_order))
            if not rollback_order:
                rollback_order = list(subgraph_txn.skill_transactions.keys())

            for skill_name in rollback_order:
                if skill_name in subgraph_txn.skill_transactions:
                    skill_txn = subgraph_txn.skill_transactions[skill_name]
                    if skill_txn.status != TransactionStatus.ROLLED_BACK:
                        self.rollback_skill(skill_txn)
                        subgraph_txn.rolled_back_skills.append(skill_name)

            subgraph_txn.status = TransactionStatus.ROLLED_BACK
            subgraph_txn.completed_at = datetime.now().isoformat()

            self._log(
                f"[Transaction] Roll back subgraph transaction {subgraph_txn.transaction_id} "
                f"({len(subgraph_txn.rolled_back_skills)} skills rolled back)",
                "info"
            )

            return True

        except Exception as e:
            self._log(f"[Transaction] Failed to roll back subgraph transaction: {e}", "error")
            subgraph_txn.status = TransactionStatus.FAILED
            subgraph_txn.error_message = str(e)
            return False

    def partial_rollback(
        self,
        subgraph_txn: SubgraphTransaction,
        skills_to_rollback: List[str]
    ) -> bool:
        """
        Partial rollback: only roll back the specified skills.

        Args:
            subgraph_txn: Subgraph transaction
            skills_to_rollback: List of skills to roll back

        Returns:
            bool: Whether the operation succeeded
        """
        try:
            for skill_name in skills_to_rollback:
                if skill_name in subgraph_txn.skill_transactions:
                    skill_txn = subgraph_txn.skill_transactions[skill_name]
                    if skill_txn.status != TransactionStatus.ROLLED_BACK:
                        self.rollback_skill(skill_txn)
                        subgraph_txn.rolled_back_skills.append(skill_name)

            subgraph_txn.status = TransactionStatus.PARTIAL_ROLLBACK

            self._log(
                f"[Transaction] Partial rollback of subgraph transaction {subgraph_txn.transaction_id} "
                f"({len(skills_to_rollback)} skills rolled back)",
                "info"
            )

            return True

        except Exception as e:
            self._log(f"[Transaction] Partial rollback failed: {e}", "error")
            return False

    # ==================== Skill-level transactions ====================

    def begin_skill_transaction(
        self,
        subgraph_txn: SubgraphTransaction,
        skill_name: str,
        transaction_type: Optional[str] = None,
    ) -> Optional[SkillTransaction]:
        """
        Begin a transaction for a single skill.

        Args:
            subgraph_txn: Parent subgraph transaction
            skill_name: Skill name
            transaction_type: Transaction type (defaults to inheriting the subgraph type)

        Returns:
            Optional[SkillTransaction]: Skill transaction; returns None if the skill does not exist
        """
        if not self.skill_graph_manager:
            self._log("[Transaction] Cannot create skill transaction: no skill_graph_manager", "error")
            return None

        node = self.skill_graph_manager.get_node(skill_name)
        if not node:
            self._log(f"[Transaction] Cannot create transaction: skill '{skill_name}' does not exist", "warning")
            return None

        txn_type = (
            TransactionType(transaction_type)
            if transaction_type
            else subgraph_txn.transaction_type
        )

        # Create the snapshot
        snapshot = SkillSnapshot.from_node(node)

        skill_txn = SkillTransaction(
            transaction_id=f"skill_{self._generate_id()}",
            skill_name=skill_name,
            snapshot_before=snapshot,
            transaction_type=txn_type,
            status=TransactionStatus.IN_PROGRESS,
        )

        # Add to the subgraph transaction
        subgraph_txn.skill_transactions[skill_name] = skill_txn

        # Update the dependency order
        if skill_name not in subgraph_txn.dependency_order:
            subgraph_txn.dependency_order.append(skill_name)

        self._log(
            f"[Transaction] Begin skill transaction {skill_txn.transaction_id} "
            f"for '{skill_name}'",
            "info"
        )

        return skill_txn

    def commit_skill(self, skill_txn: SkillTransaction) -> bool:
        """
        Commit the skill transaction.

        Args:
            skill_txn: Skill transaction

        Returns:
            bool: Whether the operation succeeded
        """
        try:
            if not self.skill_graph_manager:
                return False

            # Get the current state as the after snapshot
            node = self.skill_graph_manager.get_node(skill_txn.skill_name)
            if node:
                skill_txn.snapshot_after = SkillSnapshot.from_node(node)

            skill_txn.status = TransactionStatus.COMMITTED
            skill_txn.completed_at = datetime.now().isoformat()

            self._log(
                f"[Transaction] Commit skill transaction {skill_txn.transaction_id} "
                f"for '{skill_txn.skill_name}'",
                "info"
            )

            return True

        except Exception as e:
            self._log(f"[Transaction] Failed to commit skill transaction: {e}", "error")
            skill_txn.status = TransactionStatus.FAILED
            skill_txn.error_message = str(e)
            return False

    def rollback_skill(self, skill_txn: SkillTransaction) -> bool:
        """
        Roll back the skill transaction.

        Restores the skill to its state before the transaction started.

        Args:
            skill_txn: Skill transaction

        Returns:
            bool: Whether the operation succeeded
        """
        try:
            if not self.skill_graph_manager:
                return False

            node = self.skill_graph_manager.get_node(skill_txn.skill_name)
            if not node:
                # The skill may have been deleted; recreate it
                self._restore_deleted_skill(skill_txn.snapshot_before)
            else:
                # Restore the skill state
                self._restore_skill_from_snapshot(node, skill_txn.snapshot_before)

            skill_txn.status = TransactionStatus.ROLLED_BACK
            skill_txn.completed_at = datetime.now().isoformat()

            self._log(
                f"[Transaction] Roll back skill transaction {skill_txn.transaction_id} "
                f"for '{skill_txn.skill_name}'",
                "info"
            )

            return True

        except Exception as e:
            self._log(f"[Transaction] Failed to roll back skill transaction: {e}", "error")
            skill_txn.status = TransactionStatus.FAILED
            skill_txn.error_message = str(e)
            return False

    def _restore_skill_from_snapshot(
        self,
        node: 'SkillNode',
        snapshot: SkillSnapshot
    ):
        """Restore the skill state from a snapshot."""
        node.code = snapshot.code
        node.version = snapshot.version
        node.description = snapshot.description
        node.is_covered = snapshot.is_covered
        node.covered_by = snapshot.covered_by
        # Restore coverage_type from string
        from skillnet.agents.skill_graph.models.coverage import CoverageType
        node.coverage_type = CoverageType.from_refactor_type(snapshot.coverage_type) if snapshot.coverage_type else None

        # Restore parameters
        node.parameters = dict(snapshot.parameters)

        # Restore dependency relationships
        node.dependencies = set(snapshot.dependencies)
        node.dependents = set(snapshot.dependents)

        # Note: restoring preconditions and effects requires recreating objects;
        # only recorded here — full restoration may need more complex logic

    def _restore_deleted_skill(self, snapshot: SkillSnapshot):
        """Restore a deleted skill."""
        if self.skill_graph_manager:
            node = SkillNode(
                name=snapshot.skill_name,
                code=snapshot.code,
                description=snapshot.description,
            )
            self.skill_graph_manager.add_skill_node(node)
            # Restore other attributes
            restored_node = self.skill_graph_manager.get_node(snapshot.skill_name)
            if restored_node:
                self._restore_skill_from_snapshot(restored_node, snapshot)

    # ==================== Persistence ====================

    def _save_session_log(self, session: SessionTransaction):
        """Save the session log to file."""
        if not self.txn_log_dir:
            return

        try:
            log_file = os.path.join(
                self.txn_log_dir,
                f"{session.session_id}.json"
            )

            # Convert to a serializable dict
            log_data = self._session_to_dict(session)

            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)

            self._log(f"[Transaction] Saved session log to {log_file}", "info")

        except Exception as e:
            self._log(f"[Transaction] Failed to save session log: {e}", "warning")

    def _session_to_dict(self, session: SessionTransaction) -> Dict[str, Any]:
        """Convert a session to a serializable dict."""
        return {
            "session_id": session.session_id,
            "session_type": session.session_type.value,
            "primary_skill": session.primary_skill,
            "task": session.task,
            "status": session.status.value,
            "started_at": session.started_at,
            "completed_at": session.completed_at,
            "total_skills_modified": session.total_skills_modified,
            "total_skills_rolled_back": session.total_skills_rolled_back,
            "subgraph_transactions": {
                txn_id: self._subgraph_to_dict(txn)
                for txn_id, txn in session.subgraph_transactions.items()
            }
        }

    def _subgraph_to_dict(self, txn: SubgraphTransaction) -> Dict[str, Any]:
        """Convert a subgraph transaction to a serializable dict."""
        return {
            "transaction_id": txn.transaction_id,
            "primary_skill": txn.primary_skill,
            "transaction_type": txn.transaction_type.value,
            "task": txn.task,
            "status": txn.status.value,
            "started_at": txn.started_at,
            "completed_at": txn.completed_at,
            "dependency_order": txn.dependency_order,
            "committed_skills": txn.committed_skills,
            "rolled_back_skills": txn.rolled_back_skills,
            "skill_transactions": {
                skill: self._skill_txn_to_dict(skill_txn)
                for skill, skill_txn in txn.skill_transactions.items()
            }
        }

    def _skill_txn_to_dict(self, txn: SkillTransaction) -> Dict[str, Any]:
        """Convert a skill transaction to a serializable dict."""
        return {
            "transaction_id": txn.transaction_id,
            "skill_name": txn.skill_name,
            "transaction_type": txn.transaction_type.value,
            "status": txn.status.value,
            "started_at": txn.started_at,
            "completed_at": txn.completed_at,
            "changes": txn.changes,
            "error_message": txn.error_message,
            "snapshot_before": asdict(txn.snapshot_before),
            "snapshot_after": asdict(txn.snapshot_after) if txn.snapshot_after else None,
        }

    def load_session_log(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session log."""
        if not self.txn_log_dir:
            return None

        try:
            log_file = os.path.join(self.txn_log_dir, f"{session_id}.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self._log(f"[Transaction] Failed to load session log: {e}", "warning")

        return None

    def list_session_logs(self) -> List[str]:
        """List all session logs."""
        if not self.txn_log_dir or not os.path.exists(self.txn_log_dir):
            return []

        return [
            f.replace(".json", "")
            for f in os.listdir(self.txn_log_dir)
            if f.endswith(".json")
        ]


# ==================== Convenience functions ====================

