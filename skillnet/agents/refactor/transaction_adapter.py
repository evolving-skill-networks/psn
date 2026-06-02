"""
Refactor Transaction Adapter

Unified transaction management adapter: integrate Refactor operations with TransactionManager.

Features:
- Provides a simplified Refactor transaction API
- Automatically manages transaction lifecycle
- Supports context manager syntax
- Integrates with RefactorHistoryTracker

Usage:

    # Style 1: context manager
    with RefactorTransactionContext(manager, skill_graph_manager) as ctx:
        result = parametric_refactor.execute(...)
        ctx.record_result(result)
        # Auto commit or rollback

    # Style 2: manual management
    adapter = RefactorTransactionAdapter(manager, skill_graph_manager, history_tracker)
    adapter.begin_refactor_session("craftOakBoat")
    try:
        adapter.begin_skill_refactor("craftOakBoat")
        # Run refactor...
        adapter.commit_skill_refactor("craftOakBoat", result)
        adapter.commit_session()
    except Exception:
        adapter.rollback_session()

    # Style 3: batch refactor
    with RefactorBatchContext(adapter, plan) as batch:
        for opportunity in plan.opportunities:
            batch.execute_refactor(opportunity)
"""

import uuid
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Callable, Tuple
from datetime import datetime
from contextlib import contextmanager

from .base import RefactorResult, RefactorOpportunity, RefactorType
from .history import RefactorHistoryTracker, RefactorRecord

if TYPE_CHECKING:
    from skillnet.agents.optimizer.tracking.transaction import (
        TransactionManager, SessionTransaction, SubgraphTransaction, SkillTransaction
    )
    from skillnet.agents.skill_graph import SkillGraphManager


@dataclass
class RefactorTransactionState:
    """
    Refactor transaction state

    Tracks the transaction state of a single refactor operation.
    """
    skill_name: str
    transaction_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # State
    is_active: bool = True
    committed: bool = False
    rolled_back: bool = False

    # Result
    result: Optional[RefactorResult] = None
    error: Optional[str] = None

    # Associated transaction ID
    skill_txn_id: Optional[str] = None


class RefactorTransactionAdapter:
    """
    Refactor transaction adapter

    Integrates Refactor operations with TransactionManager.
    """

    def __init__(
        self,
        transaction_manager: 'TransactionManager',
        skill_graph_manager: 'SkillGraphManager',
        history_tracker: Optional[RefactorHistoryTracker] = None,
        logger=None,
    ):
        """
        Initialize the adapter

        Args:
            transaction_manager: transaction manager
            skill_graph_manager: skill graph manager
            history_tracker: history tracker (optional)
            logger: logger
        """
        self.txn_manager = transaction_manager
        self.skill_graph_manager = skill_graph_manager
        self.history_tracker = history_tracker
        self.logger = logger

        # Thread-safety lock (RLock to allow recursive calls from the same thread)
        self._lock = threading.RLock()

        # Currently active transactions
        self._session: Optional['SessionTransaction'] = None
        self._subgraph: Optional['SubgraphTransaction'] = None
        self._skill_states: Dict[str, RefactorTransactionState] = {}

    def _log(self, message: str, level: str = "info"):
        """Emit a log message"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)

    # ==================== Session level ====================

    def begin_refactor_session(
        self,
        primary_skill: str,
        task: Optional[str] = None,
    ) -> str:
        """
        Start a refactor session

        Args:
            primary_skill: primary skill
            task: associated task

        Returns:
            str: session ID

        Raises:
            RuntimeError: if there is already an active session
        """
        with self._lock:
            # Check whether there's already an active session to prevent leaks
            if self._session is not None:
                self._log(
                    f"[RefactorTxn] Warning: there's already an active session {self._session.session_id}; auto-closing",
                    "warning"
                )
                # Attempt to roll back the old session
                try:
                    self.rollback_session()
                except Exception as e:
                    self._log(f"[RefactorTxn] Failed to close old session: {e}", "error")

        # Create the session
        self._session = self.txn_manager.begin_session(
            "refactor",
            primary_skill=primary_skill,
            task=task,
        )

        # Create the subgraph transaction
        self._subgraph = self.txn_manager.begin_subgraph_transaction(
            self._session,
            primary_skill=primary_skill,
            task=task,
        )

        self._log(
            f"[RefactorTxn] Started refactor session {self._session.session_id} "
            f"(primary={primary_skill})",
            "info"
        )

        return self._session.session_id

    def commit_session(self) -> bool:
        """
        Commit the refactor session

        Returns:
            bool: success
        """
        if not self._session:
            return False

        try:
            # Check for unfinished skill transactions
            with self._lock:
                active_skills = [
                    state.skill_name for state in self._skill_states.values()
                    if state.is_active and not state.committed and not state.rolled_back
                ]
                if active_skills:
                    self._log(
                        f"[RefactorTxn] Warning: {len(active_skills)} unfinished skill transactions: {active_skills}",
                        "warning"
                    )

            # Commit the subgraph
            if self._subgraph:
                self.txn_manager.commit_subgraph(self._subgraph)

            # End the session
            success = self.txn_manager.end_session(self._session, commit=True)

            self._log(
                f"[RefactorTxn] Committed session {self._session.session_id}",
                "info"
            )

            # Cleanup
            self._cleanup()

            return success

        except Exception as e:
            self._log(f"[RefactorTxn] Failed to commit session: {e}", "error")
            return False

    def rollback_session(self) -> bool:
        """
        Roll back the refactor session

        Returns:
            bool: success
        """
        if not self._session:
            return False

        try:
            # Roll back the subgraph
            if self._subgraph:
                self.txn_manager.rollback_subgraph(self._subgraph)

            # End the session
            success = self.txn_manager.end_session(self._session, commit=False)

            # Record the rollback to history
            if self.history_tracker:
                for state in self._skill_states.values():
                    if state.result:
                        self.history_tracker.record_rollback(
                            state.transaction_id,
                            reason="session_rollback"
                        )

            self._log(
                f"[RefactorTxn] Rolled back session {self._session.session_id}",
                "info"
            )

            # Cleanup
            self._cleanup()

            return success

        except Exception as e:
            self._log(f"[RefactorTxn] Failed to roll back session: {e}", "error")
            return False

    def _cleanup(self):
        """Clean up transaction state"""
        self._session = None
        self._subgraph = None
        self._skill_states.clear()

    # ==================== Skill level ====================

    def begin_skill_refactor(self, skill_name: str) -> Optional[str]:
        """
        Start refactoring a single skill

        Args:
            skill_name: skill name

        Returns:
            Optional[str]: transaction ID, or None on failure
        """
        with self._lock:
            if not self._subgraph:
                self._log("[RefactorTxn] No active session; cannot start skill refactor", "error")
                return None

            # Create the skill transaction
            skill_txn = self.txn_manager.begin_skill_transaction(
                self._subgraph,
                skill_name,
                transaction_type="refactor",
            )

            if not skill_txn:
                return None

            # Create state tracking
            state = RefactorTransactionState(
                skill_name=skill_name,
                transaction_id=f"refactor_{uuid.uuid4().hex[:12]}",
                skill_txn_id=skill_txn.transaction_id,
            )
            self._skill_states[skill_name] = state

            self._log(
                f"[RefactorTxn] Started skill refactor {skill_name} "
                f"(txn={skill_txn.transaction_id})",
                "info"
            )

            return state.transaction_id

    def commit_skill_refactor(
        self,
        skill_name: str,
        result: RefactorResult,
        task: Optional[str] = None,
        trigger: str = "manual",
    ) -> bool:
        """
        Commit the refactor for a single skill

        Args:
            skill_name: skill name
            result: refactor result
            task: associated task
            trigger: trigger method

        Returns:
            bool: success
        """
        with self._lock:
            state = self._skill_states.get(skill_name)
            if not state or not state.is_active:
                return False

            try:
                # Commit the transaction
                if not self._subgraph:
                    self._log(f"[RefactorTxn] No active subgraph transaction; skipping skill commit", "warning")
                    skill_txn = None
                else:
                    skill_txn = self._subgraph.skill_transactions.get(skill_name)
                if skill_txn:
                    self.txn_manager.commit_skill(skill_txn)

                # Update state
                state.result = result
                state.committed = True
                state.is_active = False

                # Record in history.
                # For a BEHAVIORAL Case B batch refactor, sub_results contains
                # an independent result for each covered_skill. We record a
                # separate history entry per sub_result instead of only the
                # merged result (avoids the r34 batch refactor information-loss bug).
                if self.history_tracker:
                    session_id = self._session.session_id if self._session else None
                    if result.sub_results:
                        # Record each sub_result individually
                        for sub_result in result.sub_results:
                            self.history_tracker.record(
                                result=sub_result,
                                task=task,
                                trigger=trigger,
                                session_id=session_id,
                            )
                    else:
                        # Standard case: only record the merged result
                        self.history_tracker.record(
                            result=result,
                            task=task,
                            trigger=trigger,
                            session_id=session_id,
                        )

                self._log(
                    f"[RefactorTxn] Committed skill refactor {skill_name} "
                    f"({'success' if result.success else 'failure'})",
                    "info"
                )

                return True

            except Exception as e:
                self._log(f"[RefactorTxn] Failed to commit skill refactor: {e}", "error")
                state.error = str(e)
                return False

    def rollback_skill_refactor(
        self,
        skill_name: str,
        reason: str = "manual",
    ) -> bool:
        """
        Roll back the refactor for a single skill

        Args:
            skill_name: skill name
            reason: rollback reason

        Returns:
            bool: success
        """
        with self._lock:
            state = self._skill_states.get(skill_name)
            if not state:
                return False

            try:
                # Roll back the transaction (check whether _subgraph exists)
                if self._subgraph:
                    skill_txn = self._subgraph.skill_transactions.get(skill_name)
                    if skill_txn:
                        self.txn_manager.rollback_skill(skill_txn)

                # Update state
                state.rolled_back = True
                state.is_active = False

                # Record the rollback to history
                if self.history_tracker and state.result:
                    self.history_tracker.record_rollback(
                        state.transaction_id,
                        reason=reason
                    )

                self._log(
                    f"[RefactorTxn] Rolled back skill refactor {skill_name}: {reason}",
                    "info"
                )

                return True

            except Exception as e:
                self._log(f"[RefactorTxn] Failed to roll back skill refactor: {e}", "error")
                return False

    # ==================== Queries ====================

    def get_session_id(self) -> Optional[str]:
        """Get the current session ID"""
        return self._session.session_id if self._session else None


class RefactorTransactionContext:
    """
    Refactor transaction context manager

    Provides a simplified context-manager syntax.

    Usage:
        with RefactorTransactionContext(adapter, "craftOakBoat") as ctx:
            result = refactor.execute(...)
            ctx.record_result(result)
    """

    def __init__(
        self,
        adapter: RefactorTransactionAdapter,
        primary_skill: str,
        task: Optional[str] = None,
        auto_commit: bool = True,
    ):
        """
        Initialize the context

        Args:
            adapter: transaction adapter
            primary_skill: primary skill
            task: associated task
            auto_commit: whether to auto-commit (when no error)
        """
        self.adapter = adapter
        self.primary_skill = primary_skill
        self.task = task
        self.auto_commit = auto_commit

        self._results: List[RefactorResult] = []
        self._error: Optional[Exception] = None

    def __enter__(self) -> 'RefactorTransactionContext':
        """Enter the context"""
        self.adapter.begin_refactor_session(
            self.primary_skill,
            task=self.task,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context"""
        if exc_type is not None:
            # Exception occurred; roll back
            self._error = exc_val
            self.adapter.rollback_session()
            return False

        if self.auto_commit:
            # Auto-commit
            self.adapter.commit_session()

        return False

    def record_result(
        self,
        result: RefactorResult,
        trigger: str = "manual",
    ):
        """
        Record a refactor result

        Args:
            result: refactor result
            trigger: trigger method
        """
        self._results.append(result)

        # If the refactor was successful, commit the skill transaction
        if result.success:
            txn_id = self.adapter.begin_skill_refactor(result.source_skill)
            if txn_id:  # Only commit if begin succeeded
                self.adapter.commit_skill_refactor(
                    result.source_skill,
                    result,
                    task=self.task,
                    trigger=trigger,
                )
        else:
            # Failed refactors (validation rejection, syntax check,
            # semantic preservation, etc.) — record the attempt to
            # history so R_cons properly counts pre-commit rejections.
            # (Without this, only commit-path successes/rollbacks are
            # tracked; validation rejections are silently dropped and
            # R_cons artificially reports near-1.0.)
            if self.adapter.history_tracker:
                session_id = (self.adapter._session.session_id
                              if self.adapter._session else None)
                self.adapter.history_tracker.record(
                    result=result,
                    task=self.task,
                    trigger=trigger,
                    session_id=session_id,
                )

    def get_results(self) -> List[RefactorResult]:
        """Get all recorded results"""
        return self._results

    @property
    def session_id(self) -> Optional[str]:
        """Get the session ID"""
        return self.adapter.get_session_id()


@contextmanager
def refactor_transaction(
    transaction_manager: 'TransactionManager',
    skill_graph_manager: 'SkillGraphManager',
    primary_skill: str,
    history_tracker: Optional[RefactorHistoryTracker] = None,
    task: Optional[str] = None,
    logger=None,
):
    """
    Refactor transaction context manager (functional API)

    Usage:
        with refactor_transaction(txn_mgr, sgm, "craftOakBoat") as adapter:
            adapter.begin_skill_refactor("craftOakBoat")
            # Run refactor...
            adapter.commit_skill_refactor("craftOakBoat", result)

    Args:
        transaction_manager: transaction manager
        skill_graph_manager: skill graph manager
        primary_skill: primary skill
        history_tracker: history tracker
        task: associated task
        logger: logger

    Yields:
        RefactorTransactionAdapter: adapter instance
    """
    adapter = RefactorTransactionAdapter(
        transaction_manager=transaction_manager,
        skill_graph_manager=skill_graph_manager,
        history_tracker=history_tracker,
        logger=logger,
    )

    adapter.begin_refactor_session(primary_skill, task=task)

    try:
        yield adapter
        adapter.commit_session()
    except Exception:
        adapter.rollback_session()
        raise


class RefactorBatchContext:
    """
    Batch refactor transaction context

    Used for handling batch refactors.

    Usage:
        with RefactorBatchContext(adapter, plan) as batch:
            for opportunity in plan.opportunities:
                result = batch.execute_refactor(opportunity, refactor_func)
    """

    def __init__(
        self,
        adapter: RefactorTransactionAdapter,
        task: Optional[str] = None,
        on_error: str = "continue",  # "continue" | "rollback_skill" | "rollback_all"
    ):
        """
        Initialize the batch context

        Args:
            adapter: transaction adapter
            task: associated task
            on_error: error-handling strategy
        """
        self.adapter = adapter
        self.task = task
        self.on_error = on_error

        self._results: List[RefactorResult] = []
        self._errors: List[Tuple[str, Exception]] = []

    def __enter__(self) -> 'RefactorBatchContext':
        """Enter the context"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context"""
        if exc_type is not None and self.on_error == "rollback_all":
            self.adapter.rollback_session()
            return False
        return False

    def execute_refactor(
        self,
        opportunity: RefactorOpportunity,
        refactor_func: Callable[[RefactorOpportunity], RefactorResult],
        trigger: str = "batch",
    ) -> Optional[RefactorResult]:
        """
        Execute a single refactor

        Args:
            opportunity: refactor opportunity
            refactor_func: refactor function
            trigger: trigger method

        Returns:
            Optional[RefactorResult]: refactor result, or None on failure
        """
        skill_name = opportunity.source_skill

        try:
            # Start the skill transaction
            self.adapter.begin_skill_refactor(skill_name)

            # Run the refactor
            result = refactor_func(opportunity)

            # Commit or rollback
            if result.success:
                self.adapter.commit_skill_refactor(
                    skill_name,
                    result,
                    task=self.task,
                    trigger=trigger,
                )
            else:
                # Record the failed attempt to history BEFORE rolling back
                # so R_cons in learning_dynamics_recorder properly counts
                # validation rejections / pre-commit failures.
                # (rollback_skill_refactor only calls record_rollback when
                # state.result is set, which never happens on the failure
                # path — so without this explicit record() call the failed
                # refactor is silently dropped from RefactorHistory and
                # `total_refactors` undercounts the actual attempt rate.
                # Net effect: R_cons appeared artificially high in any
                # run with rejected refactor proposals.)
                if self.adapter.history_tracker:
                    session_id = (self.adapter._session.session_id
                                  if self.adapter._session else None)
                    self.adapter.history_tracker.record(
                        result=result,
                        task=self.task,
                        trigger=trigger,
                        session_id=session_id,
                    )
                self.adapter.rollback_skill_refactor(
                    skill_name,
                    reason=f"refactor_failed: {result.error_message}"
                )

            self._results.append(result)
            return result

        except Exception as e:
            self._errors.append((skill_name, e))

            if self.on_error == "rollback_skill":
                self.adapter.rollback_skill_refactor(skill_name, reason=str(e))
            elif self.on_error == "rollback_all":
                raise

            return None
