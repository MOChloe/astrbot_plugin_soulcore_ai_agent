"""Bounded export and new-run restore adapters for Main Core work state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .work_checkpoint import (
    ControlledWorkResult,
    MainCoreWorkCheckpoint,
    WorkCallbackEnvelope,
    WorkCallbackLease,
    WorkCallbackOutcome,
    WorkCallbackRejection,
    WorkCheckpointError,
    WorkCheckpointStatus,
    WorkRecoveryAction,
    WorkRecoveryBaseline,
    WorkRecoveryRuntime,
    WorkReevaluationFlag,
    WorkScope,
    decide_callback,
)
from .work_continuity import (
    MainCoreWorkSession,
    MainCoreWorkSnapshot,
    WorkCompletionCondition,
    WorkStep,
)


class WorkSessionRestoreRejection(StrEnum):
    ENVELOPE_NOT_READY = "ENVELOPE_NOT_READY"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    WORK_REF_MISMATCH = "WORK_REF_MISMATCH"
    RUN_GENERATION_MISMATCH = "RUN_GENERATION_MISMATCH"
    CHECKPOINT_VERSION_MISMATCH = "CHECKPOINT_VERSION_MISMATCH"
    RESULT_REF_NO_LONGER_CONTROLLED = "RESULT_REF_NO_LONGER_CONTROLLED"
    ACTION_NO_LONGER_ALLOWED = "ACTION_NO_LONGER_ALLOWED"
    REEVALUATION_REQUIRED = "REEVALUATION_REQUIRED"
    ENVELOPE_CANNOT_AUTHORIZE = "ENVELOPE_CANNOT_AUTHORIZE"


@dataclass(frozen=True, slots=True)
class MainCoreWorkRecoveryEnvelope:
    """Internal-only input for a newly created Main Core run.

    The envelope is work memory, not an authorization object.  It intentionally
    has no reply, memo, expression, persistence projection, command execution, or
    platform dispatch method.
    """

    scope: WorkScope
    snapshot: MainCoreWorkSnapshot
    checkpoint_version: int
    source_run_generation: int
    resume_run_generation: int
    callback_sequence: int
    callback_outcome: WorkCallbackOutcome
    callback_result_summary: str
    controlled_results: tuple[ControlledWorkResult, ...]
    triggering_results: tuple[ControlledWorkResult, ...]
    remaining_steps: tuple[WorkStep, ...]
    remaining_completion_conditions: tuple[WorkCompletionCondition, ...]
    allowed_actions: tuple[WorkRecoveryAction, ...]
    reevaluation_flags: tuple[WorkReevaluationFlag, ...]
    must_reevaluate: bool = True
    authorizes_actions: bool = False
    recovery_status: WorkCheckpointStatus = WorkCheckpointStatus.RECOVERY_READY

    @property
    def work_ref(self) -> str:
        return self.snapshot.work_ref

    def __post_init__(self) -> None:
        if self.recovery_status != WorkCheckpointStatus.RECOVERY_READY:
            raise WorkCheckpointError("recovery envelope must be recovery-ready")
        if self.source_run_generation <= 0:
            raise WorkCheckpointError("source run generation must be positive")
        if self.resume_run_generation != self.source_run_generation + 1:
            raise WorkCheckpointError("resume run generation must advance exactly once")
        if self.callback_sequence <= 0:
            raise WorkCheckpointError("recovery callback sequence must be positive")
        if not self.must_reevaluate:
            raise WorkCheckpointError("a recovered Main Core run must reevaluate")
        if self.authorizes_actions:
            raise WorkCheckpointError("a recovery envelope cannot grant authority")
        if WorkRecoveryAction.REASSESS_PLAN not in self.allowed_actions:
            raise WorkCheckpointError("recovery must allow deterministic plan reassessment")
        if WorkReevaluationFlag.CALLBACK_RESULT not in self.reevaluation_flags:
            raise WorkCheckpointError("recovery must identify the callback result trigger")
        controlled = {(item.slot_id, item.resource_ref) for item in self.controlled_results}
        triggering = {(item.slot_id, item.resource_ref) for item in self.triggering_results}
        if not triggering.issubset(controlled):
            raise WorkCheckpointError("triggering results must be controlled results")


@dataclass(frozen=True, slots=True)
class WorkRecoveryDecision:
    accepted: bool
    checkpoint: MainCoreWorkCheckpoint
    envelope: MainCoreWorkRecoveryEnvelope | None = None
    rejection: WorkCallbackRejection | None = None


@dataclass(frozen=True, slots=True)
class WorkSessionRestoreDecision:
    accepted: bool
    session: MainCoreWorkSession | None = None
    rejection: WorkSessionRestoreRejection | None = None


def export_work_checkpoint(
    snapshot: MainCoreWorkSnapshot,
    *,
    scope: WorkScope,
    checkpoint_version: int,
    run_generation: int,
    callback_sequence: int,
    baseline: WorkRecoveryBaseline,
    allowed_actions: tuple[WorkRecoveryAction, ...],
    lease: WorkCallbackLease,
    created_at: datetime,
    expires_at: datetime,
    controlled_resource_refs: frozenset[str],
) -> MainCoreWorkCheckpoint:
    """Freeze one bounded TASK-0008 in-run snapshot as a waiting checkpoint."""

    if snapshot.status != "ACTIVE":
        raise WorkCheckpointError("only active Main Core work may be exported")
    controlled_results = tuple(
        ControlledWorkResult(
            slot_id=slot.slot_id,
            resource_ref=resource_ref,
            result_kind="BOUND_RESULT",
            source_callback_sequence=callback_sequence,
        )
        for slot in snapshot.result_slots
        for resource_ref in slot.resource_refs
    )
    unknown = [
        item.resource_ref
        for item in controlled_results
        if item.resource_ref not in controlled_resource_refs
    ]
    if unknown:
        raise WorkCheckpointError("checkpoint may only export currently controlled result refs")
    return MainCoreWorkCheckpoint(
        scope=scope,
        snapshot=snapshot,
        checkpoint_version=checkpoint_version,
        run_generation=run_generation,
        callback_sequence=callback_sequence,
        baseline=baseline,
        allowed_actions=allowed_actions,
        controlled_results=controlled_results,
        lease=lease,
        created_at=created_at,
        expires_at=expires_at,
    )


def decide_work_recovery(
    checkpoint: MainCoreWorkCheckpoint,
    callback: WorkCallbackEnvelope,
    runtime: WorkRecoveryRuntime,
) -> WorkRecoveryDecision:
    """Validate a callback and, only if legal, build an internal recovery input."""

    callback_decision = decide_callback(checkpoint, callback, runtime)
    if not callback_decision.accepted:
        return WorkRecoveryDecision(
            accepted=False,
            checkpoint=callback_decision.checkpoint,
            rejection=callback_decision.rejection,
        )
    accepted = callback_decision.checkpoint
    envelope = MainCoreWorkRecoveryEnvelope(
        scope=accepted.scope,
        snapshot=accepted.snapshot,
        checkpoint_version=accepted.checkpoint_version,
        source_run_generation=accepted.run_generation - 1,
        resume_run_generation=accepted.run_generation,
        callback_sequence=accepted.callback_sequence,
        callback_outcome=callback.outcome,
        callback_result_summary=callback.result_summary,
        controlled_results=accepted.controlled_results,
        triggering_results=callback_decision.triggering_results,
        remaining_steps=tuple(
            item for item in accepted.snapshot.steps if item.status not in {"COMPLETED", "SKIPPED"}
        ),
        remaining_completion_conditions=_remaining_conditions(
            accepted.snapshot,
            accepted.controlled_results,
        ),
        allowed_actions=accepted.allowed_actions,
        reevaluation_flags=callback_decision.reevaluation_flags,
    )
    return WorkRecoveryDecision(
        accepted=True,
        checkpoint=accepted,
        envelope=envelope,
    )


def restore_work_session_for_new_run(
    envelope: MainCoreWorkRecoveryEnvelope,
    *,
    expected_scope: WorkScope,
    expected_work_ref: str,
    expected_checkpoint_version: int,
    expected_run_generation: int,
    current_controlled_resource_refs: frozenset[str],
    currently_allowed_actions: frozenset[WorkRecoveryAction],
) -> WorkSessionRestoreDecision:
    """Recheck current fences and recover state for Main Core reevaluation only."""

    rejection = _restore_rejection(
        envelope,
        expected_scope=expected_scope,
        expected_work_ref=expected_work_ref,
        expected_checkpoint_version=expected_checkpoint_version,
        expected_run_generation=expected_run_generation,
        current_controlled_resource_refs=current_controlled_resource_refs,
        currently_allowed_actions=currently_allowed_actions,
    )
    if rejection is not None:
        return WorkSessionRestoreDecision(accepted=False, rejection=rejection)
    return WorkSessionRestoreDecision(
        accepted=True,
        session=MainCoreWorkSession(snapshot=envelope.snapshot),
    )


def _remaining_conditions(
    snapshot: MainCoreWorkSnapshot,
    results: tuple[ControlledWorkResult, ...],
) -> tuple[WorkCompletionCondition, ...]:
    bound_slots = {item.slot_id for item in results}
    return tuple(
        condition
        for condition in snapshot.completion_conditions
        if condition.requires_visible_output
        or any(slot_id not in bound_slots for slot_id in condition.required_slot_ids)
    )


def _restore_rejection(
    envelope: MainCoreWorkRecoveryEnvelope,
    *,
    expected_scope: WorkScope,
    expected_work_ref: str,
    expected_checkpoint_version: int,
    expected_run_generation: int,
    current_controlled_resource_refs: frozenset[str],
    currently_allowed_actions: frozenset[WorkRecoveryAction],
) -> WorkSessionRestoreRejection | None:
    if envelope.recovery_status != WorkCheckpointStatus.RECOVERY_READY:
        return WorkSessionRestoreRejection.ENVELOPE_NOT_READY
    if envelope.scope != expected_scope:
        return WorkSessionRestoreRejection.SCOPE_MISMATCH
    if envelope.work_ref != expected_work_ref:
        return WorkSessionRestoreRejection.WORK_REF_MISMATCH
    if envelope.checkpoint_version != int(expected_checkpoint_version):
        return WorkSessionRestoreRejection.CHECKPOINT_VERSION_MISMATCH
    if envelope.resume_run_generation != int(expected_run_generation):
        return WorkSessionRestoreRejection.RUN_GENERATION_MISMATCH
    if envelope.authorizes_actions:
        return WorkSessionRestoreRejection.ENVELOPE_CANNOT_AUTHORIZE
    if not envelope.must_reevaluate:
        return WorkSessionRestoreRejection.REEVALUATION_REQUIRED
    if any(
        item.resource_ref not in current_controlled_resource_refs
        for item in envelope.controlled_results
    ):
        return WorkSessionRestoreRejection.RESULT_REF_NO_LONGER_CONTROLLED
    if any(action not in currently_allowed_actions for action in envelope.allowed_actions):
        return WorkSessionRestoreRejection.ACTION_NO_LONGER_ALLOWED
    return None


__all__ = [
    "MainCoreWorkRecoveryEnvelope",
    "WorkRecoveryDecision",
    "WorkSessionRestoreDecision",
    "WorkSessionRestoreRejection",
    "decide_work_recovery",
    "export_work_checkpoint",
    "restore_work_session_for_new_run",
]
