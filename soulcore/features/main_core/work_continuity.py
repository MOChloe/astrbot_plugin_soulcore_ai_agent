"""Bounded, in-memory work continuity for one Main Core command loop.

The state in this module is deliberately not durable.  It records only the
minimum structured facts needed to continue an automatic background task inside the
current run; it never stores free-form reasoning or authorizes an operation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ...contracts.models import CoreRunResult, CoreWakeRequest, RunStatus, WakeSource
from ...shared.time import utcnow
from .command_context import DecisionCollector


@dataclass(frozen=True, slots=True)
class LockedMainCoreRunStart:
    run_id: int
    expected_state_epoch: int
    expected_activity_epoch: int
    recovery: Any | None = None
    rejection: CoreRunResult | None = None


class WorkRecoveryExecutionMixin:
    async def _start_locked_main_core_run(
        self: Any, request: CoreWakeRequest, state: Any
    ) -> LockedMainCoreRunStart:
        marker = _work_recovery_marker(request)
        run_request = {
            "route_umo": request.route_umo,
            "user_message": request.user_message,
            "requested_at": request.requested_at.isoformat(),
            "metadata": request.metadata,
        }
        if marker is None:
            expected_state = request.expected_state_epoch
            expected_activity = request.expected_activity_epoch
            expected_state = state.state_epoch if expected_state is None else expected_state
            expected_activity = (
                state.activity_epoch if expected_activity is None else expected_activity
            )
            run_id = await self._create_run(
                request,
                request.source,
                reason=request.reason,
                request=run_request,
                expected_state_epoch=expected_state,
            )
            return LockedMainCoreRunStart(run_id, expected_state, expected_activity)
        work_ref, work_version = marker
        recovery = await self.core_results.start_work_recovery_run(
            profile_id=request.profile_id,
            instance_id=str(request.instance_id or ""),
            work_ref=work_ref,
            checkpoint_version=work_version,
            wakeup_id=int(request.wakeup_id or 0),
            reason=request.reason,
            request=run_request,
            now=utcnow(),
        )
        if recovery is None:
            return _rejected_start("work_recovery_claim_rejected")
        return LockedMainCoreRunStart(
            recovery.run_id,
            recovery.expected_state_epoch,
            recovery.expected_activity_epoch,
            recovery=recovery,
        )


def work_recovery_collector_fields(recovery: Any | None) -> dict[str, Any]:
    if recovery is None:
        return {
            "work_session": MainCoreWorkSession(),
            "work_recovery_envelope": None,
            "recovered_work_resource_refs": set(),
        }
    session = _settle_automatic_recovery(recovery)
    return {
        "work_session": session,
        "work_recovery_envelope": recovery.envelope,
        "recovered_work_resource_refs": set(recovery.controlled_resource_refs),
    }


def _settle_automatic_recovery(recovery: Any) -> MainCoreWorkSession:
    """Advance file-callback work internally; the model never maintains checkpoints."""

    session = recovery.session
    snapshot = session.snapshot
    if snapshot is None or snapshot.status != "ACTIVE":
        return session
    refs = set(recovery.controlled_resource_refs)
    envelope = recovery.envelope
    outcome = envelope.callback_outcome.value
    if outcome == "SUCCEEDED":
        bound_slots = {item.slot_id for item in envelope.controlled_results}
        for step in tuple(snapshot.steps):
            if step.status not in {"PENDING", "IN_PROGRESS"} or not set(
                step.required_slot_ids
            ).issubset(bound_slots):
                continue
            if step.status == "PENDING":
                snapshot, _ = session.mutate(
                    operation="UPDATE",
                    expected_version=snapshot.version,
                    core_run_id=int(recovery.run_id),
                    known_resource_refs=refs,
                    payload={"step_updates": [{"step_id": step.step_id, "status": "IN_PROGRESS"}]},
                )
            snapshot, _ = session.mutate(
                operation="UPDATE",
                expected_version=snapshot.version,
                core_run_id=int(recovery.run_id),
                known_resource_refs=refs,
                payload={
                    "step_updates": [{"step_id": step.step_id, "status": "COMPLETED"}],
                    "next_action": "",
                },
            )
        if all(step.status in {"COMPLETED", "SKIPPED"} for step in snapshot.steps):
            session.mutate(
                operation="COMPLETE",
                expected_version=snapshot.version,
                core_run_id=int(recovery.run_id),
                known_resource_refs=refs,
                payload={},
            )
        return session
    session.mutate(
        operation="ABANDON",
        expected_version=snapshot.version,
        core_run_id=int(recovery.run_id),
        known_resource_refs=refs,
        payload={"terminal_reason": str(envelope.callback_result_summary or "后台任务失败")},
    )
    return session


def _work_recovery_marker(request: CoreWakeRequest) -> tuple[str, int] | None:
    has_ref = "work_ref" in request.metadata
    has_version = "work_version" in request.metadata
    if not has_ref and not has_version:
        return None
    allowed_metadata = {"work_ref", "work_version", "ai_task_managed", "ai_task_id"}
    if (
        request.source is not WakeSource.PLUGIN_WAKE
        or not request.instance_id
        or not request.wakeup_id
        or not has_ref
        or not has_version
        or set(request.metadata) - allowed_metadata
    ):
        return ("invalid", -1)
    work_ref = str(request.metadata.get("work_ref") or "")
    try:
        version = int(str(request.metadata.get("work_version")))
    except (TypeError, ValueError):
        return ("invalid", -1)
    if not work_ref or len(work_ref) > 160 or version <= 0:
        return ("invalid", -1)
    return work_ref, version


def _rejected_start(error: str) -> LockedMainCoreRunStart:
    return LockedMainCoreRunStart(
        0,
        0,
        0,
        rejection=CoreRunResult(
            0,
            RunStatus.SUPERSEDED,
            superseded=True,
            error=error,
        ),
    )


def validate_terminal_work(
    collector: DecisionCollector,
    *,
    has_visible_output: bool,
    visible_text: str = "",
) -> str | None:
    session = collector.work_session
    snapshot = session.snapshot if session is not None else None
    if snapshot is None:
        return None
    del visible_text
    file_requests = list(collector.file_generation_requests)
    if file_requests:
        deferred_error = _deferred_file_work_error(snapshot, file_requests)
        if deferred_error:
            collector.work_internal_errors.append(deferred_error)
            return "error: 文件请求没有正确保存；暂时不要结束这次交流"
        return None
    if snapshot.status not in TERMINAL_WORK_STATUSES:
        return "error: 还有先前开始的事项没有处理完；请先完成，或明确取消并说明原因"
    if snapshot.status == "COMPLETED":
        error = work_completion_error(
            snapshot,
            known_work_resource_refs(collector),
            has_visible_output=has_visible_output,
        )
        if error:
            collector.work_internal_errors.append(error)
            return "error: 当前事项还缺少必要结果；请补齐后再结束"
    elif not snapshot.terminal_reason:
        return "error: 取消先前事项时必须说明原因"
    return None


def work_completion_error(
    snapshot: MainCoreWorkSnapshot,
    known_resource_refs: set[str],
    *,
    has_visible_output: bool | None,
) -> str:
    checks = (
        _unfinished_step_error(snapshot),
        _slot_error(snapshot, known_resource_refs),
        _deliverable_error(snapshot),
        _condition_error(snapshot, has_visible_output),
    )
    return next((item for item in checks if item), "")


def _unfinished_step_error(snapshot: MainCoreWorkSnapshot) -> str:
    incomplete = [
        item.step_id for item in snapshot.steps if item.status not in {"COMPLETED", "SKIPPED"}
    ]
    return "仍有未完成步骤：" + "、".join(incomplete) if incomplete else ""


def _slot_error(snapshot: MainCoreWorkSnapshot, known_resource_refs: set[str]) -> str:
    missing_required = [
        item.slot_id for item in snapshot.result_slots if item.required and not item.resource_refs
    ]
    if missing_required:
        return "必需的结果位置为空：" + "、".join(missing_required)
    stale_refs = [
        ref
        for item in snapshot.result_slots
        for ref in item.resource_refs
        if ref not in known_resource_refs
    ]
    if stale_refs:
        return "这些结果短引用在本轮已失效：" + "、".join(stale_refs)
    return ""


def _deliverable_error(snapshot: MainCoreWorkSnapshot) -> str:
    slots = {item.slot_id: item for item in snapshot.result_slots}
    for deliverable in snapshot.deliverables:
        missing = [slot for slot in deliverable.required_slot_ids if not slots[slot].resource_refs]
        if missing:
            return f"交付项 {deliverable.deliverable_id} 缺少结果：{'、'.join(missing)}"
    return ""


def _condition_error(snapshot: MainCoreWorkSnapshot, has_visible_output: bool | None) -> str:
    slots = {item.slot_id: item for item in snapshot.result_slots}
    for condition in snapshot.completion_conditions:
        missing = [slot for slot in condition.required_slot_ids if not slots[slot].resource_refs]
        if missing:
            return f"完成条件 {condition.condition_id} 缺少结果：{'、'.join(missing)}"
        if (
            condition.requires_visible_output
            and has_visible_output is False
            and not _file_artifact_condition(condition)
        ):
            return f"完成条件 {condition.condition_id} 要求最终发送可见内容"
    return ""


def _file_artifact_condition(condition: Any) -> bool:
    """Older recovered file work must not turn generation into forced delivery."""

    return (
        str(condition.condition_id) == "file-ready"
        and bool(condition.required_slot_ids)
        and all(
            "file" in str(slot_id).casefold() or "artifact" in str(slot_id).casefold()
            for slot_id in condition.required_slot_ids
        )
    )


def known_work_resource_refs(collector: DecisionCollector) -> set[str]:
    refs = _media_resource_refs(collector)
    refs.update(_context_resource_refs(collector))
    refs.update(_file_resource_refs(collector))
    refs.update(_recovered_resource_refs(collector))
    return refs


def _media_resource_refs(collector: DecisionCollector) -> set[str]:
    return {
        str(item)
        for item in (
            *collector.generated_media_asset_ids,
            *collector.inspected_search_media_asset_ids,
        )
        if valid_work_resource_ref(item)
    }


def _context_resource_refs(collector: DecisionCollector) -> set[str]:
    web_context = collector.web_command_context
    refs = {
        str(item)
        for item in (web_context.resource_ids if web_context is not None else ())
        if valid_work_resource_ref(item)
    }
    refs.update(
        str(item) for item in collector.important_todo_refs if valid_work_resource_ref(item)
    )
    return refs


def _file_resource_refs(collector: DecisionCollector) -> set[str]:
    refs = {
        str(item.get("request_ref") or "")
        for item in collector.file_generation_requests
        if valid_work_resource_ref(item.get("request_ref"))
    }
    for item in collector.important_todo_refs.values():
        for key in ("todo_id", "file_asset_id"):
            value = item.get(key) if isinstance(item, dict) else None
            if valid_work_resource_ref(value):
                refs.add(str(value))
    return refs


def _recovered_resource_refs(collector: DecisionCollector) -> set[str]:
    return {
        str(item)
        for item in collector.recovered_work_resource_refs
        if valid_work_resource_ref(item)
    }


def _deferred_file_work_error(
    snapshot: MainCoreWorkSnapshot, file_requests: list[dict[str, Any]]
) -> str:
    if snapshot.status != "ACTIVE":
        return "文件生成完成回调前，持续工作必须保持进行中"
    expected = [str(item.get("request_ref") or "") for item in file_requests]
    if not expected or any(not valid_work_resource_ref(item) for item in expected):
        return "文件请求短引用无效"
    occurrences = {
        ref: sum(ref in slot.resource_refs for slot in snapshot.result_slots) for ref in expected
    }
    unbound = [ref for ref, count in occurrences.items() if count != 1]
    if unbound:
        return "每个文件请求必须且只能绑定一个工作结果位置"
    return ""


MAX_DELIVERABLES = 6
MAX_COMPLETION_CONDITIONS = 8
MAX_CONSTRAINTS = 8
MAX_STEPS = 12
MAX_RESULT_SLOTS = 12
MAX_REFS_PER_SLOT = 5
MAX_GOAL_CHARS = 800
MAX_TEXT_CHARS = 320
MAX_PURPOSE_CHARS = 240
MAX_NEXT_ACTION_CHARS = 400
MAX_TERMINAL_REASON_CHARS = 400
MAX_RESOURCE_REF_CHARS = 240

WORK_STATUSES = frozenset({"ACTIVE", "COMPLETED", "CANCELLED", "ABANDONED"})
TERMINAL_WORK_STATUSES = frozenset({"COMPLETED", "CANCELLED", "ABANDONED"})
STEP_STATUSES = frozenset({"PENDING", "IN_PROGRESS", "COMPLETED", "SKIPPED"})
_STEP_TRANSITIONS = {
    "PENDING": {"IN_PROGRESS", "SKIPPED"},
    "IN_PROGRESS": {"COMPLETED", "SKIPPED"},
    "COMPLETED": set(),
    "SKIPPED": set(),
}
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


class WorkContinuityError(ValueError):
    """An attempted work-state mutation violated a deterministic contract."""


@dataclass(frozen=True, slots=True)
class WorkDeliverable:
    deliverable_id: str
    description: str
    required_slot_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkCompletionCondition:
    condition_id: str
    description: str
    required_slot_ids: tuple[str, ...] = ()
    requires_visible_output: bool = False


@dataclass(frozen=True, slots=True)
class WorkStep:
    step_id: str
    title: str
    purpose: str
    status: str
    required_slot_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkResultSlot:
    slot_id: str
    description: str
    required: bool
    resource_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MainCoreWorkSnapshot:
    work_ref: str
    goal: str
    deliverables: tuple[WorkDeliverable, ...]
    completion_conditions: tuple[WorkCompletionCondition, ...]
    constraints: tuple[str, ...]
    steps: tuple[WorkStep, ...]
    result_slots: tuple[WorkResultSlot, ...]
    next_action: str
    status: str
    version: int
    terminal_reason: str = ""


@dataclass(slots=True)
class MainCoreWorkSession:
    snapshot: MainCoreWorkSnapshot | None = None
    last_base_version: int | None = None
    last_operation_fingerprint: str = ""

    def mutate(
        self,
        *,
        operation: str,
        expected_version: int,
        core_run_id: int,
        known_resource_refs: set[str],
        payload: dict[str, Any],
    ) -> tuple[MainCoreWorkSnapshot, bool]:
        normalized_operation = str(operation or "").strip().upper()
        if normalized_operation not in {"CREATE", "UPDATE", "COMPLETE", "CANCEL", "ABANDON"}:
            raise WorkContinuityError(
                "operation must be CREATE, UPDATE, COMPLETE, CANCEL or ABANDON"
            )
        base_version = int(expected_version)
        fingerprint = _fingerprint(normalized_operation, base_version, payload)
        current_version = self.snapshot.version if self.snapshot is not None else 0
        if base_version != current_version:
            if (
                self.snapshot is not None
                and self.last_base_version == base_version
                and self.last_operation_fingerprint == fingerprint
            ):
                return self.snapshot, True
            raise WorkContinuityError(
                f"stale work version: expected {current_version}, received {base_version}"
            )
        if normalized_operation == "CREATE":
            if self.snapshot is not None:
                raise WorkContinuityError("only one Main Core work may exist in a run")
            updated = _create_snapshot(core_run_id=core_run_id, payload=payload)
        else:
            if self.snapshot is None:
                raise WorkContinuityError("no Main Core work exists in this run")
            if self.snapshot.status in TERMINAL_WORK_STATUSES:
                raise WorkContinuityError("a terminal Main Core work cannot be changed")
            if normalized_operation == "UPDATE":
                updated = _update_snapshot(
                    self.snapshot,
                    payload=payload,
                    known_resource_refs=known_resource_refs,
                )
            elif normalized_operation == "COMPLETE":
                updated = _complete_snapshot(
                    self.snapshot,
                    payload=payload,
                    known_resource_refs=known_resource_refs,
                )
            else:
                updated = _stop_snapshot(
                    self.snapshot,
                    status="CANCELLED" if normalized_operation == "CANCEL" else "ABANDONED",
                    payload=payload,
                )
        self.snapshot = updated
        self.last_base_version = base_version
        self.last_operation_fingerprint = fingerprint
        return updated, False


def _create_snapshot(*, core_run_id: int, payload: dict[str, Any]) -> MainCoreWorkSnapshot:
    goal = _bounded_text(payload.get("goal"), "goal", MAX_GOAL_CHARS, required=True)
    deliverables = _deliverables(payload.get("deliverables"))
    conditions = _conditions(payload.get("completion_conditions"))
    constraints = _string_list(
        payload.get("constraints"), "constraints", MAX_CONSTRAINTS, MAX_TEXT_CHARS
    )
    steps = _steps(payload.get("steps"))
    slots = _slots(payload.get("result_slots"))
    slot_ids = {item.slot_id for item in slots}
    _validate_slot_links(deliverables, conditions, steps, slot_ids)
    if sum(item.status == "IN_PROGRESS" for item in steps) > 1:
        raise WorkContinuityError("at most one step may be IN_PROGRESS")
    if any(item.status in {"COMPLETED", "SKIPPED"} for item in steps):
        raise WorkContinuityError("new work steps cannot start in a terminal step status")
    next_action = _bounded_text(
        payload.get("next_action"), "next_action", MAX_NEXT_ACTION_CHARS, required=True
    )
    return MainCoreWorkSnapshot(
        work_ref=f"work:{int(core_run_id)}:1",
        goal=goal,
        deliverables=deliverables,
        completion_conditions=conditions,
        constraints=constraints,
        steps=steps,
        result_slots=slots,
        next_action=next_action,
        status="ACTIVE",
        version=1,
    )


def _update_snapshot(
    snapshot: MainCoreWorkSnapshot,
    *,
    payload: dict[str, Any],
    known_resource_refs: set[str],
) -> MainCoreWorkSnapshot:
    slot_updates = list(payload.get("slot_bindings") or [])
    step_updates = list(payload.get("step_updates") or [])
    next_action_supplied = "next_action" in payload and payload.get("next_action") is not None
    if not slot_updates and not step_updates and not next_action_supplied:
        raise WorkContinuityError("UPDATE requires a slot, step or next_action change")
    slots = _apply_slot_bindings(snapshot.result_slots, slot_updates, known_resource_refs)
    steps = _apply_step_updates(snapshot.steps, step_updates, slots)
    next_action = (
        _bounded_text(
            payload.get("next_action"),
            "next_action",
            MAX_NEXT_ACTION_CHARS,
            required=True,
        )
        if next_action_supplied
        else snapshot.next_action
    )
    if (
        slots == snapshot.result_slots
        and steps == snapshot.steps
        and next_action == snapshot.next_action
    ):
        raise WorkContinuityError("UPDATE must change the current work snapshot")
    return MainCoreWorkSnapshot(
        work_ref=snapshot.work_ref,
        goal=snapshot.goal,
        deliverables=snapshot.deliverables,
        completion_conditions=snapshot.completion_conditions,
        constraints=snapshot.constraints,
        steps=steps,
        result_slots=slots,
        next_action=next_action,
        status=snapshot.status,
        version=snapshot.version + 1,
    )


def _complete_snapshot(
    snapshot: MainCoreWorkSnapshot,
    *,
    payload: dict[str, Any],
    known_resource_refs: set[str],
) -> MainCoreWorkSnapshot:
    reason = _bounded_text(
        payload.get("terminal_reason"),
        "terminal_reason",
        MAX_TERMINAL_REASON_CHARS,
        required=False,
    )
    error = work_completion_error(snapshot, known_resource_refs, has_visible_output=None)
    if error:
        raise WorkContinuityError(error)
    return MainCoreWorkSnapshot(
        work_ref=snapshot.work_ref,
        goal=snapshot.goal,
        deliverables=snapshot.deliverables,
        completion_conditions=snapshot.completion_conditions,
        constraints=snapshot.constraints,
        steps=snapshot.steps,
        result_slots=snapshot.result_slots,
        next_action="",
        status="COMPLETED",
        version=snapshot.version + 1,
        terminal_reason=reason,
    )


def _stop_snapshot(
    snapshot: MainCoreWorkSnapshot, *, status: str, payload: dict[str, Any]
) -> MainCoreWorkSnapshot:
    reason = _bounded_text(
        payload.get("terminal_reason"),
        "terminal_reason",
        MAX_TERMINAL_REASON_CHARS,
        required=True,
    )
    return MainCoreWorkSnapshot(
        work_ref=snapshot.work_ref,
        goal=snapshot.goal,
        deliverables=snapshot.deliverables,
        completion_conditions=snapshot.completion_conditions,
        constraints=snapshot.constraints,
        steps=snapshot.steps,
        result_slots=snapshot.result_slots,
        next_action="",
        status=status,
        version=snapshot.version + 1,
        terminal_reason=reason,
    )


def _apply_slot_bindings(
    current: tuple[WorkResultSlot, ...],
    values: list[Any],
    known_resource_refs: set[str],
) -> tuple[WorkResultSlot, ...]:
    if len(values) > MAX_RESULT_SLOTS:
        raise WorkContinuityError("too many slot bindings")
    by_id = {item.slot_id: item for item in current}
    for raw in values:
        if not isinstance(raw, dict):
            raise WorkContinuityError("each slot binding must be an object")
        slot_id = _identifier(raw.get("slot_id"), "slot_id")
        if slot_id not in by_id:
            raise WorkContinuityError(f"unknown result slot: {slot_id}")
        refs = _resource_refs(raw.get("resource_refs"))
        unknown = [item for item in refs if item not in known_resource_refs]
        if unknown:
            raise WorkContinuityError(
                "slot bindings may only use current-run controlled command results: "
                + ", ".join(unknown)
            )
        previous = by_id[slot_id]
        combined = tuple(dict.fromkeys((*previous.resource_refs, *refs)))
        if len(combined) > MAX_REFS_PER_SLOT:
            raise WorkContinuityError("a result slot contains too many resource refs")
        by_id[slot_id] = WorkResultSlot(
            slot_id=previous.slot_id,
            description=previous.description,
            required=previous.required,
            resource_refs=combined,
        )
    return tuple(by_id[item.slot_id] for item in current)


def _apply_step_updates(
    current: tuple[WorkStep, ...],
    values: list[Any],
    slots: tuple[WorkResultSlot, ...],
) -> tuple[WorkStep, ...]:
    if len(values) > MAX_STEPS:
        raise WorkContinuityError("too many step updates")
    by_id = {item.step_id: item for item in current}
    bound_slots = {item.slot_id for item in slots if item.resource_refs}
    for raw in values:
        _apply_one_step_update(by_id, raw, bound_slots)
    updated = tuple(by_id[item.step_id] for item in current)
    if sum(item.status == "IN_PROGRESS" for item in updated) > 1:
        raise WorkContinuityError("at most one step may be IN_PROGRESS")
    return updated


def _apply_one_step_update(by_id: dict[str, WorkStep], raw: Any, bound_slots: set[str]) -> None:
    if not isinstance(raw, dict):
        raise WorkContinuityError("each step update must be an object")
    step_id = _identifier(raw.get("step_id"), "step_id")
    if step_id not in by_id:
        raise WorkContinuityError(f"unknown work step: {step_id}")
    status = str(raw.get("status") or "").strip().upper()
    previous = by_id[step_id]
    if status not in STEP_STATUSES:
        raise WorkContinuityError(f"invalid step status: {status or '<empty>'}")
    if status == previous.status:
        return
    if status not in _STEP_TRANSITIONS[previous.status]:
        raise WorkContinuityError(f"invalid step transition: {previous.status} -> {status}")
    missing = [item for item in previous.required_slot_ids if item not in bound_slots]
    if status == "COMPLETED" and missing:
        raise WorkContinuityError(
            "a step cannot complete before its real result slots are bound: " + ", ".join(missing)
        )
    by_id[step_id] = WorkStep(
        step_id=previous.step_id,
        title=previous.title,
        purpose=previous.purpose,
        status=status,
        required_slot_ids=previous.required_slot_ids,
    )


def _deliverables(value: Any) -> tuple[WorkDeliverable, ...]:
    values = _object_list(value, "deliverables", MAX_DELIVERABLES, required=True)
    result = tuple(
        WorkDeliverable(
            deliverable_id=_identifier(item.get("deliverable_id"), "deliverable_id"),
            description=_bounded_text(
                item.get("description"), "deliverable description", MAX_TEXT_CHARS, required=True
            ),
            required_slot_ids=_identifier_list(item.get("required_slot_ids")),
        )
        for item in values
    )
    _unique_ids([item.deliverable_id for item in result], "deliverable")
    return result


def _conditions(value: Any) -> tuple[WorkCompletionCondition, ...]:
    values = _object_list(value, "completion_conditions", MAX_COMPLETION_CONDITIONS, required=True)
    result = tuple(
        WorkCompletionCondition(
            condition_id=_identifier(item.get("condition_id"), "condition_id"),
            description=_bounded_text(
                item.get("description"), "condition description", MAX_TEXT_CHARS, required=True
            ),
            required_slot_ids=_identifier_list(item.get("required_slot_ids")),
            requires_visible_output=bool(item.get("requires_visible_output", False)),
        )
        for item in values
    )
    _unique_ids([item.condition_id for item in result], "completion condition")
    if any(not item.required_slot_ids and not item.requires_visible_output for item in result):
        raise WorkContinuityError(
            "each completion condition must require a result slot or visible output"
        )
    return result


def _steps(value: Any) -> tuple[WorkStep, ...]:
    values = _object_list(value, "steps", MAX_STEPS, required=True)
    result = tuple(
        WorkStep(
            step_id=_identifier(item.get("step_id"), "step_id"),
            title=_bounded_text(item.get("title"), "step title", MAX_TEXT_CHARS, required=True),
            purpose=_bounded_text(
                item.get("purpose"), "step purpose", MAX_PURPOSE_CHARS, required=True
            ),
            status=str(item.get("status") or "PENDING").strip().upper(),
            required_slot_ids=_identifier_list(item.get("required_slot_ids")),
        )
        for item in values
    )
    _unique_ids([item.step_id for item in result], "step")
    invalid = [item.status for item in result if item.status not in STEP_STATUSES]
    if invalid:
        raise WorkContinuityError(f"invalid initial step status: {invalid[0]}")
    return result


def _slots(value: Any) -> tuple[WorkResultSlot, ...]:
    values = _object_list(value, "result_slots", MAX_RESULT_SLOTS, required=True)
    result = tuple(
        WorkResultSlot(
            slot_id=_identifier(item.get("slot_id"), "slot_id"),
            description=_bounded_text(
                item.get("description"), "slot description", MAX_TEXT_CHARS, required=True
            ),
            required=bool(item.get("required", False)),
        )
        for item in values
    )
    _unique_ids([item.slot_id for item in result], "result slot")
    return result


def _validate_slot_links(
    deliverables: tuple[WorkDeliverable, ...],
    conditions: tuple[WorkCompletionCondition, ...],
    steps: tuple[WorkStep, ...],
    slot_ids: set[str],
) -> None:
    referenced = {
        slot for item in (*deliverables, *conditions, *steps) for slot in item.required_slot_ids
    }
    unknown = sorted(referenced - slot_ids)
    if unknown:
        raise WorkContinuityError("unknown referenced result slots: " + ", ".join(unknown))


def _object_list(value: Any, label: str, maximum: int, *, required: bool) -> list[dict[str, Any]]:
    values = list(value or [])
    if required and not values:
        raise WorkContinuityError(f"{label} must not be empty")
    if len(values) > maximum:
        raise WorkContinuityError(f"{label} exceeds its item limit ({maximum})")
    if any(not isinstance(item, dict) for item in values):
        raise WorkContinuityError(f"each {label} item must be an object")
    return values


def _string_list(value: Any, label: str, maximum: int, char_limit: int) -> tuple[str, ...]:
    values = list(value or [])
    if len(values) > maximum:
        raise WorkContinuityError(f"{label} exceeds its item limit ({maximum})")
    return tuple(_bounded_text(item, label, char_limit, required=True) for item in values)


def _identifier_list(value: Any) -> tuple[str, ...]:
    values = list(value or [])
    if len(values) > MAX_RESULT_SLOTS:
        raise WorkContinuityError("too many referenced result slots")
    result = tuple(_identifier(item, "result slot reference") for item in values)
    if len(set(result)) != len(result):
        raise WorkContinuityError("result slot references must be unique")
    return result


def _resource_refs(value: Any) -> tuple[str, ...]:
    values = list(value or [])
    if not values:
        raise WorkContinuityError("slot binding resource_refs must not be empty")
    if len(values) > MAX_REFS_PER_SLOT:
        raise WorkContinuityError("too many resource refs in one slot binding")
    refs = tuple(str(item or "").strip() for item in values)
    if any(not valid_work_resource_ref(item) for item in refs):
        raise WorkContinuityError("resource ref is empty or exceeds its hard length limit")
    if len(set(refs)) != len(refs):
        raise WorkContinuityError("resource refs in one binding must be unique")
    return refs


def valid_work_resource_ref(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and len(text) <= MAX_RESOURCE_REF_CHARS


def _bounded_text(value: Any, label: str, maximum: int, *, required: bool) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise WorkContinuityError(f"{label} is required")
    if len(text) > maximum:
        raise WorkContinuityError(f"{label} exceeds its hard character limit ({maximum})")
    return text


def _identifier(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not _ID_PATTERN.fullmatch(text):
        raise WorkContinuityError(f"{label} must match {_ID_PATTERN.pattern}")
    return text


def _unique_ids(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise WorkContinuityError(f"{label} IDs must be unique")


def _fingerprint(operation: str, expected_version: int, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "operation": operation,
            "expected_version": int(expected_version),
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "MainCoreWorkSession",
    "MainCoreWorkSnapshot",
    "TERMINAL_WORK_STATUSES",
    "WorkContinuityError",
    "valid_work_resource_ref",
]
