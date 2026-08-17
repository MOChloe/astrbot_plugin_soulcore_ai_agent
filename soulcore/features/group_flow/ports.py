"""Narrow persistence boundary owned by the group-flow feature."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from ...contracts.group_flow import (
    GroupFlowDiagnostic,
    GroupFlowInboundMessage,
    GroupFlowPolicy,
    GroupFlowSourceMessage,
    GroupFlowWindow,
    GroupReplyRelocationCheck,
    GroupReplyRelocationResult,
    GroupRunFence,
)


class GroupFlowRepository(Protocol):
    async def get_group_flow_policy(
        self, profile_id: str, scope: str = "group"
    ) -> GroupFlowPolicy: ...

    async def update_group_flow_policy(
        self,
        profile_id: str,
        scope: str,
        patch: Mapping[str, object],
        *,
        expected_version: int,
    ) -> GroupFlowPolicy: ...

    async def append_message(
        self,
        profile_id: str,
        instance_id: str,
        message: GroupFlowInboundMessage,
        *,
        policy: GroupFlowPolicy,
        now: datetime,
    ) -> GroupFlowWindow: ...

    async def get_window(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> GroupFlowWindow | None: ...

    async def load_window_messages(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> tuple[GroupFlowSourceMessage, ...]: ...

    async def load_judge_messages(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> tuple[GroupFlowSourceMessage, ...]: ...

    async def settle_due_windows(self, *, now: datetime, limit: int = 32) -> int: ...

    async def claim_judging_windows(
        self, *, now: datetime, worker_id: str, limit: int, lease_seconds: int
    ) -> tuple[GroupFlowWindow, ...]: ...

    async def record_judgment(
        self,
        window: GroupFlowWindow,
        *,
        suitable: bool,
        error_code: str,
        now: datetime,
    ) -> GroupFlowWindow | None: ...

    async def release_judgment(self, window: GroupFlowWindow, *, now: datetime) -> bool: ...

    async def claim_ready_windows(
        self, *, now: datetime, worker_id: str, limit: int, lease_seconds: int
    ) -> tuple[GroupFlowWindow, ...]: ...

    async def attach_main_core_run(
        self, window: GroupFlowWindow, *, main_core_task_ref: str, now: datetime
    ) -> GroupRunFence | None: ...

    async def release_ready(
        self, window: GroupFlowWindow, *, retry_at: datetime, reason: str
    ) -> bool: ...

    async def mark_waiting_first_attempt(
        self, profile_id: str, instance_id: str, fence: GroupRunFence, *, now: datetime
    ) -> bool: ...

    async def is_first_attempt_protected(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> bool: ...

    async def has_protected_run(self, profile_id: str, instance_id: str) -> bool: ...

    async def next_collecting_message_id(self, profile_id: str, instance_id: str) -> int | None: ...

    async def mark_first_attempt_started(
        self, profile_id: str, instance_id: str, window_id: str, *, now: datetime
    ) -> bool: ...

    async def resolve_window(
        self,
        profile_id: str,
        instance_id: str,
        window_id: str,
        *,
        outcome: str,
        now: datetime,
    ) -> bool: ...

    async def recover(self, *, now: datetime) -> int: ...
    async def next_due_at(self) -> datetime | None: ...
    async def diagnostic(self, profile_id: str, instance_id: str) -> GroupFlowDiagnostic: ...

    async def claim_reply_relocation_checks(
        self, *, now: datetime, worker_id: str, limit: int, lease_seconds: int
    ) -> tuple[GroupReplyRelocationCheck, ...]: ...

    async def record_reply_relocation_decision(
        self,
        check: GroupReplyRelocationCheck,
        *,
        recheck_after_seconds: int,
        error_code: str,
        now: datetime,
    ) -> bool: ...

    async def release_reply_relocation_check(
        self, check: GroupReplyRelocationCheck, *, now: datetime
    ) -> bool: ...

    async def apply_reply_relocation(
        self, check: GroupReplyRelocationCheck, *, now: datetime
    ) -> GroupReplyRelocationResult: ...


class GroupInterjectionJudge(Protocol):
    async def judge(
        self,
        *,
        profile_id: str,
        instance_id: str,
        messages: Sequence[GroupFlowSourceMessage],
        token_budget: int,
        owner_id: str,
        idempotency_key: str,
    ) -> object: ...


class GroupReplyRelocationJudge(Protocol):
    async def judge(
        self,
        *,
        profile_id: str,
        instance_id: str,
        original_messages: Sequence[GroupFlowSourceMessage],
        later_messages: Sequence[GroupFlowSourceMessage],
        pending_first_text: str,
        token_budget: int,
        owner_id: str,
        idempotency_key: str,
        allow_wait: bool = True,
    ) -> object: ...


__all__ = [
    "GroupFlowRepository",
    "GroupInterjectionJudge",
    "GroupReplyRelocationJudge",
]
