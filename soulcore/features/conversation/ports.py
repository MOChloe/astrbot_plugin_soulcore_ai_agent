"""Persistence boundaries used by conversation context compilation."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

from ...contracts.persistence import ConversationProgressQueryPort
from .turn_buffer import (
    TurnBufferBatch,
    TurnBufferDialogueProjection,
    TurnBufferMessageProjection,
    TurnBufferStatus,
)


class ConversationRepositoryPort(ConversationProgressQueryPort, Protocol):
    async def get_instance_message(self, *args: object, **kwargs: object) -> Any: ...
    async def append_instance_message(self, *args: object, **kwargs: object) -> Any: ...
    async def commit_dialogue_summary(self, *args: object, **kwargs: object) -> Any: ...
    async def count_instance_messages(self, *args: object, **kwargs: object) -> Any: ...
    async def get_dialogue_summary_window(self, *args: object, **kwargs: object) -> Any: ...
    async def get_context_build_report(self, *args: object, **kwargs: object) -> Any: ...
    async def get_latest_dialogue_summary(self, *args: object, **kwargs: object) -> Any: ...
    async def list_instance_messages(self, *args: object, **kwargs: object) -> Any: ...
    async def list_instance_message_activity(
        self,
        profile_id: str,
        instance_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]: ...
    async def find_context_eligible_message_at_or_before(
        self,
        profile_id: str,
        instance_id: str,
        occurred_at: datetime,
    ) -> Any: ...
    async def set_instance_message_knowledge_eligibility(
        self, *args: object, **kwargs: object
    ) -> Any: ...
    async def save_context_build_report(self, *args: object, **kwargs: object) -> Any: ...
    async def patch_instance_message_metadata(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        *,
        metadata_patch: Mapping[str, object],
    ) -> Any: ...
    async def begin_foreground_platform_delivery(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
    ) -> bool: ...
    async def claim_foreground_delivery_preparation(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
    ) -> bool: ...
    async def update_instance_message_delivery(self, *args: object, **kwargs: object) -> Any: ...
    async def publish_context_backup(self, *args: object, **kwargs: object) -> Any: ...
    async def list_inbound_turn_messages_since_visible_assistant(
        self,
        profile_id: str,
        instance_id: str,
        *,
        through_message_id: int | None = None,
    ) -> tuple[TurnBufferMessageProjection, ...]: ...

    async def list_inbound_turn_messages_by_ids(
        self,
        profile_id: str,
        instance_id: str,
        message_ids: Sequence[int],
    ) -> tuple[TurnBufferMessageProjection, ...]: ...

    async def list_recent_turn_buffer_dialogue_before(
        self,
        profile_id: str,
        instance_id: str,
        *,
        before_message_id: int,
        limit: int = 4,
    ) -> tuple[TurnBufferDialogueProjection, ...]: ...


class TurnBufferRepositoryPort(Protocol):
    async def append_or_refresh_turn_buffer_batch(
        self,
        profile_id: str,
        instance_id: str,
        *,
        message_ids: Sequence[int],
        activity_epoch: int,
        now: datetime,
        admission_message_id: int | None = None,
        admission_lease_owner: str | None = None,
        admission_lease_token: int | None = None,
    ) -> TurnBufferBatch: ...

    async def get_active_turn_buffer_batch(
        self, profile_id: str, instance_id: str
    ) -> TurnBufferBatch | None: ...

    async def claim_turn_buffer_batches_for_classification(
        self,
        *,
        now: datetime,
        limit: int = 10,
        lease_seconds: int = 30,
        worker_id: str = "turn-buffer-classifier",
    ) -> tuple[TurnBufferBatch, ...]: ...

    async def record_turn_buffer_decision(
        self, profile_id: str, instance_id: str, batch_id: str, **values: object
    ) -> TurnBufferBatch | None: ...

    async def defer_turn_buffer_classification(
        self, profile_id: str, instance_id: str, batch_id: str, **values: object
    ) -> bool: ...

    async def claim_due_turn_buffer_batches(
        self,
        *,
        now: datetime,
        limit: int = 10,
        lease_seconds: int = 120,
        worker_id: str = "turn-buffer-admission",
    ) -> tuple[TurnBufferBatch, ...]: ...

    async def renew_turn_buffer_batch_lease(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_status: TurnBufferStatus,
        expected_generation: int,
        lease_token: int,
        lease_owner: str,
        now: datetime,
        lease_seconds: int,
    ) -> bool: ...

    async def attach_turn_buffer_main_core_task(
        self, profile_id: str, instance_id: str, batch_id: str, **values: object
    ) -> TurnBufferBatch | None: ...

    async def resolve_turn_buffer_batch(
        self, profile_id: str, instance_id: str, batch_id: str, **values: object
    ) -> bool: ...

    async def release_turn_buffer_batch(
        self, profile_id: str, instance_id: str, batch_id: str, **values: object
    ) -> bool: ...

    async def recover_turn_buffer_batches(
        self,
        *,
        now: datetime,
    ) -> int: ...

    async def reconcile_turn_buffer_switches(
        self, *, now: datetime, profile_id: str | None = None
    ) -> int: ...

    async def next_turn_buffer_due_at(self) -> datetime | None: ...


class AITaskSchedulerPort(Protocol):
    async def create_ai_task(self, profile_id: str, task_type: str, **values: object) -> object: ...

    async def list_ai_tasks(self, **values: object) -> list[Any]: ...


__all__ = [
    "ConversationRepositoryPort",
    "ConversationProgressQueryPort",
    "AITaskSchedulerPort",
    "TurnBufferBatch",
    "TurnBufferDialogueProjection",
    "TurnBufferMessageProjection",
    "TurnBufferRepositoryPort",
    "TurnBufferStatus",
]
