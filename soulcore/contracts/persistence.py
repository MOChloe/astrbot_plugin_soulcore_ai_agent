"""Small cross-feature read contracts implemented by composed storage adapters."""

from collections.abc import Sequence
from typing import Protocol


class ConversationHistoryPort(Protocol):
    async def list_instance_messages(
        self,
        profile_id: str,
        instance_id: str,
        *,
        limit: int,
        ascending: bool,
        context_eligible_only: bool,
    ) -> Sequence[object]: ...


class ConversationProgressQueryPort(Protocol):
    async def count_dialogue_turns(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None = None,
        through_message_id: int | None = None,
    ) -> int: ...

    async def get_latest_dialogue_message_id(self, profile_id: str, instance_id: str) -> int: ...

    async def count_player_inbound_messages(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None = None,
        through_message_id: int | None = None,
    ) -> int: ...

    async def get_latest_player_inbound_message_id(
        self, profile_id: str, instance_id: str
    ) -> int: ...


__all__ = ["ConversationHistoryPort", "ConversationProgressQueryPort"]
