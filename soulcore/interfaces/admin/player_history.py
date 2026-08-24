"""Player-facing dialogue history assembled from existing durable records.

The advanced AI work page intentionally exposes diagnostic structures.  This
read model is a separate boundary: it joins the existing message ledger, Core
runs, delivery links and accepted model outputs, then emits only language a
player can understand.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from ...contracts.delivery_visibility import is_dialogue_continuity_visible
from .player_output import PlayerOutputProjector
from .player_views import player_history_record_ref
from .presentation import jsonable

_LINK_PAGE_SIZE = 1_000
_MESSAGE_PAGE_SIZE = 1_000


@dataclass(frozen=True, slots=True)
class _HistoryRecord:
    record_ref: str
    messages: tuple[Any, ...]
    workflow_id: int | None
    run_id: int | None
    occurred_at: Any
    had_player_input: bool


class PlayerHistoryController:
    """Compose the existing stores into a stable, read-only player history."""

    def __init__(
        self,
        *,
        conversation_repository: Any,
        timeline_repository: Any,
        delivery_repository: Any,
        ai_repository: Any,
    ) -> None:
        self.conversation = conversation_repository
        self.timeline = timeline_repository
        self.delivery = delivery_repository
        self.ai = ai_repository
        self.output = PlayerOutputProjector()

    async def history(
        self,
        profile_id: str,
        contacts: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        *,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        bounded_page = max(1, int(page))
        bounded_size = max(1, min(int(page_size), 50))
        grouped = await asyncio.gather(
            *(self._contact_records(profile_id, internal) for internal, _public in contacts)
        )
        all_records = [
            (record, public)
            for records, (_internal, public) in zip(grouped, contacts, strict=True)
            for record in records
        ]
        all_records.sort(key=lambda item: self._time_key(item[0].occurred_at), reverse=True)
        page_count = max(1, (len(all_records) + bounded_size - 1) // bounded_size)
        bounded_page = min(bounded_page, page_count)
        start = (bounded_page - 1) * bounded_size
        selected = all_records[start : start + bounded_size]
        thought_by_workflow = await self._thought_availability(
            tuple(
                dict.fromkeys(
                    record.workflow_id
                    for record, _contact in selected
                    if record.workflow_id is not None
                )
            )
        )
        return {
            "items": [
                self._record_summary(
                    record,
                    contact,
                    has_thought=bool(
                        record.workflow_id is not None
                        and thought_by_workflow.get(record.workflow_id, False)
                    ),
                )
                for record, contact in selected
            ],
            "contacts": [dict(public) for _internal, public in contacts],
            "page": bounded_page,
            "page_size": bounded_size,
            "total": len(all_records),
            "has_more": start + bounded_size < len(all_records),
        }

    async def record(
        self,
        profile_id: str,
        internal_contact: Mapping[str, Any],
        public_contact: Mapping[str, Any],
        record_ref: str,
    ) -> dict[str, Any]:
        wanted = str(record_ref or "").strip()
        if not wanted:
            raise ValueError("record_ref is required")
        records = await self._contact_records(profile_id, internal_contact)
        selected = next((item for item in records if item.record_ref == wanted), None)
        if selected is None:
            raise ValueError("这条记录已经变化，请返回记录列表重新选择")
        return await self._record_detail(
            profile_id,
            internal_contact,
            public_contact,
            selected,
        )

    async def record_from_contacts(
        self,
        profile_id: str,
        contacts: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        record_ref: str,
    ) -> dict[str, Any]:
        """Resolve a stable record reference without requiring a contact hint."""

        wanted = str(record_ref or "").strip()
        if not wanted:
            raise ValueError("record_ref is required")
        grouped = await asyncio.gather(
            *(self._contact_records(profile_id, internal) for internal, _public in contacts)
        )
        for records, (internal, public) in zip(grouped, contacts, strict=True):
            selected = next((item for item in records if item.record_ref == wanted), None)
            if selected is not None:
                return await self._record_detail(profile_id, internal, public, selected)
        raise ValueError("这条记录已经变化，请返回记录列表重新选择")

    async def _record_detail(
        self,
        profile_id: str,
        internal_contact: Mapping[str, Any],
        public_contact: Mapping[str, Any],
        selected: _HistoryRecord,
    ) -> dict[str, Any]:
        visible_messages = await self._visible_messages(
            profile_id, str(internal_contact.get("instance_id") or "")
        )
        context_before, context_after = self._surrounding_context(visible_messages, selected)
        thought = (
            await self._workflow_thought(selected.workflow_id)
            if selected.workflow_id is not None
            else []
        )
        projected_turn = [self._message_view(item, public_contact) for item in selected.messages]
        return {
            "record": self._record_summary(selected, public_contact, has_thought=bool(thought)),
            "messages": projected_turn,
            "context_before": [self._message_view(item, public_contact) for item in context_before],
            "context_after": [self._message_view(item, public_contact) for item in context_after],
            "thought": thought,
            "actual_outbound": [
                self._message_view(item, public_contact)
                for item in selected.messages
                if self._direction(item) == "OUTBOUND"
            ],
        }

    async def _contact_records(
        self, profile_id: str, contact: Mapping[str, Any]
    ) -> list[_HistoryRecord]:
        instance_id = str(contact.get("instance_id") or "")
        messages, runs, delivery_links = await asyncio.gather(
            self._visible_messages(profile_id, instance_id),
            self._all_runs(profile_id, instance_id),
            self._all_delivery_links(profile_id, instance_id),
        )
        by_id = {int(item.message_id): item for item in messages}
        candidates = self._run_candidates(
            profile_id,
            instance_id,
            runs,
            delivery_links,
            by_id,
        )
        result, claimed = self._deduplicated_records(candidates)
        result.extend(self._standalone_records(profile_id, instance_id, messages, claimed))
        result.sort(key=lambda item: self._time_key(item.occurred_at), reverse=True)
        return result

    async def _all_runs(self, profile_id: str, instance_id: str) -> list[Mapping[str, Any]]:
        return await self._paged_rows(
            lambda offset: self.timeline.list_instance_runs(
                profile_id,
                instance_id,
                limit=_LINK_PAGE_SIZE,
                offset=offset,
            )
        )

    async def _all_delivery_links(
        self, profile_id: str, instance_id: str
    ) -> list[Mapping[str, Any]]:
        return await self._paged_rows(
            lambda offset: self.delivery.list_instance_player_history_links(
                profile_id,
                instance_id,
                limit=_LINK_PAGE_SIZE,
                offset=offset,
            )
        )

    @staticmethod
    async def _paged_rows(load: Any) -> list[Mapping[str, Any]]:
        result: list[Mapping[str, Any]] = []
        offset = 0
        while True:
            page = list(await load(offset))
            result.extend(page)
            if len(page) < _LINK_PAGE_SIZE:
                return result
            offset += len(page)

    def _run_candidates(
        self,
        profile_id: str,
        instance_id: str,
        runs: Sequence[Mapping[str, Any]],
        delivery_links: Sequence[Mapping[str, Any]],
        by_id: Mapping[int, Any],
    ) -> list[tuple[tuple[int, int, int], _HistoryRecord]]:
        candidates = []
        for run in runs:
            candidate = self._run_candidate(
                profile_id,
                instance_id,
                run,
                delivery_links,
                by_id,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _run_candidate(
        self,
        profile_id: str,
        instance_id: str,
        run: Mapping[str, Any],
        delivery_links: Sequence[Mapping[str, Any]],
        by_id: Mapping[int, Any],
    ) -> tuple[tuple[int, int, int], _HistoryRecord] | None:
        run_id = self._positive_int(run.get("run_id"))
        if run_id is None:
            return None
        workflow_id = self._positive_int(run.get("workflow_id"))
        inbound_ids = self._run_inbound_ids(run, by_id)
        outbound_ids = self._run_outbound_ids(
            delivery_links,
            by_id,
            run_id=run_id,
            workflow_id=workflow_id,
        )
        message_ids = tuple(dict.fromkeys((*inbound_ids, *outbound_ids)))
        selected_messages = tuple(
            sorted((by_id[value] for value in message_ids), key=self._message_id)
        )
        if not selected_messages:
            return None
        record = _HistoryRecord(
            record_ref=player_history_record_ref(profile_id, instance_id, f"run:{run_id}"),
            messages=selected_messages,
            workflow_id=workflow_id,
            run_id=run_id,
            occurred_at=self._latest_time(selected_messages, run.get("started_at")),
            had_player_input=bool(inbound_ids),
        )
        return (int(bool(outbound_ids)), len(message_ids), run_id), record

    def _run_inbound_ids(self, run: Mapping[str, Any], by_id: Mapping[int, Any]) -> tuple[int, ...]:
        result = []
        for message_id in self._run_context_message_ids(run):
            message = by_id.get(message_id)
            if message is None or self._direction(message) != "INBOUND":
                continue
            result.append(message_id)
        return tuple(dict.fromkeys(result))

    def _run_outbound_ids(
        self,
        delivery_links: Sequence[Mapping[str, Any]],
        by_id: Mapping[int, Any],
        *,
        run_id: int,
        workflow_id: int | None,
    ) -> tuple[int, ...]:
        result = []
        for link in delivery_links:
            if not self._link_belongs_to_run(link, run_id, workflow_id):
                continue
            message_id = self._positive_int(link.get("context_message_id"))
            if message_id is None:
                continue
            message = by_id.get(message_id)
            if message is None or self._direction(message) != "OUTBOUND":
                continue
            result.append(message_id)
        return tuple(dict.fromkeys(result))

    def _deduplicated_records(
        self, candidates: Sequence[tuple[tuple[int, int, int], _HistoryRecord]]
    ) -> tuple[list[_HistoryRecord], set[int]]:
        claimed: set[int] = set()
        result = []
        for _quality, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            fresh = tuple(
                message
                for message in candidate.messages
                if self._message_id(message) not in claimed
            )
            if not fresh:
                continue
            claimed.update(self._message_id(message) for message in fresh)
            result.append(
                _HistoryRecord(
                    record_ref=candidate.record_ref,
                    messages=fresh,
                    workflow_id=candidate.workflow_id,
                    run_id=candidate.run_id,
                    occurred_at=self._latest_time(fresh, candidate.occurred_at),
                    had_player_input=any(
                        self._direction(message) == "INBOUND" for message in fresh
                    ),
                )
            )
        return result, claimed

    def _standalone_records(
        self,
        profile_id: str,
        instance_id: str,
        messages: Sequence[Any],
        claimed: set[int],
    ) -> list[_HistoryRecord]:
        result = []
        for message in messages:
            message_id = self._message_id(message)
            if message_id in claimed:
                continue
            result.append(
                _HistoryRecord(
                    record_ref=player_history_record_ref(
                        profile_id, instance_id, f"message:{message_id}"
                    ),
                    messages=(message,),
                    workflow_id=None,
                    run_id=None,
                    occurred_at=message.occurred_at or message.created_at,
                    had_player_input=self._direction(message) == "INBOUND",
                )
            )
        return result

    async def _visible_messages(self, profile_id: str, instance_id: str) -> list[Any]:
        messages: list[Any] = []
        after_message_id: int | None = None
        while True:
            page = await self.conversation.list_instance_messages(
                profile_id,
                instance_id,
                after_message_id=after_message_id,
                limit=_MESSAGE_PAGE_SIZE,
                ascending=True,
                context_eligible_only=False,
            )
            if not page:
                break
            next_message_id = max(self._message_id(item) for item in page)
            if after_message_id is not None and next_message_id <= after_message_id:
                break
            messages.extend(page)
            if len(page) < _MESSAGE_PAGE_SIZE:
                break
            after_message_id = next_message_id
        return [
            item
            for item in messages
            if is_dialogue_continuity_visible(
                self._direction(item), str(item.delivery_status or "")
            )
        ]

    async def _thought_availability(self, workflow_ids: Sequence[int]) -> dict[int, bool]:
        if not workflow_ids:
            return {}
        projected = await asyncio.gather(
            *(self._workflow_thought(workflow_id) for workflow_id in workflow_ids)
        )
        return {
            int(workflow_id): bool(stages)
            for workflow_id, stages in zip(workflow_ids, projected, strict=True)
        }

    async def _workflow_thought(self, workflow_id: int) -> list[dict[str, Any]]:
        nodes, attempts = await asyncio.gather(
            self.ai.list_ai_work_nodes(int(workflow_id)),
            self.ai.list_ai_provider_attempts(workflow_id=int(workflow_id)),
        )
        nodes_by_id = {
            int(node["node_id"]): node
            for node in nodes
            if self._positive_int(node.get("node_id")) is not None
        }
        stages: list[dict[str, Any]] = []
        main_round = 0
        for attempt in attempts:
            projected = self._attempt_thought(attempt, nodes_by_id)
            if projected is None:
                continue
            kind, cards = projected
            if kind == "main_core":
                main_round += 1
                stages.append(
                    {
                        "kind": "main_core",
                        "title": f"主 Core 第 {main_round} 轮",
                        "cards": cards,
                    }
                )
            else:
                stages.append(
                    {
                        "kind": "response_polish",
                        "title": "润色后的输出",
                        "cards": cards,
                    }
                )
        return stages

    def _attempt_thought(
        self,
        attempt: Mapping[str, Any],
        nodes_by_id: Mapping[int, Mapping[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]] | None:
        if str(attempt.get("status") or "").upper() != "SUCCEEDED":
            return None
        node_id = self._positive_int(attempt.get("node_id"))
        node = nodes_by_id.get(node_id or 0)
        if node is None:
            return None
        evaluation = self._mapping(attempt.get("evaluation"))
        if not self.output.accepted(evaluation):
            return None
        purpose = str(node.get("purpose") or "").upper()
        if purpose == "MAIN_CORE":
            cards = self.output.main_core_cards(evaluation)
            return ("main_core", cards) if cards else None
        if purpose != "RESPONSE_POLISH":
            return None
        if str(node.get("status") or "").upper() != "SUCCEEDED":
            return None
        cards = self.output.expression_cards(evaluation.get("validated_output"))
        return ("response_polish", cards) if cards else None

    @classmethod
    def _record_summary(
        cls,
        record: _HistoryRecord,
        contact: Mapping[str, Any],
        *,
        has_thought: bool,
    ) -> dict[str, Any]:
        inbound = [item for item in record.messages if cls._direction(item) == "INBOUND"]
        outbound = [item for item in record.messages if cls._direction(item) == "OUTBOUND"]
        media = cls._media_hints(record.messages)
        kind = "conversation"
        if not record.had_player_input:
            kind = "proactive"
        elif not outbound:
            kind = "unanswered"
        return {
            "record_ref": record.record_ref,
            "contact": dict(contact),
            "occurred_at": jsonable(record.occurred_at),
            "kind": kind,
            "kind_label": {
                "conversation": "一轮对话",
                "proactive": "角色主动联系",
                "unanswered": "等待回复",
            }[kind],
            "user_preview": cls._messages_preview(inbound) if inbound else "角色主动联系",
            "ai_preview": cls._messages_preview(outbound) if outbound else "暂时没有实际回复",
            "media": media,
            "has_thought": bool(has_thought),
        }

    @classmethod
    def _message_view(cls, message: Any, contact: Mapping[str, Any]) -> dict[str, Any]:
        direction = cls._direction(message)
        sender = "角色"
        if direction == "INBOUND":
            sender = str(message.sender_name or "").strip()
            if not sender:
                sender = (
                    "一位群成员"
                    if str(contact.get("kind") or "") == "group"
                    else str(contact.get("display_name") or "对方")
                )
        return {
            "message_ref": player_history_record_ref(
                str(message.profile_id),
                str(message.instance_id),
                f"ledger:{cls._message_id(message)}",
            ),
            "side": "player" if direction == "INBOUND" else "role",
            "sender": sender,
            "text": str(message.plain_text or "").strip(),
            "media": cls._message_media(message),
            "occurred_at": jsonable(message.occurred_at or message.created_at),
        }

    @classmethod
    def _messages_preview(cls, messages: Sequence[Any]) -> str:
        parts = []
        for message in messages:
            text = re.sub(r"\s+", " ", str(message.plain_text or "").strip())
            labels = [item["label"] for item in cls._message_media(message)]
            value = " ".join(part for part in (text, *labels) if part)
            if value:
                parts.append(value)
        return cls._truncate(" / ".join(parts), 140)

    @classmethod
    def _media_hints(cls, messages: Sequence[Any]) -> list[str]:
        return list(
            dict.fromkeys(
                item["label"] for message in messages for item in cls._message_media(message)
            )
        )

    @staticmethod
    def _message_media(message: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for component in getattr(message, "components", ()) or ():
            if not isinstance(component, Mapping):
                continue
            component_type = str(component.get("type") or "").strip().lower()
            if not component_type or component_type in {
                "plain",
                "at",
                "reply",
                "inbound_reply_reference",
            }:
                continue
            candidate = PlayerHistoryController._component_media(component, component_type)
            if candidate not in result:
                result.append(candidate)
        return result

    @staticmethod
    def _component_media(component: Mapping[str, Any], component_type: str) -> dict[str, str]:
        if "sticker" in component_type:
            return {"kind": "sticker", "label": "表情"}
        if "image" in component_type:
            return {"kind": "image", "label": "图片"}
        if "file" in component_type or "document" in component_type:
            filename = PurePath(str(component.get("name") or "").replace("\\", "/")).name
            label = f"文件：{filename}" if filename and len(filename) <= 120 else "文件"
            return {"kind": "file", "label": label}
        if "voice" in component_type or "audio" in component_type:
            return {"kind": "voice", "label": "语音"}
        if "video" in component_type:
            return {"kind": "video", "label": "视频"}
        return {"kind": "media", "label": "媒体"}

    @classmethod
    def _surrounding_context(
        cls, visible_messages: Sequence[Any], record: _HistoryRecord
    ) -> tuple[list[Any], list[Any]]:
        positions = {
            cls._message_id(message): index for index, message in enumerate(visible_messages)
        }
        selected_positions = [
            positions[cls._message_id(message)]
            for message in record.messages
            if cls._message_id(message) in positions
        ]
        if not selected_positions:
            return [], []
        first, last = min(selected_positions), max(selected_positions)
        return list(visible_messages[max(0, first - 3) : first]), list(
            visible_messages[last + 1 : last + 4]
        )

    @staticmethod
    def _run_context_message_ids(run: Mapping[str, Any]) -> tuple[int, ...]:
        request = PlayerHistoryController._mapping(run.get("request"))
        metadata = PlayerHistoryController._mapping(request.get("metadata"))
        values = metadata.get("context_message_ids")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return ()
        return tuple(
            dict.fromkeys(
                value
                for raw in values
                if (value := PlayerHistoryController._positive_int(raw)) is not None
            )
        )

    @staticmethod
    def _link_belongs_to_run(link: Mapping[str, Any], run_id: int, workflow_id: int | None) -> bool:
        origin_run_id = PlayerHistoryController._positive_int(link.get("origin_run_id"))
        if origin_run_id is not None:
            return origin_run_id == run_id
        return (
            workflow_id is not None
            and PlayerHistoryController._positive_int(link.get("workflow_id")) == workflow_id
        )

    @staticmethod
    def _latest_time(messages: Sequence[Any], fallback: Any) -> Any:
        values = [message.occurred_at or message.created_at for message in messages]
        values = [value for value in values if value is not None]
        return max(values, key=PlayerHistoryController._time_key) if values else fallback

    @staticmethod
    def _time_key(value: Any) -> str:
        projected = jsonable(value)
        return str(projected or "")

    @staticmethod
    def _message_id(message: Any) -> int:
        return int(message.message_id)

    @staticmethod
    def _direction(message: Any) -> str:
        value = getattr(message, "direction", "")
        return str(getattr(value, "value", value) or "").upper()

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        return [item for item in value if isinstance(item, Mapping)]

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        text = str(value or "").strip()
        return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


__all__ = ["PlayerHistoryController"]
