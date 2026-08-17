"""Ordered, idempotent inbound voice transcription before ledger admission."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import (
    AIAudioContent,
    AICapabilityEffect,
    AICapabilityRequest,
    AIExecutionMode,
    AITranscriptionResult,
    AIWorkPurpose,
)
from ...contracts.system_notice import soulcore_system_notice
from .context_message import VOICE_MARKER, is_voice_component_kind
from .inbound_voice_repository import InboundVoiceAdmissionPort
from .media_resolution import resolve_inbound_audio

VOICE_TRANSCRIPTION_FAILURE_NOTICE = soulcore_system_notice(
    "这条语音没有识别成功，因此没有交给角色处理。请重新发送语音，或改用文字。"
)


class InboundVoiceTranscriptionError(RuntimeError):
    """A retryable STT failure that must not be committed as an empty transcript."""


_MAX_INBOUND_AUDIO_BYTES = 25 * 1024 * 1024


@dataclass(slots=True)
class _InstanceOrderState:
    pending: set[int]
    changed: asyncio.Event
    active: int | None = None


@dataclass(slots=True)
class InboundOrderToken:
    coordinator: InboundVoiceCoordinator
    key: tuple[str, str]
    sequence: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.coordinator.release(self.key, self.sequence)


@dataclass(slots=True)
class InboundRouteOrderToken:
    coordinator: InboundVoiceCoordinator
    key: str
    sequence: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.coordinator.release_route(self.key, self.sequence)


class InboundVoiceCoordinator:
    """Fence every instance admission while speech ahead of it is transcribed."""

    def __init__(
        self,
        *,
        ai_manager: Any | None,
        repository: InboundVoiceAdmissionPort | None = None,
    ) -> None:
        self.ai_manager = ai_manager
        self.repository = repository
        self._route_order: dict[str, _InstanceOrderState] = {}
        self._instance_order: dict[tuple[str, str], _InstanceOrderState] = {}
        self._settled_in_process: dict[tuple[str, str, str], tuple[str, ...]] = {}

    def register(self, profile_id: str, instance_id: str, sequence: int) -> None:
        key = (str(profile_id), str(instance_id))
        state = self._state(key)
        state.pending.add(int(sequence))
        self._notify(state)

    def register_route(self, route_key: str, sequence: int) -> None:
        state = self._route_state(route_key)
        state.pending.add(int(sequence))
        self._notify(state)

    async def acquire_route(self, route_key: str, sequence: int) -> InboundRouteOrderToken:
        key = str(route_key)
        ticket = int(sequence)
        state = self._route_state(key)
        state.pending.add(ticket)
        try:
            while state.active is not None or ticket != min(state.pending):
                changed = state.changed
                await changed.wait()
        except asyncio.CancelledError:
            self.release_route(key, ticket)
            raise
        state.active = ticket
        return InboundRouteOrderToken(self, key, ticket)

    def release_route(self, route_key: str, sequence: int) -> None:
        key = str(route_key)
        ticket = int(sequence)
        state = self._route_order.get(key)
        if state is None:
            return
        state.pending.discard(ticket)
        if state.active == ticket:
            state.active = None
        self._notify(state)
        if not state.pending and state.active is None and self._route_order.get(key) is state:
            self._route_order.pop(key, None)

    def _route_state(self, route_key: str) -> _InstanceOrderState:
        key = str(route_key)
        state = self._route_order.get(key)
        if state is None:
            state = _InstanceOrderState(set(), asyncio.Event())
            self._route_order[key] = state
        return state

    async def acquire(self, profile_id: str, instance_id: str, sequence: int) -> InboundOrderToken:
        key = (str(profile_id), str(instance_id))
        state = self._state(key)
        ticket = int(sequence)
        state.pending.add(ticket)
        while state.active is not None or ticket != min(state.pending):
            changed = state.changed
            await changed.wait()
        state.active = ticket
        return InboundOrderToken(self, key, ticket)

    def release(self, key: tuple[str, str], sequence: int) -> None:
        state = self._instance_order.get(key)
        if state is None:
            return
        ticket = int(sequence)
        state.pending.discard(ticket)
        if state.active == ticket:
            state.active = None
        self._notify(state)
        self._discard_empty_state(key, state)

    def discard(self, profile_id: str, instance_id: str, sequence: int) -> None:
        self.release((str(profile_id), str(instance_id)), int(sequence))

    def _state(self, key: tuple[str, str]) -> _InstanceOrderState:
        state = self._instance_order.get(key)
        if state is None:
            state = _InstanceOrderState(set(), asyncio.Event())
            self._instance_order[key] = state
        return state

    @staticmethod
    def _notify(state: _InstanceOrderState) -> None:
        changed = state.changed
        state.changed = asyncio.Event()
        changed.set()

    def _discard_empty_state(self, key: tuple[str, str], state: _InstanceOrderState) -> None:
        if not state.pending and state.active is None and self._instance_order.get(key) is state:
            self._instance_order.pop(key, None)

    async def transcribe_payload(
        self,
        *,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        payload: dict[str, Any],
    ) -> None:
        entries = tuple(payload.pop("inbound_voice", None) or ())
        ordered_projection = list(payload.pop("voice_ordered_projection", None) or ())
        self._remove_voice_media(payload)
        if not entries:
            self._remove_voice_components(payload)
            return

        key = (str(profile_id), str(instance_id), str(platform_message_id))
        transcripts: tuple[str, ...]
        if platform_message_id and self.repository is not None:
            admission, inserted = await self.repository.admit(
                profile_id,
                instance_id,
                platform_message_id,
                len(entries),
            )
            if inserted:
                transcripts = await self._transcribe_entries(
                    profile_id,
                    instance_id,
                    platform_message_id,
                    entries,
                )
                settled = await self.repository.settle(
                    profile_id,
                    instance_id,
                    platform_message_id,
                    transcripts,
                )
                transcripts = settled.transcripts
            elif admission.settled:
                transcripts = admission.transcripts
            else:
                # A prior attempt failed before settlement.  A replay must run
                # STT again instead of permanently converting the voice to an
                # empty marker.
                transcripts = await self._transcribe_entries(
                    profile_id,
                    instance_id,
                    platform_message_id,
                    entries,
                )
                settled = await self.repository.settle(
                    profile_id, instance_id, platform_message_id, transcripts
                )
                transcripts = settled.transcripts
        elif platform_message_id and key in self._settled_in_process:
            transcripts = self._settled_in_process[key]
        else:
            transcripts = await self._transcribe_entries(
                profile_id,
                instance_id,
                platform_message_id,
                entries,
            )
            if platform_message_id:
                self._settled_in_process[key] = transcripts

        self._apply_transcripts(payload, entries, ordered_projection, transcripts)

    def settle_without_transcription(self, payload: dict[str, Any]) -> None:
        """Remove live voice and keep marker-only text for non-STT lifecycle paths."""

        entries = tuple(payload.pop("inbound_voice", None) or ())
        ordered_projection = list(payload.pop("voice_ordered_projection", None) or ())
        self._remove_voice_media(payload)
        if entries:
            self._apply_transcripts(
                payload,
                entries,
                ordered_projection,
                tuple("" for _ in entries),
            )
        else:
            self._remove_voice_components(payload)

    async def _transcribe_entries(
        self,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        entries: tuple[Any, ...],
    ) -> tuple[str, ...]:
        results: list[str] = []
        for ordinal, entry in enumerate(entries):
            results.append(
                await self._transcribe_one(
                    profile_id,
                    instance_id,
                    platform_message_id,
                    ordinal,
                    entry,
                )
            )
        return tuple(results)

    async def _transcribe_one(
        self,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        ordinal: int,
        entry: Any,
    ) -> str:
        if self.ai_manager is None or not isinstance(entry, Mapping):
            raise InboundVoiceTranscriptionError("speech transcription is unavailable")
        try:
            resolved = await resolve_inbound_audio(
                entry.get("component"),
                str(entry.get("locator") or ""),
                max_bytes=_MAX_INBOUND_AUDIO_BYTES,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise InboundVoiceTranscriptionError("inbound audio could not be resolved") from exc
        stable_message_id = str(platform_message_id or uuid.uuid4().hex)
        stage_key = f"inbound-voice:{stable_message_id}:{ordinal}"
        result: Any = None
        failure: Exception | None = None
        for delay in (0.0, 0.2, 0.8):
            if delay:
                await asyncio.sleep(delay)
            try:
                result = await self.ai_manager.invoke_capability(
                    AICapabilityRequest(
                        invocation_id=uuid.uuid4().hex,
                        capability="audio.transcribe",
                        work_purpose=AIWorkPurpose.AUDIO_TRANSCRIPTION,
                        logical_stage_key=stage_key,
                        payload={
                            "audio": AIAudioContent(
                                data=resolved.data,
                                mime_type=resolved.mime_type,
                                filename=resolved.filename,
                                duration_seconds=resolved.duration_seconds,
                            )
                        },
                        effect=AICapabilityEffect.READ_ONLY,
                        execution_mode=AIExecutionMode.FOREGROUND_SYNC,
                        profile_id=profile_id,
                        instance_id=instance_id,
                        owner_kind="inbound_voice",
                        owner_id=stable_message_id,
                        idempotency_key=stage_key,
                    )
                )
                failure = None
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = exc
        if failure is not None:
            raise InboundVoiceTranscriptionError(
                "speech transcription failed after bounded retries"
            ) from failure
        output = result.output
        if not isinstance(output, AITranscriptionResult):
            raise InboundVoiceTranscriptionError("speech provider returned an invalid result")
        return str(output.text or "").strip()

    @classmethod
    def _apply_transcripts(
        cls,
        payload: dict[str, Any],
        entries: tuple[Any, ...],
        ordered_projection: list[str],
        transcripts: tuple[str, ...],
    ) -> None:
        components = list(payload.get("components") or [])
        values = cls._transcript_values(len(entries), transcripts)
        for entry, value in zip(entries, values, strict=True):
            cls._replace_transcribed_component(
                components,
                ordered_projection,
                entry,
                value,
            )
        payload["components"] = components
        payload["plain_text"] = cls._joined_projection(ordered_projection)

    @staticmethod
    def _transcript_values(count: int, transcripts: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            InboundVoiceCoordinator._transcript_value(transcripts, index) for index in range(count)
        )

    @staticmethod
    def _transcript_value(transcripts: tuple[str, ...], index: int) -> str:
        if index >= len(transcripts):
            return VOICE_MARKER
        transcript = str(transcripts[index] or "").strip()
        return f"{VOICE_MARKER}{transcript}" if transcript else VOICE_MARKER

    @staticmethod
    def _replace_transcribed_component(
        components: list[Any],
        ordered_projection: list[str],
        entry: Any,
        value: str,
    ) -> None:
        index = InboundVoiceCoordinator._entry_component_index(entry)
        if index is None or not 0 <= index < len(components):
            return
        components[index] = {"type": "plain", "text": value}
        if 0 <= index < len(ordered_projection):
            ordered_projection[index] = value

    @staticmethod
    def _entry_component_index(entry: Any) -> int | None:
        if not isinstance(entry, Mapping):
            return None
        try:
            return int(entry.get("component_index"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _joined_projection(ordered_projection: list[str]) -> str:
        parts: list[str] = []
        for item in ordered_projection:
            text = str(item or "").strip()
            if text:
                parts.append(text)
        return " ".join(parts)

    @staticmethod
    def _remove_voice_media(payload: dict[str, Any]) -> None:
        payload["inbound_media"] = [
            item
            for item in list(payload.get("inbound_media") or [])
            if not is_voice_component_kind(item.get("kind") if isinstance(item, Mapping) else "")
            and str(item.get("kind") if isinstance(item, Mapping) else "").lower() != "audio"
        ]

    @staticmethod
    def _remove_voice_components(payload: dict[str, Any]) -> None:
        components = []
        for item in list(payload.get("components") or []):
            if isinstance(item, Mapping) and is_voice_component_kind(item.get("type")):
                components.append({"type": "plain", "text": VOICE_MARKER})
            else:
                components.append(item)
        payload["components"] = components


__all__ = [
    "InboundVoiceTranscriptionError",
    "InboundOrderToken",
    "InboundRouteOrderToken",
    "InboundVoiceCoordinator",
    "VOICE_TRANSCRIPTION_FAILURE_NOTICE",
]
