"""Late TTS synthesis and short-lived artifact lifecycle for Outbox delivery."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, TypeVar

from ...contracts.ai_models import (
    AIAudioContent,
    AICapabilityEffect,
    AICapabilityName,
    AICapabilityRequest,
    AIExecutionMode,
    AIRetryPolicy,
    AISpeechResult,
    AIWorkPurpose,
)
from ...shared.time import utcnow
from .audio_normalization import normalize_outbound_voice_audio
from .dispatch_context import OutboxDispatchContext
from .voice_artifacts import (
    VOICE_FALLBACK_PAYLOAD_KEY,
    VOICE_FALLBACK_REASONS,
    VoiceArtifact,
    VoiceArtifactService,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class VoiceDeliveryMixin:
    async def _prepare_voice_delivery(self, ctx: OutboxDispatchContext) -> None:
        """Resolve or synthesize audio before any send permit or Outbox claim."""

        if not self._voice_presentation_requested(ctx) or not ctx.content:
            return
        ctx.voice_requested = True
        persisted_reason = str(ctx.current.payload.get(VOICE_FALLBACK_PAYLOAD_KEY) or "")
        if persisted_reason in VOICE_FALLBACK_REASONS:
            ctx.voice_fallback_reason = persisted_reason
            return
        if not self._voice_route_ready(ctx):
            ctx.voice_fallback_reason = "PLATFORM_UNSUPPORTED"
            return
        service = self.voice_artifact_service
        if not isinstance(service, VoiceArtifactService):
            raise TypeError("voice_artifact_service must be a VoiceArtifactService")
        try:
            await asyncio.to_thread(service.purge_orphans)
            if await self._reuse_voice_artifact(ctx, service):
                return
            invocation = await self._invoke_voice_synthesis(ctx)
            output = invocation.output
            if not isinstance(output, AISpeechResult):
                raise TypeError("audio.speech returned an invalid result")
            audio = output.audio
            normalized_audio = await self._normalize_synthesized_voice(ctx, service, audio)
            if normalized_audio is None:
                return
            artifact = await self._materialize_voice_artifact(
                service,
                ctx,
                data=normalized_audio,
                mime_type="audio/wav",
                filename="voice.wav",
            )
            try:
                entry = await self._register_voice_artifact(ctx, service, artifact)
            except BaseException:
                await self._delete_unregistered_voice_artifact(service, artifact)
                raise
            ctx.voice_artifact = artifact
            ctx.voice_cleanup_id = int(entry.cleanup_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            ctx.voice_fallback_reason = "SYNTHESIS_FAILED"

    @staticmethod
    async def _normalize_synthesized_voice(
        ctx: OutboxDispatchContext,
        service: VoiceArtifactService,
        audio: AIAudioContent,
    ) -> bytes | None:
        try:
            return await normalize_outbound_voice_audio(
                audio.data,
                audio.mime_type,
                filename=audio.filename,
                maximum_bytes=service.maximum_bytes,
                work_root=service.root,
                audio_metadata=audio.metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            ctx.voice_fallback_reason = "AUDIO_NORMALIZATION_FAILED"
            return None

    async def _persist_voice_text_fallback(
        self,
        ctx: OutboxDispatchContext,
        *,
        reason: str,
    ) -> bool:
        normalized = str(reason or "").strip().upper()
        if normalized not in VOICE_FALLBACK_REASONS:
            raise ValueError("voice fallback reason is not bounded")
        persisted = bool(
            await self.outbox.persist_outbox_voice_text_fallback(
                ctx.profile_id,
                ctx.instance_id,
                ctx.item.outbox_id,
                reason=normalized,
            )
        )
        if persisted:
            ctx.voice_requested = True
            ctx.voice_delivered = False
            ctx.voice_fallback_reason = normalized
        return persisted

    @staticmethod
    def _voice_presentation_requested(ctx: OutboxDispatchContext) -> bool:
        payload = ctx.current.payload
        return (
            str(payload.get("expression_kind") or "").strip().upper() == "TEXT"
            and str(payload.get("presentation") or "").strip().upper() == "VOICE"
        )

    def _voice_route_ready(self, ctx: OutboxDispatchContext) -> bool:
        try:
            return bool(self.delivery.voice_ready(ctx.item.umo))
        except Exception:
            return False

    async def _reuse_voice_artifact(
        self,
        ctx: OutboxDispatchContext,
        service: VoiceArtifactService,
    ) -> bool:
        entries = tuple(
            await self.outbox.list_outbox_voice_artifacts(
                ctx.profile_id,
                ctx.instance_id,
                ctx.item.outbox_id,
            )
        )
        for entry in entries:
            try:
                artifact = service.from_cleanup_entry(entry)
                if not service.belongs_to(
                    artifact,
                    profile_id=ctx.profile_id,
                    instance_id=ctx.instance_id,
                    outbox_id=ctx.item.outbox_id,
                    text=ctx.content,
                ):
                    continue
                await asyncio.to_thread(service.resolve, artifact, touch=True)
                refreshed = await self._register_voice_artifact(ctx, service, artifact)
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                continue
            ctx.voice_artifact = artifact
            ctx.voice_cleanup_id = int(refreshed.cleanup_id)
            return True
        if entries:
            await self._cleanup_outbox_voice_artifacts(
                ctx,
                reason="voice_artifact_stale_before_retry",
            )
        return False

    async def _invoke_voice_synthesis(self, ctx: OutboxDispatchContext) -> Any:
        stable_key = (
            f"outbox:{ctx.item.outbox_id}:audio.speech:"
            f"{VoiceArtifactService.text_fingerprint(ctx.content)}"
        )
        return await self.model_gateway.invoke_capability(
            AICapabilityRequest(
                invocation_id=uuid.uuid4().hex,
                capability=AICapabilityName.AUDIO_SPEECH.value,
                work_purpose=AIWorkPurpose.AUDIO_SPEECH_GENERATION,
                logical_stage_key=stable_key,
                payload={"text": ctx.content},
                effect=AICapabilityEffect.NON_IDEMPOTENT_WRITE,
                execution_mode=AIExecutionMode.FOREGROUND_SYNC,
                profile_id=ctx.profile_id,
                instance_id=ctx.instance_id,
                owner_kind="outbox_voice",
                owner_id=str(ctx.item.outbox_id),
                idempotency_key=stable_key,
                retry_policy=AIRetryPolicy(max_attempts=3),
                metadata={"maximum_backend_candidates": 3},
            )
        )

    async def _materialize_voice_artifact(
        self,
        service: VoiceArtifactService,
        ctx: OutboxDispatchContext,
        *,
        data: bytes,
        mime_type: str,
        filename: str,
    ) -> VoiceArtifact:
        task = asyncio.create_task(
            asyncio.to_thread(
                service.materialize,
                profile_id=ctx.profile_id,
                instance_id=ctx.instance_id,
                outbox_id=ctx.item.outbox_id,
                text=ctx.content,
                data=data,
                mime_type=mime_type,
                filename=filename,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            artifact = await _drain_cancelled_task(
                task,
                cancellation,
                operation="voice artifact materialization",
            )
            if artifact is not None:
                await self._delete_unregistered_voice_artifact(service, artifact)
            raise

    async def _register_voice_artifact(
        self,
        ctx: OutboxDispatchContext,
        service: VoiceArtifactService,
        artifact: VoiceArtifact,
    ) -> Any:
        return await self.outbox.register_outbox_voice_artifact(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            storage_relpath=artifact.storage_relpath,
            expected_sha256=artifact.sha256,
            expected_byte_size=artifact.byte_size,
            expires_at=utcnow() + service.ttl,
        )

    async def _delete_unregistered_voice_artifact(
        self,
        service: VoiceArtifactService,
        artifact: VoiceArtifact,
    ) -> None:
        cleanup = asyncio.create_task(asyncio.to_thread(service.release_artifact, artifact))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as cancellation:
            await _drain_cancelled_task(
                cleanup,
                cancellation,
                operation="unregistered voice artifact cleanup",
            )
        except Exception:
            logger.error("unregistered voice artifact cleanup failed")

    async def _cleanup_outbox_voice_artifacts(
        self,
        ctx: OutboxDispatchContext,
        *,
        reason: str,
    ) -> None:
        try:
            entries = tuple(
                await self.outbox.schedule_outbox_voice_artifact_cleanup(
                    ctx.profile_id,
                    ctx.instance_id,
                    ctx.item.outbox_id,
                    reason=reason,
                )
            )
        except Exception as exc:
            logger.error(
                "voice artifact cleanup scheduling failed (%s)",
                type(exc).__name__,
            )
            ctx.voice_artifact = None
            ctx.voice_cleanup_id = None
            return
        service = self.voice_artifact_service
        if not isinstance(service, VoiceArtifactService):
            raise TypeError("voice_artifact_service must be a VoiceArtifactService")
        for entry in entries:
            try:
                artifact = service.from_cleanup_entry(entry)
                await asyncio.to_thread(service.release_artifact, artifact)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.error(
                    "voice artifact cleanup deferred (%s)",
                    type(exc).__name__,
                )
                continue
            try:
                await self.outbox.complete_outbox_voice_artifact_cleanup(
                    ctx.profile_id,
                    ctx.instance_id,
                    ctx.item.outbox_id,
                    int(entry.cleanup_id),
                )
            except Exception as exc:
                logger.error(
                    "voice artifact cleanup completion deferred (%s)",
                    type(exc).__name__,
                )
        ctx.voice_artifact = None
        ctx.voice_cleanup_id = None

    async def _drain_voice_artifact_cleanup(
        self,
        ctx: OutboxDispatchContext,
        *,
        reason: str,
    ) -> None:
        if ctx.voice_artifact is None:
            return
        task = asyncio.create_task(
            self._cleanup_outbox_voice_artifacts(ctx, reason=reason),
            name="soulcore-outbox-voice-artifact-cleanup",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            await _drain_cancelled_task(
                task,
                cancellation,
                operation="outbox voice artifact cleanup",
            )


async def _drain_cancelled_task(
    task: asyncio.Task[_T],
    cancellation: asyncio.CancelledError,
    *,
    operation: str,
) -> _T | None:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                continue
            if task.cancelled():
                cancellation.add_note(f"{operation} was cancelled")
                return None
            try:
                return task.result()
            except BaseException as exc:
                cancellation.add_note(f"{operation} failed: {type(exc).__name__}")
                return None
        except BaseException as exc:
            cancellation.add_note(f"{operation} failed: {type(exc).__name__}")
            return None


__all__ = ["VoiceDeliveryMixin"]
