from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from typing import Any

from ...contracts.ai_models import (
    AICapabilityEffect,
    AICapabilityRequest,
    AIExecutionMode,
    AIImageContent,
    AIRetryPolicy,
    AIVisionDescription,
    AIWorkPurpose,
)
from ...contracts.runtime_limits import DURABLE_AI_MAX_ATTEMPTS
from ...contracts.vision import (
    VisionInspectionMode,
    VisionSequenceKind,
    VisionTextState,
    build_history_projection,
)
from ...shared.event_log import record_event
from ..ai.service import safe_ai_failure_details
from .domain import MediaProjectionStatus
from .visual_cache import (
    VISUAL_OBSERVATION_CONTRACT_VERSION,
    CachedVisualObservation,
    VisualCachePolicy,
)


class VisualObservationServiceMixin:
    """Persistent objective-vision reuse for ``VisualExpressionService``."""

    _background: set[asyncio.Task[Any]]
    _background_scopes: dict[asyncio.Task[Any], tuple[str, str]]
    _background_accepting: bool
    _background_blocked_profiles: set[str]
    _background_blocked_instances: set[tuple[str, str]]

    async def _asset_is_model_visible(
        self, profile_id: str, instance_id: str, asset_id: str
    ) -> bool:
        checker = getattr(self.media, "asset_is_model_visible", None)
        if not callable(checker):
            return True
        return bool(await checker(profile_id, instance_id, asset_id))

    def describe_in_background(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset_ids: Sequence[str],
        cache_policy: VisualCachePolicy | str = VisualCachePolicy.USE,
        inspection_mode: VisionInspectionMode | str = VisionInspectionMode.OBJECTIVE,
    ) -> None:
        scope = (str(profile_id), str(instance_id))
        if (
            not self._background_accepting
            or scope[0] in self._background_blocked_profiles
            or scope in self._background_blocked_instances
        ):
            return
        policy = VisualCachePolicy(str(cache_policy))
        mode = VisionInspectionMode(str(inspection_mode))

        async def run() -> None:
            if not await self.runtime_gate.is_enabled(profile_id, instance_id):
                return
            visible_ids = [
                asset_id
                for asset_id in dict.fromkeys(asset_ids)
                if await self._asset_is_model_visible(profile_id, instance_id, str(asset_id))
            ]
            if not visible_ids:
                return
            await self.ai_manager.submit_task(
                profile_id=profile_id,
                instance_id=instance_id,
                task_type="VISION_DESCRIPTION",
                capability="vision.describe",
                payload={
                    "asset_ids": visible_ids,
                    "cache_policy": policy.value,
                    "inspection_mode": mode.value,
                },
                priority=-20,
                idempotency_key=(
                    f"vision-description:{instance_id}:{mode.value}:{policy.value}:"
                    + ":".join(sorted(set(visible_ids)))
                ),
                recovery_policy="RESTART_SAFE",
                max_attempts=DURABLE_AI_MAX_ATTEMPTS,
            )

        task = asyncio.create_task(run(), name=f"soulcore-vision:{instance_id}")
        self._background.add(task)
        self._background_scopes[task] = scope

        def discard(completed: asyncio.Task[Any]) -> None:
            self._background.discard(completed)
            self._background_scopes.pop(completed, None)

        task.add_done_callback(discard)

    async def cancel_instance_background(self, profile_id: str, instance_id: str) -> None:
        """Drain pre-reset submitters so they cannot recreate deleted work."""

        async with self.quiesce_instance_background(profile_id, instance_id):
            pass

    @asynccontextmanager
    async def quiesce_instance_background(
        self,
        profile_id: str,
        instance_id: str,
    ) -> AsyncIterator[None]:
        """Block visual submitters for one instance throughout a reset."""

        scope = (str(profile_id), str(instance_id))
        self._background_blocked_instances.add(scope)
        try:
            await self._drain_background_tasks(profile_id=scope[0], instance_id=scope[1])
            yield
        finally:
            await self._drain_background_tasks(profile_id=scope[0], instance_id=scope[1])
            self._background_blocked_instances.discard(scope)

    async def cancel_profile_background(self, profile_id: str) -> None:
        """Drain every visual submitter owned by a profile before profile deletion."""

        profile_key = str(profile_id)
        self._background_blocked_profiles.add(profile_key)
        try:
            await self._drain_background_tasks(profile_id=profile_key)
        finally:
            self._background_blocked_profiles.discard(profile_key)

    async def _drain_background_tasks(
        self,
        *,
        profile_id: str | None = None,
        instance_id: str | None = None,
    ) -> None:
        while True:
            tasks = [
                task
                for task, owner in tuple(self._background_scopes.items())
                if not task.done()
                and (profile_id is None or owner[0] == profile_id)
                and (instance_id is None or owner[1] == instance_id)
            ]
            if not tasks:
                return
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def execute_description_task(
        self, task: Mapping[str, Any], control: Any
    ) -> Mapping[str, Any]:
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        await self.runtime_gate.require_enabled(profile_id, instance_id)
        payload = dict(task.get("input") or {})
        cache_policy = VisualCachePolicy(
            str(payload.get("cache_policy") or VisualCachePolicy.USE.value)
        )
        inspection_mode = VisionInspectionMode(
            str(payload.get("inspection_mode") or VisionInspectionMode.OBJECTIVE.value)
        )
        completed: list[str] = []
        for asset_id in payload.get("asset_ids") or []:
            await control.check_control()
            await self.describe_asset(
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=str(asset_id),
                foreground=False,
                cache_policy=cache_policy,
                inspection_mode=inspection_mode,
            )
            completed.append(str(asset_id))
        return {"asset_ids": completed, "count": len(completed)}

    async def describe_assets(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset_ids: Sequence[str],
        foreground: bool = True,
        cache_policy: VisualCachePolicy | str = VisualCachePolicy.USE,
        inspection_mode: VisionInspectionMode | str = VisionInspectionMode.OBJECTIVE,
    ) -> str:
        await self.runtime_gate.require_enabled(profile_id, instance_id)
        descriptions: list[str] = []
        for asset_id in asset_ids:
            result = await self.describe_asset(
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=asset_id,
                foreground=foreground,
                cache_policy=cache_policy,
                inspection_mode=inspection_mode,
            )
            descriptions.append(
                build_history_projection(
                    result.visible_facts,
                    result.ocr_text,
                    result.visible_text_state,
                    social_impression=result.social_impression,
                )
                or result.visible_facts
            )
        return "\n".join(f"[图片{index}] {value}" for index, value in enumerate(descriptions, 1))

    async def describe_asset(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        foreground: bool = True,
        cache_policy: VisualCachePolicy | str = VisualCachePolicy.USE,
        inspection_mode: VisionInspectionMode | str = VisionInspectionMode.OBJECTIVE,
    ) -> AIVisionDescription:
        await self.runtime_gate.require_enabled(profile_id, instance_id)
        policy = VisualCachePolicy(str(cache_policy))
        mode = VisionInspectionMode(str(inspection_mode))
        asset = await self.media.get_media_asset(
            asset_id, profile_id=profile_id, instance_id=instance_id
        )
        persist_media_projection = asset is not None
        if persist_media_projection and not await self._asset_is_model_visible(
            profile_id, instance_id, asset_id
        ):
            raise ValueError("source message is unavailable")
        if asset is None and self.stickers is not None:
            asset = await self.stickers.get_accessible_sticker_asset(
                asset_id,
                profile_id=profile_id,
                instance_id=instance_id,
            )
        if asset is None:
            raise ValueError("media or sticker asset is unavailable")
        if (
            mode is VisionInspectionMode.OBJECTIVE
            and policy is VisualCachePolicy.USE
            and persist_media_projection
        ):
            cached = await self._cached_description(
                profile_id=profile_id,
                instance_id=instance_id,
                asset=asset,
                asset_id=asset_id,
            )
            if cached is not None:
                return cached
            key = (
                profile_id,
                instance_id,
                str(asset.sha256),
                str(VISUAL_OBSERVATION_CONTRACT_VERSION),
            )
            async with self._observation_singleflight(key):
                cached = await self._cached_description(
                    profile_id=profile_id,
                    instance_id=instance_id,
                    asset=asset,
                    asset_id=asset_id,
                )
                if cached is not None:
                    return cached
                return await self._invoke_description(
                    profile_id=profile_id,
                    instance_id=instance_id,
                    asset=asset,
                    asset_id=asset_id,
                    foreground=foreground,
                    cache_policy=policy,
                    inspection_mode=mode,
                    persist_media_projection=True,
                )
        return await self._invoke_description(
            profile_id=profile_id,
            instance_id=instance_id,
            asset=asset,
            asset_id=asset_id,
            foreground=foreground,
            cache_policy=policy,
            inspection_mode=mode,
            persist_media_projection=persist_media_projection,
        )

    async def _invoke_description(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset: Any,
        asset_id: str,
        foreground: bool,
        cache_policy: VisualCachePolicy,
        inspection_mode: VisionInspectionMode,
        persist_media_projection: bool,
    ) -> AIVisionDescription:
        if persist_media_projection and not await self._asset_is_model_visible(
            profile_id, instance_id, asset_id
        ):
            raise ValueError("source message is unavailable")
        if not asset.storage_relpath:
            raise ValueError("media asset is unavailable")
        request = await self._description_request(
            asset,
            asset_id,
            foreground,
            inspection_mode=inspection_mode,
            profile_id=profile_id,
            instance_id=instance_id,
            projection_version=(
                int(asset.current_projection_version) if persist_media_projection else None
            ),
        )
        try:
            invocation = await self.ai_manager.invoke_capability(request)
            output = invocation.output
            if not isinstance(output, AIVisionDescription):
                raise TypeError("vision adapter returned an invalid result")
            output = self._normalize_objective_description(output)
            if persist_media_projection and not await self._asset_is_model_visible(
                profile_id, instance_id, asset_id
            ):
                raise ValueError("source message is unavailable")
            if persist_media_projection:
                await self._commit_description(
                    profile_id,
                    instance_id,
                    asset_id,
                    backend_id=invocation.backend_id,
                    output=output,
                    cache_status=self._cache_status(cache_policy),
                )
            if self._should_cache(persist_media_projection, cache_policy):
                observation = CachedVisualObservation.from_vision(
                    output,
                    backend_id=invocation.backend_id,
                )
                if observation is not None:
                    await self._save_cached_observation(
                        profile_id=profile_id,
                        instance_id=instance_id,
                        asset_id=asset_id,
                        sha256=str(asset.sha256),
                        observation=observation,
                    )
            return output
        except Exception as exc:
            if persist_media_projection:
                await self._fail_description(profile_id, instance_id, asset_id, exc)
            raise

    @staticmethod
    def _normalize_objective_description(output: AIVisionDescription) -> AIVisionDescription:
        description = str(output.visible_facts or "").strip()
        if not description:
            raise ValueError("vision adapter returned an empty objective description")
        state = VisionTextState(str(output.visible_text_state))
        content_text = str(output.ocr_text or "").strip()
        if state is VisionTextState.TRANSCRIBED and not content_text:
            raise ValueError("vision adapter declared transcribed text without text")
        if state is VisionTextState.NO_TEXT and content_text:
            raise ValueError("vision adapter declared no text while returning text")
        raw = dict(output.raw) if isinstance(output.raw, Mapping) else {}
        safe = raw.get("safe", output.safe)
        if not isinstance(safe, bool):
            raise ValueError("vision adapter returned no safety judgment")
        return replace(
            output,
            visible_facts=description,
            ocr_text=content_text,
            visible_text_state=state.value,
            safe=safe,
            raw={"safe": safe},
        )

    @staticmethod
    def _cache_status(policy: VisualCachePolicy) -> str:
        statuses = {
            VisualCachePolicy.REFRESH: "refresh",
            VisualCachePolicy.BYPASS: "bypass",
        }
        return statuses.get(policy, "miss")

    @staticmethod
    def _should_cache(
        persist_media_projection: bool,
        policy: VisualCachePolicy,
    ) -> bool:
        return persist_media_projection and policy is not VisualCachePolicy.BYPASS

    async def _cached_description(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset: Any,
        asset_id: str,
    ) -> AIVisionDescription | None:
        if not await self._asset_is_model_visible(profile_id, instance_id, asset_id):
            return None
        try:
            hit = await self.media.get_cached_visual_observation(
                profile_id,
                instance_id,
                str(asset.sha256),
                contract_version=VISUAL_OBSERVATION_CONTRACT_VERSION,
            )
        except Exception as exc:
            await self._record_cache_issue(
                profile_id,
                instance_id,
                asset_id,
                stage="read",
                exc=exc,
            )
            return None
        if not isinstance(hit, CachedVisualObservation):
            return None
        output = hit.description()
        await self._commit_description(
            profile_id,
            instance_id,
            asset_id,
            backend_id=hit.backend_id,
            output=output,
            cache_status="hit",
        )
        return output

    async def _save_cached_observation(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        sha256: str,
        observation: CachedVisualObservation,
    ) -> None:
        try:
            pruned = await self.media.save_cached_visual_observation(
                profile_id,
                instance_id,
                asset_id,
                sha256,
                observation,
                contract_version=VISUAL_OBSERVATION_CONTRACT_VERSION,
            )
        except Exception as exc:
            await self._record_cache_issue(
                profile_id,
                instance_id,
                asset_id,
                stage="write",
                exc=exc,
            )
            return
        if int(pruned or 0) > 0:
            with suppress(Exception):
                await record_event(
                    self.event_log,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    level="INFO",
                    category="vision.cache",
                    message="视觉观察缓存已按最近使用顺序收缩",
                    details={"evicted": int(pruned), "remaining": 450},
                )

    async def _record_cache_issue(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        *,
        stage: str,
        exc: BaseException,
    ) -> None:
        with suppress(Exception):
            await record_event(
                self.event_log,
                profile_id=profile_id,
                instance_id=instance_id,
                level="WARNING",
                category="vision.cache",
                message="视觉观察缓存不可用，已继续正常识别流程",
                details={
                    "asset_id": asset_id,
                    "stage": str(stage),
                    "error": type(exc).__name__,
                },
            )

    @asynccontextmanager
    async def _observation_singleflight(self, key: tuple[str, ...]) -> AsyncIterator[None]:
        async with self._observation_locks_guard:
            lock = self._observation_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._observation_locks[key] = lock
            self._observation_lock_users[key] = self._observation_lock_users.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            async with self._observation_locks_guard:
                remaining = self._observation_lock_users.get(key, 1) - 1
                if remaining <= 0:
                    self._observation_lock_users.pop(key, None)
                    if self._observation_locks.get(key) is lock:
                        self._observation_locks.pop(key, None)
                else:
                    self._observation_lock_users[key] = remaining

    async def _description_request(
        self,
        asset: Any,
        asset_id: str,
        foreground: bool,
        *,
        inspection_mode: VisionInspectionMode,
        profile_id: str,
        instance_id: str,
        projection_version: int | None,
    ) -> AICapabilityRequest:
        data = await self.read_verified_media_bytes(
            relative_path=asset.storage_relpath,
            byte_size=asset.byte_size,
            sha256=asset.sha256,
        )
        payloads = await asyncio.to_thread(self.vision_payloads, data, asset.mime_type)
        animated = int(getattr(asset, "frame_count", 1) or 1) > 1
        contact_sheet = animated and len(payloads) == 1
        sequence_kind = (
            VisionSequenceKind.ANIMATION_CONTACT_SHEET
            if contact_sheet
            else VisionSequenceKind.GIF_REPRESENTATIVE_FRAMES
            if animated
            else VisionSequenceKind.SINGLE_IMAGE
        )
        return AICapabilityRequest(
            invocation_id=uuid.uuid4().hex,
            capability="vision.describe",
            work_purpose=AIWorkPurpose.IMAGE_UNDERSTANDING,
            logical_stage_key=f"vision:{asset_id}:{uuid.uuid4().hex}",
            payload={
                "images": [
                    AIImageContent(
                        mime,
                        data=payload,
                        asset_id=asset_id,
                        metadata={
                            "representative_frame": index,
                            "animation_contact_sheet": contact_sheet,
                        },
                    )
                    for index, (payload, mime) in enumerate(payloads)
                ],
                "sequence_kind": sequence_kind.value,
                "inspection_mode": inspection_mode.value,
            },
            effect=AICapabilityEffect.READ_ONLY,
            execution_mode=(
                AIExecutionMode.FOREGROUND_SYNC
                if foreground
                else AIExecutionMode.BACKGROUND_DURABLE
            ),
            profile_id=profile_id,
            instance_id=instance_id,
            owner_kind="vision_description",
            owner_id=asset_id,
            idempotency_key=(
                f"vision:{asset_id}:{projection_version + 1}"
                if projection_version is not None
                else f"vision:{asset_id}:sticker:{uuid.uuid4().hex}"
            ),
            retry_policy=AIRetryPolicy(max_attempts=3),
        )

    async def _commit_description(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        *,
        backend_id: str,
        output: AIVisionDescription,
        cache_status: str,
    ) -> None:
        await self.runtime_gate.require_enabled(profile_id, instance_id)
        await self.media.save_media_projection(
            asset_id,
            status=MediaProjectionStatus.READY,
            visible_facts=output.visible_facts,
            history_projection=(
                build_history_projection(
                    output.visible_facts,
                    output.ocr_text,
                    output.visible_text_state,
                    social_impression=output.social_impression,
                )
                or output.visible_facts
            ),
            ocr_text=output.ocr_text,
            backend_id=backend_id,
            model_id=output.model,
        )
        try:
            await self.media.mark_media_release_if_already_summarized(asset_id)
        except Exception as exc:
            with suppress(Exception):
                await record_event(
                    self.event_log,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    level="WARNING",
                    category="vision.release",
                    message="视觉投影已提交，媒体释放记账等待后台恢复",
                    details={
                        "asset_id": asset_id,
                        "exception_type": type(exc).__name__,
                    },
                )
        with suppress(Exception):
            await record_event(
                self.event_log,
                profile_id=profile_id,
                instance_id=instance_id,
                level="INFO",
                category="vision.describe",
                message="媒体视觉投影已提交",
                details={
                    "asset_id": asset_id,
                    "backend_id": backend_id,
                    "cache_status": cache_status,
                    "observation_kind": "OBJECTIVE",
                },
            )

    async def _fail_description(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        exc: Exception,
    ) -> None:
        if not await self._asset_is_model_visible(profile_id, instance_id, asset_id):
            return
        diagnostics = safe_ai_failure_details(exc)
        values = (
            str(diagnostics.get("error_code") or "VISION_UNAVAILABLE"),
            str(diagnostics.get("exception_type") or ""),
            str(diagnostics.get("cause_type") or ""),
        )
        error_summary = ":".join(value for value in values if value)[:500]
        await self.media.save_media_projection(
            asset_id,
            status=MediaProjectionStatus.FAILED,
            error=error_summary,
            backend_id=str(diagnostics.get("backend_id") or ""),
            model_id=str(diagnostics.get("model_id") or ""),
        )
        await record_event(
            self.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="ERROR",
            category="vision.describe",
            message="媒体视觉投影失败",
            details={"asset_id": asset_id, **diagnostics},
        )


__all__ = ["VisualObservationServiceMixin"]
