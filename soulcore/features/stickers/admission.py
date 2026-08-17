"""Sticker text finishing, promotion, deduplication and Check persistence."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from ...contracts.vision import VisionInspectionMode
from ..ai import safe_ai_failure_details
from ..media.ports import VisualCachePolicy
from ..media.service import image_fingerprints
from .check_pipeline import StickerCheckResult
from .contracts import (
    DESCRIPTION_CONTRACT_VERSION,
    StickerDescriptionContractError,
    StickerGenerationSpec,
    StickerTextFinishingDeferred,
)
from .domain import STICKER_CHECK_FAILURE_LIMIT, StickerCheckVerdict
from .policy import StickerRuntimeDisabled
from .text_modes import TEXT_MODE_INTEGRATED_TEXT, TEXT_MODE_NONE


class StickerAdmissionMixin:
    async def _complete_pending_text(
        self,
        *,
        candidate: Any,
        metadata: Mapping[str, Any],
        initial_vision: Any,
        control: Any,
    ) -> tuple[Any, Any] | None:
        """Finish a retained generated candidate after vision recovers."""

        mode, spec = self._pending_text_spec(metadata)
        task_id = int(metadata.get("collector_task_id") or 0)
        await self._progress(control, "TEXT_FINISH", detail="视觉恢复，继续完成表情包文字")
        if mode != TEXT_MODE_INTEGRATED_TEXT:
            raise ValueError(f"unsupported pending sticker text mode: {mode}")
        replacement_id = str(candidate.source_asset_id)
        if not self._ocr_matches(spec.meme_text, str(initial_vision.ocr_text or "")):
            try:
                replacement_id = await self._finish_integrated_text(
                    profile_id=candidate.profile_id,
                    instance_id=candidate.instance_id,
                    task_id=task_id,
                    source_asset_id=candidate.source_asset_id,
                    spec=spec,
                    initial_vision=initial_vision,
                    release_replaced_source=False,
                )
            except StickerTextFinishingDeferred as exc:
                replacement_id = exc.asset_id
                candidate = await self.repository.replace_sticker_candidate_source(
                    candidate.profile_id,
                    candidate.instance_id,
                    candidate.candidate_id,
                    replacement_id,
                    metadata_update={"text_finish_pending": True},
                    release_old=True,
                )
                await self._mark_waiting_check(
                    candidate,
                    reason="VISION_UNAVAILABLE",
                    failure_stage="TEXT_FINISH_VISION",
                    cause=exc.cause,
                )
                return None
            if not replacement_id:
                await self._record_text_finish_rejection(
                    candidate,
                    reason="INTEGRATED_TEXT_OCR_MISMATCH",
                    vision=initial_vision,
                    expected_text=spec.meme_text,
                )
                return None

        return await self._replace_and_inspect_finished_candidate(candidate, replacement_id)

    @staticmethod
    def _pending_text_spec(metadata: Mapping[str, Any]) -> tuple[str, StickerGenerationSpec]:
        mode = str(metadata.get("text_mode") or TEXT_MODE_NONE).upper()
        return mode, StickerGenerationSpec(
            str(metadata.get("generation_prompt") or ""),
            text_mode=mode,
            meme_text=str(metadata.get("meme_text") or ""),
            text_position=str(metadata.get("text_position") or ""),
            text_safe_zone=str(metadata.get("text_safe_zone") or ""),
            character_specific=bool(metadata.get("character_specific")),
            collection_intent=(
                metadata.get("collection_intent")
                if isinstance(metadata.get("collection_intent"), Mapping)
                else {}
            ),
        )

    async def _replace_and_inspect_finished_candidate(
        self, candidate: Any, replacement_id: str
    ) -> tuple[Any, Any] | None:
        await self._require_runtime_source(
            candidate.profile_id, candidate.instance_id, candidate.source_kind
        )
        candidate = await self.repository.replace_sticker_candidate_source(
            candidate.profile_id,
            candidate.instance_id,
            candidate.candidate_id,
            replacement_id,
            metadata_update={
                "text_finish_pending": False,
                "text_finished": True,
            },
            release_old=True,
        )
        try:
            final_vision = await self.visual_service.describe_asset(
                profile_id=candidate.profile_id,
                instance_id=candidate.instance_id,
                asset_id=candidate.source_asset_id,
                foreground=False,
                cache_policy=VisualCachePolicy.USE,
                inspection_mode=VisionInspectionMode.OBJECTIVE,
            )
            await self._require_runtime_source(
                candidate.profile_id, candidate.instance_id, candidate.source_kind
            )
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            await self._mark_waiting_check(
                candidate,
                reason="VISION_UNAVAILABLE",
                failure_stage="FINAL_PRODUCT_CHECK",
                cause=exc,
            )
            return None
        return candidate, final_vision

    async def _record_text_finish_rejection(
        self,
        candidate: Any,
        *,
        reason: str,
        vision: Any,
        expected_text: str,
    ) -> None:
        raw = {
            "accepted": False,
            "rejection_category": "TEXT_QUALITY",
            "compact_description": str(vision.visible_facts or "")[:100] or "文字成品不合格",
            "reason": reason,
        }
        result = StickerCheckResult(
            "REJECTED",
            "TEXT_QUALITY",
            compact_description=str(raw["compact_description"]),
        )
        await self._record_check(
            candidate,
            result,
            raw=raw,
            reason=reason,
            visible_text=str(vision.ocr_text or ""),
        )

    async def _identity_reference(
        self,
        profile_id: str,
        instance_id: str,
        *,
        scope: Any | None = None,
    ) -> dict[str, Any] | None:
        if scope is None:
            instance = await self.profiles.get_character_instance(profile_id, instance_id)
            if instance is None:
                return None
            scope = instance.scope
        record = await self.repository.get_character_identity_reference(profile_id, scope)
        if record is None:
            return None
        snapshot: dict[str, Any] = {
            "asset_id": record.asset_id,
            "storage_relpath": record.storage_relpath,
            "mime_type": record.mime_type,
            "file_extension": record.file_extension,
            "sha256": record.sha256,
            "byte_size": record.byte_size,
            "width": record.width,
            "height": record.height,
            "frame_count": record.frame_count,
            "duration_ms": record.duration_ms,
            "identity_description": record.identity_description,
        }
        image, _notes = await self.visual_service.resolve_identity_reference(snapshot)
        if image is None or not image.data:
            return None
        snapshot["data"] = bytes(image.data)
        return snapshot

    async def _promote_checked_candidate(
        self,
        *,
        candidate: Any,
        raw: Mapping[str, Any],
        result: Any,
        persona: str,
        visible_text: str,
        ocr_text: str,
        control: Any,
        fingerprints: tuple[str, str] | None = None,
    ) -> Any | None:
        await self._require_runtime_source(
            candidate.profile_id, candidate.instance_id, candidate.source_kind
        )
        await self._progress(control, "FINGERPRINT", detail="计算多帧视觉指纹与语义去重")
        if fingerprints is None:
            path = await self.visual_service.asset_file_path(
                profile_id=candidate.profile_id,
                instance_id=candidate.instance_id,
                asset_id=candidate.source_asset_id,
            )
            if path is None:
                return None
            fingerprints = await asyncio.to_thread(image_fingerprints, str(path))
        phash, dhash = fingerprints
        await self._require_runtime_source(
            candidate.profile_id, candidate.instance_id, candidate.source_kind
        )
        await self._progress(control, "PROMOTING", detail="提升为长期正式表情包资产")
        item, _ = await self.media_storage.promote_sticker_candidate(
            candidate.profile_id,
            candidate.instance_id,
            candidate.candidate_id,
            compact_name=str(raw.get("compact_name") or "")[:80],
            compact_description=result.compact_description,
            visible_text=visible_text[:2000],
            ocr_text=ocr_text[:2000],
            usage_type=result.usage_type.value,
            vibe_tags=tuple(raw.get("vibe_tags") or ())[:20],
            search_keywords=tuple(raw.get("search_keywords") or ())[:20],
            semantic_key=result.semantic_key,
            emotion=result.emotion,
            speech_act=result.speech_act,
            intensity=result.intensity,
            persona_score=result.persona_score,
            phash=phash,
            dhash=dhash,
            frame_hashes=tuple(phash.split(".")),
            visual_group=phash.split(".", 1)[0],
            metadata={
                "persona_bound": bool(raw.get("persona_bound")),
                "persona_fingerprint": (
                    self.persona_fingerprint(persona) if raw.get("persona_bound") else ""
                ),
                "visible_text": visible_text[:2000],
                "search_keywords": list(raw.get("search_keywords") or ())[:20],
                "description_version": str(
                    raw.get("description_version") or DESCRIPTION_CONTRACT_VERSION
                ),
                "text_mode": str(candidate.metadata.get("text_mode") or TEXT_MODE_NONE),
                "collection_intent": (
                    dict(candidate.metadata.get("collection_intent") or {})
                    if isinstance(candidate.metadata.get("collection_intent"), Mapping)
                    else {}
                ),
                "structured_description": {
                    "objective_scene": str(raw.get("objective_scene") or "")[:5000],
                    "social_impression": str(raw.get("vision_social_impression") or "")[:80],
                },
            },
        )
        return item

    async def _mark_waiting_check(
        self,
        candidate: Any,
        *,
        reason: str,
        failure_stage: str,
        cause: BaseException | None = None,
    ) -> None:
        attempt = int(candidate.retry_count) + 1
        diagnostics = safe_ai_failure_details(cause) if cause is not None else {}
        reason_code = re.sub(r"[^0-9A-Z_:.-]", "_", str(reason or "WAITING_CHECK").upper())[:100]
        safe_reason = ":".join(
            value
            for value in (
                reason_code,
                str(diagnostics.get("error_code") or ""),
                str(diagnostics.get("exception_type") or ""),
                str(diagnostics.get("cause_type") or ""),
            )
            if value
        )[:500]
        if attempt >= STICKER_CHECK_FAILURE_LIMIT:
            await self.repository.quarantine_sticker_candidate(
                candidate.profile_id,
                candidate.instance_id,
                candidate.candidate_id,
                reason=safe_reason,
                failure_stage=failure_stage,
                increment_retry=True,
                diagnostics=diagnostics,
            )
            return
        await self.repository.mark_sticker_candidate_waiting_check(
            candidate.profile_id,
            candidate.instance_id,
            candidate.candidate_id,
            reason=safe_reason,
            failure_stage=failure_stage,
            next_retry_at=datetime.now(UTC) + self._waiting_retry_delay(attempt),
            recoverable=True,
            increment_retry=True,
            diagnostics=diagnostics,
        )

    async def _quarantine_check_failure(
        self,
        candidate: Any,
        *,
        reason: str,
        failure_stage: str,
        cause: BaseException | None = None,
    ) -> None:
        diagnostics = safe_ai_failure_details(cause) if cause is not None else {}
        reason_code = re.sub(r"[^0-9A-Z_:.-]", "_", str(reason or "STICKER_CHECK_FAILED").upper())[
            :100
        ]
        safe_reason = ":".join(
            value
            for value in (
                reason_code,
                str(diagnostics.get("error_code") or ""),
                str(diagnostics.get("exception_type") or ""),
                str(diagnostics.get("cause_type") or ""),
            )
            if value
        )[:500]
        await self.repository.quarantine_sticker_candidate(
            candidate.profile_id,
            candidate.instance_id,
            candidate.candidate_id,
            reason=safe_reason,
            failure_stage=failure_stage,
            increment_retry=True,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _waiting_retry_delay(attempt: int) -> timedelta:
        if attempt <= 1:
            return timedelta(minutes=15)
        if attempt == 2:
            return timedelta(hours=1)
        return timedelta(hours=6)

    @staticmethod
    def _vision_explicitly_safe(vision: Any) -> bool:
        return vision.safe is True

    @classmethod
    def _minimal_check_from_vision(cls, vision: Any) -> dict[str, Any]:
        evidence = cls._vision_description_evidence(vision)
        scene = cls._description_text(
            evidence.get("scene_description") or evidence.get("visible_facts"),
            5000,
        )
        social_impression = cls._description_text(evidence.get("social_impression"), 80)
        if not social_impression:
            raise StickerDescriptionContractError(
                "DESCRIPTION_SOCIAL_IMPRESSION_MISSING",
                "vision did not provide a reliable social impression",
            )
        description = f"{social_impression}；{scene}" if scene else social_impression
        return {
            "accepted": True,
            "rejection_category": "",
            "reason": "视觉证据已提供明确交流观感且未配置额外接纳策略",
            "compact_name": social_impression[:40],
            "compact_description": description,
            "visible_text": cls._description_text(evidence.get("visible_text"), 2000),
            "usage_contexts": [],
            "vibe_tags": [social_impression],
            "emotion": "",
            "speech_act": "",
            "intensity": 0,
            "objective_scene": scene,
            "vision_social_impression": social_impression,
        }

    @staticmethod
    def _ocr_matches(expected: str, actual: str) -> bool:
        def normalize(value: str) -> str:
            return re.sub(r"[^\w\u4e00-\u9fff]", "", str(value or "")).lower()

        wanted = normalize(expected)
        found = normalize(actual)
        return bool(wanted and wanted in found)

    @staticmethod
    def _semantic_key(value: str) -> str:
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").lower())[:120]
        return normalized

    async def _release_source_asset(self, asset_id: str, *, reason: str) -> None:
        try:
            await self.repository.mark_media_asset_release_pending(asset_id, reason=reason)
        except (KeyError, ValueError):
            return

    async def _discard_redundant_candidate(self, candidate: Any) -> None:
        """Remove a late eligible candidate after another task claimed the slot."""

        try:
            await self.repository.delete_sticker_candidate(
                candidate.profile_id,
                candidate.instance_id,
                candidate.candidate_id,
            )
            return
        except (KeyError, ValueError):
            pass
        await self._release_source_asset(
            str(candidate.source_asset_id),
            reason="RUN_ACCEPT_SLOT_ALREADY_CLAIMED",
        )

    async def _record_check(
        self,
        candidate: Any,
        result: Any,
        *,
        reason: str,
        raw: Mapping[str, Any] | None = None,
        visible_text: str = "",
        details_update: Mapping[str, Any] | None = None,
    ) -> None:
        verdict = {
            "ACCEPTED": StickerCheckVerdict.ACCEPT,
            "REJECTED": StickerCheckVerdict.REJECT,
        }.get(result.verdict, StickerCheckVerdict.QUARANTINE)
        source = dict(raw or {})
        details = self._check_details(source, result)
        details.update(dict(details_update or {}))
        await self.repository.record_sticker_check(
            candidate.profile_id,
            candidate.instance_id,
            candidate.candidate_id,
            verdict=verdict,
            compact_name=str(source.get("compact_name") or "")[:80],
            compact_description=result.compact_description,
            visible_text=visible_text[:2000],
            usage_type=result.usage_type.value,
            semantic_key=result.semantic_key,
            emotion=result.emotion,
            speech_act=result.speech_act,
            intensity=result.intensity,
            persona_score=result.persona_score,
            reason=reason[:500],
            details=details,
        )

    @staticmethod
    def _check_details(source: Mapping[str, Any], result: Any) -> dict[str, Any]:
        return {
            "reason_code": result.reason_code,
            "description_version": str(source.get("description_version") or ""),
            "search_keywords": list(source.get("search_keywords") or ())[:20],
            "usage_contexts": list(source.get("usage_contexts") or ())[:20],
            "structured_description": {
                "objective_scene": str(source.get("objective_scene") or "")[:5000],
                "social_impression": str(source.get("vision_social_impression") or "")[:80],
            },
        }

    async def promote_staged_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
    ) -> tuple[Any, bool]:
        """Promote one persisted READY candidate using its accepted Check evidence."""

        candidate = await self.repository.get_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
        )
        if candidate is None or str(candidate.status.value) != "READY":
            raise ValueError("sticker intake candidate is no longer ready")
        await self._require_runtime_source(profile_id, instance_id, candidate.source_kind)
        checks = await self.repository.list_sticker_checks(
            profile_id,
            instance_id,
            candidate_id=candidate_id,
            limit=1,
        )
        if not checks or checks[0].verdict != StickerCheckVerdict.ACCEPT:
            raise ValueError("sticker intake candidate has no accepted Check")
        check = checks[0]
        details = dict(check.details or {})
        phash, dhash = await self._staged_candidate_fingerprints(candidate, details)
        return await self._promote_staged_check(
            profile_id,
            instance_id,
            candidate_id,
            check,
            details,
            phash=phash,
            dhash=dhash,
        )

    async def _staged_candidate_fingerprints(
        self,
        candidate: Any,
        details: Mapping[str, Any],
    ) -> tuple[str, str]:
        phash = self._staged_text(details.get("phash"))
        dhash = self._staged_text(details.get("dhash"))
        if phash and dhash:
            return phash, dhash
        fingerprints = await self._candidate_fingerprints(candidate)
        if fingerprints is None:
            raise ValueError("sticker intake candidate fingerprint is unavailable")
        return fingerprints

    async def _promote_staged_check(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        check: Any,
        details: Mapping[str, Any],
        *,
        phash: str,
        dhash: str,
    ) -> tuple[Any, bool]:
        item, created = await self.media_storage.promote_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
            compact_name=self._staged_text(check.compact_name, 80),
            compact_description=self._staged_text(check.compact_description, 100),
            visible_text=self._staged_text(check.visible_text, 2000),
            ocr_text=self._staged_text(details.get("ocr_text"), 2000),
            usage_type=str(check.usage_type.value),
            vibe_tags=self._staged_items(details.get("vibe_tags"), 20),
            search_keywords=self._staged_items(details.get("search_keywords"), 20),
            semantic_key=self._staged_text(check.semantic_key, 200),
            emotion=self._staged_text(check.emotion, 48),
            speech_act=self._staged_text(check.speech_act, 48),
            intensity=self._staged_int(check.intensity),
            persona_score=self._staged_float(check.persona_score),
            phash=phash,
            dhash=dhash,
            frame_hashes=self._staged_frame_hashes(details, phash),
            visual_group=self._staged_visual_group(details, phash),
            metadata=self._staged_promotion_metadata(check, details),
        )
        return item, bool(created)

    @staticmethod
    def _staged_text(value: Any, limit: int | None = None) -> str:
        text = str(value or "")
        return text if limit is None else text[:limit]

    @staticmethod
    def _staged_items(value: Any, limit: int) -> tuple[Any, ...]:
        return tuple(value or ())[:limit]

    @staticmethod
    def _staged_int(value: Any) -> int:
        return int(value or 0)

    @staticmethod
    def _staged_float(value: Any) -> float:
        return float(value or 0.0)

    @classmethod
    def _staged_frame_hashes(
        cls,
        details: Mapping[str, Any],
        phash: str,
    ) -> tuple[Any, ...]:
        stored = cls._staged_items(details.get("frame_hashes"), 20)
        return stored if stored else tuple(phash.split("."))

    @classmethod
    def _staged_visual_group(cls, details: Mapping[str, Any], phash: str) -> str:
        stored = cls._staged_text(details.get("visual_group"))
        return stored if stored else phash.split(".", 1)[0]

    @classmethod
    def _staged_promotion_metadata(
        cls,
        check: Any,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        structured = details.get("structured_description")
        structured_mapping = structured if isinstance(structured, Mapping) else {}
        return {
            "persona_bound": bool(details.get("persona_bound")),
            "persona_fingerprint": cls._staged_text(details.get("persona_fingerprint")),
            "visible_text": cls._staged_text(check.visible_text, 2000),
            "search_keywords": list(cls._staged_items(details.get("search_keywords"), 20)),
            "description_version": cls._staged_text(details.get("description_version")),
            "text_mode": cls._staged_text(details.get("text_mode")) or TEXT_MODE_NONE,
            "structured_description": {
                "objective_scene": cls._staged_text(
                    structured_mapping.get("objective_scene"),
                    5000,
                ),
                "social_impression": cls._staged_text(
                    structured_mapping.get("social_impression"),
                    80,
                ),
            },
        }


__all__ = ["StickerAdmissionMixin"]
