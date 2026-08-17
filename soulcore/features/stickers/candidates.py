"""Sticker candidate acquisition, Check and administrator acceptance."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

from ...contracts.vision import VisionInspectionMode
from ..character_context import CharacterRunContext, CharacterRunScope
from ..media.ports import VisualCachePolicy
from ..media.service import image_fingerprints
from .check_pipeline import StickerCheckPipeline, StickerCheckResult
from .contracts import (
    DESCRIPTION_CONTRACT_VERSION,
    StickerDescriptionContractError,
)
from .domain import StickerCandidateSource
from .policy import StickerRuntimeDisabled
from .text_modes import TEXT_MODE_INTEGRATED_TEXT, TEXT_MODE_NONE


class StickerCandidateAdminMixin:
    async def admin_accept_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reason: str = "",
        control: Any | None = None,
    ) -> Any:
        """Administratively accept a candidate without AI/persona judgement.

        This is deliberately a service-level operation rather than a direct
        status write.  It still verifies ownership, metadata, retained bytes
        and decodability, and then uses the normal bounded duplicate/capacity
        promotion transaction.
        """

        candidate, existing = await self._admin_candidate(profile_id, instance_id, candidate_id)
        await self._require_runtime_source(profile_id, instance_id, candidate.source_kind)
        if existing is not None:
            return existing
        phash, dhash = await self._admin_asset_fingerprints(profile_id, instance_id, candidate)
        description_payload = await self._admin_description(
            profile_id,
            instance_id,
            candidate,
            control=control,
        )
        await self._progress(control, "ADMIN_ACCEPT", detail="管理员覆盖语义与人设判断")
        latest = await self._record_admin_override(
            profile_id, instance_id, candidate_id, reason, description_payload
        )
        return await self._promote_admin_candidate(
            profile_id,
            instance_id,
            candidate_id,
            candidate,
            latest,
            description_payload,
            phash,
            dhash,
        )

    async def _admin_candidate(
        self, profile_id: str, instance_id: str, candidate_id: str
    ) -> tuple[Any, Any | None]:
        candidate = await self.repository.get_sticker_candidate(
            profile_id, instance_id, candidate_id
        )
        if candidate is None:
            raise KeyError((profile_id, instance_id, candidate_id))
        item_id = str(candidate.accepted_item_id or "")
        if not item_id:
            return candidate, None
        existing = await self.repository.get_sticker_item(profile_id, instance_id, item_id)
        return candidate, existing

    async def _admin_asset_fingerprints(
        self, profile_id: str, instance_id: str, candidate: Any
    ) -> tuple[str, str]:
        asset = await self.media.get_media_asset(
            candidate.source_asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        deterministic = StickerCheckPipeline.check_deterministic(
            source_kind=candidate.source_kind.value,
            owner_matches=asset is not None,
            decoded_image=bool(asset and asset.storage_relpath),
            mime_type=str(asset.mime_type if asset is not None else ""),
            byte_size=int(asset.byte_size if asset is not None else 0),
            width=int(asset.width if asset is not None else 0),
            height=int(asset.height if asset is not None else 0),
        )
        if deterministic is not None:
            raise ValueError(f"ADMIN_ACCEPT_MEDIA_INVALID:{deterministic.reason_code}")
        path = await self.visual_service.asset_file_path(
            profile_id=profile_id,
            instance_id=instance_id,
            asset_id=candidate.source_asset_id,
        )
        if path is None:
            raise ValueError("ADMIN_ACCEPT_MEDIA_INVALID:MISSING_FILE")
        return image_fingerprints(str(path))

    async def _admin_description(
        self,
        profile_id: str,
        instance_id: str,
        candidate: Any,
        *,
        control: Any | None = None,
    ) -> dict[str, Any]:
        await self._require_runtime_source(profile_id, instance_id, candidate.source_kind)
        try:
            vision = await self.visual_service.describe_asset(
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=candidate.source_asset_id,
                foreground=True,
                cache_policy=VisualCachePolicy.USE,
                inspection_mode=VisionInspectionMode.STICKER_QUALITY,
            )
            await self._require_runtime_source(profile_id, instance_id, candidate.source_kind)
            self._require_admin_vision_quality(candidate, vision)
            requirements = await self._admin_requirements(profile_id, instance_id)
            self._require_admin_source_marker_policy(vision, requirements)
            return await self._admin_description_payload(
                profile_id,
                instance_id,
                candidate,
                vision,
                requirements,
                control,
            )
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            raise StickerDescriptionContractError(
                "ADMIN_ACCEPT_DESCRIPTION_UNAVAILABLE",
                "administrator override still requires a complete visual description",
                cause=exc,
            ) from exc

    def _require_admin_vision_quality(self, candidate: Any, vision: Any) -> None:
        if vision.safe is not True:
            raise ValueError("ADMIN_ACCEPT_UNSAFE_CONTENT")
        if str(vision.visible_text_state) == "UNCLEAR_TEXT":
            raise ValueError("ADMIN_ACCEPT_TEXT_QUALITY")
        expected_text = str(candidate.metadata.get("meme_text") or "").strip()
        if expected_text and not self._ocr_matches(expected_text, str(vision.ocr_text or "")):
            raise ValueError("ADMIN_ACCEPT_TEXT_QUALITY")

    async def _admin_requirements(self, profile_id: str, instance_id: str) -> str:
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        config = (
            await self.repository.get_sticker_config(profile_id, instance.scope)
            if instance is not None
            else None
        )
        return str(config.requirements if config is not None else "")

    @staticmethod
    def _require_admin_source_marker_policy(vision: Any, requirements: str) -> None:
        if vision.transient_source_marker_present is True and re.search(
            r"(?:水印|Logo|网址|URL|账号|署名|平台角标|来源角标)", requirements, re.I
        ):
            raise ValueError("ADMIN_ACCEPT_WATERMARK_PRESENT")

    async def _admin_description_payload(
        self,
        profile_id: str,
        instance_id: str,
        candidate: Any,
        vision: Any,
        requirements: str,
        control: Any | None,
    ) -> dict[str, Any]:
        if not requirements.strip():
            return self.compose_description_contract(self._minimal_check_from_vision(vision))
        persona = ""
        if self._requirements_require_persona(requirements):
            character = await CharacterRunContext.start(self.character_models, profile_id)
            with CharacterRunScope(character):
                persona = await self._intake_persona(
                    profile_id,
                    instance_id,
                    relevance_text=requirements,
                    control=control,
                )
        payload, result = await self.build_strict_description(
            profile_id,
            instance_id,
            vision=vision,
            persona=persona,
            requirements=requirements,
            source_kind=candidate.source_kind.value,
            interaction_owner_id=f"admin-accept:{candidate.candidate_id}",
        )
        if not result.accepted:
            raise ValueError(f"ADMIN_ACCEPT_POLICY_REJECTED:{result.reason_code}")
        return payload

    async def _record_admin_override(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        reason: str,
        description_payload: dict[str, Any],
    ) -> Any:
        await self.repository.admin_accept_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
            reason=str(reason or "管理员覆盖AI语义／人设判断"),
            description_payload=description_payload,
        )
        checks = await self.repository.list_sticker_checks(
            profile_id, instance_id, candidate_id=candidate_id, limit=1
        )
        if not checks:
            raise RuntimeError("administrator acceptance did not create a Check revision")
        return checks[0]

    async def _promote_admin_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        candidate: Any,
        latest: Any,
        description_payload: dict[str, Any],
        phash: str,
        dhash: str,
    ) -> Any:
        await self._require_runtime_source(profile_id, instance_id, candidate.source_kind)
        values = self._admin_promotion_values(candidate, latest, description_payload)
        item, _ = await self.media_storage.promote_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
            compact_name=values["compact_name"],
            compact_description=str(description_payload["compact_description"])[:100],
            visible_text=values["visible_text"],
            search_keywords=tuple(description_payload.get("search_keywords") or ())[:20],
            semantic_key=values["semantic_key"],
            emotion=str(latest.emotion or "")[:48],
            speech_act=str(latest.speech_act or "")[:48],
            intensity=int(latest.intensity or 0),
            persona_score=float(latest.persona_score or 0.0),
            phash=phash,
            dhash=dhash,
            frame_hashes=tuple(part for part in phash.split(".") if part),
            visual_group=phash.split(".", 1)[0],
            metadata={
                "admin_override": True,
                "persona_bound": False,
                "persona_fingerprint": "",
                "visible_text": values["visible_text"],
                "search_keywords": list(description_payload.get("search_keywords") or ())[:20],
                "description_version": values["description_version"],
                "text_mode": values["text_mode"],
                "structured_description": {
                    "objective_scene": str(description_payload.get("objective_scene") or "")[:5000],
                    "social_impression": str(
                        description_payload.get("vision_social_impression") or ""
                    )[:80],
                },
            },
        )
        return item

    @staticmethod
    def _admin_promotion_values(
        candidate: Any, latest: Any, payload: dict[str, Any]
    ) -> dict[str, str]:
        candidate_metadata = candidate.metadata
        return {
            "compact_name": str(payload.get("compact_name") or latest.compact_name or "")[:80],
            "visible_text": str(payload.get("visible_text") or "")[:2000],
            "semantic_key": str(payload.get("semantic_key") or latest.semantic_key)[:200],
            "description_version": str(
                payload.get("description_version") or DESCRIPTION_CONTRACT_VERSION
            ),
            "text_mode": str(candidate_metadata.get("text_mode") or TEXT_MODE_NONE),
        }


class StickerCandidateCheckHelpers:
    def _accepted_candidate_details(
        self,
        candidate: Any,
        vision: Any,
        raw: Mapping[str, Any],
        persona: str,
        fingerprints: tuple[str, str],
    ) -> dict[str, Any]:
        phash, dhash = fingerprints
        persona_bound = bool(raw.get("persona_bound"))
        return {
            "vibe_tags": list(raw.get("vibe_tags") or ())[:20],
            "ocr_text": str(vision.ocr_text or "")[:2000],
            "persona_bound": persona_bound,
            "persona_fingerprint": self.persona_fingerprint(persona) if persona_bound else "",
            "text_mode": str(candidate.metadata.get("text_mode") or TEXT_MODE_NONE),
            "phash": phash,
            "dhash": dhash,
            "frame_hashes": [part for part in phash.split(".") if part],
            "visual_group": phash.split(".", 1)[0],
            "structured_description": {
                "objective_scene": str(vision.visible_facts or "")[:5000],
                "social_impression": str(vision.social_impression or "")[:80],
            },
        }


class StickerCandidateMixin(StickerCandidateCheckHelpers):
    async def _admit_sources(
        self,
        *,
        profile_id: str,
        instance_id: str,
        task_id: int,
        source_assets: Sequence[StickerCandidateSource],
        control: Any,
        persona: str,
        requirements: str = "",
        target_accepts: int = 1,
        processed_before: int = 0,
        accepted_before: int = 0,
        quarantined_before: int = 0,
        rejected_before: int = 0,
    ) -> dict[str, Any]:
        """Create candidates and run the same formal Check for either source phase."""
        accepted = quarantined = rejected = waiting = 0
        item_ids: list[str] = []
        phase_total = len(source_assets)
        target = max(1, int(target_accepts))
        promotion_lock = asyncio.Lock()
        admission_state: dict[str, int] = {"accepted": 0, "target": target}
        semaphore = asyncio.Semaphore(2)

        async def bounded(index: int, source: StickerCandidateSource) -> tuple[str, str]:
            async with semaphore:
                return await self._admit_one_source(
                    profile_id=profile_id,
                    instance_id=instance_id,
                    task_id=task_id,
                    index=index,
                    source=source,
                    phase_total=phase_total,
                    control=control,
                    persona=persona,
                    requirements=requirements,
                    processed_before=processed_before,
                    accepted_before=accepted_before,
                    quarantined_before=quarantined_before,
                    rejected_before=rejected_before,
                    promotion_lock=promotion_lock,
                    admission_state=admission_state,
                )

        outcomes = (
            await asyncio.gather(
                *(
                    asyncio.create_task(bounded(index, source))
                    for index, source in enumerate(source_assets, start=1)
                )
            )
            if source_assets
            else []
        )
        for status, item_id in outcomes:
            if status == "accepted" and accepted < target:
                accepted += 1
                item_ids.append(item_id)
            elif status == "waiting":
                waiting += 1
            elif status == "rejected":
                rejected += 1
            elif status == "quarantined":
                quarantined += 1
        return {
            "processed": phase_total,
            "accepted": accepted,
            "quarantined": quarantined,
            "rejected": rejected,
            "waiting": waiting,
            "item_ids": item_ids,
        }

    async def _admit_one_source(
        self,
        *,
        profile_id: str,
        instance_id: str,
        task_id: int,
        index: int,
        source: StickerCandidateSource,
        phase_total: int,
        control: Any,
        persona: str,
        requirements: str,
        processed_before: int,
        accepted_before: int,
        quarantined_before: int,
        rejected_before: int,
        promotion_lock: asyncio.Lock,
        admission_state: dict[str, int],
    ) -> tuple[str, str]:
        asset_id = source.asset_id
        source_kind = source.source_kind
        metadata = dict(source.metadata)
        await self._progress(
            control,
            "CANDIDATE_PREPARE",
            detail="创建待检查候选",
            current=processed_before + index,
            total=processed_before + phase_total,
            accepted=accepted_before + admission_state["accepted"],
            quarantined=quarantined_before,
            rejected=rejected_before,
        )
        candidate = None
        try:
            candidate, created = await self.repository.create_sticker_candidate(
                profile_id,
                instance_id,
                asset_id,
                source_kind=source_kind,
                persona_fingerprint=self.persona_fingerprint(persona),
                metadata={"collector_task_id": task_id, **metadata},
            )
            if not created:
                return self._existing_candidate_outcome(candidate)
            item = await self.check_candidate(
                profile_id,
                instance_id,
                candidate.candidate_id,
                control=control,
                persona=persona,
                requirements=requirements,
                admission_lock=promotion_lock,
                admission_state=admission_state,
            )
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            await self._handle_candidate_failure(profile_id, instance_id, asset_id, candidate, exc)
            return "quarantined", ""
        return await self._candidate_outcome(profile_id, instance_id, candidate, item)

    @staticmethod
    def _existing_candidate_outcome(candidate: Any) -> tuple[str, str]:
        status = str(candidate.status.value)
        if status == "WAITING_CHECK":
            return "waiting", ""
        if status == "REJECTED":
            return "rejected", ""
        if status == "QUARANTINED":
            return "quarantined", ""
        return "redundant", ""

    async def _handle_candidate_failure(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        candidate: Any,
        exc: Exception,
    ) -> None:
        if candidate is None:
            await self._release_source_asset(
                asset_id, reason=f"CANDIDATE_CREATE_FAILED:{type(exc).__name__}"
            )
            return
        with suppress(Exception):
            await self.repository.quarantine_sticker_candidate(
                profile_id,
                instance_id,
                candidate.candidate_id,
                reason=f"CANDIDATE_FAILED:{type(exc).__name__}:{str(exc)[:300]}",
            )

    async def _candidate_outcome(
        self, profile_id: str, instance_id: str, candidate: Any, item: Any
    ) -> tuple[str, str]:
        if item is not None:
            return "accepted", str(item.item_id)
        checks = await self.repository.list_sticker_checks(
            profile_id,
            instance_id,
            candidate_id=candidate.candidate_id,
            limit=1,
        )
        current = await self.repository.get_sticker_candidate(
            profile_id, instance_id, candidate.candidate_id
        )
        if current is None:
            return "redundant", ""
        status = current.status.value
        if status == "WAITING_CHECK":
            return "waiting", ""
        verdict = ""
        if checks:
            verdict = checks[0].verdict.value
        return ("rejected", "") if verdict == "REJECT" else ("quarantined", "")

    async def check_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        control: Any,
        persona: str,
        requirements: str = "",
        admission_lock: asyncio.Lock | None = None,
        admission_state: dict[str, int] | None = None,
        staged_session_id: str = "",
        staged_entry_id: str = "",
    ) -> Any | None:
        preflight = await self._candidate_check_preflight(
            profile_id,
            instance_id,
            candidate_id,
        )
        if preflight is None:
            return None
        candidate, fingerprints = preflight
        evaluated = await self._evaluate_candidate_check(
            profile_id,
            instance_id,
            candidate,
            control=control,
            persona=persona,
            requirements=requirements,
        )
        if evaluated is None:
            return None
        candidate, vision, raw, result = evaluated
        return await self._settle_candidate_check(
            candidate,
            vision,
            raw,
            result,
            persona=persona,
            control=control,
            fingerprints=fingerprints,
            admission_lock=admission_lock,
            admission_state=admission_state,
            staged_session_id=staged_session_id,
            staged_entry_id=staged_entry_id,
        )

    async def _candidate_check_preflight(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
    ) -> tuple[Any, tuple[str, str]] | None:
        candidate_before = await self.repository.get_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
        )
        if candidate_before is not None:
            await self._require_runtime_source(
                profile_id,
                instance_id,
                candidate_before.source_kind,
            )
        else:
            await self._require_runtime_enabled(profile_id, instance_id)
        candidate = await self._load_checking_candidate(
            profile_id,
            instance_id,
            candidate_id,
        )
        if reason := self._automatic_collection_intent_error(candidate):
            result = StickerCheckResult("QUARANTINED", reason)
            await self._record_check(candidate, result, reason=reason)
            return None
        asset = await self.media.get_media_asset(
            candidate.source_asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        deterministic = self._deterministic_candidate_check(candidate, asset)
        if deterministic is not None:
            await self._record_check(candidate, deterministic, reason=deterministic.reason_code)
            return None
        fingerprints = await self._candidate_fingerprints(candidate)
        if fingerprints is None:
            result = self._preflight_rejection(candidate, "MEDIA_FINGERPRINT_UNAVAILABLE")
            await self._record_check(candidate, result, reason=result.reason_code)
            return None
        phash, dhash = fingerprints
        capacity = await self.repository.preflight_sticker_visual_capacity(
            profile_id,
            instance_id,
            candidate.candidate_id,
            phash=phash,
            dhash=dhash,
        )
        if not bool(capacity.get("allowed")):
            result = self._preflight_rejection(candidate, "PERCEPTUAL_DUPLICATE_CAPACITY")
            await self._record_check(candidate, result, reason=result.reason_code)
            return None
        return candidate, fingerprints

    async def _evaluate_candidate_check(
        self,
        profile_id: str,
        instance_id: str,
        candidate: Any,
        *,
        control: Any,
        persona: str,
        requirements: str,
    ) -> tuple[Any, Any, Mapping[str, Any], Any] | None:
        vision = await self._candidate_vision(profile_id, instance_id, candidate, control)
        if vision is None:
            return None
        completed = await self._prepare_candidate_text(candidate, vision, control)
        if completed is None:
            return None
        candidate, vision, expected_text, text_mode = completed
        if await self._integrated_text_mismatch(candidate, vision, expected_text, text_mode):
            return None
        await self._progress(
            control,
            "ADMISSION_CHECK",
            detail="等待人设、安全与完整描述入库 Check",
        )
        checked = await self._candidate_description_check(
            profile_id,
            instance_id,
            candidate,
            vision,
            persona,
            requirements,
        )
        if checked is None:
            return None
        raw, result = checked
        return candidate, vision, raw, result

    async def _settle_candidate_check(
        self,
        candidate: Any,
        vision: Any,
        raw: Mapping[str, Any],
        result: Any,
        *,
        persona: str,
        control: Any,
        fingerprints: tuple[str, str],
        admission_lock: asyncio.Lock | None,
        admission_state: dict[str, int] | None,
        staged_session_id: str,
        staged_entry_id: str,
    ) -> Any | None:
        details_update = (
            self._accepted_candidate_details(candidate, vision, raw, persona, fingerprints)
            if result.accepted
            else {}
        )
        await self._record_check(
            candidate,
            result,
            raw=raw,
            reason=str(raw.get("reason") or result.reason_code),
            visible_text=str(raw.get("visible_text") or ""),
            details_update=details_update,
        )
        if not result.accepted:
            return None
        if staged_session_id or staged_entry_id:
            if not staged_session_id or not staged_entry_id:
                raise ValueError("staged Check requires both session and entry identifiers")
            staged = await self.repository.stage_sticker_intake_candidate(
                staged_session_id,
                staged_entry_id,
                candidate.candidate_id,
            )
            return candidate if staged else None
        return await self._promote_candidate_with_limit(
            candidate,
            raw,
            result,
            persona,
            vision,
            control,
            admission_lock,
            admission_state,
            fingerprints,
        )

    async def _candidate_fingerprints(self, candidate: Any) -> tuple[str, str] | None:
        path = await self.visual_service.asset_file_path(
            profile_id=candidate.profile_id,
            instance_id=candidate.instance_id,
            asset_id=candidate.source_asset_id,
        )
        if path is None:
            return None
        return await asyncio.to_thread(image_fingerprints, str(path))

    @staticmethod
    def _preflight_rejection(candidate: Any, reason: str) -> Any:
        del candidate
        return StickerCheckResult("REJECTED", reason)

    @staticmethod
    def _deterministic_candidate_check(candidate: Any, asset: Any) -> Any:
        return StickerCheckPipeline.check_deterministic(
            source_kind=candidate.source_kind.value,
            owner_matches=asset is not None,
            decoded_image=bool(asset and asset.storage_relpath),
            mime_type=str(asset.mime_type if asset is not None else ""),
            byte_size=int(asset.byte_size if asset is not None else 0),
            width=int(asset.width if asset is not None else 0),
            height=int(asset.height if asset is not None else 0),
        )

    @classmethod
    def _automatic_collection_intent_error(cls, candidate: Any) -> str:
        metadata = candidate.metadata
        if (
            candidate.source_kind.value not in {"WEB", "GENERATED"}
            or "collector_task_id" not in metadata
        ):
            return ""
        intent = metadata.get("collection_intent")
        if not isinstance(intent, Mapping):
            return "COLLECTION_INTENT_MISSING"
        try:
            cls._validated_collection_intent(intent)
        except StickerDescriptionContractError:
            return "COLLECTION_INTENT_INVALID"
        return ""

    async def _load_checking_candidate(
        self, profile_id: str, instance_id: str, candidate_id: str
    ) -> Any:
        candidate = await self.repository.get_sticker_candidate(
            profile_id, instance_id, candidate_id
        )
        if candidate is None:
            raise KeyError(candidate_id)
        return await self.repository.set_sticker_candidate_checking(
            profile_id, instance_id, candidate_id
        )

    async def _candidate_vision(
        self, profile_id: str, instance_id: str, candidate: Any, control: Any
    ) -> Any | None:
        await self._require_runtime_source(profile_id, instance_id, candidate.source_kind)
        await self._progress(control, "VISION_CHECK", detail="等待图片识别模型")
        try:
            result = await self.visual_service.describe_asset(
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=candidate.source_asset_id,
                foreground=False,
                cache_policy=VisualCachePolicy.USE,
                inspection_mode=VisionInspectionMode.STICKER_QUALITY,
            )
            await self._require_runtime_source(profile_id, instance_id, candidate.source_kind)
            return result
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            await self._mark_waiting_check(
                candidate,
                reason="VISION_UNAVAILABLE",
                failure_stage="VISION_CHECK",
                cause=exc,
            )
            return None

    async def _prepare_candidate_text(
        self, candidate: Any, vision: Any, control: Any
    ) -> tuple[Any, Any, str, str] | None:
        metadata = dict(candidate.metadata)
        if bool(metadata.get("text_finish_pending")):
            try:
                completed = await self._complete_pending_text(
                    candidate=candidate,
                    metadata=metadata,
                    initial_vision=vision,
                    control=control,
                )
            except StickerRuntimeDisabled:
                raise
            except Exception as exc:
                await self._quarantine_text_failure(candidate, exc)
                return None
            if completed is None:
                return None
            candidate, vision = completed
            metadata = dict(candidate.metadata)
        expected = str(metadata.get("meme_text") or "").strip()
        mode = str(metadata.get("text_mode") or TEXT_MODE_NONE).upper()
        return candidate, vision, expected, mode

    async def _quarantine_text_failure(self, candidate: Any, exc: Exception) -> None:
        await self.repository.quarantine_sticker_candidate(
            candidate.profile_id,
            candidate.instance_id,
            candidate.candidate_id,
            reason=f"TEXT_FINISH_FAILED:{type(exc).__name__}:{str(exc)[:300]}",
        )

    async def _integrated_text_mismatch(
        self,
        candidate: Any,
        vision: Any,
        expected_text: str,
        text_mode: str,
    ) -> bool:
        if text_mode != TEXT_MODE_INTEGRATED_TEXT or not expected_text:
            return False
        if self._ocr_matches(expected_text, str(vision.ocr_text or "")):
            return False
        result = StickerCheckResult(
            "REJECTED",
            "TEXT_QUALITY",
            compact_description=str(vision.visible_facts or "")[:100],
        )
        await self._record_check(
            candidate,
            result,
            reason="INTEGRATED_TEXT_OCR_MISMATCH",
            visible_text=str(vision.ocr_text or ""),
        )
        return True

    async def _candidate_description_check(
        self,
        profile_id: str,
        instance_id: str,
        candidate: Any,
        vision: Any,
        persona: str,
        requirements: str,
    ) -> tuple[dict[str, Any], Any] | None:
        try:
            return await self.build_strict_description(
                profile_id,
                instance_id,
                vision=vision,
                persona=persona,
                requirements=requirements,
                source_kind=candidate.source_kind.value,
                interaction_owner_id=candidate.candidate_id,
                collection_intent=(
                    candidate.metadata.get("collection_intent")
                    if isinstance(candidate.metadata.get("collection_intent"), Mapping)
                    else None
                ),
            )
        except StickerDescriptionContractError as exc:
            if exc.code == "DESCRIPTION_MODEL_UNAVAILABLE":
                await self._mark_waiting_check(
                    candidate,
                    reason=exc.code,
                    failure_stage="DESCRIPTION_CONTRACT",
                    cause=exc,
                )
            else:
                await self._quarantine_check_failure(
                    candidate,
                    reason=exc.code,
                    failure_stage="DESCRIPTION_CONTRACT",
                    cause=exc,
                )
            return None

    async def _promote_candidate_with_limit(
        self,
        candidate: Any,
        raw: dict[str, Any],
        result: Any,
        persona: str,
        vision: Any,
        control: Any,
        admission_lock: asyncio.Lock | None,
        admission_state: dict[str, int] | None,
        fingerprints: tuple[str, str],
    ) -> Any | None:
        async def promote() -> Any | None:
            await self._require_runtime_source(
                candidate.profile_id, candidate.instance_id, candidate.source_kind
            )
            return await self._promote_checked_candidate(
                candidate=candidate,
                raw=raw,
                result=result,
                persona=persona,
                visible_text=str(raw.get("visible_text") or ""),
                ocr_text=str(vision.ocr_text or ""),
                control=control,
                fingerprints=fingerprints,
            )

        if admission_lock is None or admission_state is None:
            return await promote()
        async with admission_lock:
            await self._require_runtime_source(
                candidate.profile_id, candidate.instance_id, candidate.source_kind
            )
            if int(admission_state.get("accepted", 0)) >= int(admission_state.get("target", 1)):
                await self._discard_redundant_candidate(candidate)
                return None
            item = await promote()
            if item is not None:
                admission_state["accepted"] = int(admission_state.get("accepted", 0)) + 1
            return item


__all__ = ["StickerCandidateMixin"]
