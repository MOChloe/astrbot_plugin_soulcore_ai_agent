"""Web/generation collection planning, execution and progress reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ...contracts.ai_models import AIInvocationError
from ...contracts.web import (
    SearchRequest,
    WebCallerKind,
    WebSearchDepth,
    WebSearchFreshness,
    WebSearchIntensity,
    WebSearchPurpose,
)
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_markup_record,
)
from .contracts import (
    STICKER_GENERATION_DESIGN_OUTPUT_CONTRACT,
    STICKER_GENERATION_DESIGN_SYSTEM_PROMPT,
    StickerGenerationSpec,
    StickerTextFinishingDeferred,
)
from .domain import StickerCollectedAsset, StickerSourceKind
from .planning import StickerCollectionGap, StickerPlanningMixin, StickerTextFinishingMixin
from .policy import StickerRuntimeDisabled
from .text_modes import (
    GENERATED_STICKER_TEXT_MODES,
    TEXT_MODE_INTEGRATED_TEXT,
    TEXT_MODE_NONE,
)


class StickerCollectionMixin(StickerPlanningMixin, StickerTextFinishingMixin):
    async def _research_generation_references(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        queries: Sequence[Any],
        concepts: Sequence[Any],
        control: Any,
    ) -> list[dict[str, str]]:
        """Search text references before drawing; never treat results as instructions."""
        normalized = list(
            dict.fromkeys(str(value).strip()[:240] for value in queries if str(value).strip())
        )[:2]
        if not normalized:
            return []
        references: list[dict[str, str]] = []
        errors: list[Exception] = []
        for index, query in enumerate(normalized, start=1):
            await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.WEB)
            await self._progress(
                control,
                "GENERATION_RESEARCH",
                detail="等待联网角色外观与表情表达资料",
                current=index,
                total=len(normalized),
                found=len(references),
            )
            try:
                response = await self.web_research.search(
                    SearchRequest(
                        profile_id=profile_id,
                        instance_id=instance_id,
                        caller_kind=WebCallerKind.STICKER_COLLECTOR,
                        caller_id="sticker_generation_research",
                        ai_task_id=str(task_id),
                        purpose=WebSearchPurpose.SELF_EXPLORATION,
                        query=query,
                        depth=WebSearchDepth.QUICK,
                        freshness=WebSearchFreshness.AUTO,
                        intensity=WebSearchIntensity.STANDARD,
                        operation_timeout_seconds=self.operation_timeout_seconds,
                    )
                )
                await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.WEB)
            except StickerRuntimeDisabled:
                raise
            except Exception as exc:
                errors.append(exc)
                continue
            for item in response.results[:3]:
                references.append(
                    {
                        "title": item.title[:240],
                        "snippet": item.snippet[:800],
                        "domain": item.domain[:120],
                    }
                )
                if len(references) >= 6:
                    break
            if len(references) >= 6:
                break
        if not references and errors:
            raise errors[-1]
        return references

    async def _refine_generation_prompts(
        self,
        *,
        profile_id: str,
        task_id: int,
        instance: Any,
        persona: str,
        role_background: str,
        world_texture: str,
        requested_theme: str,
        concepts: Sequence[Any],
        references: Sequence[Mapping[str, str]],
        limit: int,
        web_enabled: bool,
        gap: StickerCollectionGap,
        collection_intent: Mapping[str, str],
        identity_reference_note: str = "",
        requirements: str = "",
    ) -> list[StickerGenerationSpec]:
        concept_values = [str(value).strip()[:1000] for value in concepts if str(value).strip()][
            : max(0, int(limit))
        ]
        if not concept_values:
            return []
        rendered: list[StickerGenerationSpec] = []
        for index, concept in enumerate(concept_values, start=1):
            prompt = self._generation_design_prompt(
                concept=concept,
                gap=gap,
                persona=persona,
                role_background=role_background,
                world_texture=world_texture,
                requested_theme=requested_theme,
                identity_reference_note=identity_reference_note,
                requirements=requirements,
                references=references,
            )
            raw = await self._run_model_commands(
                profile_id,
                instance,
                "sticker.collect",
                STICKER_GENERATION_DESIGN_SYSTEM_PROMPT,
                prompt,
                STICKER_GENERATION_DESIGN_OUTPUT_CONTRACT,
                owner_kind="STICKER_COLLECTION",
                owner_id=f"{task_id}:design:{index}",
                include_identity_context=gap.character_specific,
            )
            specs = raw.get("generation_specs")
            if not isinstance(specs, list) or len(specs) != 1:
                raise ValueError("sticker generation director must return exactly one design")
            spec = self._render_generation_spec(
                specs[0],
                character_specific=gap.character_specific,
                collection_intent=collection_intent,
            )
            if spec is None:
                raise ValueError("sticker generation design is incomplete")
            rendered.append(spec)
        return rendered

    def _generation_design_prompt(
        self,
        *,
        concept: str,
        gap: StickerCollectionGap,
        persona: str,
        role_background: str,
        world_texture: str,
        requested_theme: str,
        identity_reference_note: str,
        requirements: str,
        references: Sequence[Mapping[str, str]],
    ) -> TrustedPromptMarkup:
        theme = str(requested_theme).strip()
        if theme in {
            str(concept).strip(),
            str(gap.goal).strip(),
            str(gap.expected_use).strip(),
        }:
            theme = ""
        records: list[Any] = [
            prompt_markup_record(
                "当前概念",
                (
                    ("概念", concept),
                    ("本轮要补", gap.goal),
                    ("沟通用途", gap.expected_use),
                    ("文字预期", gap.expected_text),
                    ("画面方向", gap.visual_goal),
                    ("指定主题", theme[:500]),
                ),
                omit_empty=True,
            )
        ]
        if str(requirements).strip():
            records.append(
                prompt_markup_record(
                    "收藏偏好",
                    (("内容", requirements[:3000]),),
                )
            )
        if gap.character_specific:
            records.append(
                prompt_markup_record(
                    "角色创作资料",
                    (
                        ("身份与外观", str(persona)[:8000]),
                        ("背景", str(role_background)[:3000]),
                        ("画面风格", str(world_texture)[:3000]),
                        ("身份参考", identity_reference_note),
                    ),
                    omit_empty=True,
                )
            )
        if references:
            records.append(self._material_markup(list(references)[:6], available=True))
        return join_prompt_markup(records)

    @staticmethod
    def _render_generation_spec(
        raw_spec: Any,
        *,
        character_specific: bool,
        collection_intent: Mapping[str, str],
    ) -> StickerGenerationSpec | None:
        if not isinstance(raw_spec, Mapping):
            return None
        required = (
            "style",
            "subject_identity",
            "action_expression",
            "composition",
            "text_mode",
            "negative_constraints",
        )
        values = {key: str(raw_spec.get(key) or "").strip() for key in required}
        if any(not values[key] for key in required):
            return None
        mode = values["text_mode"].upper()
        if mode not in GENERATED_STICKER_TEXT_MODES:
            return None
        meme_text = str(raw_spec.get("meme_text") or "").strip()
        if mode == TEXT_MODE_NONE:
            meme_text = ""
        elif not meme_text:
            return None
        position = _caption_position(str(raw_spec.get("caption_position") or ""))
        safe_zone = str(raw_spec.get("safe_zone") or "").strip()[:500]
        position_label = _caption_position_label(position)
        text_instruction = _generation_text_instruction(
            mode,
            meme_text=meme_text,
            position_label=position_label,
            safe_zone=safe_zone,
        )
        identity_note = (
            "另附的身份参考只用于保持人物身份与外观一致。"
            "本图按这次的画面方案重新创作，不照搬参考中的画风、背景、姿势或构图。"
            if character_specific
            else "画面主体是独立的通用形象，与当前角色无关。"
        )
        prompt = (
            "请创作一张即时通讯表情贴图。"
            "表情需要在缩小显示时仍能一眼辨识情绪或言语作用。\n"
            f"{identity_note}\n"
            f"画面主体是{values['subject_identity']}，{values['action_expression']}。\n"
            f"{values['composition']}。\n"
            f"整体画面风格为{values['style']}。\n"
            f"{text_instruction}。\n"
            f"画面中不要出现{values['negative_constraints']}。"
        )
        return StickerGenerationSpec(
            prompt,
            text_mode=mode,
            meme_text=meme_text,
            text_position=position,
            text_safe_zone=safe_zone,
            character_specific=character_specific,
            collection_intent=collection_intent,
        )

    async def _collect_web(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        queries: Sequence[Any],
        control: Any,
        *,
        limit: int,
    ) -> list[StickerCollectedAsset]:
        output: list[StickerCollectedAsset] = []
        errors: list[Exception] = []
        normalized_queries = list(
            dict.fromkeys(str(v).strip()[:240] for v in queries if str(v).strip())
        )[:10]
        for index, query in enumerate(normalized_queries, start=1):
            if len(output) >= max(0, limit):
                break
            await self._progress(
                control,
                "WEB_SEARCH",
                detail="等待联网图片搜索与下载检查",
                current=index,
                total=len(normalized_queries),
                found=len(output),
            )
            try:
                await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.WEB)
                response = await self.web_research.search_images(
                    SearchRequest(
                        profile_id=profile_id,
                        instance_id=instance_id,
                        caller_kind=WebCallerKind.STICKER_COLLECTOR,
                        caller_id="sticker_collector",
                        ai_task_id=str(task_id),
                        purpose=WebSearchPurpose.SELF_EXPLORATION,
                        query=query,
                        depth=WebSearchDepth.QUICK,
                        freshness=WebSearchFreshness.AUTO,
                        intensity=WebSearchIntensity.STANDARD,
                        operation_timeout_seconds=self.operation_timeout_seconds,
                    )
                )
                await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.WEB)
                resources = list(response.results)[: max(0, limit - len(output))]
                if not resources:
                    continue
                inspected = await self.visual_service.inspect_web_search_images(
                    resources=resources,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    core_run_id=str(task_id),
                    main_core_supports_vision=False,
                    defer_inspection_to_sticker_check=True,
                )
                await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.WEB)
            except StickerRuntimeDisabled:
                raise
            except Exception as exc:
                # One unavailable search source must not suppress the drawing
                # branch of the same collection run.
                errors.append(exc)
                continue
            output.extend(
                StickerCollectedAsset(str(asset_id), StickerSourceKind.WEB)
                for asset_id in inspected.get("asset_ids", ())
            )
        if not output and errors:
            raise errors[-1]
        return output[: max(0, limit)]

    async def _collect_generated(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        prompts: Sequence[Any],
        control: Any,
        *,
        limit: int,
        identity_reference: Mapping[str, Any] | None = None,
    ) -> list[StickerCollectedAsset]:
        output: list[StickerCollectedAsset] = []
        errors: list[Exception] = []
        normalized_prompts = [value for value in prompts if str(value).strip()][: max(0, limit)]
        for index, raw_spec in enumerate(normalized_prompts, start=1):
            entry, error = await self._collect_generated_item(
                profile_id,
                instance_id,
                task_id,
                raw_spec,
                control,
                index,
                len(normalized_prompts),
                len(output),
                identity_reference,
            )
            if error is not None:
                errors.append(error)
            if entry is not None:
                output.append(entry)
        if not output and not errors and normalized_prompts:
            await self._progress(
                control,
                "GENERATION_FAILED",
                detail="图片生成调用没有返回任何可登记图片",
                current=len(normalized_prompts),
                total=len(normalized_prompts),
                found=0,
                error_type="EMPTY_IMAGE_OUTPUT",
            )
        if not output and errors:
            raise errors[-1]
        return output[: max(0, limit)]

    async def _collect_generated_item(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        raw_spec: Any,
        control: Any,
        index: int,
        total: int,
        found: int,
        identity_reference: Mapping[str, Any] | None,
    ) -> tuple[StickerCollectedAsset | None, Exception | None]:
        spec = (
            raw_spec
            if isinstance(raw_spec, StickerGenerationSpec)
            else StickerGenerationSpec(str(raw_spec))
        )
        await self._progress(
            control,
            "GENERATING",
            detail="等待图片生成、下载与媒体登记",
            current=index,
            total=total,
            found=found,
        )
        try:
            await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
            result = await self.visual_service.present_visual(
                profile_id=profile_id,
                instance_id=instance_id,
                run_id=task_id,
                counterpart_requirements="",
                scene_plan=str(spec).strip()[:5000],
                selected_visual_facts="",
                image_count=1,
                aspect_ratio="1:1",
                size="auto",
                reference_asset_ids=[],
                reference_purposes=[],
                character_visible=bool(identity_reference),
                main_core_supports_vision=False,
                defer_inspection_to_sticker_check=True,
                identity_reference=(
                    dict(identity_reference) if isinstance(identity_reference, Mapping) else None
                ),
                maximum_generation_backends=2,
            )
            await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            await self._record_generation_failure(control, exc, index, total, found)
            if self._unknown_generation_effect(exc):
                raise
            return None, exc
        generated_ids = [str(asset_id) for asset_id in result.get("asset_ids", ()) if str(asset_id)]
        if not generated_ids:
            return None, None
        final_asset = generated_ids[0]
        for unused in generated_ids[1:]:
            await self._release_source_asset(unused, reason="EXTRA_GENERATED_OUTPUT")
        return await self._finish_generated_item(
            profile_id,
            instance_id,
            task_id,
            final_asset,
            spec,
            identity_reference,
        )

    async def _record_generation_failure(
        self, control: Any, exc: Exception, index: int, total: int, found: int
    ) -> None:
        await self._progress(
            control,
            "GENERATION_FAILED",
            detail=f"图片生成入口失败：{type(exc).__name__}: {str(exc)[:240]}",
            current=index,
            total=total,
            found=found,
            error_type=type(exc).__name__,
        )

    @staticmethod
    def _unknown_generation_effect(exc: Exception) -> bool:
        if not isinstance(exc, AIInvocationError):
            return False
        details = dict(exc.info.details)
        return bool(details.get("external_side_effect_unknown") or details.get("recovery_required"))

    async def _finish_generated_item(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        final_asset: str,
        spec: StickerGenerationSpec,
        identity_reference: Mapping[str, Any] | None,
    ) -> tuple[StickerCollectedAsset | None, Exception | None]:
        try:
            final_asset = await self._finish_generated_text(
                profile_id,
                instance_id,
                task_id,
                final_asset,
                spec,
            )
            if not final_asset:
                return None, None
        except StickerTextFinishingDeferred as exc:
            metadata = self._generated_metadata(
                spec, identity_reference, pending=True, failure=str(exc)
            )
            return StickerCollectedAsset(exc.asset_id, StickerSourceKind.GENERATED, metadata), None
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            if self._unknown_generation_effect(exc):
                metadata = self._generated_metadata(
                    spec,
                    identity_reference,
                    failure=f"EXTERNAL_EFFECT_UNKNOWN:{type(exc).__name__}:{str(exc)[:300]}",
                )
                metadata["text_finish_effect_unknown"] = True
                return StickerCollectedAsset(
                    final_asset, StickerSourceKind.GENERATED, metadata
                ), None
            await self._release_source_asset(final_asset, reason="STICKER_TEXT_FINISH_FAILED")
            return None, exc
        metadata = self._generated_metadata(spec, identity_reference)
        return StickerCollectedAsset(final_asset, StickerSourceKind.GENERATED, metadata), None

    async def _finish_generated_text(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        final_asset: str,
        spec: StickerGenerationSpec,
    ) -> str:
        if spec.text_mode == TEXT_MODE_INTEGRATED_TEXT:
            return await self._finish_integrated_text(
                profile_id=profile_id,
                instance_id=instance_id,
                task_id=task_id,
                source_asset_id=final_asset,
                spec=spec,
            )
        return final_asset

    @staticmethod
    def _generated_metadata(
        spec: StickerGenerationSpec,
        identity_reference: Mapping[str, Any] | None,
        *,
        pending: bool = False,
        failure: str = "",
    ) -> dict[str, Any]:
        metadata = {
            "text_mode": spec.text_mode,
            "meme_text": spec.meme_text,
            "text_position": spec.text_position,
            "text_safe_zone": spec.text_safe_zone,
            "generation_prompt": str(spec),
            "text_finish_pending": pending,
            "character_specific": bool(spec.character_specific),
            "collection_intent": dict(spec.collection_intent),
            "identity_reference_id": (
                str(identity_reference.get("asset_id") or "")
                if isinstance(identity_reference, Mapping)
                else ""
            ),
        }
        if failure:
            metadata["text_finish_failure"] = failure
        return metadata

    @staticmethod
    async def _progress(control: Any, stage: str, **values: Any) -> None:
        """Persist a safe, user-visible stage while retaining prior counters."""

        if control is None:
            return
        previous = dict(control.progress)
        changed = str(previous.get("stage") or "") != str(stage)
        progress = {
            **previous,
            **values,
            "stage": str(stage),
            "stage_started_at": (
                datetime.now(UTC).isoformat() if changed else previous.get("stage_started_at")
            ),
            "heartbeat_at": datetime.now(UTC).isoformat(),
        }
        await control.heartbeat(progress=progress)


__all__ = ["StickerCollectionMixin"]


def _caption_position(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return ""
    if any(token in normalized for token in ("下", "bottom")):
        return "BOTTOM"
    if any(token in normalized for token in ("画面内", "中", "center", "centre")):
        return "CENTER"
    if any(token in normalized for token in ("上", "top")):
        return "TOP"
    return ""


def _generation_text_instruction(
    mode: str,
    *,
    meme_text: str,
    position_label: str,
    safe_zone: str,
) -> str:
    if mode == TEXT_MODE_NONE:
        return "画面内不出现任何正文、随机字符、伪文字或水印"
    instruction = f"在画面中逐字准确呈现原文“{meme_text}”，让文字自然融入构图"
    if position_label:
        instruction += f"，文字位于{position_label}"
    if safe_zone:
        instruction += f"，{safe_zone}不可遮挡"
    return instruction


def _caption_position_label(value: str) -> str:
    return {"TOP": "画面上部", "BOTTOM": "画面下部", "CENTER": "画面内"}.get(str(value), "")
