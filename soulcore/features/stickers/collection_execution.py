"""Quota-aware web and generated sticker collection execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .contracts import StickerGenerationSpec
from .domain import StickerCandidateSource, StickerConfig, StickerSourceKind
from .planning import StickerCollectionGap
from .policy import StickerRuntimeDisabled


@dataclass(slots=True)
class CollectionProgress:
    accepted: int = 0
    quarantined: int = 0
    rejected: int = 0
    waiting: int = 0
    processed: int = 0
    source_count: int = 0
    external_effect_unknown: bool = False
    item_ids: list[str] = None  # type: ignore[assignment]
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.item_ids = [] if self.item_ids is None else self.item_ids
        self.errors = [] if self.errors is None else self.errors

    def add_admitted(self, admitted: Mapping[str, Any]) -> None:
        self.accepted += int(admitted["accepted"])
        self.quarantined += int(admitted["quarantined"])
        self.rejected += int(admitted["rejected"])
        self.waiting += int(admitted["waiting"])
        self.processed += int(admitted["processed"])
        self.item_ids.extend(admitted["item_ids"])


@dataclass(frozen=True, slots=True)
class CollectionLimits:
    normal_web: int = 0
    normal_generated: int = 0

    @property
    def target_total(self) -> int:
        return self.normal_web + self.normal_generated


class StickerCollectionExecutionMixin:
    async def _collection_limits(
        self,
        profile_id: str,
        instance_id: str,
        config: StickerConfig,
        gap: StickerCollectionGap,
    ) -> CollectionLimits:
        midnight = (
            datetime.now()
            .astimezone()
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(UTC)
        )
        web_used = await self._daily_source_count(
            profile_id, instance_id, StickerSourceKind.WEB, midnight
        )
        generated_used = await self._daily_source_count(
            profile_id, instance_id, StickerSourceKind.GENERATED, midnight
        )
        web_remaining = max(0, int(config.web_daily_limit) - web_used)
        generated_remaining = max(0, int(config.generated_daily_limit) - generated_used)
        web_limit = min(web_remaining, 3 if gap.animated_preferred else 2)
        generated_limit = (
            0
            if gap.animated_preferred
            else min(
                generated_remaining,
                2 if gap.character_specific else 1,
            )
        )
        if not config.web_collection_enabled:
            web_limit = 0
        if not config.generation_enabled:
            generated_limit = 0
        return CollectionLimits(
            normal_web=web_limit,
            normal_generated=generated_limit,
        )

    async def _daily_source_count(
        self,
        profile_id: str,
        instance_id: str,
        source_kind: Any,
        midnight: datetime,
    ) -> int:
        return int(
            await self.repository.count_sticker_items_since(
                profile_id,
                instance_id,
                source_kind=source_kind,
                since=midnight,
            )
        )

    async def _run_web_collection(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        plan: Mapping[str, Any],
        gap: StickerCollectionGap,
        collection_intent: Mapping[str, str],
        persona: str,
        requirements: str,
        control: Any,
        progress: CollectionProgress,
        *,
        target_accepts: int,
        check_limit: int,
    ) -> None:
        tiers = self._web_query_tiers(
            plan=plan,
            gap=gap,
            requirements=requirements,
        )
        accepted_at_start = progress.accepted
        checked = 0
        target = max(0, int(target_accepts))
        for tier_index, tier_queries in enumerate(tiers, start=1):
            accepted_in_phase = progress.accepted - accepted_at_start
            if accepted_in_phase >= target or checked >= max(0, int(check_limit)):
                break
            await self._progress(
                control,
                "WEB_SEARCH",
                detail=f"联网第{tier_index}层搜索（逐步放宽）",
                current=tier_index,
                total=3,
                search_tier=tier_index,
            )
            try:
                assets = await self._collect_web(
                    profile_id,
                    instance_id,
                    task_id,
                    tier_queries,
                    control,
                    limit=max(0, int(check_limit) - checked),
                )
                assets = [
                    StickerCandidateSource(
                        asset_id=source.asset_id,
                        source_kind=source.source_kind,
                        metadata={
                            **dict(source.metadata),
                            "collection_intent": dict(collection_intent),
                        },
                    )
                    for source in assets
                ]
            except StickerRuntimeDisabled:
                raise
            except Exception as exc:
                progress.errors.append(
                    f"web_tier_{tier_index}:{type(exc).__name__}:{str(exc)[:180]}"
                )
                continue
            progress.source_count += len(assets)
            checked += len(assets)
            admitted = await self._admit_sources(
                profile_id=profile_id,
                instance_id=instance_id,
                task_id=task_id,
                source_assets=assets,
                control=control,
                persona=persona,
                requirements=requirements,
                target_accepts=max(1, target - accepted_in_phase),
                processed_before=progress.processed,
                accepted_before=progress.accepted,
                quarantined_before=progress.quarantined,
                rejected_before=progress.rejected,
            )
            progress.add_admitted(admitted)

    async def _prepare_generation_prompts(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        instance: Any,
        scope: Any,
        config: StickerConfig,
        control: Any,
        plan: Mapping[str, Any],
        persona: str,
        theme: str,
        requirements: str,
        concepts: list[Any],
        limit: int,
        progress: CollectionProgress,
        *,
        gap: StickerCollectionGap,
        collection_intent: Mapping[str, str],
        identity_reference_note: str,
    ) -> list[StickerGenerationSpec]:
        if not limit:
            return []
        concepts = [str(value).strip() for value in concepts if str(value).strip()]
        concepts = concepts[: int(limit)]
        if not concepts:
            return []
        references = await self._generation_references(
            profile_id,
            instance_id,
            task_id,
            config,
            control,
            plan,
            concepts,
            progress,
        )
        if plan.get("generation_research_queries") and not references:
            progress.errors.append("generation_research:NO_USABLE_REFERENCE")
            return []
        await self._progress(
            control,
            "PROMPT_REFINEMENT",
            detail=(
                "结合联网参考设计明确画面方案"
                if references
                else (
                    "根据角色资料设计明确画面方案"
                    if gap.character_specific
                    else "根据补缺概念设计通用画面方案"
                )
            ),
            current=0,
            total=min(limit, len(concepts)),
        )
        try:
            return await self._refine_generation_prompts(
                profile_id=profile_id,
                task_id=task_id,
                instance=instance,
                persona=persona,
                role_background=scope.extra_background if scope is not None else "",
                world_texture=scope.world_texture_prompt if scope is not None else "",
                requested_theme=theme,
                concepts=concepts,
                references=references,
                limit=limit,
                web_enabled=bool(config.web_collection_enabled),
                requirements=requirements,
                gap=gap,
                collection_intent=collection_intent,
                identity_reference_note=identity_reference_note,
            )
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            progress.errors.append(f"prompt_refinement:{type(exc).__name__}:{str(exc)[:180]}")
            return []

    async def _generation_references(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        config: StickerConfig,
        control: Any,
        plan: Mapping[str, Any],
        concepts: list[Any],
        progress: CollectionProgress,
    ) -> list[dict[str, str]]:
        if not config.web_collection_enabled:
            return []
        queries = plan.get("generation_research_queries") or []
        if not queries:
            return []
        await self._progress(
            control,
            "GENERATION_RESEARCH",
            detail="联网检索角色外观、相关梗与表达参考",
            current=0,
            total=min(2, len(queries)),
        )
        try:
            return await self._research_generation_references(
                profile_id,
                instance_id,
                task_id,
                queries,
                concepts,
                control,
            )
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            progress.errors.append(f"generation_research:{type(exc).__name__}:{str(exc)[:180]}")
            return []

    async def _run_generated_collection(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        instance: Any,
        prompts: list[StickerGenerationSpec],
        limit: int,
        persona: str,
        requirements: str,
        control: Any,
        progress: CollectionProgress,
        *,
        character_specific: bool,
        collection_intent: Mapping[str, str],
    ) -> None:
        await self._progress(
            control,
            "GENERATING",
            detail="开始生成表情包候选",
            current=0,
            total=min(limit, len(prompts)),
        )
        try:
            reference = (
                await self._identity_reference(profile_id, instance_id, scope=instance.scope)
                if character_specific
                else None
            )
            assets = await self._collect_generated(
                profile_id,
                instance_id,
                task_id,
                prompts,
                control,
                limit=limit,
                identity_reference=reference,
            )
            assets = [
                StickerCandidateSource(
                    asset_id=source.asset_id,
                    source_kind=source.source_kind,
                    metadata={
                        **dict(source.metadata),
                        "collection_intent": dict(collection_intent),
                    },
                )
                for source in assets
            ]
            progress.source_count += len(assets)
            admitted = await self._admit_sources(
                profile_id=profile_id,
                instance_id=instance_id,
                task_id=task_id,
                source_assets=assets,
                control=control,
                persona=persona,
                requirements=requirements,
                target_accepts=max(1, int(limit)),
                processed_before=progress.processed,
                accepted_before=progress.accepted,
                quarantined_before=progress.quarantined,
                rejected_before=progress.rejected,
            )
            progress.add_admitted(admitted)
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            if self._unknown_generation_effect(exc):
                progress.external_effect_unknown = True
            progress.errors.append(f"generation:{type(exc).__name__}:{str(exc)[:180]}")


__all__ = [
    "CollectionLimits",
    "CollectionProgress",
    "StickerCollectionExecutionMixin",
]
