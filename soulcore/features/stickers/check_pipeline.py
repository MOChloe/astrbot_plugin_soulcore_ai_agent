"""Run-scoped, text-only access to instance-owned sticker libraries.

This module is deliberately independent from SQLite and AstrBot.  The storage
adapter owns durable candidates/items while this application boundary makes
sure a Main Core run can only see compact descriptions and opaque references.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from ...shared.token_meter import ConservativeTokenMeter
from ..profiles.ports import ProfilesRepositoryPort
from .domain import (
    StickerConfig,
    StickerImportIntent,
    StickerItem,
    StickerRunRef,
    StickerSourceKind,
    StickerUsageType,
    sticker_import_source_ref,
)
from .policy import load_sticker_runtime_policy
from .retrieval import (
    RankedSticker,
    StickerRetrievalRanker,
    StickerUsageWindow,
    structured_search_tokens,
)
from .selection import compact_text as _compact_text
from .selection import filter_preference as _filter_preference
from .selection import fit_ranked_to_token_budget as _fit_ranked_to_token_budget
from .selection import interleaved_stickers as _interleaved_stickers
from .selection import normalize_semantic_key
from .selection import sticker_intensity as _intensity
from .selection import usage_type as _usage_type

MAX_SEARCH_CALLS = 2
DEFAULT_SEARCH_TOKEN_LIMIT = 8_192
RUN_REF_TTL = timedelta(hours=1)
AUTOMATIC_SOURCES = frozenset({"web", "generated", "upload", "WEB", "GENERATED", "UPLOAD"})
PLAYER_SOURCES = frozenset({"player", "PLAYER"})


class StickerRepository(Protocol):
    async def get_sticker_config(self, profile_id: str, scope: str) -> StickerConfig: ...

    async def list_sticker_items(
        self, profile_id: str, instance_id: str, *, query: str = "", limit: int = 500
    ) -> Sequence[StickerItem]: ...

    async def search_sticker_items_indexed(
        self,
        profile_id: str,
        instance_id: str,
        *,
        tokens: Sequence[str],
        status: str,
        limit: int,
    ) -> Sequence[StickerItem]: ...

    async def create_sticker_run_refs(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int,
        item_ids: Sequence[str],
        expires_at: datetime,
    ) -> Sequence[StickerRunRef]: ...

    async def resolve_sticker_run_refs(
        self, profile_id: str, instance_id: str, run_id: int, refs: Sequence[str]
    ) -> Sequence[StickerItem]: ...

    async def disable_sticker_item_for_instance(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StickerProjection:
    sticker_ref: str
    compact_description: str
    score: float
    item_id: str = field(repr=False)
    semantic_key: str = field(default="", repr=False)
    visual_group: str = field(default="", repr=False)
    emotion: str = ""
    speech_act: str = ""
    intensity: int = 0
    usage_type: StickerUsageType = StickerUsageType.REACTION
    visible_text: str = ""
    is_animated: bool = False
    recently_used: bool = False
    current_run_visible: bool = True

    @property
    def content(self) -> str:
        # Selection metadata stays internal. Main Core receives exactly one
        # short, usage-facing sentence describing what the sticker conveys.
        return self.compact_description.strip()


@dataclass(frozen=True, slots=True)
class StickerWorkset:
    items: tuple[StickerProjection, ...] = ()
    visible_refs: frozenset[str] = frozenset()
    token_limit: int = 0
    used_tokens: int = 0


@dataclass(frozen=True, slots=True)
class StickerCommandBatchSnapshot:
    """Opaque rollback point for one ordered Main Core command batch."""

    owner: object = field(repr=False, compare=False)
    search_calls: int
    search_token_limit: int
    projections: tuple[tuple[str, StickerProjection], ...] = field(repr=False)
    import_sources: tuple[tuple[str, tuple[str, str]], ...] = field(repr=False)
    import_source_by_asset: tuple[tuple[str, str], ...] = field(repr=False)
    import_intents: tuple[tuple[str, StickerImportIntent], ...] = field(repr=False)
    disable_intents: tuple[tuple[str, StickerProjection], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class StickerCheckResult:
    verdict: str
    reason_code: str
    compact_description: str = ""
    semantic_key: str = ""
    emotion: str = ""
    speech_act: str = ""
    intensity: int = 0
    persona_score: float = 0.0
    usage_type: StickerUsageType = StickerUsageType.REACTION

    @property
    def accepted(self) -> bool:
        return self.verdict == "ACCEPTED"


class StickerCheckPipeline:
    """Normalize deterministic and structured-AI Check results.

    File decoding, remote-download safety and ownership are performed by the
    media/storage boundaries.  This layer refuses incomplete AI judgements and
    applies the current verdict/reason-code contract. Player imports have their
    own source policy; they are not a boolean exception to a persona gate.
    """

    _REJECTION_CODES = {
        "UNSAFE_CONTENT": "UNSAFE_CONTENT",
        "NOT_A_STICKER": "NOT_A_STICKER",
        "COLLECTION_INTENT_MISMATCH": "COLLECTION_INTENT_MISMATCH",
        "CHARACTER_IDENTITY_MISMATCH": "CHARACTER_IDENTITY_MISMATCH",
        "TEXT_QUALITY": "TEXT_QUALITY",
        "ADMIN_PROHIBITION": "ADMIN_PROHIBITION",
        "WATERMARK_PRESENT": "WATERMARK_PRESENT",
        "INSUFFICIENT_VISUAL_EVIDENCE": "INSUFFICIENT_VISUAL_EVIDENCE",
    }

    @classmethod
    def check_deterministic(
        cls,
        *,
        source_kind: str,
        owner_matches: bool,
        decoded_image: bool,
        mime_type: str,
        byte_size: int,
        width: int,
        height: int,
        capacity_available: bool = True,
    ) -> StickerCheckResult | None:
        source = str(source_kind).lower().strip()
        if source not in {"web", "generated", "player", "upload"}:
            return StickerCheckResult("REJECTED", "UNSUPPORTED_SOURCE")
        if not owner_matches:
            return StickerCheckResult("REJECTED", "OWNER_MISMATCH")
        if not decoded_image or not str(mime_type).lower().startswith("image/"):
            return StickerCheckResult("REJECTED", "INVALID_IMAGE")
        if int(byte_size) <= 0 or int(width) <= 0 or int(height) <= 0:
            return StickerCheckResult("REJECTED", "INVALID_IMAGE_METADATA")
        if not capacity_available:
            return StickerCheckResult("QUARANTINED", "LIBRARY_CAPACITY")
        return None

    @classmethod
    def normalize_ai_result(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        source_kind: str,
        persona_bound: bool = False,
    ) -> StickerCheckResult:
        del source_kind
        if not isinstance(payload, Mapping) or type(payload.get("accepted")) is not bool:
            return StickerCheckResult("QUARANTINED", "INCOMPLETE_AI_CHECK")
        description = _compact_text(payload.get("compact_description"), 100)
        if not description:
            return StickerCheckResult("QUARANTINED", "INCOMPLETE_AI_CHECK")
        if payload["accepted"] is False:
            category = str(payload.get("rejection_category") or "").strip().upper()
            reason_code = cls._REJECTION_CODES.get(category)
            if reason_code is None or not _compact_text(payload.get("reason"), 500):
                return StickerCheckResult("QUARANTINED", "INCOMPLETE_AI_CHECK")
            return StickerCheckResult(
                "REJECTED",
                reason_code,
                compact_description=description,
            )
        try:
            intensity = max(0, min(5, int(payload.get("intensity", 0))))
        except (TypeError, ValueError):
            return StickerCheckResult("QUARANTINED", "INVALID_AI_CHECK")
        semantic = normalize_semantic_key(payload.get("semantic_key"))
        usage_type = _usage_type(payload, semantic=semantic)
        return StickerCheckResult(
            "ACCEPTED",
            "CHECK_PASSED",
            compact_description=description,
            semantic_key=semantic,
            emotion=_compact_text(payload.get("emotion"), 48),
            speech_act=_compact_text(payload.get("speech_act"), 48),
            intensity=intensity,
            persona_score=1.0 if persona_bound else 0.5,
            usage_type=usage_type,
        )


class StickerService:
    """Build stable randomized worksets without leaking durable identifiers."""

    def __init__(
        self,
        repository: StickerRepository,
        *,
        profiles: ProfilesRepositoryPort,
    ) -> None:
        self.repository = repository
        self.profiles = profiles
        self.ranker = StickerRetrievalRanker()

    async def is_enabled(self, profile_id: str, scope: str) -> bool:
        config = await self.repository.get_sticker_config(profile_id, scope)
        return bool(config and config.enabled)

    async def is_enabled_for_instance(self, profile_id: str, instance_id: str) -> bool:
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            return False
        return await self.is_enabled(profile_id, str(instance.scope))

    async def require_enabled(self, profile_id: str, instance_id: str) -> None:
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles,
            profile_id,
            instance_id=instance_id,
        )
        policy.require_enabled()

    async def require_source(
        self,
        profile_id: str,
        instance_id: str,
        source_kind: StickerSourceKind | str,
    ) -> None:
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles,
            profile_id,
            instance_id=instance_id,
        )
        policy.require_source(source_kind)

    async def build_workset(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        current_text: str,
        recent_texts: Sequence[str] = (),
        candidate_token_limit: int,
        meter: ConservativeTokenMeter | None = None,
    ) -> StickerWorkset:
        token_limit = max(0, int(candidate_token_limit))
        if token_limit <= 0 or not await self.is_enabled_for_instance(profile_id, instance_id):
            return StickerWorkset(token_limit=token_limit)
        library_limit, requirements = await self._library_settings(profile_id, instance_id)
        rows = await self.repository.list_sticker_items(
            profile_id, instance_id, query="", limit=library_limit
        )
        meter = meter or ConservativeTokenMeter()
        usage = await self._usage_window(profile_id, instance_id, run_id)
        ranked = self.ranker.rank(
            rows,
            current_text=current_text,
            recent_texts=recent_texts,
            requirements=requirements,
            run_id=run_id,
            usage=usage,
        )
        selected = _interleaved_stickers(ranked, limit=library_limit)
        # First determine how many compact projections fit.  Refs are only
        # allocated for rows that can actually enter the prompt.
        fitted = _fit_ranked_to_token_budget(
            selected,
            token_limit=token_limit,
            meter=meter,
            describe=self._description,
        )
        if not await self.is_enabled_for_instance(profile_id, instance_id):
            return StickerWorkset(token_limit=token_limit)
        projections = await self._project_rows(
            profile_id=profile_id,
            instance_id=instance_id,
            run_id=run_id,
            ranked=fitted,
        )
        actual_used = sum(
            meter.count_text(item.content) + meter.MESSAGE_OVERHEAD for item in projections
        )
        return StickerWorkset(
            items=tuple(projections),
            visible_refs=frozenset(item.sticker_ref for item in projections),
            token_limit=token_limit,
            used_tokens=actual_used,
        )

    async def search(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        query: str,
        preference: str = "任意",
        candidate_token_limit: int = DEFAULT_SEARCH_TOKEN_LIMIT,
        limit: int | None = None,
        meter: ConservativeTokenMeter | None = None,
    ) -> tuple[StickerProjection, ...]:
        await self.require_enabled(profile_id, instance_id)
        normalized = _compact_text(query, 240)
        library_limit, _requirements = await self._library_settings(profile_id, instance_id)
        token_query = structured_search_tokens(normalized)
        if token_query:
            rows = await self.repository.search_sticker_items_indexed(
                profile_id,
                instance_id,
                tokens=token_query,
                status="ACTIVE",
                limit=library_limit,
            )
            if not rows:
                rows = await self.repository.list_sticker_items(
                    profile_id, instance_id, query="", limit=library_limit
                )
        else:
            rows = await self.repository.list_sticker_items(
                profile_id, instance_id, query="", limit=library_limit
            )
        usage = await self._usage_window(profile_id, instance_id, run_id)
        preferred_rows = _filter_preference(rows, preference)
        ranked = self.ranker.rank(
            preferred_rows,
            current_text=normalized,
            run_id=run_id,
            usage=usage,
        )
        # Natural-language search does not silently return unrelated stickers.
        matched = (
            [item for item in ranked if item.relevance > 0 or item.affect_match > 0]
            if normalized
            else ranked
        )
        maximum = library_limit if limit is None else min(library_limit, max(0, int(limit)))
        selected = _interleaved_stickers(matched, limit=maximum)
        meter = meter or ConservativeTokenMeter()
        selected = _fit_ranked_to_token_budget(
            selected,
            token_limit=max(0, int(candidate_token_limit)),
            meter=meter,
            describe=self._description,
        )
        await self.require_enabled(profile_id, instance_id)
        return tuple(
            await self._project_rows(
                profile_id=profile_id,
                instance_id=instance_id,
                run_id=run_id,
                ranked=selected,
            )
        )

    async def record_usage(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        sticker_ref: str,
        delivery_status: str,
        outbox_id: int | None = None,
        expression_ordinal: int | None = None,
    ) -> str:
        """Persist a text projection after the delivery boundary reports status."""
        rows = await self.repository.resolve_sticker_run_refs(
            profile_id, instance_id, run_id, [str(sticker_ref)]
        )
        if len(rows) != 1:
            raise ValueError("sticker ref expired, archived, or belongs to another run")
        row = rows[0]
        projection = f"[表情包] {self._description(row)}"
        await self.repository.record_sticker_usage(
            profile_id,
            instance_id,
            item_id=self._item_id(row),
            run_id=run_id,
            sticker_ref=str(sticker_ref),
            compact_projection=projection,
            delivery_status=str(delivery_status).upper(),
            outbox_id=outbox_id,
            expression_ordinal=expression_ordinal,
        )
        return projection

    async def _project_rows(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        ranked: Sequence[RankedSticker],
    ) -> list[StickerProjection]:
        if not ranked:
            return []
        await self.require_enabled(profile_id, instance_id)
        rows = [item.row for item in ranked]
        rank_by_item = {item.item_id: item for item in ranked}
        order_score = {
            item.item_id: float(len(ranked) - index) for index, item in enumerate(ranked)
        }
        item_ids = [self._item_id(row) for row in rows]
        created = await self.repository.create_sticker_run_refs(
            profile_id,
            instance_id,
            run_id,
            item_ids,
            expires_at=datetime.now(UTC) + RUN_REF_TTL,
        )
        refs_by_item = {entry.item_id: entry.sticker_ref for entry in created}
        output: list[StickerProjection] = []
        for row in rows:
            item_id = self._item_id(row)
            ref = refs_by_item.get(item_id)
            if not ref:
                continue
            ranked_item = rank_by_item[item_id]
            output.append(
                StickerProjection(
                    sticker_ref=ref,
                    compact_description=self._description(row),
                    # ContextCompiler performs one final token fit.  A descending
                    # ordinal preserves the ranker's exact lexicographic order
                    # (including recent-use fallback pools) without allowing a
                    # weighted float to invert two ranking dimensions.
                    score=order_score[item_id],
                    item_id=item_id,
                    semantic_key=normalize_semantic_key(row.semantic_key),
                    visual_group=row.visual_group,
                    emotion=_compact_text(row.emotion, 48),
                    speech_act=_compact_text(row.speech_act, 48),
                    intensity=_intensity(row),
                    usage_type=row.usage_type,
                    visible_text=_compact_text(row.ocr_text or row.visible_text, 40),
                    is_animated=bool(row.is_animated),
                    recently_used=bool(ranked_item.recently_used),
                    current_run_visible=True,
                )
            )
        return output

    def _select(
        self, rows: Sequence[StickerItem], *, current_text: str, run_id: int, limit: int
    ) -> list[StickerItem]:
        ranked = self.ranker.rank(rows, current_text=current_text, run_id=run_id)
        return [item.row for item in _interleaved_stickers(ranked, limit=limit)]

    def _scored(
        self, rows: Sequence[StickerItem], current_text: str, run_id: int
    ) -> list[tuple[float, StickerItem]]:
        return [
            (item.display_score, item.row)
            for item in self.ranker.rank(rows, current_text=current_text, run_id=run_id)
        ]

    async def _library_settings(self, profile_id: str, instance_id: str) -> tuple[int, str]:
        library_limit = 1000
        requirements = ""
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is not None:
            config = await self.repository.get_sticker_config(profile_id, instance.scope)
            # A chat can see one full shared base library plus one equally
            # bounded private overlay.  Token fitting, not an item-count cap,
            # decides how many projections enter the model context.
            library_limit = max(1, int(config.library_limit)) * 2
            requirements = config.requirements.strip()
        return library_limit, requirements

    async def _usage_window(
        self, profile_id: str, instance_id: str, run_id: int | str
    ) -> StickerUsageWindow:
        payload = await self.repository.sticker_recent_run_usage(
            profile_id,
            instance_id,
            current_run_id=run_id,
            item_window=10,
            cluster_window=3,
        )
        return StickerUsageWindow.from_payload(payload)

    @staticmethod
    def _item_id(row: StickerItem) -> str:
        return row.item_id

    @staticmethod
    def _description(row: StickerItem) -> str:
        return _compact_text(row.compact_description, 100) or "表情包"


class StickerCommandContext:
    """Mutable per-run command state; never shared between Main Core runs."""

    def __init__(
        self, service: StickerService, *, profile_id: str, instance_id: str, run_id: int
    ) -> None:
        self.service = service
        self.profile_id = str(profile_id)
        self.instance_id = str(instance_id)
        self.run_id = int(run_id)
        self.search_calls = 0
        self.search_token_limit = DEFAULT_SEARCH_TOKEN_LIMIT
        self._projections: dict[str, StickerProjection] = {}
        self._import_sources: dict[str, tuple[str, str]] = {}
        self._import_source_by_asset: dict[str, str] = {}
        self._import_intents: dict[str, StickerImportIntent] = {}
        self._disable_intents: dict[str, StickerProjection] = {}

    def register_workset(self, workset: StickerWorkset) -> None:
        self._projections.update({item.sticker_ref: item for item in workset.items})
        self.search_token_limit = max(0, int(workset.token_limit))

    def snapshot_batch_state(self) -> StickerCommandBatchSnapshot:
        """Capture every run-scoped sticker mutation made by ordered commands."""

        return StickerCommandBatchSnapshot(
            owner=self,
            search_calls=self.search_calls,
            search_token_limit=self.search_token_limit,
            projections=tuple(self._projections.items()),
            import_sources=tuple(self._import_sources.items()),
            import_source_by_asset=tuple(self._import_source_by_asset.items()),
            import_intents=tuple(self._import_intents.items()),
            disable_intents=tuple(self._disable_intents.items()),
        )

    def restore_batch_state(self, snapshot: StickerCommandBatchSnapshot) -> None:
        """Restore a failed ordered batch without retaining hidden mutations."""

        if not isinstance(snapshot, StickerCommandBatchSnapshot):
            raise TypeError("表情包批次快照类型无效")
        if snapshot.owner is not self:
            raise ValueError("表情包批次快照不属于当前运行上下文")
        self.search_calls = snapshot.search_calls
        self.search_token_limit = snapshot.search_token_limit
        self._projections = dict(snapshot.projections)
        self._import_sources = dict(snapshot.import_sources)
        self._import_source_by_asset = dict(snapshot.import_source_by_asset)
        self._import_intents = dict(snapshot.import_intents)
        self._disable_intents = dict(snapshot.disable_intents)

    def register_import_source(self, source_kind: str, asset_id: str) -> str:
        kind = str(source_kind).lower().strip()
        asset = str(asset_id).strip()
        if kind not in {"player", "web", "generated"} or not asset:
            raise ValueError("表情包来源不属于本轮可用图片")
        existing = self._import_source_by_asset.get(asset)
        if existing:
            existing_kind, _existing_asset = self._import_sources[existing]
            if existing_kind != kind:
                raise ValueError("表情包来源类型与本轮既有来源不一致")
            return existing
        ref = sticker_import_source_ref(
            self.profile_id,
            self.instance_id,
            self.run_id,
            kind,
            asset,
        )
        self._import_sources[ref] = (kind, asset)
        self._import_source_by_asset[asset] = ref
        return ref

    async def search(
        self, query: str = "", *, preference: str = "任意"
    ) -> tuple[StickerProjection, ...]:
        await self.service.require_enabled(self.profile_id, self.instance_id)
        if self.search_calls >= MAX_SEARCH_CALLS:
            raise ValueError("本轮最多搜索两次表情包")
        self.search_calls += 1
        items = await self.service.search(
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            run_id=self.run_id,
            query=query,
            preference=preference,
            candidate_token_limit=self.search_token_limit,
        )
        disabled_item_ids = set(self._disable_intents)
        items = tuple(item for item in items if item.item_id not in disabled_item_ids)
        self._projections.update({item.sticker_ref: item for item in items})
        return items

    @property
    def import_intents(self) -> tuple[StickerImportIntent, ...]:
        return tuple(self._import_intents.values())

    @property
    def disable_intents(self) -> tuple[StickerProjection, ...]:
        return tuple(self._disable_intents.values())

    @property
    def pending_disable_item_ids(self) -> tuple[str, ...]:
        """Return a read-only, de-duplicated envelope for the final transaction."""

        return tuple(self._disable_intents)

    async def propose_import(self, source_ref: str) -> StickerImportIntent:
        ref = str(source_ref).strip()
        canonical_ref = (
            ref if ref in self._import_sources else self._import_source_by_asset.get(ref)
        )
        if not canonical_ref:
            raise ValueError("来源短引用不属于本轮可用图片")
        source_kind, asset_id = self._import_sources[canonical_ref]
        await self.service.require_source(self.profile_id, self.instance_id, source_kind)
        intent = StickerImportIntent(
            source_ref=canonical_ref,
            source_kind=StickerSourceKind(source_kind.upper()),
            source_asset_id=asset_id,
        )
        self._import_intents[canonical_ref] = intent
        return intent

    async def validate_selection(self, refs: Iterable[str]) -> list[StickerProjection]:
        await self.service.require_enabled(self.profile_id, self.instance_id)
        selected_refs = [str(ref).strip() for ref in refs if str(ref).strip()]
        unknown = [
            ref
            for ref in selected_refs
            if ref not in self._projections
            or self._projections[ref].item_id in self._disable_intents
        ]
        if unknown:
            raise ValueError("表情包短引用不属于本轮可见范围：" + "、".join(unknown))
        if selected_refs:
            await self.service.require_enabled(self.profile_id, self.instance_id)
            resolved = await self.service.repository.resolve_sticker_run_refs(
                self.profile_id, self.instance_id, self.run_id, selected_refs
            )
            if len(resolved) != len(selected_refs):
                raise ValueError("表情包短引用已失效或不属于本轮")
            await self.service.require_enabled(self.profile_id, self.instance_id)
        return [self._projections[ref] for ref in selected_refs]

    async def validate_reinforcement(self, sticker_ref: str) -> StickerProjection:
        selected = await self.validate_selection([sticker_ref])
        if not selected:
            raise ValueError("表情包短引用不能为空")
        return selected[0]

    async def propose_disable(self, sticker_ref: str) -> StickerProjection:
        """Stage one exact instance disable for the successful commit boundary."""

        selected = await self.validate_selection([sticker_ref])
        if not selected:
            raise ValueError("表情包短引用不能为空")
        projection = selected[0]
        self._disable_intents[projection.item_id] = projection
        self._projections.pop(projection.sticker_ref, None)
        return projection

    async def apply_disables(self) -> None:
        """Persist staged instance disables after the Main Core decision commits."""

        for projection in self.disable_intents:
            await self.service.repository.disable_sticker_item_for_instance(
                self.profile_id,
                self.instance_id,
                projection.item_id,
            )
        self._disable_intents.clear()

    async def apply_reinforcements(self, intents: Iterable[Mapping[str, Any]]) -> None:
        for intent in intents:
            projection = await self.validate_reinforcement(str(intent.get("sticker_ref") or ""))
            await self.service.repository.reinforce_sticker_item(
                self.profile_id,
                self.instance_id,
                projection.item_id,
                strength=max(1, min(5, int(intent.get("strength") or 1))),
                reason=_compact_text(intent.get("reason"), 240),
                run_id=self.run_id,
            )


__all__ = [
    "MAX_SEARCH_CALLS",
    "DEFAULT_SEARCH_TOKEN_LIMIT",
    "StickerProjection",
    "StickerWorkset",
    "StickerCommandBatchSnapshot",
    "StickerCheckResult",
    "StickerCheckPipeline",
    "StickerService",
    "StickerCommandContext",
    "normalize_semantic_key",
]
