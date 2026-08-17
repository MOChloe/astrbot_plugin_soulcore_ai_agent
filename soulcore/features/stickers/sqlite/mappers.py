from __future__ import annotations

from .support import (
    CharacterIdentityReference,
    StickerAsset,
    StickerCandidate,
    StickerCandidateStatus,
    StickerCheckRevision,
    StickerCheckVerdict,
    StickerConfig,
    StickerItem,
    StickerItemStatus,
    StickerLibraryKind,
    StickerRunRef,
    StickerSourceKind,
    StickerUsage,
    StickerUsageType,
    _load,
    _parse,
    re,
    sqlite3,
    unicodedata,
)


class StickerRecordMappers:
    @staticmethod
    def _sticker_asset(row: sqlite3.Row) -> StickerAsset:
        return StickerAsset(
            sticker_asset_id=str(row["sticker_asset_id"]),
            profile_id=str(row["profile_id"]),
            canonical_sha256=str(row["canonical_sha256"]),
            storage_relpath=str(row["storage_relpath"]),
            mime_type=str(row["mime_type"]),
            file_extension=str(row["file_extension"]),
            byte_size=int(row["byte_size"]),
            width=int(row["width"]),
            height=int(row["height"]),
            is_animated=bool(row["is_animated"]),
            frame_count=max(1, int(row["frame_count"] or 1)),
            duration_ms=max(0, int(row["duration_ms"] or 0)),
            file_status=str(row["file_status"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _sticker_config(row: sqlite3.Row) -> StickerConfig:
        return StickerConfig(
            profile_id=str(row["profile_id"]),
            scope=str(row["scope"]),
            enabled=bool(row["enabled"]),
            player_collection_enabled=bool(row["player_collection_enabled"]),
            web_collection_enabled=bool(row["web_collection_enabled"]),
            generation_enabled=bool(row["generation_enabled"]),
            trigger_mode=str(row["trigger_mode"]),
            turn_threshold=int(row["turn_threshold"]),
            elapsed_hours=float(row["elapsed_hours"]),
            library_limit=int(row["library_limit"]),
            web_daily_limit=int(row["web_daily_limit"]),
            generated_daily_limit=int(row["generated_daily_limit"]),
            requirements=str(row["requirements"] or ""),
            version=int(row["version"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _character_identity_reference(row: sqlite3.Row) -> CharacterIdentityReference:
        return CharacterIdentityReference(
            reference_id=str(row["reference_id"]),
            profile_id=str(row["profile_id"]),
            scope=str(row["scope"]),
            asset_id=str(row["asset_id"]),
            storage_relpath=str(row["storage_relpath"]),
            mime_type=str(row["mime_type"]),
            file_extension=str(row["file_extension"]),
            sha256=str(row["sha256"]),
            byte_size=int(row["byte_size"]),
            width=int(row["width"]),
            height=int(row["height"]),
            frame_count=int(row["frame_count"]),
            duration_ms=int(row["duration_ms"]),
            label=str(row["label"]),
            identity_description=str(row["identity_description"]),
            metadata=_load(row["metadata_json"]) or {},
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _sticker_candidate(row: sqlite3.Row) -> StickerCandidate:
        return StickerCandidate(
            candidate_id=str(row["candidate_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            source_kind=StickerSourceKind(str(row["source_kind"])),
            source_asset_id=str(row["source_asset_id"]),
            source_ref=str(row["source_ref"]),
            target_library_id=str(row["target_library_id"]),
            status=StickerCandidateStatus(str(row["status"])),
            import_count=int(row["import_count"]),
            persona_fingerprint=str(row["persona_fingerprint"]),
            metadata=_load(row["metadata_json"]) or {},
            last_error=str(row["last_error"]),
            failure_stage=str(row["failure_stage"] or ""),
            retry_count=int(row["retry_count"] or 0),
            next_retry_at=_parse(row["next_retry_at"]),
            recoverable=bool(row["recoverable"]),
            accepted_item_id=str(row["accepted_item_id"] or ""),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _sticker_check(row: sqlite3.Row) -> StickerCheckRevision:
        return StickerCheckRevision(
            check_id=int(row["check_id"]),
            candidate_id=str(row["candidate_id"]),
            revision=int(row["revision"]),
            verdict=StickerCheckVerdict(str(row["verdict"])),
            compact_name=str(row["compact_name"]),
            compact_description=str(row["compact_description"]),
            visible_text=str(row["visible_text"]),
            usage_type=StickerUsageType(str(row["usage_type"])),
            semantic_key=str(row["semantic_key"]),
            emotion=str(row["emotion"]),
            speech_act=str(row["speech_act"]),
            intensity=int(row["intensity"]),
            persona_score=float(row["persona_score"]),
            reason=str(row["reason"]),
            backend_id=str(row["backend_id"]),
            model_id=str(row["model_id"]),
            details=_load(row["details_json"]) or {},
            created_at=_parse(row["created_at"]),
        )

    @staticmethod
    def _sticker_item(row: sqlite3.Row) -> StickerItem:
        return StickerItem(
            item_id=str(row["item_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            asset_id=str(row["asset_id"]),
            canonical_sha256=str(row["canonical_sha256"]),
            source_kind=StickerSourceKind(str(row["source_kind"])),
            library_id=str(row["library_id"]),
            library_kind=StickerLibraryKind(str(row["library_kind"])),
            scope=str(row["scope"]),
            usage_type=StickerUsageType(str(row["usage_type"])),
            compact_name=str(row["compact_name"]),
            compact_description=str(row["compact_description"]),
            visible_text=str(row["visible_text"] or ""),
            ocr_text=str(row["ocr_text"] or ""),
            vibe_tags=tuple(_load(row["vibe_tags_json"]) or ()),
            search_keywords=tuple(_load(row["search_keywords_json"]) or ()),
            search_index=str(row["search_index"] or ""),
            semantic_key=str(row["semantic_key"]),
            cluster_id=str(row["cluster_id"]),
            emotion=str(row["emotion"]),
            speech_act=str(row["speech_act"]),
            intensity=int(row["intensity"]),
            persona_score=float(row["persona_score"]),
            status=StickerItemStatus(str(row["status"])),
            import_count=int(row["import_count"]),
            reinforcement_score=float(row["reinforcement_score"]),
            usage_count=int(row["usage_count"]),
            last_used_at=_parse(row["last_used_at"]),
            metadata=_load(row["metadata_json"]) or {},
            created_at=_parse(row["created_at"]),
            phash=str(row["phash"] or ""),
            dhash=str(row["dhash"] or ""),
            visual_group=str(row["visual_group"] or ""),
            mime_type=str(row["mime_type"] or "image/png"),
            is_animated=bool(row["is_animated"]),
            frame_count=max(1, int(row["frame_count"] or 1)),
            duration_ms=max(0, int(row["duration_ms"] or 0)),
            representative_frame_hashes=tuple(_load(row["representative_frame_hashes_json"]) or ()),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _sticker_run_ref(row: sqlite3.Row) -> StickerRunRef:
        return StickerRunRef(
            sticker_ref=str(row["sticker_ref"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            run_id=str(row["run_id"]),
            item_id=str(row["item_id"]),
            compact_description=str(row["compact_description"]),
            expires_at=_parse(row["expires_at"]),
            created_at=_parse(row["created_at"]),
        )

    @staticmethod
    def _sticker_usage(row: sqlite3.Row) -> StickerUsage:
        return StickerUsage(
            usage_id=int(row["usage_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            item_id=str(row["item_id"]),
            run_id=str(row["run_id"]),
            sticker_ref=str(row["sticker_ref"]),
            compact_projection=str(row["compact_projection"]),
            delivery_status=str(row["delivery_status"]),
            outbox_id=(int(row["outbox_id"]) if row["outbox_id"] is not None else None),
            expression_ordinal=(
                int(row["expression_ordinal"]) if row["expression_ordinal"] is not None else None
            ),
            message_id=row["message_id"],
            created_at=_parse(row["created_at"]),
        )

    @staticmethod
    def _normalize_sticker_semantic(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(value)).casefold()
        return " ".join(re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized).split())

    @staticmethod
    def _sticker_hash_distance(first: str, second: str) -> int:
        left_frames = [value for value in str(first or "").split(".") if value]
        right_frames = [value for value in str(second or "").split(".") if value]
        distances: list[int] = []
        for left in left_frames[:4]:
            for right in right_frames[:4]:
                if len(left) != len(right):
                    continue
                try:
                    distances.append((int(left, 16) ^ int(right, 16)).bit_count())
                except ValueError:
                    continue
        return min(distances) if distances else 10_000
