from __future__ import annotations

from .item_acceptance import StickerAcceptanceContext, StickerAcceptanceTransaction
from .item_clear import StickerClearTransaction
from .item_description import StickerDescriptionContext, StickerDescriptionTransaction
from .support import (
    Any,
    Mapping,
    StickerCheckRevision,
    StickerItem,
    _dt,
    _now,
    math,
    uuid,
)


class StickerItemRecords:
    async def preflight_sticker_visual_capacity(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        phash: str,
        dhash: str,
    ) -> dict[str, Any]:
        """Reject saturated perceptual groups before any model-backed Check.

        This is an inexpensive early gate.  The acceptance transaction repeats
        the same capacity rule as the authoritative concurrency boundary.
        """

        candidate = await self.db.fetch_one(
            """SELECT target_library_id, source_kind FROM sticker_candidates
            WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?""",
            (candidate_id, profile_id, instance_id),
        )
        if candidate is None:
            raise KeyError((profile_id, instance_id, candidate_id))
        rows = await self.db.fetch_all(
            """SELECT f.visual_group, f.phash, f.dhash, i.source_kind
            FROM sticker_fingerprints f JOIN sticker_items i ON i.item_id = f.item_id
            WHERE f.library_id = ? AND i.status IN ('ACTIVE', 'NEEDS_REVIEW')""",
            (candidate["target_library_id"],),
        )
        visual_group = _nearest_visual_group(self, rows, phash=phash, dhash=dhash)
        total, auto_count = _visual_group_counts(rows, visual_group)
        is_auto = str(candidate["source_kind"]) != "PLAYER"
        return {
            "allowed": total < 6 and (not is_auto or auto_count < 4),
            "visual_group": visual_group,
            "total": total,
            "auto_count": auto_count,
        }

    async def clear_sticker_instance_data(
        self, profile_id: str, instance_id: str
    ) -> dict[str, Any]:
        instance = await self._profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        transaction = StickerClearTransaction(
            profile_id=profile_id,
            instance_id=instance_id,
            scope=str(instance.scope),
            now=_dt(_now()),
        )
        return await self.uow.run(transaction)

    async def list_sticker_checks(
        self,
        profile_id: str,
        instance_id: str,
        *,
        candidate_id: str | None = None,
        limit: int = 100,
    ) -> list[StickerCheckRevision]:
        values: list[Any] = [profile_id, instance_id]
        clause = ""
        if candidate_id is not None:
            clause = " AND c.candidate_id = ?"
            values.append(candidate_id)
        values.append(max(1, min(500, int(limit))))
        rows = await self.db.fetch_all(
            f"""SELECT r.* FROM sticker_check_revisions r
            JOIN sticker_candidates c ON c.candidate_id = r.candidate_id
            WHERE c.profile_id = ? AND c.instance_id = ? {clause}
            ORDER BY r.check_id DESC LIMIT ?""",
            values,
        )
        return [self._sticker_check(row) for row in rows]

    async def page_sticker_checks(
        self,
        profile_id: str,
        instance_id: str,
        *,
        statuses: tuple[str, ...] | list[str] = (),
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        verdicts = tuple(dict.fromkeys(str(value).upper() for value in statuses))
        size = max(1, min(100, int(page_size)))
        current = max(1, int(page))
        clauses = ["c.profile_id = ?", "c.instance_id = ?"]
        values: list[Any] = [profile_id, instance_id]
        if verdicts:
            placeholders = ",".join("?" for _ in verdicts)
            clauses.append(f"r.verdict IN ({placeholders})")
            values.extend(verdicts)
        where = " AND ".join(clauses)
        count = await self.db.fetch_one(
            f"""SELECT COUNT(*) amount FROM sticker_check_revisions r
            JOIN sticker_candidates c ON c.candidate_id = r.candidate_id
            WHERE {where}""",
            values,
        )
        rows = await self.db.fetch_all(
            f"""SELECT r.* FROM sticker_check_revisions r
            JOIN sticker_candidates c ON c.candidate_id = r.candidate_id
            WHERE {where} ORDER BY r.check_id DESC LIMIT ? OFFSET ?""",
            (*values, size, (current - 1) * size),
        )
        total = int(count["amount"] or 0) if count else 0
        return {
            "items": [self._sticker_check(row) for row in rows],
            "total": total,
            "page": current,
            "page_size": size,
            "page_count": max(1, math.ceil(total / size)),
        }

    async def find_sticker_item_by_sha(
        self,
        profile_id: str,
        instance_id: str,
        sha256: str,
    ) -> StickerItem | None:
        row = await self.db.fetch_one(
            """SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group
            FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            WHERE i.profile_id = ? AND i.library_id IN (
                SELECT visible.library_id FROM sticker_libraries visible
                JOIN character_instances current
                  ON current.profile_id = ? AND current.instance_id = ?
                WHERE visible.profile_id = current.profile_id AND (
                  (visible.library_kind = 'CORE' AND visible.scope = current.scope)
                  OR (visible.library_kind = 'PRIVATE'
                      AND visible.instance_id = current.instance_id)
                )
            )
            AND i.canonical_sha256 = ? AND i.status <> 'DELETED'""",
            (profile_id, profile_id, instance_id, str(sha256).lower()),
        )
        return self._sticker_item(row) if row is not None else None

    async def accept_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reserved_asset_id: str,
        compact_description: str = "",
        compact_name: str = "",
        visible_text: str = "",
        ocr_text: str = "",
        usage_type: str = "",
        vibe_tags: list[str] | tuple[str, ...] = (),
        search_keywords: list[str] | tuple[str, ...] = (),
        search_index: str = "",
        semantic_key: str = "",
        emotion: str = "",
        speech_act: str = "",
        intensity: int = 0,
        persona_score: float = 0.0,
        phash: str = "",
        dhash: str = "",
        frame_hashes: list[str] | tuple[str, ...] = (),
        representative_frame_hashes: list[str] | tuple[str, ...] = (),
        visual_group: str = "",
        metadata: Mapping[str, Any] | None = None,
        item_id: str = "",
    ) -> tuple[StickerItem, bool]:
        context = StickerAcceptanceContext(
            profile_id=profile_id,
            instance_id=instance_id,
            candidate_id=candidate_id,
            reserved_asset_id=reserved_asset_id,
            identifier=str(item_id).strip() or "si_" + uuid.uuid4().hex,
            compact_description=compact_description,
            compact_name=compact_name,
            visible_text=visible_text,
            ocr_text=ocr_text,
            usage_type=usage_type,
            vibe_tags=tuple(vibe_tags),
            search_keywords=tuple(search_keywords),
            search_index=search_index,
            semantic_key=semantic_key,
            emotion=emotion,
            speech_act=speech_act,
            intensity=intensity,
            persona_score=persona_score,
            phash=phash,
            dhash=dhash,
            frame_hashes=tuple(frame_hashes),
            representative_frame_hashes=tuple(representative_frame_hashes),
            visual_group=visual_group,
            metadata=metadata or {},
            now=_dt(_now()),
        )
        stored_id, created = await self.uow.run(StickerAcceptanceTransaction(self, context))
        item = await self.get_sticker_item(profile_id, instance_id, stored_id)
        assert item is not None
        return item, created

    async def get_sticker_item(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
    ) -> StickerItem | None:
        row = await self.db.fetch_one(
            """SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group
            FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            JOIN character_instances current
              ON current.profile_id = ? AND current.instance_id = ?
            WHERE i.item_id = ? AND i.profile_id = ? AND (
              (l.library_kind = 'CORE' AND l.scope = current.scope)
              OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id)
            )""",
            (profile_id, instance_id, item_id, profile_id),
        )
        return self._sticker_item(row) if row is not None else None

    async def update_sticker_item_description(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
        *,
        compact_description: str,
        visible_text: str = "",
        search_keywords: list[str] | tuple[str, ...] = (),
        metadata_update: Mapping[str, Any] | None = None,
        expected_description: str,
    ) -> StickerItem:
        description = str(compact_description or "").strip()[:100]
        if not description:
            raise ValueError("sticker description must not be empty")
        keywords = tuple(
            dict.fromkeys(
                str(value).strip()[:100] for value in search_keywords if str(value).strip()
            )
        )[:100]
        context = StickerDescriptionContext(
            profile_id=profile_id,
            instance_id=instance_id,
            item_id=item_id,
            description=description,
            visible_text=str(visible_text or "").strip()[:500],
            search_keywords=keywords,
            metadata_update=metadata_update or {},
            expected_description=str(expected_description or ""),
            now=_dt(_now()),
        )
        refreshed = await self.uow.run(StickerDescriptionTransaction(self, context))
        return self._sticker_item(refreshed)


def _nearest_visual_group(owner: Any, rows: list[Any], *, phash: str, dhash: str) -> str:
    visual_group = str(phash or dhash).split(".", 1)[0]
    nearest_distance = 10_000
    for row in rows:
        distance = min(
            owner._sticker_hash_distance(phash, str(row["phash"] or "")),
            owner._sticker_hash_distance(dhash, str(row["dhash"] or "")),
        )
        if distance <= 6 and distance < nearest_distance:
            nearest_distance = distance
            visual_group = str(row["visual_group"] or row["phash"] or visual_group)
    return visual_group


def _visual_group_counts(rows: list[Any], visual_group: str) -> tuple[int, int]:
    total = 0
    auto_count = 0
    for row in rows:
        if not visual_group or str(row["visual_group"] or "") != visual_group:
            continue
        total += 1
        if str(row["source_kind"]) != "PLAYER":
            auto_count += 1
    return total, auto_count
