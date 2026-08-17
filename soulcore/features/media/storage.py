from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import re
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .domain import (
    InboundMediaRegistrationState,
    MediaAsset,
    MediaFileStatus,
    MediaInspectionStatus,
    MediaOrigin,
    MediaPurpose,
)
from .files import MediaFileStore, await_cancellation_safe_file_store
from .inbound import InboundMediaSource
from .inspection import (
    _now,
    generate_media_asset_id,
    infer_media_root,
    inspect_animation_bytes,
)
from .locator_io import download_public_attachment
from .ports import MediaRepositoryPort, StickerPromotionRepositoryPort

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_INBOUND_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_ANIMATION_DURATION_MS = 30_000
MAX_ANIMATION_DECODED_PIXELS = 240_000_000
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/x-png": "image/png"}
_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_INBOUND_MEDIA_KINDS = frozenset({"audio", "file", "video"})


class _StickerAcceptanceState(Enum):
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    UNKNOWN = "UNKNOWN"


class _MediaRegistrationState(Enum):
    OWNED = "OWNED"
    UNOWNED = "UNOWNED"
    UNKNOWN = "UNKNOWN"


class MediaStorageCoordinator:
    """Crash-safe orchestration of DB metadata and stable media files."""

    def __init__(
        self,
        repository: MediaRepositoryPort,
        store: MediaFileStore,
        sticker_repository: StickerPromotionRepositoryPort,
    ) -> None:
        self.repository = repository
        self.store = store
        self.stickers = sticker_repository

    @classmethod
    def for_repository(
        cls,
        repository: MediaRepositoryPort,
        sticker_repository: StickerPromotionRepositoryPort,
    ) -> MediaStorageCoordinator:
        return cls(
            repository,
            MediaFileStore(infer_media_root(repository.db.path)),
            sticker_repository,
        )

    async def ingest_bytes(
        self,
        profile_id: str,
        instance_id: str,
        data: bytes,
        *,
        origin: MediaOrigin = MediaOrigin.USER_INPUT,
        purpose: MediaPurpose = MediaPurpose.NORMAL_IMAGE,
        declared_mime: str | None = None,
        asset_id: str | None = None,
        delivery_status: str = "NOT_SENT",
        inspection_status: MediaInspectionStatus = MediaInspectionStatus.PENDING,
        core_run_id: int | None = None,
        ai_task_id: int | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaAsset:
        opaque_id = asset_id or generate_media_asset_id()
        planned = self.store.plan_store_bytes(
            asset_id=opaque_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=declared_mime,
        )
        effective_metadata = dict(metadata or {})
        if int(planned.frame_count or 1) > 1:
            animation = inspect_animation_bytes(data)
            effective_metadata.update(
                {
                    "is_animated": bool(animation["animated"]),
                    "duration_ms": int(animation["duration_ms"]),
                }
            )
        cleanup_guard = await self.repository.guard_unregistered_media_file(
            profile_id,
            instance_id,
            planned,
            reason="MEDIA_ASSET_REGISTRATION",
        )
        stored = self.store.store_bytes(
            asset_id=opaque_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=declared_mime,
        )
        if stored != planned:
            raise RuntimeError("media storage changed after durable planning")
        try:
            return await self.repository.create_media_asset(
                profile_id,
                instance_id,
                stored,
                origin=origin,
                purpose=purpose,
                delivery_status=delivery_status,
                inspection_status=inspection_status,
                core_run_id=core_run_id,
                ai_task_id=ai_task_id,
                expires_at=expires_at,
                metadata=effective_metadata,
                cleanup_guard_id=cleanup_guard.cleanup_id,
            )
        except BaseException as original:
            state, authority_error = await self._media_registration_state(
                profile_id,
                instance_id,
                stored,
            )
            if authority_error is not None:
                original.add_note(
                    "media registration state could not be confirmed: "
                    f"{type(authority_error).__name__}: {authority_error}"
                )
            if state is _MediaRegistrationState.UNOWNED:
                try:
                    self.store.delete(stored.relative_path)
                    await self.repository.complete_runtime_file_cleanup(cleanup_guard.cleanup_id)
                except BaseException as cleanup_error:
                    original.add_note(
                        "unregistered media cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise

    async def ingest_generated_bytes(
        self,
        profile_id: str,
        instance_id: str,
        data: bytes,
        *,
        core_run_id: int,
        declared_mime: str | None = None,
        ai_task_id: int | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaAsset:
        return await self.ingest_bytes(
            profile_id,
            instance_id,
            data,
            origin=MediaOrigin.GENERATED,
            purpose=MediaPurpose.GENERATED_IMAGE,
            declared_mime=declared_mime,
            core_run_id=core_run_id,
            ai_task_id=ai_task_id,
            expires_at=expires_at or (_now() + timedelta(hours=24)),
            metadata=metadata,
        )

    async def ingest_inbound_attachment(
        self,
        profile_id: str,
        instance_id: str,
        *,
        message_id: int,
        source: InboundMediaSource,
        media_kind: str,
        ordinal: int,
        display_name: str = "",
    ) -> MediaAsset:
        """Resolve one live platform component into an owned opaque asset.

        Platform locators are consumed only inside this immediate ingest
        boundary.  The database receives a safe leaf display name and asset ID,
        never the URL or local path used to obtain the bytes.
        """

        kind = str(media_kind or "").strip().lower()
        if kind not in _INBOUND_MEDIA_KINDS:
            raise ValueError("unsupported inbound media kind")
        data, declared_mime = await self._read_inbound_attachment(source)
        opaque_id = (
            "ma_in_"
            + hashlib.sha256(
                f"{profile_id}\0{instance_id}\0{int(message_id)}\0{kind}\0{int(ordinal)}".encode()
            ).hexdigest()[:32]
        )

        async def cleanup_cancelled_write(stored_file: Any) -> None:
            existing = await self.repository.get_media_asset(
                opaque_id,
                profile_id=profile_id,
                instance_id=instance_id,
            )
            if existing is None or str(existing.storage_relpath or "") != stored_file.relative_path:
                self.store.delete(stored_file.relative_path)

        stored = await await_cancellation_safe_file_store(
            self.store.store_inbound_attachment_bytes,
            cleanup_after_cancel=cleanup_cancelled_write,
            asset_id=opaque_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            media_kind=kind,
            declared_mime=declared_mime,
        )
        safe_name = self._safe_attachment_name(display_name)
        registration_attempted = False
        try:
            if not await asyncio.to_thread(self.store.verify, stored):
                raise OSError("stored inbound attachment failed integrity verification")
            registration_attempted = True
            asset = await self.repository.register_inbound_media_asset(
                profile_id,
                instance_id,
                stored,
                message_id=int(message_id),
                ordinal=int(ordinal),
                delivery_status="RECEIVED",
                inspection_status=MediaInspectionStatus.READY,
                metadata={
                    "source": "inbound",
                    "media_kind": kind,
                    "display_name": safe_name,
                    "message_id": int(message_id),
                    "ordinal": int(ordinal),
                },
            )
            return asset
        except BaseException as original:
            try:
                state, authoritative = await self.repository.inspect_inbound_media_registration(
                    profile_id,
                    instance_id,
                    stored,
                    message_id=int(message_id),
                    ordinal=int(ordinal),
                )
            except BaseException as inspection_error:
                original.add_note(
                    "inbound attachment registration state could not be confirmed: "
                    f"{type(inspection_error).__name__}: {inspection_error}"
                )
                raise original from inspection_error
            if state is InboundMediaRegistrationState.UNOWNED:
                self.store.delete(stored.relative_path)
            if (
                registration_attempted
                and state is InboundMediaRegistrationState.COMMITTED
                and authoritative is not None
                and isinstance(original, Exception)
            ):
                return authoritative
            raise

    async def _media_registration_state(
        self,
        profile_id: str,
        instance_id: str,
        stored: Any,
    ) -> tuple[_MediaRegistrationState, BaseException | None]:
        """Classify ownership without unlinking through an uncertain commit."""

        try:
            existing = await asyncio.shield(
                self.repository.get_media_asset(
                    stored.asset_id,
                    profile_id=profile_id,
                    instance_id=instance_id,
                )
            )
        except BaseException as exc:
            return _MediaRegistrationState.UNKNOWN, exc
        if existing is None:
            return _MediaRegistrationState.UNOWNED, None
        if (
            existing.sha256 == stored.sha256
            and existing.storage_relpath == stored.relative_path
            and int(existing.byte_size) == int(stored.byte_size)
        ):
            return _MediaRegistrationState.OWNED, None
        if existing.storage_relpath != stored.relative_path:
            return _MediaRegistrationState.UNOWNED, None
        return _MediaRegistrationState.UNKNOWN, None

    async def _read_inbound_attachment(
        self, source: InboundMediaSource
    ) -> tuple[bytes, str | None]:
        if source.data:
            if len(source.data) > MAX_INBOUND_ATTACHMENT_BYTES:
                raise ValueError("media size is invalid")
            return source.data, source.mime_type
        if source.locator:
            return await self._read_attachment_locator(source.locator, allow_local=False)
        raise ValueError("no usable media source")

    async def _read_attachment_locator(
        self, locator: str, *, allow_local: bool
    ) -> tuple[bytes, str | None]:
        value = str(locator or "").strip()
        if not value:
            raise ValueError("empty media locator")
        if value.startswith("base64://"):
            return self._decode_attachment_base64(value[9:]), None
        if value.startswith("data:") and ";base64," in value[:256]:
            header, encoded = value.split(",", 1)
            return self._decode_attachment_base64(encoded), header[5:].split(";", 1)[0]
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme in {"http", "https"}:
            return await asyncio.to_thread(self._download_public_attachment, value)
        value = self._local_attachment_path(parsed, value)
        return await self._read_local_attachment(value, allow_local=allow_local)

    @staticmethod
    def _decode_attachment_base64(value: str) -> bytes:
        data = base64.b64decode(value, validate=True)
        if not data or len(data) > MAX_INBOUND_ATTACHMENT_BYTES:
            raise ValueError("media size is invalid")
        return data

    @staticmethod
    def _local_attachment_path(parsed: Any, value: str) -> str:
        if parsed.scheme == "file":
            value = urllib.request.url2pathname(parsed.path)
            if parsed.netloc:
                value = f"//{parsed.netloc}{value}"
            if len(value) >= 3 and value[0] in {"/", "\\"} and value[2] == ":":
                value = value[1:]
        elif parsed.scheme and not (
            len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in {"/", "\\"}
        ):
            raise ValueError("unsupported media locator scheme")
        return value

    @staticmethod
    async def _read_local_attachment(value: str, *, allow_local: bool) -> tuple[bytes, str | None]:
        if not allow_local:
            raise ValueError("serialized local media paths are not trusted")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError("resolved media file does not exist")
        if path.stat().st_size > MAX_INBOUND_ATTACHMENT_BYTES:
            raise ValueError("resolved media file is too large")
        data = await asyncio.to_thread(path.read_bytes)
        if not data:
            raise ValueError("resolved media file is empty")
        mime, _ = mimetypes.guess_type(path.name)
        return data, mime

    @staticmethod
    def _safe_attachment_name(value: str) -> str:
        leaf = re.split(r"[\\/]", str(value or ""))[-1]
        leaf = "".join(ch for ch in leaf if ch >= " " and ch != "\x7f").strip()
        return leaf[:128]

    @classmethod
    def _download_public_attachment(cls, url: str) -> tuple[bytes, str | None]:
        return download_public_attachment(
            url,
            max_bytes=MAX_INBOUND_ATTACHMENT_BYTES,
            timeout_seconds=20,
        )

    async def reserve_sticker_asset(
        self,
        profile_id: str,
        instance_id: str,
        source_asset_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        """Copy checked bytes into the profile-owned, hash-deduplicated sticker store."""

        source = await self.repository.get_media_asset(
            source_asset_id, profile_id=profile_id, instance_id=instance_id
        )
        if source is None or source.file_status is not MediaFileStatus.AVAILABLE:
            raise ValueError("sticker source asset is unavailable or has a different owner")
        existing_asset = await self.stickers.find_sticker_asset_by_sha(profile_id, source.sha256)
        if existing_asset is not None and self.store.verify(existing_asset):
            return existing_asset, False
        if not source.storage_relpath:
            raise ValueError("sticker source has no retained file")
        path = self.store.absolute_path(source.storage_relpath)
        if not path.is_file() or not self.store.verify(source):
            raise ValueError("sticker source file is missing or failed hash verification")
        data = path.read_bytes()
        stored = await asyncio.to_thread(
            self.store.store_sticker_bytes,
            asset_id=f"sa_{source.sha256[:24]}_{uuid.uuid4().hex[:16]}",
            profile_id=profile_id,
            data=data,
            declared_mime=source.mime_type,
        )
        animation = inspect_animation_bytes(data)
        asset, created = await self.stickers.create_sticker_asset(
            profile_id,
            stored,
            duration_ms=int(animation["duration_ms"]),
        )
        if not created and asset.storage_relpath != stored.relative_path:
            self.store.delete(stored.relative_path)
        return asset, created

    async def promote_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
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
        visual_group: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        candidate = await self.stickers.get_sticker_candidate(profile_id, instance_id, candidate_id)
        if candidate is None:
            raise KeyError((profile_id, instance_id, candidate_id))
        promotion_metadata = dict(metadata or {})
        reserved, created_asset = await self.reserve_sticker_asset(
            profile_id,
            instance_id,
            candidate.source_asset_id,
            metadata={"sticker_candidate_id": candidate_id, **promotion_metadata},
        )
        try:
            result = await self.stickers.accept_sticker_candidate(
                profile_id,
                instance_id,
                candidate_id,
                reserved_asset_id=reserved.asset_id,
                compact_description=compact_description,
                compact_name=compact_name,
                visible_text=visible_text,
                ocr_text=ocr_text,
                usage_type=usage_type,
                vibe_tags=vibe_tags,
                search_keywords=search_keywords,
                search_index=search_index,
                semantic_key=semantic_key,
                emotion=emotion,
                speech_act=speech_act,
                intensity=intensity,
                persona_score=persona_score,
                phash=phash,
                dhash=dhash,
                frame_hashes=frame_hashes,
                visual_group=visual_group,
                metadata=promotion_metadata,
            )
        except BaseException as exc:
            state, read_error, interrupted = await self._authoritative_sticker_acceptance_state(
                profile_id,
                instance_id,
                candidate_id,
            )
            if read_error is not None:
                exc.add_note(
                    "sticker acceptance state could not be read authoritatively: "
                    f"{type(read_error).__name__}: {read_error}"
                )
            if state is _StickerAcceptanceState.ROLLED_BACK:
                # Acceptance did not commit. Remove the DB reservation first so a
                # concurrently accepted item can never be left pointing at a
                # deleted file.
                if created_asset:
                    deleted = await self.stickers.delete_unreferenced_sticker_asset(
                        reserved.asset_id
                    )
                    if deleted:
                        self.store.delete(reserved.storage_relpath)
                await self.stickers.quarantine_sticker_candidate(
                    profile_id,
                    instance_id,
                    candidate_id,
                    reason=f"PROMOTION_REJECTED:{type(exc).__name__}:{exc}",
                )
            if interrupted is not None:
                raise interrupted from exc
            raise
        # The acceptance transaction is now authoritative. Release bookkeeping
        # is independently idempotent: cancellation or failure here must leave
        # the accepted item and its durable file intact for a safe retry.
        await self.repository.mark_media_release_if_already_summarized(candidate.source_asset_id)
        return result

    async def _authoritative_sticker_acceptance_state(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
    ) -> tuple[
        _StickerAcceptanceState,
        BaseException | None,
        asyncio.CancelledError | None,
    ]:
        """Classify a lost acceptance result without cancelling its durable read."""

        read = asyncio.create_task(
            self.stickers.get_sticker_candidate(profile_id, instance_id, candidate_id)
        )
        interrupted: asyncio.CancelledError | None = None
        while True:
            try:
                current = await asyncio.shield(read)
                break
            except asyncio.CancelledError as exc:
                if read.cancelled():
                    return _StickerAcceptanceState.UNKNOWN, exc, interrupted
                interrupted = exc
            except BaseException as exc:
                return _StickerAcceptanceState.UNKNOWN, exc, interrupted
        if current is None:
            return _StickerAcceptanceState.UNKNOWN, None, interrupted
        if current.status.value == "ACCEPTED":
            return _StickerAcceptanceState.COMMITTED, None, interrupted
        return _StickerAcceptanceState.ROLLED_BACK, None, interrupted

    async def release_pending(self, *, limit: int = 100) -> list[Any]:
        pending = await self.repository.list_pending_media_releases(limit=limit)
        results: list[Any] = []
        for asset in pending:
            try:
                self.store.delete(asset.storage_relpath)
                results.append(
                    await self.repository.finalize_media_release(asset.asset_id, success=True)
                )
            except Exception as exc:
                results.append(
                    await self.repository.finalize_media_release(
                        asset.asset_id, success=False, error=f"{type(exc).__name__}: {exc}"
                    )
                )
        sticker_pending = await self.stickers.list_pending_sticker_releases(limit=limit)
        for sticker_asset in sticker_pending:
            claimed = await self.stickers.claim_sticker_asset_release(sticker_asset.asset_id)
            if claimed is None:
                continue
            self.store.delete(claimed.storage_relpath)
            results.append(claimed)
        return results

    async def cleanup_expired(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> list[MediaAsset]:
        await self.repository.mark_expired_media_for_release(now=now, limit=limit)
        await self.repository.mark_retention_elapsed_media_for_release(now=now, limit=limit)
        return await self.release_pending(limit=limit)

    async def reconcile(self, profile_id: str, instance_id: str) -> list[str]:
        missing: list[str] = []
        offset = 0
        while True:
            page = await self.repository.list_media_assets(
                profile_id,
                instance_id,
                file_status=MediaFileStatus.AVAILABLE,
                limit=1000,
                offset=offset,
            )
            if not page:
                break
            for asset in page:
                if not self.store.verify(asset):
                    missing.append(asset.asset_id)
                    await self.repository.mark_media_missing(
                        asset.asset_id, reason="media_file_missing_or_hash_mismatch"
                    )
            # Rows marked MISSING disappear from the filtered result; advancing
            # by the full page would skip later AVAILABLE rows. Only advance by
            # rows that remained in the result set.
            offset += sum(1 for asset in page if asset.asset_id not in missing)
            if len(page) < 1000:
                break
        return missing
