"""Single-file upload handling for explicit sticker intake batches."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from ....features.media.domain import MediaInspectionStatus, MediaOrigin, MediaPurpose
from ....features.media.inspection import inspect_image_bytes
from ....features.stickers.domain import StickerIntakeEntryStatus, StickerSourceKind
from ....features.stickers.policy import load_sticker_runtime_policy


class StickerIntakeUploadMixin:
    @staticmethod
    def _intake_user_prompt(raw: Any) -> str:
        prompt = str(raw or "").strip()
        if len(prompt) > 500:
            raise ValueError("一次性搜索提示词不得超过500字")
        return prompt

    @staticmethod
    def _intake_manifest_entry(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("上传清单格式无效")
        client_id = str(raw.get("client_entry_id") or "").strip()
        if not client_id or len(client_id) > 128:
            raise ValueError("每个文件都需要有效的 client_entry_id")
        size = int(raw.get("byte_size") or 0)
        if size < 1 or size > 5 * 1024 * 1024:
            raise ValueError("单张图片不得超过5 MiB")
        mime = str(raw.get("mime_type") or "").strip().lower()
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise ValueError("仅支持 JPEG、PNG、WebP 和 GIF")
        return {
            "client_entry_id": client_id,
            "filename": Path(str(raw.get("filename") or "")).name[:160],
            "mime_type": mime,
            "byte_size": size,
        }

    async def upload_sticker_intake_image(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        session, entry = await self._resolve_intake_upload(
            profile_id,
            instance_id,
            payload,
        )
        session_id = str(session["session_id"])
        if str(entry["status"]) in {"UPLOADED", "DUPLICATE"}:
            return await self.sticker_intake_snapshot(
                profile_id,
                instance_id,
                session_id=session_id,
            )
        asset: Any | None = None
        try:
            asset = await self._ingest_intake_upload_asset(
                profile_id,
                session,
                entry,
                payload,
            )
            await self._attach_intake_asset(
                session,
                entry,
                asset,
                filename=Path(str(payload.get("filename") or "")).name[:160],
            )
        except Exception as exc:
            completed = await self._settle_intake_upload_failure(
                session,
                entry,
                asset,
                exc,
            )
            if completed:
                return await self.sticker_intake_snapshot(
                    profile_id,
                    instance_id,
                    session_id=session_id,
                )
        return await self.sticker_intake_snapshot(
            profile_id,
            instance_id,
            session_id=session_id,
        )

    async def _resolve_intake_upload(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        session_id = str(payload.get("session_id") or "").strip()
        client_id = str(payload.get("client_entry_id") or "").strip()
        entry_id = str(payload.get("entry_id") or "").strip()
        if instance is None or not session_id or (not client_id and not entry_id):
            raise ValueError("session_id and upload entry identifier are required")
        session = await self.repository.get_sticker_intake_session(session_id)
        if session is None:
            raise ValueError("快速注入批次不存在")
        self._require_intake_owner(session, profile_id, str(instance.scope), instance_id)
        (
            await load_sticker_runtime_policy(
                self.repository,
                self.profiles_repository,
                profile_id,
                instance_id=str(session["instance_id"]),
            )
        ).require_enabled()
        if str(session["intake_kind"]) != "UPLOAD" or str(session["status"]) != "UPLOADING":
            raise ValueError("当前批次不再接收上传")
        entry = (
            await self.repository.get_sticker_intake_entry_by_client_id(session_id, client_id)
            if client_id
            else None
        )
        if entry is None and entry_id:
            entry = await self.repository.get_sticker_intake_entry(session_id, entry_id)
        if entry is None:
            raise ValueError("上传文件不在本批次清单中")
        return session, entry

    async def _ingest_intake_upload_asset(
        self,
        profile_id: str,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> Any:
        raw, declared_mime = self._decode_intake_upload(payload)
        mime, _extension, _width, _height, _frames = inspect_image_bytes(
            raw,
            declared_mime=declared_mime or None,
            maximum_bytes=5 * 1024 * 1024,
        )
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise ValueError("仅支持 JPEG、PNG、WebP 和 GIF")
        content_sha256 = hashlib.sha256(raw).hexdigest()
        session_id = str(session["session_id"])
        client_id = str(entry["client_entry_id"])
        asset_id = (
            "ma_si_"
            + hashlib.sha256(
                f"{profile_id}\0{session_id}\0{client_id}\0{content_sha256}".encode()
            ).hexdigest()[:32]
        )
        return await self.media_storage.ingest_bytes(
            profile_id,
            str(session["instance_id"]),
            raw,
            origin=MediaOrigin.USER_INPUT,
            purpose=MediaPurpose.STICKER,
            declared_mime=mime,
            asset_id=asset_id,
            inspection_status=MediaInspectionStatus.PENDING,
            metadata={
                "source": "sticker_intake_upload",
                "session_id": session_id,
                "client_entry_id": client_id,
                "display_name": Path(str(payload.get("filename") or "")).name[:160],
            },
        )

    async def _settle_intake_upload_failure(
        self,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
        asset: Any | None,
        exc: Exception,
    ) -> bool:
        try:
            if await self._intake_upload_result_won_race(session, entry, asset):
                return True
            await self._record_intake_upload_error(session, entry, asset, exc)
        except Exception:
            await self._release_failed_intake_upload(asset)
        return False

    async def _intake_upload_result_won_race(
        self,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
        asset: Any | None,
    ) -> bool:
        latest = await self.repository.get_sticker_intake_entry(
            str(session["session_id"]),
            str(entry["entry_id"]),
        )
        if latest is None or str(latest["status"]) not in {
            StickerIntakeEntryStatus.UPLOADED.value,
            StickerIntakeEntryStatus.DUPLICATE.value,
        }:
            return False
        winning_asset_id = str(dict(latest.get("metadata") or {}).get("upload_asset_id") or "")
        if asset is not None and winning_asset_id and str(asset.asset_id) != winning_asset_id:
            with suppress(Exception):
                await self.repository.mark_media_asset_release_pending(
                    str(asset.asset_id),
                    reason="sticker_intake_concurrent_upload_lost",
                )
        return True

    async def _record_intake_upload_error(
        self,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
        asset: Any | None,
        exc: Exception,
    ) -> None:
        retained = (
            await self.repository.get_sticker_candidate_by_source_asset(
                str(session["profile_id"]),
                str(session["instance_id"]),
                str(asset.asset_id),
            )
            if asset is not None
            else None
        )
        await self.repository.attach_sticker_intake_upload(
            str(session["session_id"]),
            str(entry["client_entry_id"]),
            candidate_id=str(retained.candidate_id) if retained is not None else None,
            status=StickerIntakeEntryStatus.ERROR,
            reason_code=type(exc).__name__.upper()[:100],
            error_message=str(exc)[:500],
            metadata_update=(
                {"upload_asset_id": str(asset.asset_id)} if asset is not None else None
            ),
        )

    async def _release_failed_intake_upload(self, asset: Any | None) -> None:
        if asset is None:
            return
        with suppress(Exception):
            await self.repository.mark_media_asset_release_pending(
                str(asset.asset_id),
                reason="sticker_intake_upload_after_freeze",
            )

    @staticmethod
    def _decode_intake_upload(payload: Mapping[str, Any]) -> tuple[bytes, str]:
        encoded = str(payload.get("data_url") or "").strip()
        if not encoded:
            raise ValueError("图片数据为空")
        declared_mime = str(payload.get("mime_type") or "").strip().lower()
        if encoded.startswith("data:"):
            header, separator, encoded = encoded.partition(",")
            if not separator or ";base64" not in header.lower():
                raise ValueError("图片必须使用Base64数据")
            declared_mime = header[5:].split(";", 1)[0].strip().lower()
        if len(encoded) > 7_100_000:
            raise ValueError("单张图片不得超过5 MiB")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("图片Base64无效") from exc
        if not raw or len(raw) > 5 * 1024 * 1024:
            raise ValueError("单张图片不得超过5 MiB")
        return raw, declared_mime

    async def _attach_intake_asset(
        self,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
        asset: Any,
        *,
        filename: str,
    ) -> None:
        profile_id = str(session["profile_id"])
        instance_id = str(session["instance_id"])
        session_id = str(session["session_id"])
        claimed = await self.repository.claim_sticker_intake_upload_content(
            session_id,
            str(entry["entry_id"]),
            str(asset.sha256),
        )
        await self._release_replaced_intake_upload(
            session,
            entry,
            new_asset_id=str(asset.asset_id),
        )
        if not claimed:
            await self.repository.attach_sticker_intake_upload(
                session_id,
                str(entry["client_entry_id"]),
                candidate_id=None,
                status=StickerIntakeEntryStatus.DUPLICATE,
                reason_code="EXACT_DUPLICATE",
                metadata_update={
                    "mime_type": asset.mime_type,
                    "byte_size": asset.byte_size,
                    "upload_asset_id": asset.asset_id,
                },
            )
            await self.repository.mark_media_asset_release_pending(
                asset.asset_id, reason="sticker_intake_exact_duplicate"
            )
            return
        duplicate = await self.repository.find_sticker_item_by_sha(
            profile_id, instance_id, asset.sha256
        )
        duplicate = duplicate or await self.repository.find_sticker_intake_entry_by_sha(
            session_id,
            asset.sha256,
            exclude_entry_id=str(entry["entry_id"]),
        )
        if duplicate is not None:
            await self.repository.attach_sticker_intake_upload(
                session_id,
                str(entry["client_entry_id"]),
                candidate_id=None,
                status=StickerIntakeEntryStatus.DUPLICATE,
                reason_code="EXACT_DUPLICATE",
                metadata_update={
                    "mime_type": asset.mime_type,
                    "byte_size": asset.byte_size,
                    "upload_asset_id": asset.asset_id,
                },
            )
            await self.repository.mark_media_asset_release_pending(
                asset.asset_id, reason="sticker_intake_exact_duplicate"
            )
            return
        candidate, created = await self.repository.create_sticker_candidate(
            profile_id,
            instance_id,
            asset.asset_id,
            source_kind=StickerSourceKind.UPLOAD,
            source_ref=filename,
            metadata={
                "intake_session_id": session_id,
                "intake_entry_id": str(entry["entry_id"]),
                "original_name": filename,
            },
        )
        if not created and str(candidate.metadata.get("intake_session_id") or "") != session_id:
            await self.repository.attach_sticker_intake_upload(
                session_id,
                str(entry["client_entry_id"]),
                candidate_id=None,
                status=StickerIntakeEntryStatus.DUPLICATE,
                reason_code="EXACT_DUPLICATE",
            )
            return
        await self.repository.attach_sticker_intake_upload(
            session_id,
            str(entry["client_entry_id"]),
            candidate_id=candidate.candidate_id,
            status=StickerIntakeEntryStatus.UPLOADED,
            metadata_update={
                "mime_type": asset.mime_type,
                "byte_size": asset.byte_size,
                "upload_asset_id": asset.asset_id,
            },
        )

    async def _release_replaced_intake_upload(
        self,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
        *,
        new_asset_id: str,
    ) -> None:
        previous_asset_id = str(dict(entry.get("metadata") or {}).get("upload_asset_id") or "")
        if not previous_asset_id or previous_asset_id == new_asset_id:
            return
        previous_candidate_id = str(entry.get("candidate_id") or "")
        if previous_candidate_id:
            await self._discard_intake_candidate(
                str(session["profile_id"]),
                str(session["instance_id"]),
                previous_candidate_id,
            )
            return
        await self.repository.mark_media_asset_release_pending(
            previous_asset_id,
            reason="sticker_intake_upload_replaced",
        )


__all__ = ["StickerIntakeUploadMixin"]
