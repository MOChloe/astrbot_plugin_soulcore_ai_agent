"""The character illustration reference and safe sticker previews."""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ....contracts.models import CharacterInstance
from ....features.ai.service import safe_ai_failure_details
from ....features.media.image_service import VisualExpressionService
from ....features.media.inspection import (
    MAX_ANIMATION_DECODED_PIXELS,
    MAX_ANIMATION_DURATION_MS,
    MAX_ANIMATION_FRAMES,
    MAX_IMAGE_BYTES,
    generate_media_asset_id,
    inspect_animation_bytes,
    inspect_image_bytes,
)
from ....features.media.ports import MediaRepositoryPort
from ....features.media.storage import MediaStorageCoordinator
from ....features.media.visual_cache import VisualCachePolicy
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.stickers.contracts import (
    DESCRIPTION_CONTRACT_VERSION,
    StickerDescriptionContractError,
)
from ....features.stickers.domain import StickerAsset, StickerItem, StickerItemStatus
from ....features.stickers.policy import load_sticker_runtime_policy
from ....features.stickers.ports import StickerRepositoryPort
from ....features.stickers.service import StickerCollectorPlugin
from ..console_errors import ConsoleValidationError
from ..presentation import jsonable
from .reference_uploads import ReferenceUploadTransport


class StickerReferenceController:
    def __init__(
        self,
        repository: StickerRepositoryPort,
        profiles_repository: ProfilesRepositoryPort,
        media_repository: MediaRepositoryPort,
        media_storage: MediaStorageCoordinator,
        visual_service: VisualExpressionService,
        sticker_collector: StickerCollectorPlugin,
    ) -> None:
        self.repository = repository
        self.profiles_repository = profiles_repository
        self.media_repository = media_repository
        self.media_storage = media_storage
        self.visual_service = visual_service
        self.sticker_collector = sticker_collector
        self.reference_uploads = ReferenceUploadTransport(self.media_storage.store.root)

    @staticmethod
    def _public_identity_reference(record: Any) -> dict[str, Any]:
        view = dict(jsonable(record) or {})
        # Local storage paths are implementation details and must never be
        # returned to the browser or copied into prompts.
        view.pop("storage_relpath", None)
        view.pop("sha256", None)
        return view

    def _identity_reference_preview(self, storage_relpath: str) -> str:
        if self.media_storage is None or not storage_relpath:
            return ""
        path = self.media_storage.store.absolute_path(storage_relpath)
        if not path.is_file():
            return ""
        try:
            from PIL import Image

            with Image.open(path) as source:
                source.seek(0)
                frame = source.convert("RGBA")
                frame.thumbnail((320, 320), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                frame.save(output, format="WEBP", quality=78, method=4)
            return "data:image/webp;base64," + base64.b64encode(output.getvalue()).decode("ascii")
        except Exception:
            return ""

    def _sticker_thumbnail(self, storage_relpath: str) -> str:
        """Render a bounded still preview without exposing the stored file."""

        if self.media_storage is None or not storage_relpath:
            return ""
        path = self.media_storage.store.absolute_path(storage_relpath)
        if not path.is_file():
            return ""
        try:
            from PIL import Image

            with Image.open(path) as source:
                frame_count = max(1, int(getattr(source, "n_frames", 1) or 1))
                source.seek(frame_count // 2 if frame_count > 1 else 0)
                frame = source.convert("RGBA")
                frame.thumbnail((192, 192), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                frame.save(output, format="WEBP", quality=76, method=4)
                data = output.getvalue()
                if len(data) > 160 * 1024:
                    output = io.BytesIO()
                    frame.save(output, format="WEBP", quality=58, method=4)
                    data = output.getvalue()
                if len(data) > 160 * 1024:
                    return ""
            return "data:image/webp;base64," + base64.b64encode(data).decode("ascii")
        except Exception:
            return ""

    async def sticker_item_thumbnail(self, profile_id: str, instance_id: str, item_id: str) -> str:
        """Resolve the owner tuple before reading a formal sticker file."""

        item = await self.repository.get_sticker_item(profile_id, instance_id, item_id)
        if item is None or item.status == StickerItemStatus.DELETED:
            return ""
        asset = await self.repository.get_sticker_asset(item.asset_id, profile_id=profile_id)
        if asset is None or not asset.storage_relpath or asset.file_status != "AVAILABLE":
            return ""
        if not await asyncio.to_thread(self.media_storage.store.verify, asset):
            return ""
        return await asyncio.to_thread(self._sticker_thumbnail, asset.storage_relpath)

    async def sticker_candidate_thumbnail(
        self, profile_id: str, instance_id: str, candidate_id: str
    ) -> str:
        """Resolve one intake candidate and return only a bounded still WebP."""

        candidate = await self.repository.get_sticker_candidate(
            profile_id, instance_id, candidate_id
        )
        if candidate is None:
            return ""
        asset = await self.media_repository.get_media_asset(
            candidate.source_asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        if (
            asset is None
            or not asset.storage_relpath
            or str(asset.file_status.value) != "AVAILABLE"
            or not await asyncio.to_thread(self.media_storage.store.verify, asset)
        ):
            return ""
        return await asyncio.to_thread(self._sticker_thumbnail, asset.storage_relpath)

    async def sticker_reference_snapshot(self, profile_id: str, scope: str) -> dict[str, Any]:
        row = await self.repository.get_character_identity_reference(profile_id, scope)
        reference: dict[str, Any] | None = None
        if row is not None:
            view = self._public_identity_reference(row)
            raw = dict(jsonable(row) or {})
            view["preview_data_url"] = await asyncio.to_thread(
                self._identity_reference_preview,
                str(raw.get("storage_relpath") or ""),
            )
            reference = view
        return {
            "profile_id": profile_id,
            "scope": scope,
            "reference": reference,
            "purpose": "CHARACTER_IDENTITY",
        }

    async def sticker_reference_action(
        self, profile_id: str, scope: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        action = str(payload.get("action") or "upload").strip().lower()
        reference_id = str(payload.get("reference_id") or "").strip()
        if action == "delete":
            return await self._delete_reference(profile_id, scope, reference_id)
        if action == "upload_begin":
            return await self._begin_reference_upload(profile_id, scope, payload)
        if action == "upload_chunk":
            return await self._append_reference_upload(profile_id, scope, payload)
        if action == "upload_finish":
            return await self._finish_reference_upload(profile_id, scope, payload)
        if action == "upload_abort":
            await self.reference_uploads.abort(
                profile_id=profile_id,
                scope=scope,
                upload_id=str(payload.get("upload_id") or "").strip(),
            )
            return {"ok": True, "message": "角色立绘参考上传已取消"}
        if action != "upload":
            raise ValueError("unknown character identity reference action")
        return await self._upload_reference(profile_id, scope, payload)

    async def _begin_reference_upload(
        self,
        profile_id: str,
        scope: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            byte_size = int(payload.get("byte_size") or 0)
        except (TypeError, ValueError) as exc:
            raise ConsoleValidationError("角色立绘参考的文件大小信息无效。") from exc
        return dict(
            await self.reference_uploads.begin(
                profile_id=profile_id,
                scope=scope,
                expected_size=byte_size,
                filename=str(payload.get("filename") or ""),
                label=str(payload.get("label") or ""),
                mime_type=str(payload.get("mime_type") or ""),
            )
        )

    async def _append_reference_upload(
        self,
        profile_id: str,
        scope: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            chunk_index = int(payload.get("chunk_index"))
            chunk_byte_size = (
                int(payload["chunk_byte_size"])
                if payload.get("chunk_byte_size") is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ConsoleValidationError("角色立绘参考的上传分块信息无效。") from exc
        return dict(
            await self.reference_uploads.append(
                profile_id=profile_id,
                scope=scope,
                upload_id=str(payload.get("upload_id") or "").strip(),
                chunk_index=chunk_index,
                encoded_chunk=str(payload.get("data_base64") or ""),
                expected_chunk_size=chunk_byte_size,
                expected_chunk_crc32=str(payload.get("chunk_crc32") or ""),
            )
        )

    async def _finish_reference_upload(
        self,
        profile_id: str,
        scope: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        upload = await self.reference_uploads.finish(
            profile_id=profile_id,
            scope=scope,
            upload_id=str(payload.get("upload_id") or "").strip(),
            expected_crc32=str(payload.get("file_crc32") or ""),
        )
        result = await self._commit_reference_upload(
            profile_id,
            scope,
            {
                "filename": upload.filename,
                "label": upload.label,
                "mime_type": upload.mime_type,
            },
            upload.data,
        )
        result["transport_crc32"] = upload.crc32
        return result

    async def _delete_reference(
        self, profile_id: str, scope: str, reference_id: str
    ) -> dict[str, Any]:
        if not reference_id:
            raise ValueError("reference_id is required")
        await self.repository.delete_character_identity_reference(profile_id, scope, reference_id)
        return {"ok": True, "message": "角色立绘参考已删除"}

    async def _upload_reference(
        self, profile_id: str, scope: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        raw = self._decode_reference_upload(payload)
        return await self._commit_reference_upload(profile_id, scope, payload, raw)

    async def _commit_reference_upload(
        self,
        profile_id: str,
        scope: str,
        payload: Mapping[str, Any],
        raw: bytes,
    ) -> dict[str, Any]:
        try:
            mime, extension, width, height, frame_count = inspect_image_bytes(
                raw,
                maximum_bytes=MAX_IMAGE_BYTES,
                enforce_animation_limits=True,
                validate_decoded=True,
            )
            duration_ms = (
                int(
                    inspect_animation_bytes(
                        raw,
                        maximum_frames=MAX_ANIMATION_FRAMES,
                        maximum_duration_ms=MAX_ANIMATION_DURATION_MS,
                        maximum_decoded_pixels=MAX_ANIMATION_DECODED_PIXELS,
                    )["duration_ms"]
                )
                if frame_count > 1
                else 0
            )
        except (OSError, ValueError) as exc:
            raise ConsoleValidationError(
                "上传内容不是可完整读取的 JPEG、PNG、WebP 或 GIF 图片。"
            ) from exc
        stored = await self._plan_reference_bytes(profile_id, scope, raw)
        cleanup_guard = await self.media_repository.guard_unregistered_media_file(
            profile_id,
            f"character-identity:{scope}",
            stored,
            reason="IDENTITY_REFERENCE_REGISTRATION",
        )
        try:
            await asyncio.to_thread(self.media_storage.store.write_planned_bytes, stored, raw)
            row, replaced = await self._replace_reference_row(
                profile_id,
                scope,
                payload,
                stored,
                mime,
                extension,
                width,
                height,
                frame_count,
                duration_ms,
                cleanup_guard.cleanup_id,
            )
        except Exception:
            # The durable guard owns cleanup even across a crash or a locked file.
            raise
        return {
            "ok": True,
            "reference": self._public_identity_reference(row),
            "replaced": replaced,
            "message": "角色立绘参考已更换" if replaced else "角色立绘参考已上传",
        }

    @staticmethod
    def _decode_reference_upload(payload: Mapping[str, Any]) -> bytes:
        encoded = str(payload.get("data_url") or "").strip()
        if not encoded:
            raise ConsoleValidationError("请选择要上传的角色立绘参考。")
        if encoded.startswith("data:"):
            header, separator, encoded = encoded.partition(",")
            if not separator or ";base64" not in header.lower():
                raise ConsoleValidationError("角色立绘参考没有形成有效的图片数据。")
        if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
            raise ConsoleValidationError(
                f"角色立绘参考不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)} MiB。"
            )
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ConsoleValidationError("角色立绘参考的图片数据无法读取。") from exc
        if len(raw) > MAX_IMAGE_BYTES:
            raise ConsoleValidationError(
                f"角色立绘参考不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)} MiB。"
            )
        return raw

    async def _plan_reference_bytes(self, profile_id: str, scope: str, raw: bytes) -> Any:
        asset_id = generate_media_asset_id()
        return await asyncio.to_thread(
            self.media_storage.store.plan_reference_bytes,
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=f"character-identity:{scope}",
            data=raw,
        )

    async def _replace_reference_row(
        self,
        profile_id: str,
        scope: str,
        payload: Mapping[str, Any],
        stored: Any,
        mime: str,
        extension: str,
        width: int,
        height: int,
        frame_count: int,
        duration_ms: int,
        cleanup_guard_id: int,
    ) -> Any:
        return await self.repository.replace_character_identity_reference(
            profile_id,
            scope,
            asset_id=stored.asset_id,
            storage_relpath=stored.relative_path,
            mime_type=mime,
            file_extension=extension,
            sha256=stored.sha256,
            byte_size=stored.byte_size,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_ms=duration_ms,
            label=str(payload.get("label") or "").strip()[:80],
            identity_description="",
            metadata={
                "purpose": "CHARACTER_IDENTITY",
                "animated": frame_count > 1,
                "original_name": Path(str(payload.get("filename") or "")).name[:160],
            },
            cleanup_guard_id=cleanup_guard_id,
        )

    async def regenerate_item_description(
        self, profile_id: str, instance_id: str, item_id: str
    ) -> Any:
        """Refresh one formal item's description through the shared Check contract."""

        if self.visual_service is None or self.sticker_collector is None:
            raise RuntimeError("表情包视觉与 Check 服务尚未就绪，旧描述已保留")
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        item = await self.repository.get_sticker_item(profile_id, instance_id, item_id)
        if instance is None or item is None:
            raise ValueError("表情包不属于当前档案与实例")
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles_repository,
            profile_id,
            instance_id=instance_id,
        )
        policy.require_enabled()
        if item.status == StickerItemStatus.DELETED:
            raise ValueError("已删除的表情包不能重新生成描述")
        asset = await self.repository.get_sticker_asset(item.asset_id, profile_id=profile_id)
        if not self._formal_sticker_asset_available(asset):
            raise ValueError("正式表情包原始媒体不可用，旧描述已保留")
        try:
            payload, description = await self._generate_sticker_description(
                profile_id,
                instance_id,
                instance,
                item,
                self.sticker_collector.build_strict_description,
            )
            (
                await load_sticker_runtime_policy(
                    self.repository,
                    self.profiles_repository,
                    profile_id,
                    instance_id=instance_id,
                )
            ).require_enabled()
            return await self._save_sticker_description(
                profile_id, instance_id, item_id, item, payload, description
            )
        except Exception as exc:
            diagnostics = safe_ai_failure_details(exc)
            code = str(
                exc.code
                if isinstance(exc, StickerDescriptionContractError)
                else diagnostics.get("error_code") or type(exc).__name__
            )[:80]
            raise RuntimeError(f"重新生成描述失败，旧描述已保留（{code}）") from exc

    @staticmethod
    def _formal_sticker_asset_available(asset: StickerAsset | None) -> bool:
        if asset is None or not asset.storage_relpath:
            return False
        return asset.file_status == "AVAILABLE"

    async def _generate_sticker_description(
        self,
        profile_id: str,
        instance_id: str,
        instance: CharacterInstance,
        item: StickerItem,
        build_description: Any,
    ) -> tuple[dict[str, Any], str]:
        (
            await load_sticker_runtime_policy(
                self.repository,
                self.profiles_repository,
                profile_id,
                instance_id=instance_id,
            )
        ).require_enabled()
        vision = await self.visual_service.describe_asset(
            profile_id=profile_id,
            instance_id=instance_id,
            asset_id=item.asset_id,
            foreground=True,
            cache_policy=VisualCachePolicy.REFRESH,
        )
        (
            await load_sticker_runtime_policy(
                self.repository,
                self.profiles_repository,
                profile_id,
                instance_id=instance_id,
            )
        ).require_enabled()
        persona = await self.sticker_collector.persona_resolver(instance.route_umo)
        config = await self.repository.get_sticker_config(profile_id, instance.scope)
        structured, result = await build_description(
            profile_id,
            instance_id,
            vision=vision,
            persona=str(persona or ""),
            requirements=config.requirements,
            source_kind=item.source_kind.value,
        )
        if not result.accepted:
            reason = result.reason_code or "CHECK_REJECTED"
            raise RuntimeError(f"DESCRIPTION_{reason}")
        payload = dict(structured)
        description = str(
            result.compact_description or payload.get("compact_description") or ""
        ).strip()
        if not description:
            raise RuntimeError("DESCRIPTION_EMPTY")
        return payload, description

    async def _save_sticker_description(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
        item: Any,
        payload: Mapping[str, Any],
        description: str,
    ) -> Any:
        return await self.repository.update_sticker_item_description(
            profile_id,
            instance_id,
            item_id,
            compact_description=description,
            visible_text=str(payload.get("visible_text") or "").strip(),
            search_keywords=tuple(payload.get("search_keywords") or ())[:20],
            metadata_update={
                "description_version": str(
                    payload.get("description_version") or DESCRIPTION_CONTRACT_VERSION
                ),
                "structured_description": {
                    "objective_scene": str(payload.get("objective_scene") or "")[:5000],
                    "social_impression": str(payload.get("vision_social_impression") or "")[:80],
                },
            },
            expected_description=item.compact_description,
        )
