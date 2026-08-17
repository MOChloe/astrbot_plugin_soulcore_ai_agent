from __future__ import annotations

import asyncio
import base64
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...contracts.ai_models import AIBackendDescriptor, AIImageContent
from ...shared.event_log import EventLogPort, record_event
from ..character_model.ports import CharacterModelReadPort
from ..profiles.service import ProfileRuntimeGate
from ..recall import RecallService
from . import MAX_IMAGE_BYTES, MediaFileStore
from .domain import (
    InboundMediaRegistrationState,
    MediaInspectionStatus,
    MediaOrigin,
)
from .errors import (
    IMAGE_INGEST_FAILED,
    ImageGenerationDisabledError,
    ImageGenerationRequestError,
    InboundImageIngestResult,
)
from .files import await_cancellation_safe_file_store
from .image_generation_prompt import ImageReferenceBinding
from .image_generation_prompt import generation_prompt as _generation_prompt
from .inbound import InboundMediaSource
from .inspection import inspect_animation_bytes
from .locator_io import (
    download_http,
    validate_remote_url,
    vision_payload,
    vision_payloads,
)
from .main_core_projection import (
    MainCoreMediaProjection,
    inspect_current_media,
    main_core_media_semantic_note,
    project_main_core_media,
)
from .ports import MediaRepositoryPort, VisualProfilesPort, VisualWorldPort
from .sticker_likelihood import classify_possible_sticker
from .visual_observation_service import VisualObservationServiceMixin


def _utcnow() -> datetime:
    return datetime.now(UTC)


_IDENTITY_REFERENCE_PURPOSE = (
    "仅保持当前角色的稳定身份与固定视觉特征；不得复制服装、画风、背景、姿势、表情或构图"
)


def _normalized_reference_inputs(
    asset_ids: Sequence[str], purposes: Sequence[str]
) -> tuple[list[str], list[str]]:
    normalized_ids = [str(value or "").strip() for value in asset_ids]
    normalized_purposes = [str(value or "").strip() for value in purposes]
    if len(normalized_ids) != len(normalized_purposes):
        raise ImageGenerationRequestError(
            "REFERENCE_PURPOSE_COUNT_MISMATCH",
            "“参考图片”和“逐张参考用途”必须数量相同并按顺序一一对应。",
        )
    if len(normalized_ids) > 5:
        raise ImageGenerationRequestError(
            "REFERENCE_LIMIT_EXCEEDED",
            "一次生成最多携带五张参考图；当前角色已有的固定身份参考也计入这个总数。",
        )
    if not all(normalized_ids) or not all(normalized_purposes):
        raise ImageGenerationRequestError(
            "REFERENCE_PURPOSE_EMPTY",
            "每张参考图都必须提供非空用途。",
        )
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ImageGenerationRequestError(
            "REFERENCE_DUPLICATE",
            "同一张参考图不能在一次请求中重复填写；请把它的用途合并成一项。",
        )
    return normalized_ids, normalized_purposes


def _reference_unavailable(index: int, detail: str) -> ImageGenerationRequestError:
    return ImageGenerationRequestError(
        "REFERENCE_UNAVAILABLE",
        f"参考图{index}{detail}",
    )


def _reference_objective_content(asset: Any, projection: Any) -> str:
    objective_content = (
        str(projection.visible_facts or "").strip() if projection is not None else ""
    )
    return objective_content or f"图片尺寸 {asset.width or '?'}×{asset.height or '?'}"


class VisualExpressionService(VisualObservationServiceMixin):
    """Instance-bound media ingest, description, generation and selection."""

    def __init__(
        self,
        *,
        media_repository: MediaRepositoryPort,
        profiles_repository: VisualProfilesPort,
        world_repository: VisualWorldPort,
        event_log: EventLogPort,
        ai_manager: Any,
        identity: Any,
        file_store: MediaFileStore,
        runtime_gate: ProfileRuntimeGate,
        recall: RecallService,
        sticker_repository: Any | None = None,
        trusted_local_roots: Sequence[str | Path] = (),
    ) -> None:
        self.media = media_repository
        self.profiles = profiles_repository
        self.worlds = world_repository
        self.event_log = event_log
        self.ai_manager = ai_manager
        self.identity = identity
        self.file_store = file_store
        self.runtime_gate = runtime_gate
        self.recall = recall
        self.stickers = sticker_repository
        self.character_models: CharacterModelReadPort | None = None
        roots = [file_store.root, *(Path(item) for item in trusted_local_roots)]
        self._trusted_local_roots = tuple(dict.fromkeys(path.resolve() for path in roots))
        self._background: set[asyncio.Task[Any]] = set()
        self._background_scopes: dict[asyncio.Task[Any], tuple[str, str]] = {}
        self._background_accepting = True
        self._background_blocked_profiles: set[str] = set()
        self._background_blocked_instances: set[tuple[str, str]] = set()
        self._observation_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self._observation_lock_users: dict[tuple[str, ...], int] = {}
        self._observation_locks_guard = asyncio.Lock()

    def bind_character_models(self, port: CharacterModelReadPort) -> None:
        if self.character_models is not None and self.character_models is not port:
            raise RuntimeError("character model read port is already bound")
        self.character_models = port

    async def close(self) -> None:
        self._background_accepting = False
        await self._drain_background_tasks()
        self._background.clear()
        self._background_scopes.clear()

    async def require_image_generation_enabled(
        self,
        profile_id: str,
        instance_id: str = "",
    ) -> None:
        """Re-read the profile switch at the image-generation trust boundary."""
        await self.runtime_gate.require_enabled(profile_id, instance_id)
        if instance_id and not await self.runtime_gate.image_send_enabled(
            profile_id,
            instance_id,
        ):
            raise ImageGenerationDisabledError()
        if not await self.profiles.get_profile_image_generation_enabled(profile_id):
            raise ImageGenerationDisabledError()

    async def has_image_generation_provider(self, profile_id: str) -> bool:
        """Return whether this profile currently has a usable generation route."""

        return bool(
            await self.ai_manager.has_capability_provider(
                "image.generate",
                str(profile_id),
            )
        )

    @staticmethod
    def backend_supports_vision(descriptor: AIBackendDescriptor | None) -> bool:
        if descriptor is None:
            return False
        metadata = dict(descriptor.metadata)
        value = metadata["supports_vision"]
        if not isinstance(value, bool):
            raise ValueError("supports_vision must be a boolean")
        return value

    async def project_main_core_media(self, **values: Any) -> MainCoreMediaProjection:
        """Build bounded MainCore previews without exposing authoritative paths."""
        values["asset_ids"] = await self._model_visible_asset_ids(
            str(values.get("profile_id") or ""),
            str(values.get("instance_id") or ""),
            values.get("asset_ids") or (),
        )
        return await project_main_core_media(self.media, self.file_store, **values)

    async def main_core_media_semantic_note(self, **values: Any) -> str:
        """Project optional media semantics without reading authoritative image bytes."""
        values["asset_ids"] = await self._model_visible_asset_ids(
            str(values.get("profile_id") or ""),
            str(values.get("instance_id") or ""),
            values.get("asset_ids") or (),
        )
        return await main_core_media_semantic_note(self.media, **values)

    async def inspect_current_media(self, **values: Any) -> Any:
        """Resolve one authorized asset to exact static bytes or native animation frames."""
        if not await self._asset_is_model_visible(
            str(values.get("profile_id") or ""),
            str(values.get("instance_id") or ""),
            str(values.get("asset_id") or ""),
        ):
            raise ValueError("source message is unavailable")
        return await inspect_current_media(self.media, self.file_store, **values)

    async def current_media_timing(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset_id: str,
    ) -> Mapping[str, Any]:
        """Return bounded animation timing used to resolve natural frame positions."""

        if not await self._asset_is_model_visible(profile_id, instance_id, asset_id):
            raise ValueError("source message is unavailable")
        asset = await self.media.get_media_asset(
            asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        if asset is None:
            raise ValueError("current media asset is unavailable")
        frame_count = max(1, int(asset.frame_count or 1))
        duration_ms = max(0, int((asset.metadata or {}).get("duration_ms") or 0))
        if (
            frame_count > 1
            and duration_ms <= 0
            and asset.storage_relpath
            and await asyncio.to_thread(self.file_store.verify, asset)
        ):
            path = self.file_store.absolute_path(asset.storage_relpath)
            data = await asyncio.to_thread(path.read_bytes)
            timing = await asyncio.to_thread(inspect_animation_bytes, data)
            frame_count = max(1, int(timing.get("frame_count") or frame_count))
            duration_ms = max(0, int(timing.get("duration_ms") or 0))
        return {
            "frame_count": frame_count,
            "duration_ms": duration_ms,
            "mime_type": str(asset.mime_type or ""),
        }

    async def _model_visible_asset_ids(
        self, profile_id: str, instance_id: str, asset_ids: Sequence[str]
    ) -> list[str]:
        result: list[str] = []
        for asset_id in dict.fromkeys(str(value) for value in asset_ids if str(value)):
            if await self._asset_is_model_visible(profile_id, instance_id, asset_id):
                result.append(asset_id)
        return result

    async def ingest_inbound(
        self,
        *,
        profile_id: str,
        instance_id: str,
        message_id: int,
        platform_message_id: str,
        sources: Sequence[InboundMediaSource],
        source_ordinals: Sequence[int] | None = None,
    ) -> InboundImageIngestResult:
        await self.runtime_gate.require_enabled(profile_id, instance_id)
        asset_ids: list[str] = []
        failures: list[dict[str, Any]] = []
        source_count = min(5, len(sources))
        for ordinal in range(source_count):
            source = sources[ordinal]
            source_ordinal = (
                int(source_ordinals[ordinal])
                if source_ordinals is not None and ordinal < len(source_ordinals)
                else ordinal
            )
            try:
                asset_id = await self._ingest_inbound_one(
                    profile_id=profile_id,
                    instance_id=instance_id,
                    message_id=message_id,
                    platform_message_id=platform_message_id,
                    source=source,
                    ordinal=source_ordinal,
                )
            except Exception as exc:
                failure = {
                    "ordinal": source_ordinal,
                    "source": "inbound_media",
                    "error": type(exc).__name__,
                    "detail": self._safe_ingest_error(exc),
                }
                failures.append(failure)
                await record_event(
                    self.event_log,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    level="ERROR",
                    category="media.ingest",
                    message="一张入站图片未能安全写入媒体资产层",
                    details=failure,
                )
                continue
            asset_ids.append(asset_id)
        return InboundImageIngestResult(
            asset_ids=tuple(asset_ids),
            failure_categories=tuple(IMAGE_INGEST_FAILED for _ in failures),
        )

    async def _ingest_inbound_one(
        self,
        *,
        profile_id: str,
        instance_id: str,
        message_id: int,
        platform_message_id: str,
        source: InboundMediaSource,
        ordinal: int,
    ) -> str:
        data, declared_mime = await self._read_inbound_image(source)
        asset_id = (
            "ma_in_"
            + hashlib.sha256(
                f"{profile_id}\0{instance_id}\0{message_id}\0{ordinal}".encode()
            ).hexdigest()[:32]
        )

        async def cleanup_cancelled_write(stored_file: Any) -> None:
            existing = await self.media.get_media_asset(
                asset_id,
                profile_id=profile_id,
                instance_id=instance_id,
            )
            if existing is None or str(existing.storage_relpath or "") != stored_file.relative_path:
                self.file_store.delete(stored_file.relative_path)

        stored = await await_cancellation_safe_file_store(
            self.file_store.store_bytes,
            cleanup_after_cancel=cleanup_cancelled_write,
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=declared_mime,
        )
        registration_attempted = False
        try:
            sticker = classify_possible_sticker(
                mime_type=stored.mime_type,
                width=stored.width,
                height=stored.height,
                frame_count=stored.frame_count,
                evidence=source.sticker_evidence,
            )
            if not await asyncio.to_thread(self.file_store.verify, stored):
                raise OSError("stored inbound image failed integrity verification")
            registration_attempted = True
            await self.media.register_inbound_media_asset(
                profile_id,
                instance_id,
                stored,
                message_id=message_id,
                ordinal=ordinal,
                platform_message_id=platform_message_id,
                inspection_status=MediaInspectionStatus.PENDING,
                metadata={
                    "source": "inbound",
                    "ordinal": ordinal,
                    "media_semantic_version": 1,
                    "possible_sticker": sticker.possible,
                    "sticker_evidence": list(sticker.evidence),
                },
            )
        except BaseException as original:
            try:
                state, _asset = await self.media.inspect_inbound_media_registration(
                    profile_id=profile_id,
                    instance_id=instance_id,
                    stored=stored,
                    message_id=message_id,
                    ordinal=ordinal,
                    platform_message_id=platform_message_id,
                )
            except BaseException as inspection_error:
                original.add_note(
                    "inbound media registration state could not be confirmed: "
                    f"{type(inspection_error).__name__}: {inspection_error}"
                )
                raise original from inspection_error
            if state is InboundMediaRegistrationState.UNOWNED:
                self.file_store.delete(stored.relative_path)
            if not (
                registration_attempted
                and state is InboundMediaRegistrationState.COMMITTED
                and isinstance(original, Exception)
            ):
                raise
        await record_event(
            self.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="INFO",
            category="media.ingest",
            message="图片已写入SoulCore媒体资产层",
            details={"asset_id": asset_id, "origin": "USER_INPUT"},
        )
        return asset_id

    async def _read_inbound_image(self, source: InboundMediaSource) -> tuple[bytes, str | None]:
        if source.data:
            if len(source.data) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds 20 MiB")
            return source.data, source.mime_type
        if source.locator:
            return await self.read_locator(source.locator)
        raise ValueError("no usable image source")

    @staticmethod
    def _safe_ingest_error(exc: BaseException) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            return f"HTTP {exc.code}"
        if isinstance(exc, urllib.error.URLError):
            return f"network:{type(exc.reason).__name__}"
        if isinstance(exc, OSError):
            code = getattr(exc, "errno", None) or getattr(exc, "winerror", None)
            return f"{type(exc).__name__}:{code}" if code else type(exc).__name__
        text = str(exc).strip()
        # All ValueError messages produced above are URL/token free.  Unknown
        # exceptions are reduced to their class so signed QQ URLs cannot leak.
        if isinstance(exc, (ValueError, TypeError)) and len(text) <= 240:
            return text
        return type(exc).__name__

    async def inspect_web_search_images(
        self,
        *,
        resources: Sequence[Any],
        profile_id: str,
        instance_id: str,
        core_run_id: str,
        main_core_supports_vision: bool = False,
        defer_inspection_to_sticker_check: bool = False,
    ) -> Mapping[str, Any]:
        from .web_image_inspection import inspect_web_search_images

        return await inspect_web_search_images(
            self,
            resources=resources,
            profile_id=profile_id,
            instance_id=instance_id,
            core_run_id=core_run_id,
            main_core_supports_vision=main_core_supports_vision,
            defer_inspection_to_sticker_check=defer_inspection_to_sticker_check,
        )

    async def present_visual(self, **request: Any) -> Mapping[str, Any]:
        await self.require_image_generation_enabled(
            str(request["profile_id"]),
            str(request.get("instance_id") or ""),
        )
        if str(request.get("output_type") or "generated_image") == "social_snapshot":
            from .social_snapshot import present_social_snapshot

            return await present_social_snapshot(self, request)
        from .generation import present_visual

        return await present_visual(self, request)

    async def compile_social_snapshot_scene(self, **request: Any) -> str:
        """Run the hidden, constrained natural-intent to scene compilation stage."""

        from .social_snapshot_intent import compile_social_snapshot_intent

        await self.require_image_generation_enabled(
            str(request["profile_id"]),
            str(request.get("instance_id") or ""),
        )
        return await compile_social_snapshot_intent(
            self.ai_manager,
            profile_id=str(request["profile_id"]),
            instance_id=str(request["instance_id"]),
            run_id=int(request["run_id"]),
            preset=request["preset"],
            content=str(request.get("content") or ""),
            reference_images=str(request.get("reference_images") or ""),
        )

    async def validate_selection(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        asset_ids: Sequence[str],
    ) -> bool | str:
        if not await self.runtime_gate.is_enabled(profile_id, instance_id):
            return "profile_disabled"
        rows = await self.media.list_media_assets(
            profile_id,
            instance_id,
            core_run_id=run_id,
            limit=100,
        )
        by_id = {item.asset_id: item for item in rows}
        for asset_id in asset_ids:
            asset = by_id.get(str(asset_id))
            if (
                asset is None
                or asset.origin is not MediaOrigin.GENERATED
                or asset.file_status.value != "AVAILABLE"
                or asset.inspection_status.value != "READY"
            ):
                return "selected assets must be inspected outputs of the current run"
        return True

    async def asset_file_path(
        self, *, profile_id: str, instance_id: str, asset_id: str
    ) -> Path | None:
        asset = await self.media.get_media_asset(
            asset_id, profile_id=profile_id, instance_id=instance_id
        )
        if asset is not None and asset.storage_relpath and asset.file_status.value == "AVAILABLE":
            path = self.file_store.absolute_path(asset.storage_relpath)
            verified = await asyncio.to_thread(self.file_store.verify, asset)
            return path if verified else None
        if self.stickers is None:
            return None
        sticker = await self.stickers.get_accessible_sticker_asset(
            asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        if sticker is None or sticker.file_status != "AVAILABLE":
            return None
        path = self.file_store.absolute_path(sticker.storage_relpath)
        if not path.is_file() or path.stat().st_size != sticker.byte_size:
            return None
        digest = await asyncio.to_thread(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
        return path if digest == sticker.canonical_sha256 else None

    async def resolve_references(
        self,
        profile_id: str,
        instance_id: str,
        asset_ids: Sequence[str],
        purposes: Sequence[str],
    ) -> tuple[list[AIImageContent], list[ImageReferenceBinding]]:
        normalized_ids, normalized_purposes = _normalized_reference_inputs(asset_ids, purposes)
        images: list[AIImageContent] = []
        bindings: list[ImageReferenceBinding] = []
        for index, (asset_id, purpose) in enumerate(
            zip(normalized_ids, normalized_purposes, strict=True),
            start=1,
        ):
            image, binding = await self._resolve_reference(
                profile_id, instance_id, asset_id, purpose, index
            )
            images.append(image)
            bindings.append(binding)
        return images, bindings

    async def _resolve_reference(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        purpose: str,
        index: int,
    ) -> tuple[AIImageContent, ImageReferenceBinding]:
        asset = await self.media.get_media_asset(
            asset_id, profile_id=profile_id, instance_id=instance_id
        )
        if asset is None:
            raise _reference_unavailable(
                index, "不属于当前会话或已经不可用，请换用本轮可见的图片引用。"
            )
        data = await self._verified_reference_bytes(asset, index)
        projection = await self.media.get_latest_media_projection(asset_id)
        objective_content = _reference_objective_content(asset, projection)
        return (
            AIImageContent(
                asset.mime_type,
                data=data,
                asset_id=asset_id,
                metadata={"declared_purpose": purpose, "provider_ordinal": index},
            ),
            ImageReferenceBinding(
                label=f"参考图{index}",
                purpose=purpose,
                objective_content=objective_content,
            ),
        )

    async def _verified_reference_bytes(self, asset: Any, index: int) -> bytes:
        if not asset.storage_relpath:
            raise _reference_unavailable(index, "没有可发送的原始图片，请换用另一张本轮可见图片。")
        try:
            return await self.read_verified_media_bytes(
                relative_path=asset.storage_relpath,
                byte_size=asset.byte_size,
                sha256=asset.sha256,
            )
        except OSError:
            raise _reference_unavailable(
                index, "的原始图片已经不可用，请换用另一张本轮可见图片。"
            ) from None
        except ValueError:
            raise _reference_unavailable(
                index, "的原始图片完整性校验失败，请换用另一张图片。"
            ) from None

    async def read_verified_media_bytes(
        self,
        *,
        relative_path: str,
        byte_size: int,
        sha256: str,
    ) -> bytes:
        path = self.file_store.absolute_path(str(relative_path))
        data = await asyncio.to_thread(path.read_bytes)
        self._require_verified_bytes(data, byte_size=byte_size, sha256=sha256)
        return data

    @staticmethod
    def _require_verified_bytes(data: bytes, *, byte_size: int, sha256: str) -> None:
        if len(data) != int(byte_size) or hashlib.sha256(data).hexdigest() != str(sha256):
            raise ValueError("media bytes failed size or SHA-256 verification")

    async def resolve_identity_reference(
        self, record: Mapping[str, Any] | None
    ) -> tuple[AIImageContent | None, list[str]]:
        """Read the config-owned identity asset without borrowing instance media."""

        if not isinstance(record, Mapping):
            return None, []
        notes = _identity_reference_notes(record)
        data = await self._identity_reference_bytes(record)
        if data is None:
            return None, notes
        return _identity_reference_content(record, data), notes

    async def _identity_reference_bytes(self, record: Mapping[str, Any]) -> bytes | None:
        frozen = record.get("data")
        if isinstance(frozen, (bytes, bytearray)) and frozen:
            data = bytes(frozen)
            try:
                self._require_verified_bytes(
                    data,
                    byte_size=int(record.get("byte_size") or 0),
                    sha256=str(record.get("sha256") or ""),
                )
            except (TypeError, ValueError):
                return None
            return data
        relative = str(record.get("storage_relpath") or "").strip()
        if not relative:
            return None
        try:
            return await self.read_verified_media_bytes(
                relative_path=relative,
                byte_size=int(record.get("byte_size") or 0),
                sha256=str(record.get("sha256") or ""),
            )
        except (OSError, ValueError):
            return None

    generation_prompt = staticmethod(_generation_prompt)

    async def image_content_bytes(self, image: AIImageContent) -> tuple[bytes, str | None]:
        if image.data:
            return image.data, image.mime_type
        return await self.read_locator(image.url)

    async def read_locator(
        self, locator: str, *, referer: str | None = None
    ) -> tuple[bytes, str | None]:
        value = str(locator or "").strip()
        if value.startswith("data:image/"):
            header, encoded = value.split(",", 1)
            mime = header[5:].split(";", 1)[0]
            data = base64.b64decode(encoded, validate=True)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds 20 MiB")
            return data, mime
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme in {"http", "https"}:
            return await asyncio.to_thread(self.download_http, value, referer=referer)
        windows_absolute = bool(
            len(value) >= 3 and value[1] == ":" and value[0].isalpha() and value[2] in {"\\", "/"}
        )
        if parsed.scheme and not windows_absolute:
            raise ValueError("unsupported image locator scheme")
        path = Path(value).expanduser().resolve()
        if not any(path == root or root in path.parents for root in self._trusted_local_roots):
            raise ValueError("local image locator is outside trusted storage")
        data = await asyncio.to_thread(path.read_bytes)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 20 MiB")
        return data, None

    download_http = classmethod(download_http)
    validate_remote_url = staticmethod(validate_remote_url)
    vision_payloads = staticmethod(vision_payloads)
    vision_payload = staticmethod(vision_payload)


def _identity_reference_notes(record: Mapping[str, Any]) -> list[str]:
    identity = str(record.get("identity_description") or "").strip()
    return [identity[:1000]] if identity else []


def _identity_reference_content(
    record: Mapping[str, Any],
    data: bytes,
) -> AIImageContent:
    return AIImageContent(
        str(record.get("mime_type") or "image/png"),
        data=data,
        asset_id=str(record.get("asset_id") or ""),
        metadata={
            "purpose": "CHARACTER_IDENTITY",
            "identity_only": True,
            "do_not_copy_clothing": True,
            "do_not_copy_style_background_pose": True,
        },
    )


__all__ = ["ImageReferenceBinding", "VisualExpressionService"]
