from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from ..identity import IdentityService
from ..social_snapshot import SocialSnapshotError, SocialSnapshotErrorCode
from ..social_snapshot.ports import ControlledAssetResolverPort, RenderedSnapshotPart
from ..social_snapshot.service import PreparedSnapshotAssets, SocialSnapshotService
from .domain import MediaFileStatus, MediaProjectionStatus
from .errors import ImageGenerationDisabledError
from .files import MediaFileStore
from .ports import MediaRepositoryPort


class _RuntimeGatePort(Protocol):
    async def require_enabled(self, profile_id: str, instance_id: str = "") -> object: ...


class _VisualServicePort(Protocol):
    media: MediaRepositoryPort
    file_store: MediaFileStore
    runtime_gate: _RuntimeGatePort
    identity: IdentityService

    async def require_image_generation_enabled(
        self,
        profile_id: str,
        instance_id: str = "",
    ) -> None: ...

    async def read_verified_media_bytes(
        self,
        *,
        relative_path: str,
        byte_size: int,
        sha256: str,
    ) -> bytes: ...


class _PreparedResolver(ControlledAssetResolverPort):
    def __init__(self, values: Mapping[str, bytes]) -> None:
        self._values = dict(values)

    def resolve(self, asset_ref: str) -> bytes | None:
        return self._values.get(asset_ref)


class SocialSnapshotMediaAdapter:
    """Bind deterministic snapshot rendering to the existing owned media lifecycle."""

    def __init__(self, visual_service: _VisualServicePort) -> None:
        self._service = visual_service
        self._created: dict[str, str] = {}

    async def prepare_assets(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset_refs: Sequence[str],
        semantic_media_refs: Sequence[str],
    ) -> PreparedSnapshotAssets:
        values: dict[str, bytes] = {}
        descriptions: dict[str, str] = {}
        semantic = set(semantic_media_refs)
        for asset_ref in dict.fromkeys(asset_refs):
            asset = await self._service.media.get_media_asset(
                asset_ref,
                profile_id=profile_id,
                instance_id=instance_id,
            )
            if (
                asset is None
                or asset.file_status is not MediaFileStatus.AVAILABLE
                or not asset.storage_relpath
            ):
                raise SocialSnapshotError(
                    SocialSnapshotErrorCode.ASSET_MISSING,
                    "scene avatar or media references an unavailable controlled asset",
                )
            try:
                values[asset_ref] = await self._service.read_verified_media_bytes(
                    relative_path=asset.storage_relpath,
                    byte_size=asset.byte_size,
                    sha256=asset.sha256,
                )
            except (OSError, ValueError):
                raise SocialSnapshotError(
                    SocialSnapshotErrorCode.ASSET_MISSING,
                    "scene avatar or media references an unavailable controlled asset",
                ) from None
            if asset_ref in semantic:
                descriptions[asset_ref] = await self._source_description(asset_ref)
        return PreparedSnapshotAssets(_PreparedResolver(values), descriptions)

    async def register_parts(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        request_fingerprint: str,
        parts: Sequence[RenderedSnapshotPart],
        projection: Any,
    ) -> tuple[str, ...]:
        asset_ids: list[str] = []
        try:
            for part in parts:
                asset_ids.append(
                    await self._register_part(
                        profile_id=profile_id,
                        instance_id=instance_id,
                        run_id=run_id,
                        request_fingerprint=request_fingerprint,
                        part=part,
                        part_count=len(parts),
                        projection=projection,
                    )
                )
        except BaseException as original:
            try:
                await self.discard_created(reason="social_snapshot_batch_registration_failed")
            except Exception as cleanup_error:
                original.add_note(
                    "social snapshot batch cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        return tuple(asset_ids)

    async def discard_created(self, *, reason: str = "image_generation_disabled") -> None:
        created = tuple(self._created.items())
        self._created.clear()
        first_error: Exception | None = None
        for asset_id, relative_path in created:
            try:
                await self._service.media.mark_media_missing(
                    asset_id,
                    reason=reason,
                )
            except Exception as exc:
                # Do not unlink while the authoritative row may still say
                # AVAILABLE.  A later reconciliation can safely close it.
                if first_error is None:
                    first_error = exc
                continue
            self._service.file_store.delete(relative_path)
        if first_error is not None:
            raise first_error

    async def _source_description(self, asset_ref: str) -> str:
        projection = await self._service.media.get_latest_media_projection(asset_ref)
        if projection is None:
            return ""
        return str(projection.history_projection or projection.visible_facts or "").strip()

    async def _register_part(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        request_fingerprint: str,
        part: RenderedSnapshotPart,
        part_count: int,
        projection: Any,
    ) -> str:
        await self._service.require_image_generation_enabled(profile_id, instance_id)
        asset_id = _snapshot_asset_id(
            profile_id,
            instance_id,
            run_id,
            request_fingerprint,
            part.index,
            projection.text,
        )
        existing = await self._service.media.get_media_asset(
            asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        await self._service.require_image_generation_enabled(profile_id, instance_id)
        planned = await asyncio.to_thread(
            self._service.file_store.plan_store_bytes,
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=part.png_bytes,
            declared_mime="image/png",
        )
        cleanup_guard = await self._service.media.guard_unregistered_media_file(
            profile_id,
            instance_id,
            planned,
            reason="SOCIAL_SNAPSHOT_REGISTRATION",
        )
        stored = await asyncio.to_thread(
            self._service.file_store.store_bytes,
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=part.png_bytes,
            declared_mime="image/png",
        )
        if stored != planned:
            raise RuntimeError("social snapshot storage changed after durable planning")
        registration_attempted = False
        try:
            if not await asyncio.to_thread(self._service.file_store.verify, stored):
                raise RuntimeError("stored social snapshot part failed integrity verification")
            await self._service.require_image_generation_enabled(profile_id, instance_id)
            registration_attempted = True
            await self._register_exact_part(
                profile_id=profile_id,
                instance_id=instance_id,
                run_id=run_id,
                part=part,
                part_count=part_count,
                projection=projection,
                existing=existing,
                stored=stored,
                cleanup_guard_id=cleanup_guard.cleanup_id,
            )
            await self._service.require_image_generation_enabled(profile_id, instance_id)
            await self._save_projection_once(asset_id, projection.text)
            await self._service.require_image_generation_enabled(profile_id, instance_id)
        except ImageGenerationDisabledError:
            if asset_id in self._created:
                await self.discard_created()
            elif not registration_attempted:
                self._service.file_store.delete(stored.relative_path)
                await self._service.media.complete_runtime_file_cleanup(cleanup_guard.cleanup_id)
            raise
        except BaseException:
            if not registration_attempted:
                self._service.file_store.delete(stored.relative_path)
                await self._service.media.complete_runtime_file_cleanup(cleanup_guard.cleanup_id)
            raise
        return asset_id

    async def _register_exact_part(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        part: RenderedSnapshotPart,
        part_count: int,
        projection: Any,
        existing: Any,
        stored: Any,
        cleanup_guard_id: int,
    ) -> None:
        try:
            registered = await self._service.media.register_generated_media_asset(
                profile_id,
                instance_id,
                stored,
                core_run_id=run_id,
                expires_at=datetime.now(UTC) + timedelta(hours=24),
                metadata={
                    "source": "social_snapshot",
                    "theme": projection.theme,
                    "mode": projection.mode,
                    "part_index": part.index,
                    "part_count": part_count,
                },
                revive_missing_file=True,
                cleanup_guard_id=cleanup_guard_id,
            )
        except BaseException as registration_error:
            registered = await self._reconcile_registration_error(
                registration_error,
                profile_id=profile_id,
                instance_id=instance_id,
                existing=existing,
                stored=stored,
                cleanup_guard_id=cleanup_guard_id,
            )
        if not _is_exact_available_registration(
            registered,
            profile_id=profile_id,
            instance_id=instance_id,
            stored=stored,
        ):
            self._service.file_store.delete(stored.relative_path)
            await self._service.media.complete_runtime_file_cleanup(cleanup_guard_id)
            raise RuntimeError(
                "social snapshot registration did not commit the exact AVAILABLE asset"
            )
        if existing is None or existing.file_status is MediaFileStatus.MISSING:
            self._created[stored.asset_id] = stored.relative_path

    async def _reconcile_registration_error(
        self,
        registration_error: BaseException,
        *,
        profile_id: str,
        instance_id: str,
        existing: Any,
        stored: Any,
        cleanup_guard_id: int,
    ) -> Any:
        try:
            authoritative = await self._service.media.get_media_asset(
                stored.asset_id,
                profile_id=profile_id,
                instance_id=instance_id,
            )
        except Exception as authority_error:
            registration_error.add_note(
                "social snapshot registration state could not be confirmed: "
                f"{type(authority_error).__name__}: {authority_error}"
            )
            # The file may back a committed AVAILABLE row. Preserve it until
            # reconciliation can resolve the unknown outcome.
            raise registration_error from authority_error
        if not _is_exact_available_registration(
            authoritative,
            profile_id=profile_id,
            instance_id=instance_id,
            stored=stored,
        ):
            self._service.file_store.delete(stored.relative_path)
            await self._service.media.complete_runtime_file_cleanup(cleanup_guard_id)
            raise registration_error
        if not isinstance(registration_error, Exception):
            if existing is None or existing.file_status is MediaFileStatus.MISSING:
                self._created[stored.asset_id] = stored.relative_path
            # Control-flow exceptions must not become successful snapshots merely
            # because commit can be confirmed. Batch compensation owns the row.
            raise registration_error
        return authoritative

    async def _save_projection_once(self, asset_id: str, text: str) -> None:
        current = await self._service.media.get_latest_media_projection(asset_id)
        if (
            current is not None
            and current.status is MediaProjectionStatus.READY
            and str(current.history_projection) == text
        ):
            return
        await self._service.media.save_media_projection(
            asset_id,
            status=MediaProjectionStatus.READY,
            visible_facts=text,
            history_projection=text,
            ocr_text="",
            backend_id="deterministic_social_snapshot",
            model_id="",
        )


async def present_social_snapshot(
    visual_service: _VisualServicePort,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    profile_id = str(request["profile_id"])
    instance_id = str(request["instance_id"])
    await visual_service.require_image_generation_enabled(profile_id, instance_id)
    scene = request.get("scene")
    if not isinstance(scene, Mapping):
        raise ValueError("social snapshot scene is required")
    identity_context = await visual_service.identity.context(
        str(request["profile_id"]),
        str(request["instance_id"]),
    )
    rendered_scene = visual_service.identity.render_data(dict(scene), identity_context)
    adapter = SocialSnapshotMediaAdapter(visual_service)
    result = await SocialSnapshotService(adapter).render(
        profile_id=profile_id,
        instance_id=str(request["instance_id"]),
        run_id=int(request["run_id"]),
        scene_payload=rendered_scene,
        maximum_parts=int(request.get("maximum_parts") or 5),
    )
    try:
        await visual_service.require_image_generation_enabled(profile_id, instance_id)
    except ImageGenerationDisabledError:
        await adapter.discard_created()
        raise
    projection_payload = result.projection.as_payload()
    return {
        "content": result.projection.text,
        "content_parts": [],
        "asset_ids": list(result.asset_ids),
        "media_asset_ids": list(result.asset_ids),
        "generated_count": len(result.asset_ids),
        "registered_count": len(result.asset_ids),
        "semantic_projection": projection_payload,
        "part_count": len(result.part_dimensions),
        "part_dimensions": [
            {"width": width, "height": height} for width, height in result.part_dimensions
        ],
        "failures": [],
    }


def _snapshot_asset_id(
    profile_id: str,
    instance_id: str,
    run_id: int,
    request_fingerprint: str,
    part_index: int,
    projection_text: str,
) -> str:
    digest = hashlib.sha256(
        (
            f"{profile_id}\0{instance_id}\0{run_id}\0{request_fingerprint}\0"
            f"{part_index}\0{projection_text}"
        ).encode()
    ).hexdigest()
    return "ma_ss_" + digest[:32]


def _is_exact_available_registration(
    asset: Any,
    *,
    profile_id: str,
    instance_id: str,
    stored: Any,
) -> bool:
    return bool(
        asset is not None
        and asset.asset_id == stored.asset_id
        and asset.profile_id == profile_id
        and asset.instance_id == instance_id
        and asset.file_status is MediaFileStatus.AVAILABLE
        and asset.storage_relpath == stored.relative_path
        and asset.byte_size == stored.byte_size
        and asset.sha256 == stored.sha256
    )


__all__ = ["SocialSnapshotMediaAdapter", "present_social_snapshot"]
