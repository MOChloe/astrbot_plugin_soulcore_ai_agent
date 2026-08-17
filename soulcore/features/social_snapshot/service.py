from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .dto import (
    CompactItem,
    CompactPerson,
    CompactQuote,
    CompactUi,
    SocialSnapshotRequest,
)
from .normalize import normalize_request
from .pillow_renderer import PillowSocialSnapshotRenderer
from .ports import ControlledAssetResolverPort, RenderedSnapshotPart
from .projection import SocialSnapshotProjection, compile_social_snapshot_projection


@dataclass(frozen=True, slots=True)
class PreparedSnapshotAssets:
    resolver: ControlledAssetResolverPort
    media_descriptions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SocialSnapshotServiceResult:
    projection: SocialSnapshotProjection
    asset_ids: tuple[str, ...]
    part_dimensions: tuple[tuple[int, int], ...]


class SocialSnapshotMediaPort(Protocol):
    async def prepare_assets(
        self,
        *,
        profile_id: str,
        instance_id: str,
        asset_refs: Sequence[str],
        semantic_media_refs: Sequence[str],
    ) -> PreparedSnapshotAssets: ...

    async def register_parts(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        request_fingerprint: str,
        parts: Sequence[RenderedSnapshotPart],
        projection: SocialSnapshotProjection,
    ) -> tuple[str, ...]: ...


class SocialSnapshotService:
    def __init__(self, media: SocialSnapshotMediaPort) -> None:
        self._media = media

    async def render(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        scene_payload: Mapping[str, Any],
        maximum_parts: int = 5,
    ) -> SocialSnapshotServiceResult:
        request = social_snapshot_request_from_payload(scene_payload)
        # Validate the compact scene before interpreting optional avatar/media strings as
        # controlled asset IDs. This keeps malformed model arguments out of repository lookup
        # and preserves an actionable, redaction-safe domain error for the command caller.
        normalize_request(request)
        all_refs, semantic_refs = _asset_refs(request)
        prepared = await self._media.prepare_assets(
            profile_id=profile_id,
            instance_id=instance_id,
            asset_refs=all_refs,
            semantic_media_refs=semantic_refs,
        )
        renderer = PillowSocialSnapshotRenderer(prepared.resolver)
        rendered = await asyncio.to_thread(renderer.render, request)
        if not 1 <= len(rendered.parts) <= max(0, int(maximum_parts)):
            raise ValueError("social snapshot exceeds the remaining visual-output budget")
        dimensions = tuple((part.width, part.height) for part in rendered.parts)
        projection = compile_social_snapshot_projection(
            rendered.scene,
            media_descriptions=prepared.media_descriptions,
            part_dimensions=dimensions,
        )
        asset_ids = await self._media.register_parts(
            profile_id=profile_id,
            instance_id=instance_id,
            run_id=run_id,
            request_fingerprint=rendered.scene.request_fingerprint,
            parts=rendered.parts,
            projection=projection,
        )
        return SocialSnapshotServiceResult(projection, asset_ids, dimensions)


def social_snapshot_request_from_payload(payload: Mapping[str, Any]) -> SocialSnapshotRequest:
    scene = _mapping(
        payload,
        label="scene",
        allowed={"theme", "mode", "title", "people", "items", "draft", "ui"},
        required={"theme", "mode", "title", "people", "items"},
    )
    people = _sequence(scene["people"], label="scene.people")
    items = _sequence(scene["items"], label="scene.items")
    return SocialSnapshotRequest(
        theme=scene["theme"],
        mode=scene["mode"],
        title=scene["title"],
        people=tuple(_person(item) for item in people),
        items=tuple(_item(item) for item in items),
        draft=scene.get("draft", ""),
        ui=_ui(scene.get("ui", {})),
    )


def _person(value: object) -> CompactPerson:
    item = _mapping(
        value,
        label="person",
        allowed={"id", "name", "avatar", "side", "badge", "color"},
        required={"id", "name"},
    )
    return CompactPerson(
        id=item["id"],
        name=item["name"],
        avatar=item.get("avatar"),
        side=item.get("side", "left"),
        badge=item.get("badge", ""),
        color=item.get("color", "#7f8c9a"),
    )


def _item(value: object) -> CompactItem:
    item = _mapping(
        value,
        label="item",
        allowed={"k", "by", "text", "time", "media", "quote"},
        required={"k"},
    )
    return CompactItem(
        k=item["k"],
        by=item.get("by"),
        text=item.get("text", ""),
        time=item.get("time", ""),
        media=item.get("media"),
        quote=_quote(item.get("quote")),
    )


def _quote(value: object) -> CompactQuote | None:
    if value is None:
        return None
    item = _mapping(
        value,
        label="quote",
        allowed={"sender", "text", "media_label", "time"},
        required={"sender"},
    )
    return CompactQuote(
        sender=item["sender"],
        text=item.get("text", ""),
        media_label=item.get("media_label", ""),
        time=item.get("time", ""),
    )


def _ui(value: object) -> CompactUi:
    item = _mapping(
        value,
        label="ui",
        allowed={
            "subtitle",
            "clock",
            "battery_percent",
            "battery_charging",
            "width",
            "height",
            "segment_height",
        },
        required=set(),
    )
    return CompactUi(
        subtitle=item.get("subtitle", ""),
        clock=item.get("clock", "00:00"),
        battery_percent=item.get("battery_percent", 100),
        battery_charging=item.get("battery_charging", False),
        width=item.get("width", 873),
        height=item.get("height"),
        segment_height=item.get("segment_height", 1920),
    )


def _mapping(
    value: object,
    *,
    label: str,
    allowed: set[str],
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result = dict(value)
    unknown = set(result) - allowed
    missing = required - set(result)
    if unknown:
        raise ValueError(f"{label} contains unsupported fields")
    if missing:
        raise ValueError(f"{label} is missing required fields")
    return result


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return list(value)


def _asset_refs(request: SocialSnapshotRequest) -> tuple[tuple[str, ...], tuple[str, ...]]:
    avatars = [person.avatar for person in request.people if person.avatar]
    media = [item.media for item in request.items if item.media]
    return tuple(dict.fromkeys((*avatars, *media))), tuple(dict.fromkeys(media))


__all__ = [
    "PreparedSnapshotAssets",
    "SocialSnapshotMediaPort",
    "SocialSnapshotService",
    "SocialSnapshotServiceResult",
    "social_snapshot_request_from_payload",
]
