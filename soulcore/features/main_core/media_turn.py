"""Small current-media helpers shared by MainCore preparation phases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def current_media_asset_ids(metadata: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in metadata.get("media_asset_ids", ()) or () if str(value).strip()]


def main_core_supports_vision(visual_service: Any, backend_hint: Any) -> bool:
    return bool(visual_service is not None and visual_service.backend_supports_vision(backend_hint))


__all__ = ["current_media_asset_ids", "main_core_supports_vision"]
