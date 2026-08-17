"""Conservative, explainable sticker likelihood for inbound images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

ANIMATED_IMAGE = "ANIMATED_IMAGE"
COMPACT_IMAGE = "COMPACT_IMAGE"
PLATFORM_EMOJI_METADATA = "PLATFORM_EMOJI_METADATA"
PLATFORM_STICKER_SUMMARY = "PLATFORM_STICKER_SUMMARY"
PREVIOUS_CLASSIFICATION = "PREVIOUS_CLASSIFICATION"

_PLATFORM_EVIDENCE = frozenset({PLATFORM_EMOJI_METADATA, PLATFORM_STICKER_SUMMARY})
# Published platform assets cluster between tiny custom emoji and 512px stickers,
# while received chat images may be re-encoded larger.  The area limit admits
# common 720/768px square variants without treating ordinary 1024x768 photos as
# compact chat media.
_MIN_STICKER_EDGE = 32
_MAX_STICKER_EDGE = 1024
_MAX_STICKER_PIXELS = 1024 * 640
_MAX_STICKER_ASPECT_RATIO = 3.0
_KNOWN_EVIDENCE = frozenset(
    {
        *_PLATFORM_EVIDENCE,
        ANIMATED_IMAGE,
        COMPACT_IMAGE,
        PREVIOUS_CLASSIFICATION,
    }
)


@dataclass(frozen=True, slots=True)
class StickerLikelihood:
    possible: bool
    evidence: tuple[str, ...]


def classify_possible_sticker(
    *,
    mime_type: str,
    width: int | None,
    height: int | None,
    frame_count: int | None,
    evidence: Sequence[str] = (),
    previously_possible: bool = False,
) -> StickerLikelihood:
    """Classify only conspicuous cases; absence deliberately means unknown."""

    observed = _known_evidence(evidence)
    animated, compact, reasonable_animation = _image_shape(mime_type, width, height, frame_count)
    if animated:
        observed.append(ANIMATED_IMAGE)
    if compact:
        observed.append(COMPACT_IMAGE)

    platform_marked = bool(_PLATFORM_EVIDENCE.intersection(observed))
    possible = bool(previously_possible or platform_marked or reasonable_animation or compact)
    if previously_possible and not observed:
        observed.append(PREVIOUS_CLASSIFICATION)
    return StickerLikelihood(possible, tuple(dict.fromkeys(observed)))


def asset_sticker_likelihood(asset: Any) -> StickerLikelihood:
    metadata = dict(getattr(asset, "metadata", None) or {})
    evidence = metadata.get("sticker_evidence") or ()
    if isinstance(evidence, str):
        evidence = (evidence,)
    return classify_possible_sticker(
        mime_type=str(getattr(asset, "mime_type", "") or ""),
        width=getattr(asset, "width", None),
        height=getattr(asset, "height", None),
        frame_count=getattr(asset, "frame_count", None),
        evidence=evidence,
        previously_possible=metadata.get("possible_sticker") is True,
    )


def _image_shape(
    mime_type: str,
    width: int | None,
    height: int | None,
    frame_count: int | None,
) -> tuple[bool, bool, bool]:
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    animated = (_positive_int(frame_count) or 0) > 1 or normalized_mime in {
        "image/gif",
        "image/apng",
    }
    width_value = _positive_int(width)
    height_value = _positive_int(height)
    if width_value is None or height_value is None:
        return animated, False, animated

    max_edge = max(width_value, height_value)
    min_edge = min(width_value, height_value)
    aspect_ratio = max_edge / min_edge
    compact = (
        min_edge >= _MIN_STICKER_EDGE
        and max_edge <= _MAX_STICKER_EDGE
        and width_value * height_value <= _MAX_STICKER_PIXELS
        and aspect_ratio <= _MAX_STICKER_ASPECT_RATIO
    )
    reasonable_animation = animated and max_edge <= 1280 and aspect_ratio <= 3.0
    return animated, compact, reasonable_animation


def _known_evidence(values: Sequence[str]) -> list[str]:
    if isinstance(values, str):
        values = (values,)
    return [str(value) for value in values if str(value) in _KNOWN_EVIDENCE]


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "ANIMATED_IMAGE",
    "COMPACT_IMAGE",
    "PLATFORM_EMOJI_METADATA",
    "PLATFORM_STICKER_SUMMARY",
    "StickerLikelihood",
    "asset_sticker_likelihood",
    "classify_possible_sticker",
]
