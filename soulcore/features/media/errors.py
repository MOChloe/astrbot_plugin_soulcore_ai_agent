"""Stable, model-safe media workflow errors."""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_INGEST_FAILED = "IMAGE_INGEST_FAILED"


@dataclass(frozen=True, slots=True)
class InboundImageIngestResult:
    """Bounded image ingest outcome without locators or exception text."""

    asset_ids: tuple[str, ...]
    failure_categories: tuple[str, ...]


class ImageGenerationDisabledError(RuntimeError):
    """The profile disabled creation of new game-visible image assets."""

    code = "IMAGE_GENERATION_DISABLED"

    def __init__(self) -> None:
        super().__init__("image generation is disabled for this profile")


class ImageGenerationRequestError(RuntimeError):
    """Actionable, model-safe failure in a requested image specification."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.safe_message = str(message)
        super().__init__(self.safe_message)


class WebImageInspectionError(RuntimeError):
    """Safe, model-visible classification for the selected-image pipeline."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.safe_message = str(message)
        super().__init__(self.safe_message)


__all__ = [
    "IMAGE_INGEST_FAILED",
    "ImageGenerationDisabledError",
    "ImageGenerationRequestError",
    "InboundImageIngestResult",
    "WebImageInspectionError",
]
