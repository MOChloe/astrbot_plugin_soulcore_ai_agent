from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .dto import SocialSnapshotScene


class ControlledAssetResolverPort(Protocol):
    def resolve(self, asset_ref: str) -> bytes | None:
        """Resolve one already-authorized opaque asset reference to bytes."""


@dataclass(frozen=True, slots=True)
class RenderedSnapshotPart:
    index: int
    width: int
    height: int
    png_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class SocialSnapshotManifest:
    theme: str
    mode: str
    request_fingerprint: str
    parts: tuple[tuple[int, int, int, str], ...]


@dataclass(frozen=True, slots=True)
class SocialSnapshotRenderResult:
    scene: SocialSnapshotScene
    parts: tuple[RenderedSnapshotPart, ...]
    manifest: SocialSnapshotManifest
