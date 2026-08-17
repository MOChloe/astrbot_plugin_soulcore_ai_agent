from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageDraw

from .assets import AssetImages
from .chat_renderer import ChatRenderer
from .drawing import FontBook
from .dto import SnapshotTheme, SocialSnapshotRequest, SocialSnapshotScene
from .errors import SocialSnapshotError, SocialSnapshotErrorCode
from .feed_renderer import FeedRenderer
from .normalize import MAX_PARTS, MAX_TOTAL_PIXELS, normalize_request
from .ports import (
    ControlledAssetResolverPort,
    RenderedSnapshotPart,
    SocialSnapshotManifest,
    SocialSnapshotRenderResult,
)

CHAT_THEMES = {
    SnapshotTheme.MOBILE_CHAT,
    SnapshotTheme.WECHAT,
    SnapshotTheme.DINGTALK,
}


class PillowSocialSnapshotRenderer:
    def __init__(
        self,
        asset_resolver: ControlledAssetResolverPort,
        *,
        font_path: Path | None = None,
    ) -> None:
        self._fonts = FontBook(font_path) if font_path is not None else FontBook()
        self._asset_resolver = asset_resolver

    def render(self, request: SocialSnapshotRequest) -> SocialSnapshotRenderResult:
        scene = normalize_request(request)
        try:
            image = self._render_scene(scene, AssetImages(self._asset_resolver))
            parts = self._encode_parts(scene, image)
        except SocialSnapshotError:
            raise
        except Exception as exc:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.RENDER_FAILED, "snapshot rendering failed"
            ) from exc
        manifest = SocialSnapshotManifest(
            theme=scene.theme,
            mode=scene.mode,
            request_fingerprint=scene.request_fingerprint,
            parts=tuple((part.index, part.width, part.height, part.sha256) for part in parts),
        )
        return SocialSnapshotRenderResult(scene=scene, parts=parts, manifest=manifest)

    def _render_scene(self, scene: SocialSnapshotScene, assets: AssetImages) -> Image.Image:
        if scene.theme in CHAT_THEMES:
            return ChatRenderer(self._fonts, assets).render(scene)
        return FeedRenderer(self._fonts, assets).render(scene)

    def _encode_parts(
        self, scene: SocialSnapshotScene, image: Image.Image
    ) -> tuple[RenderedSnapshotPart, ...]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.width * image.height > MAX_TOTAL_PIXELS:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.LIMIT_EXCEEDED, "rendered pixel limit exceeded"
            )
        segments = self._segments(scene, image)
        if len(segments) > MAX_PARTS:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.LIMIT_EXCEEDED, "rendered part limit exceeded"
            )
        return tuple(
            self._encode(index, segment, len(segments)) for index, segment in enumerate(segments, 1)
        )

    @staticmethod
    def _segments(scene: SocialSnapshotScene, image: Image.Image) -> tuple[Image.Image, ...]:
        if scene.ui.height is not None:
            return (image,)
        height = scene.ui.segment_height
        return tuple(
            image.crop((0, top, image.width, min(image.height, top + height))).convert("RGB")
            for top in range(0, image.height, height)
        )

    def _encode(self, index: int, image: Image.Image, total: int) -> RenderedSnapshotPart:
        if total > 1:
            self._draw_part_marker(image, index, total)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False, compress_level=9)
        payload = output.getvalue()
        return RenderedSnapshotPart(
            index=index,
            width=image.width,
            height=image.height,
            png_bytes=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def _draw_part_marker(self, image: Image.Image, index: int, total: int) -> None:
        draw = ImageDraw.Draw(image)
        marker_font = self._fonts.get(max(15, round(image.width * 0.02)))
        marker = f"AI演绎 · {index}/{total}"
        box = draw.textbbox((0, 0), marker, font=marker_font)
        width = box[2] - box[0] + 24
        draw.rounded_rectangle(
            (image.width - width - 16, 12, image.width - 16, 12 + box[3] - box[1] + 16),
            radius=10,
            fill="#ededed",
        )
        draw.text((image.width - width - 4, 18 - box[1]), marker, font=marker_font, fill="#676767")
