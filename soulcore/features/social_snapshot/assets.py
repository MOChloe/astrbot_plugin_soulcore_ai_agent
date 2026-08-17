from __future__ import annotations

import io
import warnings

from PIL import Image, UnidentifiedImageError

from .errors import SocialSnapshotError, SocialSnapshotErrorCode
from .ports import ControlledAssetResolverPort

MAX_ASSET_BYTES = 20 * 1024 * 1024
MAX_DECODED_PIXELS = 16_000_000


class AssetImages:
    def __init__(self, resolver: ControlledAssetResolverPort) -> None:
        self._resolver = resolver
        self._cache: dict[str, Image.Image] = {}

    def get(self, asset_ref: str | None) -> Image.Image | None:
        if asset_ref is None:
            return None
        cached = self._cache.get(asset_ref)
        if cached is not None:
            return cached.copy()
        try:
            payload = self._resolver.resolve(asset_ref)
        except Exception as exc:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.ASSET_MISSING, "controlled asset is unavailable"
            ) from exc
        if payload is None:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.ASSET_MISSING, "controlled asset is unavailable"
            )
        if not isinstance(payload, bytes) or not payload:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.ASSET_INVALID, "controlled asset is invalid"
            )
        if len(payload) > MAX_ASSET_BYTES:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.ASSET_TOO_LARGE, "controlled asset exceeds limits"
            )
        image = self._decode(payload)
        self._cache[asset_ref] = image
        return image.copy()

    @staticmethod
    def _decode(payload: bytes) -> Image.Image:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as opened:
                    if opened.width * opened.height > MAX_DECODED_PIXELS:
                        raise SocialSnapshotError(
                            SocialSnapshotErrorCode.ASSET_TOO_LARGE,
                            "controlled asset exceeds limits",
                        )
                    opened.load()
                    return opened.convert("RGB")
        except SocialSnapshotError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.ASSET_TOO_LARGE, "controlled asset exceeds limits"
            ) from exc
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.ASSET_INVALID, "controlled asset is invalid"
            ) from exc
