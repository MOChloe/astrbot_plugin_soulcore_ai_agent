"""Normalize deliberately different web-provider response shapes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts.web import ProviderImageItem, ProviderSearchItem


def bearer_headers(key: str) -> Mapping[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def search_items(
    rows: Any,
    *,
    title: str,
    url: str,
    snippet: str | tuple[str, ...],
    favicon: str = "",
    published: str = "",
) -> tuple[ProviderSearchItem, ...]:
    output: list[ProviderSearchItem] = []
    if not isinstance(rows, (list, tuple)):
        return ()
    snippet_keys = (snippet,) if isinstance(snippet, str) else snippet
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or not row.get(url):
            continue
        snippet_value = _first_snippet(row, snippet_keys)
        output.append(
            ProviderSearchItem(
                title=str(row.get(title) or ""),
                url=str(row.get(url) or ""),
                snippet=str(snippet_value or ""),
                published_at=str(row.get(published) or "") if published else "",
                favicon=str(row.get(favicon) or "") if favicon else "",
                provider_rank=rank,
            )
        )
    return tuple(output)


def image_items(rows: Any, *, limit: int) -> tuple[ProviderImageItem, ...]:
    output: list[ProviderImageItem] = []
    if not isinstance(rows, (list, tuple)):
        return ()
    for rank, row in enumerate(rows, 1):
        normalized = _image_row(row)
        if normalized is None:
            continue
        item = _image_item(normalized, rank)
        if item is None:
            continue
        output.append(item)
        if len(output) >= limit:
            break
    return tuple(output)


def bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _first_snippet(row: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            value = value[0] if value else ""
        if value:
            return value
    return ""


def _image_row(row: Any) -> Mapping[str, Any] | None:
    if isinstance(row, str):
        return {"url": row}
    return row if isinstance(row, Mapping) else None


def _image_item(row: Mapping[str, Any], rank: int) -> ProviderImageItem | None:
    properties = row.get("properties") if isinstance(row.get("properties"), Mapping) else {}
    thumbnail = row.get("thumbnail") if isinstance(row.get("thumbnail"), Mapping) else {}
    nested = row.get("image") if isinstance(row.get("image"), Mapping) else {}
    image_url, source_url, thumbnail_url = _image_urls(row, properties, thumbnail, nested)
    if not image_url:
        return None
    return ProviderImageItem(
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        source_url=source_url,
        title=_first(row, "title", "name"),
        description=_first(row, "description", "content", "snippet"),
        width=_dimension(row, nested or properties, "width"),
        height=_dimension(row, nested or properties, "height"),
        provider_rank=rank,
    )


def _image_urls(
    row: Mapping[str, Any],
    properties: Mapping[str, Any],
    thumbnail: Mapping[str, Any],
    nested: Mapping[str, Any],
) -> tuple[str, str, str]:
    image_url = (
        _first(row, "image_url", "imageUrl", "contentUrl", "original", "src")
        or _first(nested, "url", "image_url", "imageUrl", "contentUrl", "src")
        or _first(properties, "url")
        or _first(row, "url")
    )
    source_url = _first(row, "source_url", "sourceUrl", "hostPageUrl", "page_url", "pageUrl")
    if not source_url and (properties or nested):
        source_url = _first(row, "url")
    thumbnail_url = (
        _first(row, "thumbnail_url", "thumbnailUrl", "thumbUrl")
        or _first(thumbnail, "src", "url")
        or _first(nested, "thumbnail_url", "thumbnailUrl", "thumbUrl", "thumbnail")
        or image_url
    )
    return image_url, source_url, thumbnail_url


def _first(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _dimension(row: Mapping[str, Any], properties: Mapping[str, Any], key: str) -> int:
    return bounded_int(row.get(key, properties.get(key)), 0, 0, 1_000_000)


__all__ = ["bearer_headers", "bounded_int", "image_items", "search_items"]
