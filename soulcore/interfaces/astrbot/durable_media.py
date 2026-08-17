"""Rebuild model-facing inbound media from durable owned records."""

from __future__ import annotations

from typing import Any

from ...features.media.errors import IMAGE_INGEST_FAILED

MEDIA_OUTCOME_METADATA_KEY = "inbound_media_outcome"
MEDIA_OUTCOME_VERSION = 1
IMAGE_FAILURE_CATEGORIES = frozenset(
    {
        "IMAGE_LIMIT_EXCEEDED",
        "IMAGE_RESOLUTION_FAILED",
        IMAGE_INGEST_FAILED,
    }
)
ATTACHMENT_FAILURE_CATEGORIES = frozenset(
    {
        "ATTACHMENT_LIMIT_EXCEEDED",
        "ATTACHMENT_RESOLUTION_FAILED",
        "ATTACHMENT_INGEST_FAILED",
    }
)
_ALL_FAILURE_CATEGORIES = IMAGE_FAILURE_CATEGORIES | ATTACHMENT_FAILURE_CATEGORIES


def inbound_media_outcome(
    *,
    image_input_count: int,
    image_success_count: int,
    attachment_input_count: int,
    attachment_success_count: int,
    failure_categories: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    return {
        "version": MEDIA_OUTCOME_VERSION,
        "image_input_count": max(0, int(image_input_count)),
        "image_success_count": max(0, int(image_success_count)),
        "attachment_input_count": max(0, int(attachment_input_count)),
        "attachment_success_count": max(0, int(attachment_success_count)),
        "failure_categories": list(
            dict.fromkeys(
                category
                for category in (str(value).strip().upper() for value in failure_categories)
                if category in _ALL_FAILURE_CATEGORIES
            )
        ),
    }


def apply_media_outcome_projection(
    payload: dict[str, Any],
    outcomes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> None:
    categories: set[str] = set()
    for outcome in outcomes:
        if int(outcome.get("version") or 0) != MEDIA_OUTCOME_VERSION:
            continue
        raw_categories = outcome.get("failure_categories")
        if not isinstance(raw_categories, list):
            continue
        categories.update(
            category
            for category in (str(value).strip().upper() for value in raw_categories)
            if category in _ALL_FAILURE_CATEGORIES
        )
    payload["media_ingest_error"] = (
        "inbound_image_ingest_incomplete" if categories & IMAGE_FAILURE_CATEGORIES else ""
    )
    payload["inbound_media_error"] = (
        "inbound_attachment_ingest_incomplete" if categories & ATTACHMENT_FAILURE_CATEGORIES else ""
    )


def _image_source_projection(
    media_history: dict[Any, Any], available_asset_ids: list[str]
) -> dict[str, int]:
    candidates: dict[str, set[int]] = {}
    available = set(available_asset_ids)
    for raw_message_id, rows in media_history.items():
        message_id = int(raw_message_id)
        if message_id <= 0:
            continue
        for row in rows:
            asset_id = str(row.get("asset_id") or "").strip()
            if asset_id in available:
                candidates.setdefault(asset_id, set()).add(message_id)
    return {
        asset_id: next(iter(source_ids))
        for asset_id, source_ids in candidates.items()
        if len(source_ids) == 1
    }


def _safe_attachment_refs(refs: list[dict[str, Any]]) -> list[dict[str, str]]:
    safe_refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in refs:
        asset_id = str(item.get("asset_id") or "")
        if not asset_id or asset_id in seen:
            continue
        seen.add(asset_id)
        safe_refs.append(
            {
                "asset_id": asset_id,
                "kind": str(item.get("kind") or ""),
                "display_name": str(item.get("display_name") or ""),
            }
        )
    return safe_refs[:20]


async def _durable_media_outcomes(
    conversation_repository: Any,
    *,
    profile_id: str,
    instance_id: str,
    message_ids: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for message_id in dict.fromkeys(int(value) for value in message_ids):
        message = await conversation_repository.get_instance_message(
            profile_id,
            instance_id,
            message_id,
        )
        if message is None or str(message.delivery_status) != "RECEIVED":
            continue
        raw = message.metadata.get(MEDIA_OUTCOME_METADATA_KEY)
        if isinstance(raw, dict):
            outcomes.append(raw)
    return outcomes


async def reconstruct_durable_media_payload(
    media_repository: Any,
    conversation_repository: Any,
    *,
    profile_id: str,
    instance_id: str,
    message_ids: list[int] | tuple[int, ...],
    payload: dict[str, Any],
) -> None:
    """Replace transient platform locators with bounded safe asset projections."""

    available_image_ids = list(
        await media_repository.list_available_image_asset_ids_for_messages(
            profile_id,
            instance_id,
            message_ids,
            limit=100,
        )
    )
    payload["image_urls"] = []
    # Transient payload order is neither durable nor scoped.  Final model order
    # comes exclusively from the ledger-owned query above.
    payload["media_asset_ids"] = list(dict.fromkeys(available_image_ids))[:5]
    media_history = await media_repository.media_history_projections_for_messages(
        profile_id,
        instance_id,
        message_ids,
    )
    payload["media_asset_message_ids"] = _image_source_projection(
        media_history, payload["media_asset_ids"]
    )
    refs = await media_repository.list_available_attachment_refs_for_messages(
        profile_id,
        instance_id,
        message_ids,
        limit=100,
    )
    payload["inbound_media_refs"] = _safe_attachment_refs(refs)
    outcomes = await _durable_media_outcomes(
        conversation_repository,
        profile_id=profile_id,
        instance_id=instance_id,
        message_ids=message_ids,
    )
    apply_media_outcome_projection(payload, outcomes)


__all__ = [
    "MEDIA_OUTCOME_METADATA_KEY",
    "apply_media_outcome_projection",
    "inbound_media_outcome",
    "reconstruct_durable_media_payload",
]
