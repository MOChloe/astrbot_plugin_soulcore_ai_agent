"""Private fitting helpers for the conversation context compiler."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ...shared.token_meter import ConservativeTokenMeter, TokenMeter

if TYPE_CHECKING:
    from .context import ContextItem

MIN_DIALOGUE_MESSAGES = 20


def _ordered(items: Iterable[ContextItem]) -> list[ContextItem]:
    return sorted(items, key=lambda item: (item.sequence, item.item_id))


def _fit_newest(
    items: Sequence[ContextItem],
    token_limit: int,
    meter: TokenMeter,
) -> tuple[list[ContextItem], list[ContextItem]]:
    kept_reversed: list[ContextItem] = []
    used = 0
    for item in reversed(_ordered(items)):
        cost = meter.measure_item(item).tokens
        if used + cost <= token_limit:
            kept_reversed.append(item)
            used += cost
    kept = list(reversed(kept_reversed))
    kept_ids = {item.item_id for item in kept}
    return kept, [item for item in items if item.item_id not in kept_ids]


def _fit_chronological_history(
    items: Sequence[ContextItem],
    token_limit: int,
    meter: TokenMeter,
) -> tuple[list[ContextItem], list[ContextItem], list[str]]:
    """Fit one uninterrupted newest-to-oldest suffix of compressed history.

    Historical prose is a timeline, not a bag of search hits.  Once the next
    older record cannot fit, no still-older record may jump across that gap.
    A record's own ultra-brief projection may be used at the boundary because
    it is still the same story record, only shorter.
    """

    ordered = _ordered(items)
    kept_reversed: list[ContextItem] = []
    shortened: list[str] = []
    used = 0
    for item in reversed(ordered):
        candidate = item
        cost = meter.measure_item(candidate).tokens
        if used + cost > token_limit:
            compact = item.metadata.get("compact_content")
            if not isinstance(compact, str) or not compact.strip() or compact == item.body:
                break
            candidate = replace(
                item,
                body=compact,
                metadata={**dict(item.metadata), "render_level": "compact"},
            )
            cost = meter.measure_item(candidate).tokens
            if used + cost > token_limit:
                break
            shortened.append(item.item_id)
        kept_reversed.append(candidate)
        used += cost

    kept = list(reversed(kept_reversed))
    kept_ids = {item.item_id for item in kept}
    return kept, [item for item in ordered if item.item_id not in kept_ids], shortened


def _fit_scored(
    items: Sequence[ContextItem],
    token_limit: int,
    meter: TokenMeter,
) -> tuple[list[ContextItem], list[ContextItem]]:
    """Fit independent knowledge records by relevance, not creation order."""

    ranked = sorted(
        items,
        key=lambda item: (
            float(item.metadata.get("score", 0.0)),
            int(item.metadata.get("rank", 0)),
            item.sequence,
            item.item_id,
        ),
        reverse=True,
    )
    kept: list[ContextItem] = []
    used = 0
    for item in ranked:
        cost = meter.measure_item(item).tokens
        if used + cost <= token_limit:
            kept.append(item)
            used += cost
    kept_ids = {item.item_id for item in kept}
    return _ordered(kept), [item for item in items if item.item_id not in kept_ids]


def _fit_dialogue_with_floor(
    items: Sequence[ContextItem],
    token_limit: int,
    meter: TokenMeter,
) -> tuple[list[ContextItem], list[ContextItem], list[str]]:
    """Fit one contiguous newest suffix and preserve its last 20 structures.

    Dialogue is chronological context, not a bag of independently useful
    snippets. A large recent message may be shortened, but it must never be
    skipped in favour of an older small message.
    """

    ordered = _ordered(items)
    newest = list(ordered[-min(MIN_DIALOGUE_MESSAGES, len(ordered)) :])
    anchors = [item for item in ordered if bool(item.metadata.get("dialogue_anchor"))]
    kept_by_id = {item.item_id: item for item in (*anchors, *newest)}
    kept = [item for item in ordered if item.item_id in kept_by_id]
    shortened: list[str] = []
    fallback = meter if isinstance(meter, ConservativeTokenMeter) else ConservativeTokenMeter()
    _shorten_dialogue_floor(kept, shortened, token_limit, meter, fallback)
    _shave_dialogue_floor(kept, shortened, token_limit, meter, fallback)
    _prepend_newest_dialogue(ordered, kept, token_limit, meter)
    kept_ids = {item.item_id for item in kept}
    return kept, [item for item in ordered if item.item_id not in kept_ids], shortened


def _shorten_dialogue_floor(
    kept: list[ContextItem],
    shortened: list[str],
    token_limit: int,
    meter: TokenMeter,
    fallback: ConservativeTokenMeter,
) -> None:
    for index in range(len(kept)):
        total = sum(meter.measure_item(item).tokens for item in kept)
        if total <= token_limit:
            break
        item = kept[index]
        if bool(item.metadata.get("dialogue_anchor")) or not isinstance(item.body, str):
            continue
        excess = total - token_limit
        current = meter.measure_item(item).tokens
        target = max(
            0,
            current - excess - ConservativeTokenMeter.MESSAGE_OVERHEAD - 8,
        )
        body = _truncate_string(item.body, target, fallback)
        if body != item.body:
            kept[index] = replace(item, body=body)
            shortened.append(item.item_id)


def _shave_dialogue_floor(
    kept: list[ContextItem],
    shortened: list[str],
    token_limit: int,
    meter: TokenMeter,
    fallback: ConservativeTokenMeter,
) -> None:
    # Mixed-script rounding and per-message overhead can leave a few tokens
    # above the exact source ceiling; shave content until the fixed cap holds.
    while sum(meter.measure_item(item).tokens for item in kept) > token_limit:
        excess = sum(meter.measure_item(item).tokens for item in kept) - token_limit
        changed = False
        for index, item in enumerate(kept):
            if (
                bool(item.metadata.get("dialogue_anchor"))
                or not isinstance(item.body, str)
                or not item.body
            ):
                continue
            body_tokens = fallback.count_text(item.body)
            body = _truncate_string(
                item.body,
                max(0, body_tokens - max(1, excess)),
                fallback,
            )
            if body != item.body:
                kept[index] = replace(item, body=body)
                if item.item_id not in shortened:
                    shortened.append(item.item_id)
                changed = True
                break
        if not changed:
            break


def _prepend_newest_dialogue(
    ordered: list[ContextItem],
    kept: list[ContextItem],
    token_limit: int,
    meter: TokenMeter,
) -> None:
    # Only after reply/mention anchors and the newest floor fit may other recent
    # messages be added. Anchors are intentionally allowed to make the result
    # non-contiguous so a direct reply chain is not lost.
    used = sum(meter.measure_item(item).tokens for item in kept)
    if used <= token_limit:
        kept_ids = {item.item_id for item in kept}
        for candidate in reversed(ordered):
            if candidate.item_id in kept_ids:
                continue
            cost = meter.measure_item(candidate).tokens
            if used + cost > token_limit:
                continue
            kept.append(candidate)
            kept.sort(key=lambda item: (item.sequence, item.item_id))
            kept_ids.add(candidate.item_id)
            used += cost


def _truncate_string(
    text: str,
    target_tokens: int,
    meter: ConservativeTokenMeter,
) -> str:
    if target_tokens <= 0:
        return ""
    if meter.count_text(text) <= target_tokens:
        return text
    # Binary search avoids assuming one tokenizer ratio for mixed CJK/Latin.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if meter.count_text(text[:mid]) <= max(0, target_tokens - 1):
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + "…"
