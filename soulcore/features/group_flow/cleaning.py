"""Pure, traceable projection of group messages for judging and Main Core input."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from ...contracts.group_flow import GroupFlowSourceMessage
from ..media.service import MediaFingerprintSet, are_strongly_similar

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CleanedGroupMessage:
    representative_text: str
    message_ids: tuple[int, ...]
    sender_labels: tuple[str, ...]
    sender_identities: tuple[tuple[str, str], ...]
    occurrence_count: int
    participant_count: int
    first_at: datetime
    last_at: datetime
    media_kinds: tuple[str, ...] = ()
    blank: bool = False


@dataclass(frozen=True, slots=True)
class CleanedGroupProjection:
    messages: tuple[CleanedGroupMessage, ...]
    source_message_ids: tuple[int, ...]
    repeat_ratio: float


def normalize_group_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = _CONTROL.sub("", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalized_text_fingerprint(
    value: str,
    media_kinds: tuple[str, ...] = (),
    media_cluster_keys: tuple[str, ...] = (),
    *,
    message_id: int | None = None,
) -> str:
    text = normalize_group_text(value)
    media = ",".join(sorted(str(kind).strip().upper() for kind in media_kinds if str(kind)))
    clusters = tuple(sorted(str(key).strip() for key in media_cluster_keys if str(key).strip()))
    if media and not clusters:
        clusters = (f"message:{int(message_id or 0)}",)
    material = f"{text}\x1f{media}\x1f{','.join(clusters)}".encode()
    return hashlib.sha256(material).hexdigest()


def clean_group_messages(
    messages: tuple[GroupFlowSourceMessage, ...], *, sample_limit: int = 4
) -> CleanedGroupProjection:
    if not messages:
        return CleanedGroupProjection((), (), 0.0)
    runs: list[
        tuple[tuple[str, tuple[str, ...], tuple[str, ...]], list[GroupFlowSourceMessage]]
    ] = []
    for message in messages:
        key = _cleaning_key(message)
        if runs and runs[-1][0] == key:
            runs[-1][1].append(message)
        else:
            runs.append((key, [message]))
    # Only collapse a contiguous run.  Global grouping moves a later intent
    # before the messages that preceded it (for example STOP, GO, STOP).
    cleaned = tuple(_clean_group(key, members, sample_limit) for key, members in runs)
    repeated = sum(max(0, item.occurrence_count - 1) for item in cleaned)
    return CleanedGroupProjection(
        messages=cleaned,
        source_message_ids=tuple(message.message_id for message in messages),
        repeat_ratio=repeated / len(messages),
    )


def media_cluster_keys_match(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    if not first or not second:
        return False
    if tuple(sorted(first)) == tuple(sorted(second)):
        return True
    left_sha, left_fingerprints = _parsed_media_keys(first)
    right_sha, right_fingerprints = _parsed_media_keys(second)
    if left_sha.intersection(right_sha):
        return True
    return _fingerprints_match(left_fingerprints, right_fingerprints)


def _parsed_media_keys(
    values: tuple[str, ...],
) -> tuple[set[str], tuple[MediaFingerprintSet, ...]]:
    hashes: set[str] = set()
    fingerprints: list[MediaFingerprintSet] = []
    for value in values:
        sha, fingerprint = _parse_media_key(value)
        if sha:
            hashes.add(sha)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return hashes, tuple(fingerprints)


def _fingerprints_match(
    left: tuple[MediaFingerprintSet, ...], right: tuple[MediaFingerprintSet, ...]
) -> bool:
    return any(
        are_strongly_similar(left_value, right_value)
        for left_value in left
        for right_value in right
    )


def _cleaning_key(
    message: GroupFlowSourceMessage,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    media = tuple(sorted(str(kind).strip().upper() for kind in message.media_kinds))
    clusters = tuple(
        sorted(str(value).strip() for value in message.media_cluster_keys if str(value).strip())
    )
    if media and not clusters:
        clusters = (f"message:{message.message_id}",)
    return normalize_group_text(message.plain_text), media, clusters


def _parse_media_key(value: str) -> tuple[str, MediaFingerprintSet | None]:
    fields = {}
    for part in str(value).split(";"):
        name, separator, raw = part.partition("=")
        if separator:
            fields[name.strip().lower()] = raw.strip()
    phash, dhash = fields.get("phash", ""), fields.get("dhash", "")
    if not phash and not dhash:
        return fields.get("sha256", ""), None
    try:
        count = max(1, int(fields.get("frames", "1") or 1))
    except ValueError:
        count = 1
    return fields.get("sha256", ""), MediaFingerprintSet(
        phash=phash,
        dhash=dhash,
        frame_indexes=tuple(range(min(count, 4))),
        source_frame_count=count,
    )


def _safe_sender_label(value: object) -> str:
    name = str(value or "").strip()[:80]
    opaque = (len(name) >= 16 and all(char in "0123456789abcdefABCDEF" for char in name)) or (
        len(name) >= 5 and name.isdigit()
    )
    return "群成员" if not name or opaque else name


def _clean_group(
    key: tuple[str, tuple[str, ...], tuple[str, ...]],
    members: list[GroupFlowSourceMessage],
    sample_limit: int,
) -> CleanedGroupMessage:
    identity_samples: list[tuple[str, str]] = []
    seen_identities: set[str] = set()
    for item in members:
        sender_id = str(item.sender_id or "").strip()
        if not sender_id or sender_id in seen_identities:
            continue
        seen_identities.add(sender_id)
        identity_samples.append((sender_id, _safe_sender_label(item.sender_name)))
        if len(identity_samples) >= max(1, sample_limit):
            break
    labels = tuple(
        dict.fromkeys(
            _safe_sender_label(item.sender_name) for item in members[: max(1, sample_limit)]
        )
    )
    participant_ids = {
        str(item.sender_id or "").strip() or f"unknown-message:{int(item.message_id)}"
        for item in members
    }
    return CleanedGroupMessage(
        representative_text=key[0],
        message_ids=tuple(item.message_id for item in members),
        sender_labels=labels,
        sender_identities=tuple(identity_samples),
        occurrence_count=len(members),
        participant_count=len(participant_ids),
        first_at=members[0].occurred_at,
        last_at=members[-1].occurred_at,
        media_kinds=key[1],
        blank=not key[0] and not key[1],
    )


__all__ = [
    "CleanedGroupMessage",
    "CleanedGroupProjection",
    "clean_group_messages",
    "media_cluster_keys_match",
    "normalize_group_text",
    "normalized_text_fingerprint",
]
