"""Deterministic projection from authoritative SoulCore records into Recall indexes.

The projection never mutates source records.  Every summary, edge and search token
is disposable and can be rebuilt from the snapshot returned by the repository.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .ports import RecallProjectionSnapshot
from .projection_graph import build_heterogeneous_graph
from .tokenization import fts_document, lexical_tokens, normalize_text, token_overlap

_CORRECTION_WORDS = re.compile(r"(纠正|修正|错误|不再|已经不是|取代|冲突|更正)")


def build_recall_projection(snapshot: RecallProjectionSnapshot) -> dict[str, Any]:
    """Build the complete disposable projection for one role instance."""

    documents: list[dict[str, Any]] = []
    documents.extend(_memory_documents(snapshot))
    documents.extend(_world_documents(snapshot))
    documents.extend(_role_event_documents(snapshot))
    documents.extend(_role_current_documents(snapshot))
    documents.extend(_summary_documents(snapshot))
    documents.extend(_message_documents(snapshot))
    documents.sort(key=lambda item: str(item["document_key"]))

    edges = _revision_edges(documents)
    edges.extend(_evidence_edges(documents))
    edges.extend(_current_state_edges(documents, snapshot.role_current))
    edges.extend(_temporal_edges(documents))
    edges = _deduplicate_edges(edges)
    scenes, scene_members = _build_scenes(documents)
    graph_nodes, graph_edges = build_heterogeneous_graph(
        snapshot,
        documents,
        edges,
        scenes,
        scene_members,
    )
    return {
        "documents": documents,
        "fts_rows": {
            str(item["document_key"]): fts_document(
                item["search_text"], aliases=item.get("entity_names", ())
            )
            for item in documents
        },
        "edges": edges,
        "scenes": scenes,
        "scene_members": scene_members,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
    }


def _memory_documents(snapshot: RecallProjectionSnapshot) -> list[dict[str, Any]]:
    terms = _group(snapshot.memory_terms, "memory_id", "revision")
    sources = _group(snapshot.memory_sources, "memory_id", "revision")
    groups = _group(snapshot.memories, "memory_id")
    result: list[dict[str, Any]] = []
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: int(row["revision"]))
        for index, row in enumerate(ordered):
            revision = int(row["revision"])
            projected = _memory_revision_document(
                row,
                next_row=ordered[index + 1] if index + 1 < len(ordered) else None,
                term_rows=terms.get((row["memory_id"], revision), ()),
                source_rows=sources.get((row["memory_id"], revision), ()),
            )
            if projected is not None:
                result.append(projected)
    return result


def _memory_revision_document(
    row: Mapping[str, Any],
    *,
    next_row: Mapping[str, Any] | None,
    term_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    revision = int(row["revision"])
    content = _first_text(row, "brief", "ultra_brief")
    if not content:
        return None
    current = str(row.get("status")) == "ACTIVE" and revision == int(
        _present_value(row, "current_revision", 0)
    )
    return _document(
        document_key=f"memory:{row['memory_id']}:r{revision}",
        source_type="MEMORY",
        source_key=str(row["memory_id"]),
        source_revision=revision,
        authority_status="CURRENT" if current else "HISTORICAL",
        title=_first_text(row, "ultra_brief") or "记忆",
        content=content,
        entity_names=_unique(item.get("term") for item in term_rows),
        occurred_at=_first_text(row, "event_time"),
        recorded_from=_first_text(row, "created_at"),
        recorded_until=_first_text(next_row, "created_at") if next_row else "",
        evidence=_message_evidence(source_rows),
        created_at=_first_text(row, "created_at", "entry_created_at"),
        extra={
            "importance": _float_value(row, "importance"),
            "change_reason": _first_text(row, "change_reason"),
        },
    )


def _world_documents(snapshot: RecallProjectionSnapshot) -> list[dict[str, Any]]:
    terms = _group(snapshot.world_terms, "knowledge_fact_id", "revision")
    sources = _group(snapshot.world_sources, "knowledge_fact_id", "revision")
    groups = _group(snapshot.world_info, "knowledge_fact_id")
    result: list[dict[str, Any]] = []
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: int(row["revision"]))
        for index, row in enumerate(ordered):
            revision = int(row["revision"])
            projected = _world_revision_document(
                row,
                next_row=ordered[index + 1] if index + 1 < len(ordered) else None,
                term_rows=terms.get((row["knowledge_fact_id"], revision), ()),
                source_rows=sources.get((row["knowledge_fact_id"], revision), ()),
            )
            if projected is not None:
                result.append(projected)
    return result


def _world_revision_document(
    row: Mapping[str, Any],
    *,
    next_row: Mapping[str, Any] | None,
    term_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    definition = _first_text(row, "definition")
    brief = _first_text(row, "brief")
    content = _combined_brief(brief, definition)
    if not content:
        return None
    revision = int(row["revision"])
    names = [row.get("name"), *_json_strings(row.get("aliases_json"))]
    names.extend(item.get("term") for item in term_rows)
    current = str(row.get("status")) == "ACTIVE" and revision == int(
        _present_value(row, "current_revision", 0)
    )
    return _document(
        document_key=f"world:{row['knowledge_fact_id']}:r{revision}",
        source_type="WORLD_INFO",
        source_key=str(row["knowledge_fact_id"]),
        source_revision=revision,
        authority_status="CURRENT" if current else "HISTORICAL",
        title=_first_text(row, "name") or "世界资料",
        content=content,
        entity_names=_unique(names),
        valid_from=_first_text(row, "valid_from"),
        valid_until=_first_text(row, "valid_until"),
        recorded_from=_first_text(row, "created_at"),
        recorded_until=_first_text(next_row, "created_at") if next_row else "",
        evidence=_message_evidence(source_rows),
        created_at=_first_text(row, "created_at", "entry_created_at"),
        extra={
            "importance": _float_value(row, "importance"),
            "category": _first_text(row, "category"),
            "change_reason": _first_text(row, "change_reason"),
        },
    )


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _present_value(row: Mapping[str, Any], key: str, default: Any) -> Any:
    value = row.get(key)
    return default if value is None or value == "" else value


def _float_value(row: Mapping[str, Any], key: str) -> float:
    return float(_present_value(row, key, 0.0))


def _combined_brief(brief: str, definition: str) -> str:
    if brief and definition and brief not in definition:
        return f"{brief}。{definition}"
    return definition or brief


def _role_event_documents(snapshot: RecallProjectionSnapshot) -> list[dict[str, Any]]:
    return [
        _document(
            document_key=f"role-event:{row['event_id']}",
            source_type="ROLE_EVENT",
            source_key=str(row["event_id"]),
            source_revision=1,
            authority_status="HISTORICAL",
            title="角色经历",
            content=str(row.get("content") or "").strip(),
            entity_names=(),
            valid_from=str(row.get("frame_start_at") or ""),
            valid_until=str(row.get("frame_end_at") or ""),
            occurred_at=str(row.get("frame_start_at") or ""),
            recorded_from=str(row.get("created_at") or ""),
            evidence=(
                {
                    "kind": "ROLE_TIMELINE",
                    "note": "来自已发布的角色经历",
                    "public_ref": str(row.get("public_ref") or ""),
                },
            ),
            created_at=str(row.get("created_at") or ""),
            extra={"source": str(row.get("source") or "")},
        )
        for row in snapshot.role_events
        if str(row.get("content") or "").strip()
    ]


def _role_current_documents(snapshot: RecallProjectionSnapshot) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in snapshot.role_current:
        fields = (
            ("叙事时间", row.get("narrative_time")),
            ("地点", row.get("location")),
            ("正在做", row.get("doing")),
            ("身体状态", row.get("body_state")),
            ("心情", row.get("mood")),
            ("意图", row.get("intention")),
            ("当前关切", row.get("current_concern")),
        )
        content = "；".join(
            f"{label}：{value}" for label, value in fields if str(value or "").strip()
        )
        if not content:
            continue
        revision = int(row.get("revision") or 0)
        result.append(
            _document(
                document_key=_scope_key(snapshot, "role-current", f"r{revision}"),
                source_type="ROLE_CURRENT",
                source_key=snapshot.instance_id,
                source_revision=revision,
                authority_status="CURRENT",
                title="角色当前状态",
                content=content,
                entity_names=(),
                valid_from=str(row.get("as_of") or ""),
                recorded_from=str(row.get("created_at") or ""),
                occurred_at=str(row.get("as_of") or ""),
                evidence=({"kind": "ROLE_CURRENT", "note": "来自角色当前状态视图"},),
                created_at=str(row.get("created_at") or ""),
                extra={"source_event_id": row.get("source_event_id")},
            )
        )
    return result


def _summary_documents(snapshot: RecallProjectionSnapshot) -> list[dict[str, Any]]:
    return [
        _document(
            document_key=f"summary:{row['summary_id']}:v{row['version']}",
            source_type="DIALOGUE_SUMMARY",
            source_key=str(row["summary_id"]),
            source_revision=int(row["version"]),
            authority_status="HISTORICAL",
            title="会话摘要",
            content=str(row.get("rendered_text") or "").strip(),
            entity_names=(),
            recorded_from=str(row.get("created_at") or ""),
            evidence=(
                {
                    "kind": "SUMMARY_MEMBERS",
                    "message_range": [
                        int(row.get("covered_from_message_id") or 0),
                        int(row.get("covered_through_message_id") or 0),
                    ],
                    "note": "摘要命中后需回到成员消息或正式事实",
                },
            ),
            created_at=str(row.get("created_at") or ""),
            extra={},
        )
        for row in snapshot.summaries
        if str(row.get("rendered_text") or "").strip()
    ]


def _message_documents(snapshot: RecallProjectionSnapshot) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in snapshot.messages:
        content = str(row.get("plain_text") or "").strip()
        if not content:
            continue
        sender = str(row.get("sender_name") or "").strip()
        participant_ref = _participant_reference(snapshot, row.get("sender_id"))
        speaker = (
            f"{sender}（{participant_ref}）"
            if sender and participant_ref
            else sender or ("角色" if str(row.get("role") or "") == "assistant" else "对方")
        )
        statement = f"{speaker}说：{content}"
        names = _unique((sender, participant_ref, speaker))
        result.append(
            _document(
                document_key=f"message:{row['message_id']}",
                source_type="MESSAGE",
                source_key=str(row["message_id"]),
                source_revision=1,
                authority_status="HISTORICAL",
                title=f"{speaker}的原话",
                content=statement,
                entity_names=names,
                occurred_at=str(row.get("occurred_at") or ""),
                recorded_from=str(row.get("created_at") or ""),
                evidence=(
                    {
                        "kind": "MESSAGE",
                        "message_ids": [int(row["message_id"])],
                        "note": "来自可见聊天原话",
                    },
                ),
                created_at=str(row.get("created_at") or ""),
                dense_eligible=False,
                extra={"direction": str(row.get("direction") or "")},
            )
        )
    return result


def _participant_reference(snapshot: RecallProjectionSnapshot, sender_id: object) -> str:
    value = str(sender_id or "").strip()
    if not value:
        return ""
    digest = hashlib.sha256(
        f"{snapshot.profile_id}\0{snapshot.instance_id}\0{value}".encode()
    ).hexdigest()[:8]
    return f"成员-{digest}"


def _document(
    *,
    document_key: str,
    source_type: str,
    source_key: str,
    source_revision: int,
    authority_status: str,
    title: str,
    content: str,
    entity_names: Iterable[object],
    evidence: Sequence[Mapping[str, Any]],
    created_at: str,
    extra: Mapping[str, Any],
    valid_from: str = "",
    valid_until: str = "",
    recorded_from: str = "",
    recorded_until: str = "",
    occurred_at: str = "",
    dense_eligible: bool = True,
) -> dict[str, Any]:
    names = _unique(entity_names)
    payload = {
        "source_type": source_type,
        "source_key": source_key,
        "source_revision": int(source_revision),
        "authority_status": authority_status,
        "title": title,
        "content": content,
        "entity_names": names,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "recorded_from": recorded_from,
        "recorded_until": recorded_until,
        "occurred_at": occurred_at,
        "evidence": tuple(dict(item) for item in evidence),
        "dense_eligible": bool(dense_eligible),
        "extra": dict(extra),
    }
    fingerprint = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    search_parts = [title, content, *names]
    for value in extra.values():
        if isinstance(value, str):
            search_parts.append(value)
    return {
        "document_key": document_key,
        **payload,
        "search_text": "\n".join(part for part in search_parts if part),
        "source_fingerprint": fingerprint,
        "created_at": created_at,
    }


def _revision_edges(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = _group(
        (item for item in documents if item["source_type"] in {"MEMORY", "WORLD_INFO"}),
        "source_type",
        "source_key",
    )
    result: list[dict[str, Any]] = []
    for rows in groups.values():
        ordered = sorted(rows, key=lambda item: int(item["source_revision"]))
        for before, after in zip(ordered, ordered[1:], strict=False):
            evidence = ({"kind": "REVISION", "note": "同一权威记录的相邻修订"},)
            result.extend(
                (
                    _edge(before, after, "SUPERSEDED_BY", evidence=evidence),
                    _edge(after, before, "REVISED_BY", evidence=evidence),
                )
            )
            reason = str(after.get("extra", {}).get("change_reason") or "")
            if _CORRECTION_WORDS.search(reason):
                result.extend(
                    (
                        _edge(before, after, "CONFLICTS_WITH", weight=0.9, evidence=evidence),
                        _edge(after, before, "CONFLICTS_WITH", weight=0.9, evidence=evidence),
                    )
                )
    return result


def _evidence_edges(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {str(item["document_key"]): item for item in documents}
    result: list[dict[str, Any]] = []
    for item in documents:
        if item["source_type"] == "MESSAGE":
            continue
        for message_id in _evidence_message_ids(item.get("evidence", ())):
            source = by_key.get(f"message:{message_id}")
            if source is not None:
                result.append(
                    _edge(
                        source,
                        item,
                        "EVIDENCE_FOR",
                        weight=1.0,
                        evidence=({"kind": "MESSAGE", "message_ids": [message_id]},),
                    )
                )
    return result


def _current_state_edges(
    documents: Sequence[Mapping[str, Any]], current_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    events = {
        str(item["source_key"]): item for item in documents if item["source_type"] == "ROLE_EVENT"
    }
    current_docs = [item for item in documents if item["source_type"] == "ROLE_CURRENT"]
    result: list[dict[str, Any]] = []
    for item, row in zip(current_docs, current_rows, strict=False):
        source = events.get(str(row.get("source_event_id") or ""))
        if source is not None:
            result.append(
                _edge(
                    item,
                    source,
                    "DERIVED_CURRENT_STATE",
                    evidence=({"kind": "ROLE_TIMELINE", "note": "当前状态的来源经历"},),
                )
            )
    return result


def _temporal_edges(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    timed = [(item, _document_time(item)) for item in documents]
    ordered = sorted(
        ((item, stamp) for item, stamp in timed if stamp is not None),
        key=lambda pair: (pair[1], str(pair[0]["document_key"])),
    )
    result: list[dict[str, Any]] = []
    for (before, before_time), (after, after_time) in zip(ordered, ordered[1:], strict=False):
        if before["document_key"] == after["document_key"] or before_time == after_time:
            continue
        evidence = ({"kind": "TIME", "note": "由权威记录中的明确时间确定"},)
        result.extend(
            (
                _edge(before, after, "BEFORE", weight=0.75, evidence=evidence),
                _edge(after, before, "AFTER", weight=0.75, evidence=evidence),
            )
        )
    return result


def _build_scenes(
    documents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        item
        for item in documents
        if item["source_type"] != "MESSAGE" and item["authority_status"] == "HISTORICAL"
    ]
    event_groups = _cluster_event_groups(eligible)
    topic_groups = _topic_groups(eligible)
    scenes, members, topic_keys = _project_topic_scenes(topic_groups)
    event_scenes, event_members = _project_event_scenes(event_groups, topic_keys)
    scenes.extend(event_scenes)
    members.extend(event_members)
    return scenes, members


def _cluster_event_groups(
    documents: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    timed = sorted(
        ((item, _document_time(item)) for item in documents),
        key=lambda pair: pair[1] or datetime.max.replace(tzinfo=UTC),
    )
    for item, stamp in timed:
        if stamp is None:
            continue
        group = _matching_event_group(item, stamp, groups[-8:])
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    return groups


def _matching_event_group(
    item: Mapping[str, Any],
    stamp: datetime,
    groups: Sequence[list[Mapping[str, Any]]],
) -> list[Mapping[str, Any]] | None:
    for group in reversed(groups):
        if _belongs_to_event_group(item, stamp, group):
            return group
    return None


def _belongs_to_event_group(
    item: Mapping[str, Any], stamp: datetime, group: Sequence[Mapping[str, Any]]
) -> bool:
    latest = _document_time(group[-1])
    if latest is None or stamp - latest > timedelta(hours=36):
        return False
    names = {name for member in group for name in member.get("entity_names", ())}
    shares_entity = bool(set(item.get("entity_names", ())) & names)
    query_tokens = lexical_tokens(item.get("content", ""))
    semantic = max(
        (
            token_overlap(query_tokens, lexical_tokens(member.get("content", "")))
            for member in group[-8:]
        ),
        default=0.0,
    )
    return shares_entity or semantic >= 0.2 or stamp - latest <= timedelta(hours=6)


def _topic_groups(
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in documents:
        names = tuple(item.get("entity_names", ()))
        topic = normalize_text(names[0]) if names else str(item.get("source_type") or "事件")
        groups[topic or "事件"].append(item)
    return groups


def _project_topic_scenes(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    scenes: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    keys: dict[str, str] = {}
    for topic, rows in sorted(groups.items()):
        key = _scene_key("topic", tuple(str(item["document_key"]) for item in rows))
        keys[topic] = key
        scenes.append(_scene(key, "TOPIC", topic, rows))
        members.extend(_scene_members(key, rows, weight=0.7))
    return scenes, members, keys


def _project_event_scenes(
    groups: Sequence[Sequence[Mapping[str, Any]]], topic_keys: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenes: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for rows in groups:
        names = [
            normalize_text(tuple(item.get("entity_names", ()))[0])
            for item in rows
            if item.get("entity_names")
        ]
        key = _scene_key("event", tuple(str(item["document_key"]) for item in rows))
        scene = _scene(key, "EVENT", _event_title(rows), rows)
        scene["parent_scene_key"] = topic_keys.get(names[0]) if names else None
        scenes.append(scene)
        members.extend(_scene_members(key, rows, weight=1.0))
    return scenes, members


def _scene(key: str, level: str, title: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    times = sorted(stamp for row in rows if (stamp := _document_time(row)) is not None)
    excerpts = [str(row.get("content") or "").strip()[:160] for row in rows[:8]]
    return {
        "scene_key": key,
        "parent_scene_key": None,
        "scene_level": level,
        "title": str(title or "相关经历")[:120],
        "summary": "；".join(value for value in excerpts if value)[:1200],
        "search_text": "\n".join(
            [str(title or ""), *(str(row.get("search_text") or "") for row in rows)]
        )[:8000],
        "occurred_from": times[0].isoformat() if times else "",
        "occurred_until": times[-1].isoformat() if times else "",
        "evidence": tuple(
            {"kind": "SCENE_MEMBER", "document_key": str(row["document_key"])} for row in rows
        ),
    }


def _scene_members(
    key: str, rows: Sequence[Mapping[str, Any]], *, weight: float
) -> list[dict[str, Any]]:
    return [
        {
            "scene_key": key,
            "document_key": str(row["document_key"]),
            "membership_weight": weight,
            "evidence": ({"kind": "SCENE_MEMBER", "note": "场景由成员证据重建"},),
        }
        for row in rows
    ]


def _event_title(rows: Sequence[Mapping[str, Any]]) -> str:
    names = _unique(name for row in rows for name in row.get("entity_names", ()))
    stamp = _document_time(rows[0]) if rows else None
    prefix = stamp.date().isoformat() if stamp is not None else "一段经历"
    return f"{prefix} · {'、'.join(names[:3])}" if names else prefix


def _edge(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    edge_type: str,
    *,
    weight: float = 1.0,
    evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "source_document_key": str(source["document_key"]),
        "target_document_key": str(target["document_key"]),
        "edge_type": edge_type,
        "weight": max(0.01, min(float(weight), 1.0)),
        "valid_from": str(target.get("valid_from") or target.get("occurred_at") or ""),
        "valid_until": str(target.get("valid_until") or ""),
        "evidence": tuple(dict(item) for item in evidence),
    }


def _deduplicate_edges(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in edges:
        source, target = str(item["source_document_key"]), str(item["target_document_key"])
        if not source or not target or source == target:
            continue
        key = (source, target, str(item["edge_type"]))
        candidate = dict(item)
        if key not in result or float(candidate.get("weight") or 0) > float(
            result[key].get("weight") or 0
        ):
            result[key] = candidate
    return [result[key] for key in sorted(result)]


def _message_evidence(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "kind": "MESSAGE",
            "message_ids": [int(row["message_id"])],
            "quote": str(row.get("quote") or "")[:500],
            "occurred_at": str(row.get("occurred_at") or ""),
            "note": "来自正式记忆或世界资料保存的原话证据",
        }
        for row in rows
    )


def _evidence_message_ids(evidence: Iterable[Mapping[str, Any]]) -> tuple[int, ...]:
    result: list[int] = []
    for item in evidence:
        result.extend(int(value) for value in item.get("message_ids", ()) if int(value) > 0)
    return tuple(dict.fromkeys(result))


def _document_time(item: Mapping[str, Any]) -> datetime | None:
    for key in ("occurred_at", "valid_from", "recorded_from"):
        parsed = _parse_datetime(item.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)


def _group(rows: Iterable[Mapping[str, Any]], *keys: str) -> dict[Any, list[Mapping[str, Any]]]:
    result: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_key: Any = tuple(row[key] for key in keys)
        if len(group_key) == 1:
            group_key = group_key[0]
        result[group_key].append(row)
    return result


def _unique(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip())
    )


def _json_strings(value: Any) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    return (
        tuple(str(item) for item in parsed if str(item).strip()) if isinstance(parsed, list) else ()
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _scope_key(snapshot: RecallProjectionSnapshot, kind: str, suffix: str) -> str:
    scope = hashlib.sha256(f"{snapshot.profile_id}\0{snapshot.instance_id}".encode()).hexdigest()[
        :20
    ]
    return f"{kind}:{scope}:{suffix}"


def _scene_key(kind: str, members: Sequence[str]) -> str:
    digest = hashlib.sha256("\0".join(sorted(members)).encode("utf-8")).hexdigest()[:24]
    return f"scene:{kind}:{digest}"


__all__ = ["build_recall_projection"]
