"""Typed, evidence-backed graph projection for Recall candidate expansion."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .ports import RecallProjectionSnapshot
from .tokenization import normalize_text


def build_heterogeneous_graph(
    snapshot: RecallProjectionSnapshot,
    documents: Sequence[Mapping[str, Any]],
    document_edges: Sequence[Mapping[str, Any]],
    scenes: Sequence[Mapping[str, Any]],
    scene_members: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a typed graph without merging people by display name alone."""

    builder = _GraphProjectionBuilder(snapshot)
    world_entities, aliases = _authoritative_entity_nodes(builder, documents)
    semantic_nodes = _document_graph_nodes(
        builder,
        documents,
        world_entities=world_entities,
        aliases=aliases,
    )
    _document_graph_relations(builder, document_edges, semantic_nodes)
    _scene_graph_nodes(builder, scenes, scene_members, semantic_nodes)
    return builder.result()


class _GraphProjectionBuilder:
    def __init__(self, snapshot: RecallProjectionSnapshot) -> None:
        self.snapshot = snapshot
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(
        self,
        node_type: str,
        stable_ref: str,
        label: str,
        *,
        document_key: str = "",
        scene_key: str = "",
        valid_from: str = "",
        valid_until: str = "",
        evidence: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        key = _graph_node_key(self.snapshot, node_type, stable_ref)
        candidate = {
            "node_key": key,
            "node_type": node_type,
            "stable_ref": stable_ref,
            "label": str(label or "")[:240],
            "document_key": document_key,
            "scene_key": scene_key,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "evidence": tuple(dict(item) for item in evidence),
        }
        existing = self.nodes.get(key)
        if existing is None or (not existing.get("document_key") and document_key):
            self.nodes[key] = candidate
        return key

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        weight: float,
        evidence_document_key: str = "",
        valid_from: str = "",
        valid_until: str = "",
        evidence: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, edge_type)
        candidate = {
            "source_node_key": source,
            "target_node_key": target,
            "edge_type": edge_type,
            "weight": max(0.01, min(float(weight), 1.0)),
            "evidence_document_key": evidence_document_key,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "evidence": tuple(dict(item) for item in evidence),
        }
        if key not in self.edges or candidate["weight"] > self.edges[key]["weight"]:
            self.edges[key] = candidate

    def result(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [self.nodes[key] for key in sorted(self.nodes)],
            [self.edges[key] for key in sorted(self.edges)],
        )


def _authoritative_entity_nodes(
    builder: _GraphProjectionBuilder,
    documents: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    world_entities: dict[str, str] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    world_documents = [item for item in documents if item["source_type"] == "WORLD_INFO"]
    for item in world_documents:
        source_key = str(item["source_key"])
        category = str(item.get("extra", {}).get("category") or "")
        entity_key = builder.add_node(
            "PERSON" if _person_category(category) else "ENTITY",
            f"world:{source_key}",
            str(item.get("title") or "实体"),
            document_key=str(item["document_key"]),
            valid_from=str(item.get("valid_from") or ""),
            valid_until=str(item.get("valid_until") or ""),
            evidence=item.get("evidence", ()),
        )
        world_entities[source_key] = entity_key
        for name in item.get("entity_names", ()):
            if normalized := normalize_text(name):
                aliases[normalized].add(entity_key)
    return world_entities, aliases


def _document_graph_nodes(
    builder: _GraphProjectionBuilder,
    documents: Sequence[Mapping[str, Any]],
    *,
    world_entities: Mapping[str, str],
    aliases: Mapping[str, set[str]],
) -> dict[str, str]:
    semantic_nodes: dict[str, str] = {}
    for item in documents:
        document_key = str(item["document_key"])
        semantic_type = "FACT" if item["source_type"] in {"WORLD_INFO", "ROLE_CURRENT"} else "EVENT"
        semantic = builder.add_node(
            semantic_type,
            f"semantic:{document_key}",
            str(item.get("title") or item.get("content") or "资料"),
            document_key=document_key,
            valid_from=str(item.get("valid_from") or item.get("occurred_at") or ""),
            valid_until=str(item.get("valid_until") or ""),
            evidence=item.get("evidence", ()),
        )
        evidence_node = builder.add_node(
            "EVIDENCE",
            f"document:{document_key}",
            str(item.get("title") or "证据文档"),
            document_key=document_key,
            evidence=item.get("evidence", ()),
        )
        semantic_nodes[document_key] = semantic
        builder.add_edge(
            evidence_node,
            semantic,
            "EVIDENCE_FOR",
            weight=1.0,
            evidence_document_key=document_key,
            evidence=item.get("evidence", ()),
        )
        _document_entity_relations(
            builder,
            item,
            document_key=document_key,
            semantic=semantic,
            semantic_type=semantic_type,
            world_entities=world_entities,
            aliases=aliases,
        )
    return semantic_nodes


def _document_entity_relations(
    builder: _GraphProjectionBuilder,
    item: Mapping[str, Any],
    *,
    document_key: str,
    semantic: str,
    semantic_type: str,
    world_entities: Mapping[str, str],
    aliases: Mapping[str, set[str]],
) -> None:
    entity_keys: set[str] = set()
    if item["source_type"] == "WORLD_INFO":
        entity_keys.add(world_entities[str(item["source_key"])])
    for name in item.get("entity_names", ()):
        normalized = normalize_text(name)
        if not normalized:
            continue
        authoritative = aliases.get(normalized, set())
        if len(authoritative) == 1:
            entity_keys.update(authoritative)
            continue
        participant = re.fullmatch(r"成员-[0-9a-f]{8}", str(name).strip())
        if participant:
            entity_keys.add(
                builder.add_node(
                    "PERSON",
                    f"participant:{participant.group(0)}",
                    participant.group(0),
                )
            )
            continue
        entity_keys.add(
            builder.add_node(
                "ENTITY",
                f"mention:{document_key}:{normalized}",
                str(name),
                evidence=({"kind": "DOCUMENT_MENTION", "document_key": document_key},),
            )
        )
    for entity in sorted(entity_keys):
        builder.add_edge(
            semantic,
            entity,
            "MENTIONS_ENTITY",
            weight=0.82,
            evidence_document_key=document_key,
        )
        builder.add_edge(
            entity,
            semantic,
            "PARTICIPATED_IN" if semantic_type == "EVENT" else "MENTIONS_ENTITY",
            weight=0.82,
            evidence_document_key=document_key,
        )


def _document_graph_relations(
    builder: _GraphProjectionBuilder,
    document_edges: Sequence[Mapping[str, Any]],
    semantic_nodes: Mapping[str, str],
) -> None:
    for item in document_edges:
        source_document = str(item["source_document_key"])
        target_document = str(item["target_document_key"])
        builder.add_edge(
            semantic_nodes.get(source_document, ""),
            semantic_nodes.get(target_document, ""),
            str(item["edge_type"]),
            weight=float(item.get("weight") or 1.0),
            evidence_document_key=target_document,
            valid_from=str(item.get("valid_from") or ""),
            valid_until=str(item.get("valid_until") or ""),
            evidence=item.get("evidence", ()),
        )


def _scene_graph_nodes(
    builder: _GraphProjectionBuilder,
    scenes: Sequence[Mapping[str, Any]],
    scene_members: Sequence[Mapping[str, Any]],
    semantic_nodes: Mapping[str, str],
) -> None:
    scene_nodes = {
        str(scene["scene_key"]): builder.add_node(
            "SCENE",
            str(scene["scene_key"]),
            str(scene.get("title") or "场景"),
            scene_key=str(scene["scene_key"]),
            valid_from=str(scene.get("occurred_from") or ""),
            valid_until=str(scene.get("occurred_until") or ""),
            evidence=scene.get("evidence", ()),
        )
        for scene in scenes
    }
    for member in scene_members:
        document_key = str(member["document_key"])
        source = semantic_nodes.get(document_key, "")
        target = scene_nodes.get(str(member["scene_key"]), "")
        details = {
            "weight": float(member.get("membership_weight") or 1.0),
            "evidence_document_key": document_key,
            "evidence": member.get("evidence", ()),
        }
        builder.add_edge(source, target, "BELONGS_TO_SCENE", **details)
        builder.add_edge(target, source, "BELONGS_TO_SCENE", **details)
    for scene in scenes:
        parent = str(scene.get("parent_scene_key") or "")
        if not parent:
            continue
        child_node = scene_nodes.get(str(scene["scene_key"]), "")
        parent_node = scene_nodes.get(parent, "")
        builder.add_edge(child_node, parent_node, "BELONGS_TO_TOPIC", weight=0.8)
        builder.add_edge(parent_node, child_node, "BELONGS_TO_TOPIC", weight=0.8)


def _person_category(category: str) -> bool:
    normalized = normalize_text(category)
    return any(token in normalized for token in ("人物", "角色", "用户", "成员", "person"))


def _graph_node_key(snapshot: RecallProjectionSnapshot, node_type: str, stable_ref: str) -> str:
    digest = hashlib.sha256(
        f"{snapshot.profile_id}\0{snapshot.instance_id}\0{node_type}\0{stable_ref}".encode()
    ).hexdigest()
    return f"node:{digest[:32]}"


__all__ = ["build_heterogeneous_graph"]
