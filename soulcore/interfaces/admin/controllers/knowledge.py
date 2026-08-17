"""Searchable history and WorldInfo administrator controller."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ....features.knowledge.ports import KnowledgeRepositoryPort
from ....features.knowledge.service import KnowledgeFormationPlugin
from ....features.recall import RecallMode, RecallRequest, RecallService
from ..presentation import jsonable


class KnowledgeAdminController:
    def __init__(
        self,
        repository: KnowledgeRepositoryPort,
        knowledge_plugin: KnowledgeFormationPlugin,
        recall: RecallService,
        identity: Any,
    ) -> None:
        self.repository = repository
        self.knowledge_plugin = knowledge_plugin
        self.recall = recall
        self.identity = identity

    async def knowledge_snapshot(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        """Return one instance-scoped administrator knowledge snapshot."""

        memories = await self.repository.list_memories(profile_id, instance_id, limit=1000)
        world_info = await self.repository.list_knowledge_facts(profile_id, instance_id, limit=1000)
        result = {
            "ok": True,
            "available": True,
            "profile_id": profile_id,
            "instance_id": instance_id,
            "status": jsonable(await self.repository.get_knowledge_status(profile_id, instance_id)),
            "memories": jsonable(memories),
            "world_info": jsonable([self._world_info_view(item) for item in world_info]),
            "batches": jsonable(
                await self.repository.list_knowledge_batches(profile_id, instance_id, limit=20)
            ),
            "audit": jsonable(
                await self.repository.list_knowledge_audit(profile_id, instance_id, limit=100)
            ),
            "recall_reports": jsonable(
                await self.recall.repository.list_recall_reports(profile_id, instance_id, limit=20)
            ),
        }
        context = await self.identity.context(profile_id, instance_id)
        return self.identity.render_data(result, context)

    async def knowledge_form(self, profile_id: str, instance_id: str, mode: str) -> dict[str, Any]:
        if mode not in {"dry", "commit"}:
            raise ValueError("mode must be dry or commit")
        if mode == "dry":
            result = await self.knowledge_plugin.dry_run(profile_id, instance_id)
            return {"ok": True, "mode": "dry", **(jsonable(result) or {})}
        task = await self.knowledge_plugin.enqueue(profile_id, instance_id, force=True)
        result = {
            "ok": True,
            "mode": "commit",
            "queued": task is not None,
            "task": jsonable(task),
        }
        return result

    async def knowledge_support_snapshot(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        """Return knowledge diagnostics without private content or evidence."""

        snapshot = await self.knowledge_snapshot(profile_id, instance_id)
        if snapshot.get("unavailable"):
            return snapshot
        memories = list(snapshot.get("memories") or [])
        world_info = list(snapshot.get("world_info") or [])
        result = {
            "ok": True,
            "available": True,
            "redacted": True,
            "status": self._safe_status(snapshot.get("status")),
            "memory_count": len(memories),
            "world_info_count": len(world_info),
            "memories": [self._memory_meta(row) for row in memories],
            "world_info": [self._knowledge_fact_meta(row) for row in world_info],
            "batches": self._safe_batches(snapshot.get("batches")),
            "audit": self._safe_audit(snapshot.get("audit")),
            "recall_reports": self._safe_recall(snapshot.get("recall_reports")),
        }
        return result

    @staticmethod
    def _safe_status(raw_status: Any) -> dict[str, Any]:
        status = dict(raw_status or {})
        safe_status = {
            key: status.get(key)
            for key in (
                "baseline_message_id",
                "committed_through_message_id",
                "desired_through_message_id",
                "unprocessed_message_count",
                "unprocessed_max_message_id",
                "processing_version",
                "active_task_id",
                "updated_at",
            )
        }
        active_task = status.get("active_task")
        if active_task:
            safe_status["active_task"] = {
                key: active_task.get(key)
                for key in (
                    "task_id",
                    "task_type",
                    "status",
                    "attempt_count",
                    "generation",
                    "due_at",
                    "error_code",
                    "created_at",
                    "updated_at",
                )
            }
        return safe_status

    @staticmethod
    def _memory_meta(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: record.get(key)
            for key in (
                "memory_id",
                "status",
                "revision",
                "importance",
                "event_time",
                "origin",
                "created_at",
                "updated_at",
            )
        }

    @staticmethod
    def _knowledge_fact_meta(record: Mapping[str, Any]) -> dict[str, Any]:
        result = {
            key: record.get(key)
            for key in (
                "status",
                "revision",
                "importance",
                "category",
                "origin",
                "created_at",
                "updated_at",
            )
        }
        result["world_info_id"] = record.get("world_info_id")
        return result

    @staticmethod
    def _safe_batches(raw_batches: Any) -> list[dict[str, Any]]:
        safe_batches = []
        for row in raw_batches or []:
            batch = {
                key: row.get(key)
                for key in (
                    "batch_id",
                    "ai_task_id",
                    "processing_version",
                    "status",
                    "first_message_id",
                    "last_message_id",
                    "message_count",
                    "estimated_tokens",
                    "created_at",
                    "completed_at",
                )
            }
            raw_rejections = row.get("rejection") or []
            if isinstance(raw_rejections, Mapping):
                raw_rejections = raw_rejections.get("rejections") or []
            batch["rejections"] = [
                {key: item.get(key) for key in ("kind", "index", "reason")}
                for item in raw_rejections
                if isinstance(item, Mapping)
            ]
            safe_batches.append(batch)
        return safe_batches

    @staticmethod
    def _safe_audit(raw_audit: Any) -> list[dict[str, Any]]:
        return [
            {
                key: row.get(key)
                for key in (
                    "audit_id",
                    "entity_type",
                    "entity_id",
                    "action",
                    "actor_type",
                    "actor_id",
                    "created_at",
                )
            }
            for row in raw_audit or []
        ]

    @staticmethod
    def _safe_recall(raw_reports: Any) -> list[dict[str, Any]]:
        safe_recall = []
        for row in raw_reports or []:
            report = dict(row.get("report") or {})
            selected = dict(report.get("selected") or {})
            safe_recall.append(
                {
                    "report_id": row.get("report_id"),
                    "current_message_id": row.get("current_message_id"),
                    "version": report.get("version"),
                    "selected_history_ids": selected.get("history_ids") or [],
                    "selected_world_info_ids": selected.get("world_info_ids") or [],
                    "created_at": row.get("created_at"),
                }
            )
        return safe_recall

    async def recall_probe(self, profile_id: str, instance_id: str, query: str) -> dict[str, Any]:
        if self.recall is None:
            return {
                "ok": False,
                "available": False,
                "unavailable": True,
                "error": "recall_unavailable",
            }
        query = str(query or "").strip()
        if not query:
            raise ValueError("query is required")
        bundle = await self.recall.recall(
            RecallRequest(
                profile_id=profile_id,
                instance_id=instance_id,
                need=query,
                mode=RecallMode.ADMIN_PROBE,
                current_time=datetime.now(UTC),
                token_budget=1600,
            )
        )
        result = {
            "ok": True,
            "query": query,
            "conclusion": self.recall.render(bundle, token_budget=1600),
            "recall": bundle.public_view(include_diagnostics=True),
        }
        context = await self.identity.context(profile_id, instance_id)
        return self.identity.render_data(result, context)

    async def recall_configuration(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        settings = await self.recall.repository.get_role_settings(profile_id)
        selection = await self.recall.providers.selection(self.recall.repository, profile_id)
        readiness = await self.recall.readiness(profile_id, instance_id)
        return {
            "ok": True,
            "settings": settings,
            "effective": {
                "embedding_provider_id": selection.embedding_provider_id,
                "embedding_source": selection.embedding_source,
                "rerank_provider_id": selection.rerank_provider_id,
                "rerank_source": selection.rerank_source,
            },
            "provider_options": self.recall.providers.provider_options(),
            "readiness": readiness.public_view(),
        }

    async def recall_configuration_update(
        self, profile_id: str, instance_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        settings = await self.recall.repository.save_role_settings(
            profile_id,
            embedding_provider_id=self._provider_override(payload, "embedding_provider_id"),
            rerank_provider_id=self._provider_override(payload, "rerank_provider_id"),
            expected_version=int(payload.get("expected_version") or 0),
        )
        await self.recall.repository.enqueue_rebuild(profile_id, instance_id)
        return {"ok": True, "settings": settings}

    async def recall_rebuild(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        await self.recall.repository.enqueue_rebuild(profile_id, instance_id)
        return {"ok": True, "queued": True}

    async def recall_integrity(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        await self.recall.ensure_projection(profile_id, instance_id, verify_integrity=True)
        documents = await self.recall.repository.list_documents(profile_id, instance_id)
        edges = await self.recall.repository.list_edges(profile_id, instance_id)
        scenes = await self.recall.repository.list_scenes(profile_id, instance_id)
        graph_nodes, graph_edges = await self.recall.repository.list_graph(profile_id, instance_id)
        keys = {str(item["document_key"]) for item in documents}
        broken_edges = [
            item
            for item in edges
            if str(item["source_document_key"]) not in keys
            or str(item["target_document_key"]) not in keys
        ]
        graph_node_keys = {str(item["node_key"]) for item in graph_nodes}
        broken_graph_edges = [
            item
            for item in graph_edges
            if str(item["source_node_key"]) not in graph_node_keys
            or str(item["target_node_key"]) not in graph_node_keys
            or (
                item.get("evidence_document_key") and str(item["evidence_document_key"]) not in keys
            )
        ]
        readiness = await self.recall.readiness(profile_id, instance_id)
        return {
            "ok": not broken_edges and not broken_graph_edges,
            "summary": (
                "统一记忆索引完整"
                if not broken_edges and not broken_graph_edges
                else "发现无法落回证据文档的关系边"
            ),
            "document_count": len(documents),
            "edge_count": len(edges),
            "scene_count": len(scenes),
            "graph_node_count": len(graph_nodes),
            "graph_edge_count": len(graph_edges),
            "broken_edge_count": len(broken_edges) + len(broken_graph_edges),
            "readiness": readiness.public_view(),
        }

    async def recall_benchmark(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        return await self.recall.benchmark(profile_id, instance_id)

    @staticmethod
    def _provider_override(payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None or str(value).strip().lower() == "inherit":
            return None
        return str(value).strip()

    async def knowledge_record(
        self,
        profile_id: str,
        instance_id: str,
        kind: str,
        record_id: int,
    ) -> dict[str, Any]:
        if kind not in {"memory", "world_info"} or int(record_id) < 1:
            raise ValueError("kind and positive record_id are required")
        record = (
            await self.repository.get_memory(int(record_id))
            if kind == "memory"
            else await self.repository.get_knowledge_fact(int(record_id))
        )
        if (
            record is None
            or str(record.get("profile_id") or "") != profile_id
            or str(record.get("instance_id") or "") != instance_id
        ):
            raise ValueError("unknown knowledge record in this instance")
        revisions = (
            await self.repository.list_memory_revisions(record_id)
            if kind == "memory"
            else await self.repository.list_knowledge_fact_revisions(record_id)
        )
        result = {
            "ok": True,
            "kind": kind,
            "record": jsonable(self._world_info_view(record) if kind == "world_info" else record),
            "revisions": jsonable(
                [self._world_info_view(item) for item in revisions]
                if kind == "world_info"
                else revisions
            ),
        }
        context = await self.identity.context(profile_id, instance_id)
        return self.identity.render_data(result, context)

    async def knowledge_record_action(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply one confirmed, revision-checked administrator mutation."""
        kind, action, reason, record_id, expected_revision = self._action_fields(payload)
        await self._check_revision(
            profile_id, instance_id, kind, action, record_id, expected_revision
        )
        terminal = await self._terminal_action(kind, action, reason, record_id, expected_revision)
        if terminal is not None:
            return terminal
        result = await self._write_record(
            profile_id,
            instance_id,
            kind,
            reason,
            record_id,
            expected_revision,
            payload,
        )
        return {
            "ok": True,
            "kind": kind,
            "action": action,
            "record": jsonable(self._world_info_view(result) if kind == "world_info" else result),
        }

    @staticmethod
    def _action_fields(payload: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
        kind = str(payload.get("kind") or "").strip().lower()
        action = str(payload.get("action") or "").strip().lower()
        reason = str(payload.get("reason") or "").strip()
        if kind not in {"memory", "world_info"}:
            raise ValueError("kind must be memory or world_info")
        if action not in {"create", "update", "archive", "restore", "retract", "delete"}:
            raise ValueError("unsupported knowledge action")
        if not reason:
            raise ValueError("reason is required")
        if action in {"delete", "retract"} and payload.get("confirm") is not True:
            raise ValueError("explicit confirmation is required")
        if "expected_revision" not in payload:
            raise ValueError("expected_revision is required")
        expected_revision = int(payload.get("expected_revision") or 0)
        record_id = int(payload.get("record_id") or 0)
        return kind, action, reason, record_id, expected_revision

    async def _check_revision(
        self,
        profile_id: str,
        instance_id: str,
        kind: str,
        action: str,
        record_id: int,
        expected_revision: int,
    ) -> None:
        if action == "create":
            if record_id != 0 or expected_revision != 0:
                raise ValueError("create requires record_id=0 and expected_revision=0")
            return
        current = await self.knowledge_record(profile_id, instance_id, kind, record_id)
        actual_revision = int(current["record"].get("revision") or 0)
        if actual_revision != expected_revision:
            raise ValueError("knowledge record revision conflict; refresh first")

    async def _terminal_action(
        self,
        kind: str,
        action: str,
        reason: str,
        record_id: int,
        expected_revision: int,
    ) -> dict[str, Any] | None:
        actor = "astrbot-admin-page"
        if action in {"archive", "restore", "retract"}:
            status = {
                "archive": "DISABLED",
                "restore": "ACTIVE",
                "retract": "RETRACTED",
            }[action]
            changed = (
                await self.repository.set_memory_status(
                    record_id,
                    status,
                    reason=reason,
                    actor_id=actor,
                    expected_revision=expected_revision,
                )
                if kind == "memory"
                else await self.repository.set_knowledge_fact_status(
                    record_id,
                    status,
                    reason=reason,
                    actor_id=actor,
                    expected_revision=expected_revision,
                )
            )
            return {"ok": bool(changed), "kind": kind, "action": action}
        if action == "delete":
            deleted = (
                await self.repository.delete_memory(
                    record_id,
                    reason=reason,
                    actor_id=actor,
                    expected_revision=expected_revision,
                )
                if kind == "memory"
                else await self.repository.delete_knowledge_fact(
                    record_id,
                    reason=reason,
                    actor_id=actor,
                    expected_revision=expected_revision,
                )
            )
            return {"ok": bool(deleted), "kind": kind, "action": action}
        return None

    async def _write_record(
        self,
        profile_id: str,
        instance_id: str,
        kind: str,
        reason: str,
        record_id: int,
        expected_revision: int,
        payload: Mapping[str, Any],
    ) -> Any:
        actor = "astrbot-admin-page"
        record = payload.get("record")
        if not isinstance(record, Mapping):
            raise ValueError("record must be an object")
        values = dict(record)
        if kind == "memory":
            return await self._write_memory(
                writer_name="create_or_revise_memory",
                profile_id=profile_id,
                instance_id=instance_id,
                values=values,
                record_id=record_id,
                expected_revision=expected_revision,
                reason=reason,
                actor=actor,
            )
        return await self._write_knowledge_fact(
            profile_id, instance_id, values, record_id, expected_revision, reason, actor
        )

    async def _write_memory(self, **params: Any) -> Any:
        params.pop("writer_name")
        values = params.pop("values")
        record_id = int(params.pop("record_id"))
        expected_revision = int(params.pop("expected_revision"))
        return await self.repository.create_or_revise_memory(
            params.pop("profile_id"),
            params.pop("instance_id"),
            brief=self._text(values, "brief"),
            ultra_brief=self._optional_text(values, "ultra_brief"),
            keywords=self._strings(values, "keywords"),
            importance=float(values.get("importance", 0.5)),
            event_time=self._optional_text(values, "event_time"),
            memory_id=record_id or None,
            expected_revision=expected_revision if record_id else None,
            reason=params["reason"],
            actor_id=params["actor"],
        )

    async def _write_knowledge_fact(
        self,
        profile_id: str,
        instance_id: str,
        values: dict[str, Any],
        record_id: int,
        expected_revision: int,
        reason: str,
        actor: str,
    ) -> Any:
        return await self.repository.create_or_revise_knowledge_fact(
            profile_id,
            instance_id,
            name=self._text(values, "name"),
            aliases=self._strings(values, "aliases"),
            trigger_keywords=self._strings(values, "trigger_keywords"),
            definition=self._text(values, "definition"),
            brief=self._text(values, "brief"),
            importance=float(values.get("importance", 0.5)),
            category=self._text(values, "category", "其他会话特有概念"),
            session_specific_reason=self._text(values, "session_specific_reason", reason),
            valid_from=self._optional_text(values, "valid_from"),
            valid_until=self._optional_text(values, "valid_until"),
            knowledge_fact_id=record_id or None,
            expected_revision=expected_revision if record_id else None,
            reason=reason,
            actor_id=actor,
        )

    @staticmethod
    def _text(values: Mapping[str, Any], key: str, default: str = "") -> str:
        value = values.get(key)
        return str(default if value in (None, "") else value).strip()

    @classmethod
    def _optional_text(cls, values: Mapping[str, Any], key: str) -> str | None:
        return cls._text(values, key) or None

    @staticmethod
    def _strings(values: Mapping[str, Any], key: str) -> list[str]:
        return [str(item) for item in values.get(key, ())]

    @staticmethod
    def _world_info_view(record: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(record)
        if "knowledge_fact_id" in result:
            result["world_info_id"] = result.pop("knowledge_fact_id")
        if "knowledge_fact_revision_id" in result:
            result["world_info_revision_id"] = result.pop("knowledge_fact_revision_id")
        return result
