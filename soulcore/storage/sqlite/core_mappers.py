from __future__ import annotations

import sqlite3
from typing import Any

from ...contracts.ai_task_payload import decode_task_payload
from ...contracts.models import (
    CharacterInstance,
    ContextBuildReport,
    ConversationMessage,
    CoreState,
    DialogueSummary,
    ExpressionBatch,
    ExpressionBatchStatus,
    InstanceInitializationState,
    MessageDirection,
    OutboxInterruptPolicy,
    OutboxItem,
    OutboxStatus,
    RoleProfile,
    RouteReadiness,
    ScopeConfig,
    WakeSource,
    Wakeup,
    WakeupStatus,
)
from ...contracts.web import WebCallerKind, WebReadStatus, WebSearchPurpose
from ...features.web.domain import (
    WebImageSearchResultRecord,
    WebPageSnapshotRecord,
    WebSearchKind,
    WebSearchProviderRecord,
    WebSearchResultRecord,
    WebSearchSessionRecord,
    WebSearchSessionStatus,
)
from .codec import _load, _parse, _record


class CoreRecordMappers:
    @staticmethod
    def _parse_umo(umo: str) -> tuple[str, str, str]:
        for message_type in ("FriendMessage", "PrivateMessage", "GroupMessage", "GuildMessage"):
            anchor = f":{message_type}:"
            index = umo.find(anchor)
            if index > 0:
                platform_id = umo[:index]
                target_id = umo[index + len(anchor) :]
                if target_id:
                    return platform_id, message_type, target_id
        parts = umo.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError("UMO must be '<platform instance id>:<message type>:<target id>'")
        return parts[0], parts[1], parts[2]

    @staticmethod
    def _validate_scope(scope: str) -> None:
        if scope not in {"private", "group"}:
            raise ValueError("scope must be 'private' or 'group'")

    @staticmethod
    def _conversation_message(row: sqlite3.Row) -> ConversationMessage:
        return ConversationMessage(
            message_id=int(row["message_id"]),
            profile_id=row["profile_id"],
            instance_id=row["instance_id"],
            direction=MessageDirection(row["direction"]),
            role=row["role"],
            internal_memo=str(row["internal_memo"] or ""),
            expression_batch_id=(
                str(row["expression_batch_id"]) if row["expression_batch_id"] is not None else None
            ),
            expression_ordinal=(
                int(row["expression_ordinal"]) if row["expression_ordinal"] is not None else None
            ),
            sender_id=row["sender_id"],
            sender_name=row["sender_name"],
            plain_text=row["plain_text"],
            identity_template=row["identity_template"],
            components=_load(row["components_json"]) or [],
            delivery_status=row["delivery_status"],
            idempotency_key=row["idempotency_key"],
            metadata=_load(row["metadata_json"]) or {},
            occurred_at=_parse(row["occurred_at"]),
            created_at=_parse(row["created_at"]),
            knowledge_eligibility=row["knowledge_eligibility"],
            knowledge_eligibility_reason=row["knowledge_eligibility_reason"],
        )

    @staticmethod
    def _ai_task(row: sqlite3.Row) -> dict[str, Any]:
        result = _record(row, json_columns=())
        for column, kind in (
            ("input_json", "input"),
            ("checkpoint_json", "checkpoint"),
            ("result_json", "result"),
            ("progress_json", "progress"),
            ("retry_policy_json", "retry_policy"),
        ):
            raw = result.pop(column)
            result[column.removesuffix("_json")] = (
                decode_task_payload(kind, raw) if raw is not None else None
            )
        return result

    @staticmethod
    def _ai_backend(row: sqlite3.Row) -> dict[str, Any]:
        return _record(row, json_columns=("metadata_json",))

    @staticmethod
    def _ai_api_package(row: sqlite3.Row) -> dict[str, Any]:
        result = _record(row, json_columns=("config_json",))
        result["enabled"] = bool(result["enabled"])
        return result

    @staticmethod
    def _ai_api_model(row: sqlite3.Row) -> dict[str, Any]:
        result = _record(row, json_columns=("capabilities_json", "config_json"))
        result["enabled"] = bool(result["enabled"])
        result["priority"] = int(result["priority"])
        return result

    @staticmethod
    def _web_provider(row: sqlite3.Row) -> WebSearchProviderRecord:
        return WebSearchProviderRecord(
            provider_id=str(row["provider_id"]),
            profile_id=str(row["profile_id"]),
            provider_kind=str(row["provider_kind"]),
            display_name=str(row["display_name"]),
            backend_id=str(row["backend_id"]),
            credential_id=str(row["credential_id"]),
            priority=int(row["priority"]),
            enabled=bool(row["enabled"]),
            read_enabled=bool(row["read_enabled"]),
            config=_load(row["config_json"]) or {},
            version=int(row["version"]),
            archived_at=_parse(row["archived_at"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _web_session(row: sqlite3.Row) -> WebSearchSessionRecord:
        return WebSearchSessionRecord(
            session_id=str(row["session_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            caller_kind=WebCallerKind(
                row["effective_caller_kind"] if row["effective_caller_kind"] else row["caller_kind"]
            ),
            caller_id=str(row["caller_id"]),
            core_run_id=row["core_run_id"],
            ai_task_id=row["ai_task_id"],
            purpose=WebSearchPurpose(row["purpose"]),
            query=str(row["query"]),
            search_kind=WebSearchKind(row["search_kind"]),
            depth=str(row["depth"]),
            freshness=str(row["freshness"]),
            status=WebSearchSessionStatus(row["status"]),
            deadline_at=_parse(row["deadline_at"]),
            partial_warning=str(row["partial_warning"]),
            provider_count=int(row["provider_count"]),
            result_count=int(row["result_count"]),
            diagnostics=_load(row["diagnostics_json"]) or {},
            error=str(row["error"]),
            started_at=_parse(row["started_at"]),
            finished_at=_parse(row["finished_at"]),
            expires_at=_parse(row["expires_at"]),
            redacted_at=_parse(row["redacted_at"]),
        )

    @staticmethod
    def _web_result(row: sqlite3.Row) -> WebSearchResultRecord:
        return WebSearchResultRecord(
            resource_id=str(row["resource_id"]),
            session_id=str(row["session_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            title=str(row["title"]),
            canonical_url=str(row["canonical_url"]),
            domain=str(row["domain"]),
            snippet=str(row["snippet"]),
            published_at=_parse(row["published_at"]),
            retrieved_at=_parse(row["retrieved_at"]),
            provider_id=str(row["provider_id"]),
            provider_rank=int(row["provider_rank"]),
            cross_source_count=int(row["cross_source_count"]),
            read_status=WebReadStatus(row["read_status"]),
            metadata=_load(row["metadata_json"]) or {},
            expires_at=_parse(row["expires_at"]),
            redacted_at=_parse(row["redacted_at"]),
        )

    @staticmethod
    def _web_image_result(row: sqlite3.Row) -> WebImageSearchResultRecord:
        return WebImageSearchResultRecord(
            image_resource_id=str(row["image_resource_id"]),
            session_id=str(row["session_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            original_url=str(row["original_url"]),
            thumbnail_url=str(row["thumbnail_url"]),
            source_page_url=str(row["source_page_url"]),
            source_domain=str(row["source_domain"]),
            title=str(row["title"]),
            description=str(row["description"]),
            provider_id=str(row["provider_id"]),
            provider_rank=int(row["provider_rank"]),
            cross_source_count=int(row["cross_source_count"]),
            width=row["width"],
            height=row["height"],
            mime_type=str(row["mime_type"]),
            metadata=_load(row["metadata_json"]) or {},
            retrieved_at=_parse(row["retrieved_at"]),
            expires_at=_parse(row["expires_at"]),
            redacted_at=_parse(row["redacted_at"]),
        )

    @staticmethod
    def _web_page(row: sqlite3.Row) -> WebPageSnapshotRecord:
        return WebPageSnapshotRecord(
            snapshot_id=int(row["snapshot_id"]),
            resource_id=str(row["resource_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            content=str(row["content"]),
            content_hash=str(row["content_hash"]),
            token_estimate=int(row["token_estimate"]),
            status=WebReadStatus(row["status"]),
            error=str(row["error"]),
            metadata=_load(row["metadata_json"]) or {},
            retrieved_at=_parse(row["retrieved_at"]),
            expires_at=_parse(row["expires_at"]),
            redacted_at=_parse(row["redacted_at"]),
        )

    @staticmethod
    def _dialogue_summary(row: sqlite3.Row) -> DialogueSummary:
        return DialogueSummary(
            summary_id=int(row["summary_id"]),
            profile_id=row["profile_id"],
            instance_id=row["instance_id"],
            version=int(row["version"]),
            strategy_id=row["strategy_id"],
            strategy_version=int(row["strategy_version"]),
            covered_from_message_id=row["covered_from_message_id"],
            covered_through_message_id=int(row["covered_through_message_id"]),
            structured=_load(row["structured_json"]) or {},
            rendered_text=row["rendered_text"],
            token_count=int(row["token_count"]),
            created_at=_parse(row["created_at"]),
        )

    @staticmethod
    def _context_build_report(row: sqlite3.Row) -> ContextBuildReport:
        return ContextBuildReport(
            profile_id=row["profile_id"],
            instance_id=row["instance_id"],
            model_id=row["model_id"],
            token_count_mode=row["token_count_mode"],
            hard_token_limit=int(row["hard_token_limit"]),
            target_token_budget=int(row["target_token_budget"]),
            fill_budget=int(row["fill_budget"]),
            total_tokens=int(row["total_tokens"]),
            report=_load(row["report_json"]) or {},
            created_at=_parse(row["created_at"]),
        )

    @staticmethod
    def _scope_config(row: sqlite3.Row) -> ScopeConfig:
        return ScopeConfig(
            profile_id=row["profile_id"],
            scope=row["scope"],
            proactive_enabled=bool(row["proactive_enabled"]),
            extra_background=row["extra_background"],
            world_texture_prompt=row["world_texture_prompt"],
            media_original_retention_days=int(row["media_original_retention_days"]),
            min_wakeup_minutes=row["min_wakeup_minutes"],
            max_wakeup_minutes=row["max_wakeup_minutes"],
            low_frequency_min_wakeup_minutes=row["low_frequency_min_wakeup_minutes"],
            low_frequency_max_wakeup_minutes=row["low_frequency_max_wakeup_minutes"],
            max_context_tokens=row["max_context_tokens"],
            target_context_tokens=row["target_context_tokens"],
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _character_instance(row: sqlite3.Row) -> CharacterInstance:
        return CharacterInstance(
            profile_id=row["profile_id"],
            instance_id=row["instance_id"],
            route_umo=row["route_umo"],
            platform_id=row["platform_id"],
            message_type=row["message_type"],
            target_id=row["target_id"],
            scope=row["scope"],
            session_kind=row["session_kind"],
            readiness=RouteReadiness(row["readiness"]),
            initialization_state=InstanceInitializationState(row["initialization_state"]),
            initialization_completed_at=_parse(row["initialization_completed_at"]),
            proactive_enabled=bool(row["proactive_enabled"]),
            extra_background=row["extra_background"],
            min_wakeup_minutes=row["min_wakeup_minutes"],
            max_wakeup_minutes=row["max_wakeup_minutes"],
            low_frequency_min_wakeup_minutes=row["low_frequency_min_wakeup_minutes"],
            low_frequency_max_wakeup_minutes=row["low_frequency_max_wakeup_minutes"],
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _profile(row: sqlite3.Row) -> RoleProfile:
        return RoleProfile(
            profile_id=row["profile_id"],
            name=row["name"],
            enabled=bool(row["enabled"]),
            quick_setup_decided=bool(row["quick_setup_decided"]),
            thinking_complexity=str(row["thinking_complexity"]),
            background_life_enabled=bool(row["background_life_enabled"]),
            background_life_version=int(row["background_life_version"]),
            turn_buffer_enabled=bool(row["turn_buffer_enabled"]),
            image_generation_enabled=bool(row["image_generation_enabled"]),
            file_artifacts_enabled=bool(row["file_artifacts_enabled"]),
            web_search_enabled=bool(row["web_search_enabled"]),
            web_search_intensity=str(row["web_search_intensity"]),
            proactive_enabled=bool(row["proactive_enabled"]),
            extra_background=row["extra_background"],
            min_wakeup_minutes=row["min_wakeup_minutes"],
            max_wakeup_minutes=row["max_wakeup_minutes"],
            low_frequency_min_wakeup_minutes=row["low_frequency_min_wakeup_minutes"],
            low_frequency_max_wakeup_minutes=row["low_frequency_max_wakeup_minutes"],
            orphaned=bool(row["orphaned"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _state(row: sqlite3.Row) -> CoreState:
        return CoreState(
            profile_id=row["profile_id"],
            state_epoch=row["state_epoch"],
            activity_epoch=row["activity_epoch"],
            low_frequency_mode=bool(row["low_frequency_mode"]),
            low_frequency_reason=row["low_frequency_reason"],
            low_frequency_since=_parse(row["low_frequency_since"]),
            updated_at=_parse(row["updated_at"]),
            instance_id=row["instance_id"],
        )

    @staticmethod
    def _wakeup(row: sqlite3.Row) -> Wakeup:
        return Wakeup(
            wakeup_id=row["wakeup_id"],
            profile_id=row["profile_id"],
            source=WakeSource(row["source"]),
            due_at=_parse(row["due_at"]),  # type: ignore[arg-type]
            reason=row["reason"],
            conversation_ref=row["conversation_ref"],
            idempotency_key=row["idempotency_key"],
            payload=_load(row["payload_json"]) or {},
            status=WakeupStatus(row["status"]),
            attempts=row["attempts"],
            lease_until=_parse(row["lease_until"]),
            last_error=row["last_error"],
            instance_id=row["instance_id"],
            generation=int(row["generation"] or 0),
            lease_token=int(row["lease_token"] or 0),
            version=int(row["version"] or 0),
            intent_kind=str(row["intent_kind"] or ""),
            linked_task_id=(
                int(row["linked_task_id"]) if row["linked_task_id"] is not None else None
            ),
        )

    @staticmethod
    def _outbox(row: sqlite3.Row) -> OutboxItem:
        return OutboxItem(
            outbox_id=row["outbox_id"],
            profile_id=row["profile_id"],
            umo=row["umo"],
            payload=_load(row["payload_json"]) or {},
            status=OutboxStatus(row["status"]),
            idempotency_key=row["idempotency_key"],
            attempts=row["attempts"],
            activity_epoch=row["activity_epoch"],
            expression_batch_id=(
                str(row["expression_batch_id"]) if row["expression_batch_id"] is not None else None
            ),
            expression_ordinal=(
                int(row["expression_ordinal"]) if row["expression_ordinal"] is not None else None
            ),
            expression_step_ordinal=(
                int(row["expression_step_ordinal"])
                if row["expression_step_ordinal"] is not None
                else None
            ),
            not_before_at=_parse(row["not_before_at"]),
            interrupt_policy=OutboxInterruptPolicy(row["interrupt_policy"]),
            depends_on_idempotency_key=(
                str(row["depends_on_idempotency_key"])
                if row["depends_on_idempotency_key"] is not None
                else None
            ),
            context_message_id=(
                int(row["context_message_id"]) if row["context_message_id"] is not None else None
            ),
            last_error_code=str(row["last_error_code"] or ""),
            last_error=row["last_error"],
            last_diagnostic_code=str(row["last_diagnostic_code"] or ""),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
            instance_id=row["instance_id"],
        )

    @staticmethod
    def _expression_batch(row: sqlite3.Row) -> ExpressionBatch:
        return ExpressionBatch(
            batch_id=str(row["batch_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            source_run_id=int(row["source_run_id"]),
            activity_epoch=int(row["activity_epoch"]),
            route_umo=str(row["route_umo"]),
            status=ExpressionBatchStatus(row["status"]),
            output_count=int(row["output_count"]),
            retraction_count=int(row["retraction_count"]),
            step_count=int(row["step_count"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
            settled_at=_parse(row["settled_at"]),
        )

    @staticmethod
    def _record(row: sqlite3.Row, *, json_columns: tuple[str, ...]) -> dict[str, Any]:
        result = dict(row)
        for column in json_columns:
            result[column.removesuffix("_json")] = _load(result.pop(column))
        for key in tuple(result):
            if key.endswith("_at") and isinstance(result[key], str):
                result[key] = _parse(result[key])
        return result
