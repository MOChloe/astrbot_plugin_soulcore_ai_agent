"""Administrator read model for causal AI work records."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from ....features.ai.ports import AIAdminQueryRepositoryPort
from ....features.ai.recording_health import ai_recording_health
from ....features.ai.work_taxonomy import (
    AIWorkNodeRole,
    work_purpose_options,
    work_purpose_spec,
)
from ..presentation import jsonable
from .ai_work_attempt_views import (
    debug_attempt_view,
    debug_available,
    looks_like_internal_identifier,
    raw_attempt_view,
    record_duration_ms,
)
from .ai_work_audio_views import audio_attempt_summary
from .ai_work_error_views import error_view, known_error_guidance


def attempt_token_view(attempt: Mapping[str, Any]) -> dict[str, int]:
    return {
        "input": _integer(attempt.get("input_tokens")),
        "output": _integer(attempt.get("output_tokens")),
        "cache_read": _integer(attempt.get("cache_read_tokens")),
        "cache_write": _integer(attempt.get("cache_write_tokens")),
    }


def attempt_cache_view(attempt: Mapping[str, Any]) -> dict[str, str]:
    status = str(attempt.get("cache_status") or "")
    labels = {
        "HIT": "缓存命中",
        "WRITE": "缓存写入",
        "CONFIRMED_NO_HIT": "已确认支持，当前未命中",
        "ACCEPTED_UNVERIFIED": "已接受，尚无用量证据",
        "QUALITY_ANOMALY": "缓存质量异常",
        "QUALITY_SUSPENDED": "已触发缓存暂停",
        "AUTO_QUALITY_WARNING": "自动缓存不可控",
        "CACHE_REPORTED_WHILE_DISABLED": "自动缓存不可控",
        "QUALITY_RECOVERED": "缓存复探已恢复",
    }
    return {
        "mode": str(attempt.get("cache_mode") or ""),
        "status": status,
        "label": labels.get(status, ""),
    }


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


_SPECIAL_NODE_LABELS = {
    "FINAL_EXPRESSION_CONFIRMATION": "最终表达确认",
    "FINAL_RESULT_COMMIT": "最终结果提交",
}

_DELIVERY_STATUS_LABELS = {
    "PENDING": "等待平台投递",
    "SENDING": "正在投递",
    "PLATFORM_ACCEPTED_UNCONFIRMED": "平台已接收（最终送达未确认）",
    "DISPATCH_ATTEMPTED_UNKNOWN": "已尝试投递（平台结果未知）",
    "PARTIALLY_ATTEMPTED": "部分消息已尝试投递（其余未发送）",
    "UNKNOWN_AFTER_CRASH": "已发送，结果未知",
    "FAILED": "平台投递失败",
    "CANCELLED": "投递已取消",
}

_UNKNOWN_PLATFORM_CALL_PREFIXES = (
    "send_exception:",
    "delivery_exception_after_platform_boundary:",
)


def _platform_call_result_unknown(item: Mapping[str, Any]) -> bool:
    if str(item.get("status") or "") != "PLATFORM_ACCEPTED_UNCONFIRMED":
        return False
    diagnostic = str(item.get("last_diagnostic_code") or "")
    return diagnostic.startswith(_UNKNOWN_PLATFORM_CALL_PREFIXES)


_NODE_KIND_LABELS = {
    "MODEL": "模型处理",
    "AUDIO": "语音处理",
    "WEB": "联网处理",
    "IMAGE": "图片处理",
    "FILE": "文件处理",
    "COMMAND": "内部指令",
    "SYSTEM": "系统处理",
}

_EVENT_CATEGORY_LABELS = {
    "VALIDATION": "结果校验",
    "TRANSACTION": "结果提交",
    "ROUTING": "处理路由",
    "DATA_CLEANED": "数据清理",
    "COMMAND": "内部动作",
    "NOTE": "处理记录",
}

_EVENT_SEVERITY_LABELS = {
    "INFO": "信息",
    "WARNING": "警告",
    "ERROR": "错误",
}

_RECORDING_OPERATION_LABELS = {
    "start_ai_work_node": "开始记录处理阶段",
    "start_ai_provider_attempt": "开始记录模型接口尝试",
    "enrich_ai_provider_attempt": "补充模型接口请求记录",
    "mark_ai_provider_attempt_sent": "记录模型请求已发送",
    "finish_ai_provider_attempt": "完成模型接口尝试记录",
    "annotate_ai_provider_attempt": "补充模型接口尝试结果",
    "create_ai_workflow": "创建 AI 工作记录",
    "finish_ai_workflow": "完成 AI 工作记录",
    "record_ai_work_event": "记录工作事件",
    "finish_ai_work_node": "完成处理阶段记录",
    "get_ai_work_node_by_model_invocation": "查找模型调用阶段",
    "record_ai_backend_health": "记录模型后端健康状态",
    "record_ai_circuit_health": "记录模型熔断状态",
}


def _cursor_encode(started_at: Any, workflow_id: int) -> str:
    timestamp = started_at.isoformat() if isinstance(started_at, datetime) else str(started_at)
    payload = json.dumps([timestamp, int(workflow_id)], separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _cursor_decode(value: Any) -> tuple[str, int]:
    text = str(value or "").strip()
    if not text:
        return "", 0
    try:
        padded = text + "=" * (-len(text) % 4)
        raw = json.loads(base64.urlsafe_b64decode(padded).decode())
        started_at, workflow_id = raw
        return str(started_at), max(0, int(workflow_id))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid AI work record cursor") from exc


def _text_value(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key)
    return str(value if value is not None and value != "" else default)


def _int_value(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    return int(value if value is not None else default)


def _dict_value(row: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _list_value(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    return list(value) if isinstance(value, (list, tuple)) else []


class AIWorkRecordsController:
    def __init__(self, repository: AIAdminQueryRepositoryPort) -> None:
        self.repository = repository

    async def list(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        query = self._list_query(payload)
        limit = int(query["limit"])
        rows = list(
            await self.repository.list_ai_workflow_summaries(
                profile_id=profile_id,
                **query,
            )
        )
        page, next_cursor = self._page(rows, limit)
        filters = await self.repository.list_ai_work_filter_values(
            profile_id=profile_id, instance_id=query["instance_id"]
        )
        return {
            "items": [self._summary(row) for row in page],
            "next_cursor": next_cursor,
            "filter_options": self._filter_options(filters),
            "recording_health": self._recording_health_view(),
        }

    @staticmethod
    def _list_query(payload: Mapping[str, Any]) -> dict[str, Any]:
        limit = max(1, min(_int_value(payload, "limit", 50), 100))
        cursor_started_at, cursor_workflow_id = _cursor_decode(payload.get("cursor"))
        since_hours = max(0, min(_int_value(payload, "since_hours"), 24 * 365))
        view_scope = _text_value(payload, "view_scope").lower()
        selected_instance = _text_value(payload, "instance_id").strip()
        return {
            "instance_id": None if view_scope == "profile" else selected_instance or None,
            "run_status": _text_value(payload, "run_status"),
            "delivery_status": _text_value(payload, "delivery_status"),
            "purpose": _text_value(payload, "purpose"),
            "model": _text_value(payload, "model"),
            "issue_type": _text_value(payload, "issue_type"),
            "since": (datetime.now(UTC) - timedelta(hours=since_hours) if since_hours else None),
            "issues_only": bool(payload.get("issues_only")),
            "cursor_started_at": cursor_started_at,
            "cursor_workflow_id": cursor_workflow_id,
            "limit": limit,
        }

    @staticmethod
    def _page(
        rows: list[Mapping[str, Any]], limit: int
    ) -> tuple[list[Mapping[str, Any]], str | None]:
        page = rows[:limit]
        if len(rows) <= limit or not page:
            return page, None
        last = page[-1]
        return page, _cursor_encode(last["started_at"], int(last["workflow_id"]))

    def _filter_options(self, filters: Mapping[str, Any]) -> dict[str, Any]:
        issue_types = [
            {"value": "fallback", "label": "发生降级"},
            {"value": "retried", "label": "发生 Provider 重试"},
        ]
        for raw_code in _list_value(filters, "issue_codes"):
            code = str(raw_code)
            error = self._error_view(code, "")
            issue_types.append({"value": code, "label": error["title"] if error else code})
        return {
            "run_statuses": ["RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"],
            "delivery_statuses": [
                "NONE",
                "PENDING",
                "PLATFORM_ACCEPTED_UNCONFIRMED",
                "PARTIALLY_ATTEMPTED",
                "FAILED",
                "UNKNOWN",
                "CANCELLED",
            ],
            "purposes": work_purpose_options(),
            "models": _list_value(filters, "models"),
            "issue_types": issue_types,
        }

    async def detail(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        work_ref = str(payload.get("work_ref") or "").strip()
        instance_id = self._required_instance_id(payload)
        if not work_ref:
            raise ValueError("work_ref is required")
        workflow = await self.repository.get_ai_workflow_by_ref(
            profile_id=profile_id, work_ref=work_ref
        )
        if workflow is None or str(workflow.get("instance_id") or "") != instance_id:
            raise KeyError("AI work record not found")
        workflow_id = int(workflow["workflow_id"])
        nodes = list(await self.repository.list_ai_work_nodes(workflow_id))
        attempts = list(await self.repository.list_ai_provider_attempts(workflow_id=workflow_id))
        events = list(await self.repository.list_ai_work_events(workflow_id))
        deliveries = list(await self.repository.list_ai_workflow_deliveries(workflow_id))
        timeline = self._timeline(nodes, attempts, events)
        return {
            "record": self._detail_summary(workflow, nodes, attempts, deliveries),
            "timeline": timeline,
            "workflow_events": [
                self._event_view(event) for event in events if event.get("node_id") is None
            ],
            "delivery": {
                "status": self._delivery_status(deliveries),
                "summary": self._delivery_summary(deliveries),
                "items": [self._delivery_view(item) for item in deliveries],
            },
            "recording_health": self._recording_health_view(),
        }

    async def debug(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        attempt = await self._attempt_by_payload(profile_id, payload)
        audio = audio_attempt_summary(attempt)
        if audio is not None:
            return {"audio": audio}
        return debug_attempt_view(attempt)

    async def raw(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        attempt = await self._attempt_by_payload(profile_id, payload)
        if audio_attempt_summary(attempt) is not None:
            return {"request": None, "response": None}
        return raw_attempt_view(attempt)

    async def _attempt_by_payload(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        work_ref = str(payload.get("work_ref") or "").strip()
        attempt_ref = str(payload.get("attempt_ref") or "").strip()
        instance_id = self._required_instance_id(payload)
        if not work_ref or not attempt_ref:
            raise ValueError("work_ref and attempt_ref are required")
        workflow = await self.repository.get_ai_workflow_by_ref(
            profile_id=profile_id, work_ref=work_ref
        )
        if workflow is None or str(workflow.get("instance_id") or "") != instance_id:
            raise KeyError("AI provider attempt not found")
        attempt = await self.repository.get_ai_provider_attempt_by_ref(
            profile_id=profile_id,
            work_ref=work_ref,
            attempt_ref=attempt_ref,
        )
        if attempt is None:
            raise KeyError("AI provider attempt not found")
        return attempt

    @staticmethod
    def _required_instance_id(payload: Mapping[str, Any]) -> str:
        instance_id = str(payload.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError("instance_id is required")
        return instance_id

    def _summary(self, row: Mapping[str, Any]) -> dict[str, Any]:
        purpose = _text_value(row, "primary_purpose", "MODEL_REQUEST")
        purpose_spec = work_purpose_spec(purpose)
        run_status = _text_value(row, "status", "RUNNING")
        activity_status = self._activity_status(row, run_status)
        duration_start = row.get("started_at")
        duration_end = row.get("finished_at")
        if activity_status == "WAITING_RETRY":
            duration_start = row.get("latest_stage_started_at") or duration_start
            duration_end = (
                row.get("latest_stage_finished_at")
                or row.get("durable_task_updated_at")
                or duration_end
            )
        return {
            "work_ref": _text_value(row, "public_ref"),
            "owner_instance_id": _text_value(row, "instance_id"),
            "purpose": purpose,
            "purpose_label": purpose_spec.label,
            "reason": _text_value(row, "reason", purpose_spec.reason),
            "run_status": run_status,
            "activity_status": activity_status,
            "next_retry_at": (
                jsonable(row.get("durable_task_due_at"))
                if activity_status == "WAITING_RETRY"
                else None
            ),
            "delivery_status": self._delivery_status_from_summary(row),
            "started_at": jsonable(row.get("started_at")),
            "finished_at": jsonable(row.get("finished_at")),
            "duration_ms": record_duration_ms(duration_start, duration_end),
            "object": self._object_view(row),
            "business_stage_count": _int_value(row, "business_stage_count"),
            "internal_action_count": _int_value(row, "internal_action_count"),
            "provider_send_count": _int_value(row, "provider_send_count"),
            "retry_count": _int_value(row, "retry_count"),
            "tokens": self._token_summary(row),
            "models": list(filter(None, _text_value(row, "models_csv").split(","))),
            "composition": self._composition_view(row),
            "fallback_count": _int_value(row, "fallback_count"),
            "issue_count": _int_value(row, "issue_count"),
            "final_error": self._error_view(
                _text_value(row, "final_error_code"),
                _text_value(row, "final_message"),
            ),
        }

    @staticmethod
    def _activity_status(row: Mapping[str, Any], run_status: str) -> str:
        if run_status == "RUNNING" and _text_value(row, "durable_task_status") == "RETRY_WAIT":
            return "WAITING_RETRY"
        return run_status

    @staticmethod
    def _token_summary(row: Mapping[str, Any]) -> dict[str, int]:
        input_tokens = _int_value(row, "input_tokens")
        output_tokens = _int_value(row, "output_tokens")
        cache_read_tokens = _int_value(row, "cache_read_tokens")
        cache_write_tokens = _int_value(row, "cache_write_tokens")
        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "cache_read": cache_read_tokens,
            "cache_write": cache_write_tokens,
        }

    @staticmethod
    def _composition_view(row: Mapping[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw_item in _list_value(row, "composition"):
            item = dict(raw_item)
            purpose = _text_value(item, "purpose")
            item["label"] = work_purpose_spec(purpose).label
            result.append(item)
        return result

    def _detail_summary(
        self,
        workflow: Mapping[str, Any],
        nodes: list[Mapping[str, Any]],
        attempts: list[Mapping[str, Any]],
        deliveries: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        values = dict(workflow)
        values.update(self._node_aggregates(nodes))
        values.update(self._attempt_aggregates(attempts))
        values.update(self._delivery_aggregates(deliveries))
        summary = self._summary(values)
        summary["delivery_status"] = self._delivery_status(deliveries)
        return summary

    def _node_aggregates(self, nodes: list[Mapping[str, Any]]) -> dict[str, Any]:
        business_count = internal_count = fallback_count = 0
        latest_stage: Mapping[str, Any] | None = None
        for node in nodes:
            role = _text_value(node, "node_role")
            business_count += int(role == AIWorkNodeRole.BUSINESS_STAGE.value)
            internal_count += int(role == AIWorkNodeRole.INTERNAL_ACTION.value)
            fallback_count += int(_text_value(node, "status") == "FALLBACK")
            if role == AIWorkNodeRole.BUSINESS_STAGE.value:
                latest_stage = node
        return {
            "business_stage_count": business_count,
            "internal_action_count": internal_count,
            "composition": self._composition(nodes),
            "fallback_count": fallback_count,
            "issue_count": 0,
            "latest_stage_started_at": (
                latest_stage.get("started_at") if latest_stage is not None else None
            ),
            "latest_stage_finished_at": (
                latest_stage.get("finished_at") if latest_stage is not None else None
            ),
        }

    @staticmethod
    def _attempt_aggregates(attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
        send_count = retry_count = input_tokens = output_tokens = 0
        cache_read_tokens = cache_write_tokens = 0
        models: dict[str, None] = {}
        for attempt in attempts:
            sent = attempt.get("sent_at") is not None
            send_count += int(sent)
            retry_count += int(sent and _int_value(attempt, "attempt_no", 1) > 1)
            input_tokens += _int_value(attempt, "input_tokens")
            output_tokens += _int_value(attempt, "output_tokens")
            cache_read_tokens += _int_value(attempt, "cache_read_tokens")
            cache_write_tokens += _int_value(attempt, "cache_write_tokens")
            model = _text_value(attempt, "model_id")
            if model:
                models[model] = None
        return {
            "provider_send_count": send_count,
            "retry_count": retry_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "models_csv": ",".join(models),
        }

    @staticmethod
    def _delivery_aggregates(deliveries: list[Mapping[str, Any]]) -> dict[str, int]:
        counts = {
            "delivery_count": len(deliveries),
            "delivery_failed_count": 0,
            "delivery_accepted_count": 0,
            "delivery_partial_count": 0,
            "delivery_pending_count": 0,
            "delivery_unknown_count": 0,
        }
        keys = {
            "FAILED": "delivery_failed_count",
            "PLATFORM_ACCEPTED_UNCONFIRMED": "delivery_accepted_count",
            "PARTIALLY_ATTEMPTED": "delivery_partial_count",
            "PENDING": "delivery_pending_count",
            "SENDING": "delivery_pending_count",
            "UNKNOWN_AFTER_CRASH": "delivery_unknown_count",
        }
        for delivery in deliveries:
            key = (
                "delivery_unknown_count"
                if _platform_call_result_unknown(delivery)
                else keys.get(_text_value(delivery, "status"))
            )
            if key:
                counts[key] += 1
        return counts

    def _timeline(
        self,
        nodes: list[Mapping[str, Any]],
        attempts: list[Mapping[str, Any]],
        events: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        node_purposes = {int(node["node_id"]): _text_value(node, "purpose") for node in nodes}
        attempts_by_node: dict[int, list[dict[str, Any]]] = {}
        first_backend_by_node: dict[int, str] = {}
        for attempt in attempts:
            node_id = int(attempt["node_id"])
            backend_id = _text_value(attempt, "backend_id")
            first_backend = first_backend_by_node.setdefault(node_id, backend_id)
            attempts_by_node.setdefault(node_id, []).append(
                self._attempt_view(
                    attempt,
                    purpose=node_purposes.get(node_id, ""),
                    fallback=bool(first_backend and backend_id and backend_id != first_backend),
                )
            )
        events_by_node: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            if event.get("node_id") is not None:
                events_by_node.setdefault(int(event["node_id"]), []).append(self._event_view(event))
        views: dict[int, dict[str, Any]] = {}
        for node in nodes:
            node_id = int(node["node_id"])
            views[node_id] = self._node_view(
                node,
                attempts_by_node.get(node_id, []),
                events_by_node.get(node_id, []),
            )
        roots: list[dict[str, Any]] = []
        for node in nodes:
            node_id = int(node["node_id"])
            parent_id = node.get("parent_node_id")
            if parent_id is not None and int(parent_id) in views:
                views[int(parent_id)]["children"].append(views[node_id])
            else:
                roots.append(views[node_id])
        return roots

    def _node_view(
        self,
        node: Mapping[str, Any],
        attempts: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        role = _text_value(node, "node_role", "SYSTEM_STAGE")
        purpose = _text_value(node, "purpose", role)
        kind = _text_value(node, "node_kind", "SYSTEM")
        return {
            "node_ref": _text_value(node, "public_ref"),
            "sequence": _int_value(node, "sequence"),
            "role": role,
            "kind": kind,
            "kind_label": _NODE_KIND_LABELS.get(kind, "处理步骤"),
            "purpose": purpose,
            "label": self._node_label(role, purpose),
            "collapsed_by_default": role == AIWorkNodeRole.INTERNAL_ACTION.value,
            "status": _text_value(node, "status", "RUNNING"),
            "started_at": jsonable(node.get("started_at")),
            "finished_at": jsonable(node.get("finished_at")),
            "duration_ms": record_duration_ms(node.get("started_at"), node.get("finished_at")),
            "summary": _text_value(node, "summary"),
            "error": self._error_view(
                _text_value(node, "error_code"), _text_value(node, "error_message")
            ),
            "warning": self._node_warning(node),
            "attempts": attempts,
            "events": events,
            "children": [],
        }

    @staticmethod
    def _node_label(role: str, purpose: str) -> str:
        special = _SPECIAL_NODE_LABELS.get(purpose)
        if special:
            return special
        if role == AIWorkNodeRole.INTERNAL_ACTION.value:
            command_label = purpose.removeprefix("COMMAND:").strip()
            if purpose.startswith("COMMAND:") and not looks_like_internal_identifier(command_label):
                return command_label or "内部动作"
            return "内部动作"
        if role == AIWorkNodeRole.SYSTEM_STAGE.value:
            return "系统处理"
        return work_purpose_spec(purpose).label

    @staticmethod
    def _node_warning(node: Mapping[str, Any]) -> dict[str, str] | None:
        code = _text_value(node, "warning_code")
        message = _text_value(node, "warning_message")
        return {"code": code, "message": message} if code or message else None

    def _attempt_view(
        self,
        attempt: Mapping[str, Any],
        *,
        purpose: str = "",
        fallback: bool = False,
    ) -> dict[str, Any]:
        request = _dict_value(attempt, "request")
        response = _dict_value(attempt, "response")
        audio = audio_attempt_summary(attempt, purpose=purpose, fallback=fallback)
        return {
            "attempt_ref": _text_value(attempt, "public_ref"),
            "round": _int_value(attempt, "round_no", 1),
            "attempt": _int_value(attempt, "attempt_no", 1),
            "status": _text_value(attempt, "status", "PREPARING"),
            "sent": attempt.get("sent_at") is not None,
            "sent_at": jsonable(attempt.get("sent_at")),
            "model": _text_value(attempt, "model_id"),
            "started_at": jsonable(attempt.get("started_at")),
            "finished_at": jsonable(attempt.get("finished_at")),
            "duration_ms": record_duration_ms(
                attempt.get("started_at"), attempt.get("finished_at")
            ),
            "tokens": attempt_token_view(attempt),
            "cache": None if audio is not None else attempt_cache_view(attempt),
            "cache_applicable": audio is None,
            "fallback": bool(fallback),
            "audio": audio,
            "error": self._error_view(
                _text_value(attempt, "error_code"),
                _text_value(attempt, "error_message"),
            ),
            "debug_available": (
                False
                if audio is not None
                else debug_available(request, response, attempt.get("evaluation"))
            ),
            "raw_available": (
                False
                if audio is not None
                else request.get("provider_envelope") is not None
                or response.get("provider_envelope") is not None
            ),
        }

    def _event_view(self, event: Mapping[str, Any]) -> dict[str, Any]:
        code = str(event.get("code") or "")
        category = str(event.get("event_category") or "NOTE")
        severity = str(event.get("severity") or "INFO")
        return {
            "event_ref": str(event.get("public_ref") or ""),
            "category": category,
            "category_label": _EVENT_CATEGORY_LABELS.get(category, "处理记录"),
            "severity": severity,
            "severity_label": _EVENT_SEVERITY_LABELS.get(severity, "状态待确认"),
            "code": code,
            "summary": str(event.get("summary") or ""),
            "occurred_at": jsonable(event.get("occurred_at")),
            "guidance": self._known_error_guidance(code, str(event.get("summary") or "")),
        }

    @staticmethod
    def _recording_health_view() -> dict[str, Any]:
        view = ai_recording_health.snapshot().as_dict()
        operation = str(view.get("failed_operation") or "")
        view["failed_operation_label"] = _RECORDING_OPERATION_LABELS.get(
            operation, "写入 AI 工作记录"
        )
        return view

    @staticmethod
    def _known_error_guidance(code: str, message: str) -> dict[str, Any] | None:
        return known_error_guidance(code, message)

    @staticmethod
    def _error_view(code: str, message: str) -> dict[str, Any] | None:
        return error_view(code, message)

    @staticmethod
    def _object_view(row: Mapping[str, Any]) -> dict[str, Any]:
        scope = str(row.get("object_scope") or "")
        return {
            "kind": "群聊" if scope == "group" else "好友" if scope == "private" else "后台任务",
        }

    @staticmethod
    def _composition(nodes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for node in nodes:
            if str(node.get("node_role")) != AIWorkNodeRole.BUSINESS_STAGE.value:
                continue
            purpose = str(node.get("purpose") or "MODEL_REQUEST")
            counts[purpose] = counts.get(purpose, 0) + 1
        return [{"purpose": key, "count": value} for key, value in counts.items()]

    @staticmethod
    def _delivery_status_from_summary(row: Mapping[str, Any]) -> str:
        if int(row.get("delivery_count") or 0) == 0:
            return "NONE"
        if int(row.get("delivery_failed_count") or 0):
            return "FAILED"
        if int(row.get("delivery_partial_count") or 0):
            return "PARTIALLY_ATTEMPTED"
        if int(row.get("delivery_unknown_count") or 0):
            return "UNKNOWN"
        if int(row.get("delivery_pending_count") or 0):
            return "PENDING"
        if int(row.get("delivery_accepted_count") or 0):
            return "PLATFORM_ACCEPTED_UNCONFIRMED"
        return "CANCELLED"

    @staticmethod
    def _delivery_status(deliveries: list[Mapping[str, Any]]) -> str:
        if not deliveries:
            return "NONE"
        statuses = {str(item.get("status") or "") for item in deliveries}
        if "FAILED" in statuses:
            return "FAILED"
        if "PARTIALLY_ATTEMPTED" in statuses:
            return "PARTIALLY_ATTEMPTED"
        if "UNKNOWN_AFTER_CRASH" in statuses:
            return "UNKNOWN"
        if any(_platform_call_result_unknown(item) for item in deliveries):
            return "UNKNOWN"
        if statuses & {"PENDING", "SENDING"}:
            return "PENDING"
        if "PLATFORM_ACCEPTED_UNCONFIRMED" in statuses:
            return "PLATFORM_ACCEPTED_UNCONFIRMED"
        return "CANCELLED"

    @staticmethod
    def _delivery_summary(deliveries: list[Mapping[str, Any]]) -> str:
        status = AIWorkRecordsController._delivery_status(deliveries)
        return {
            "NONE": "本次工作没有生成平台投递。",
            "PENDING": "消息正在等待平台投递。",
            "PLATFORM_ACCEPTED_UNCONFIRMED": "平台已接收（最终送达未确认）。",
            "PARTIALLY_ATTEMPTED": "部分消息已尝试投递，其余消息未发送。",
            "FAILED": "至少一条消息投递失败。",
            "UNKNOWN": "至少一条消息已经发起平台调用，但结果无法确认。",
            "CANCELLED": "平台投递已取消。",
        }[status]

    @staticmethod
    def _delivery_view(item: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(item.get("payload") or {})
        status = (
            "DISPATCH_ATTEMPTED_UNKNOWN"
            if _platform_call_result_unknown(item)
            else str(item.get("status") or "PENDING")
        )
        return {
            "status": status,
            "status_label": _DELIVERY_STATUS_LABELS.get(status, "投递状态待确认"),
            "content": str(payload.get("content") or payload.get("text") or ""),
            "scheduled_at": jsonable(item.get("not_before_at")),
            "error": AIWorkRecordsController._error_view(
                str(item.get("last_error_code") or ""), str(item.get("last_error") or "")
            ),
            "diagnostic_code": str(item.get("last_diagnostic_code") or ""),
        }


__all__ = ["AIWorkRecordsController"]
