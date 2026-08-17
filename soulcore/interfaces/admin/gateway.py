"""AstrBot Plugin Page HTTP gateway."""

from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Protocol

from .console_errors import ConsoleValidationError, console_error_envelope
from .routes import PAGE_ROUTES

PLUGIN_NAME = "astrbot_plugin_soulcore_ai_agent"
PAGE_API_PREFIX = f"/{PLUGIN_NAME}/page"
_BOOTSTRAP_READ_ACTIONS = frozenset(
    {
        "console_bootstrap",
        "settings_snapshot",
        "player_bootstrap",
        "player_now",
        "player_contacts",
        "player_relationship",
        "player_about",
        "release_notes",
        "quick_setup_status",
        "quick_setup_snapshot",
    }
)


class PageApiCallbacks(Protocol):
    async def call(self, method: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    async def download(self, method: str, payload: Mapping[str, Any]) -> Any: ...


class PageApiFacade:
    """Register all Page endpoints and delegate business work to one callback."""

    def __init__(self, callbacks: PageApiCallbacks) -> None:
        self.callbacks = callbacks

    def register(self, context: Any) -> None:
        _remove_previous_page_routes(context)
        for route in PAGE_ROUTES:
            context.register_web_api(
                f"{PAGE_API_PREFIX}/{route.suffix}",
                getattr(self, route.handler),
                list(route.methods),
                route.description,
            )

    async def console_bootstrap(self) -> Any:
        return await self._handle("console_bootstrap")

    async def player_bootstrap(self) -> Any:
        return await self._handle("player_bootstrap")

    async def player_guide_acknowledge(self) -> Any:
        return await self._handle("player_guide_acknowledge")

    async def advanced_guide_acknowledge(self) -> Any:
        return await self._handle("advanced_guide_acknowledge")

    async def player_now(self) -> Any:
        return await self._handle("player_now")

    async def player_contacts(self) -> Any:
        return await self._handle("player_contacts")

    async def player_relationship(self) -> Any:
        return await self._handle("player_relationship")

    async def player_about(self) -> Any:
        return await self._handle("player_about")

    async def role_package_export_prepare(self) -> Any:
        return await self._handle("role_package_export_prepare")

    async def role_package_download(self) -> Any:
        return await self._handle_download("role_package_download")

    async def role_package_import_upload(self) -> Any:
        return await self._handle("role_package_import_upload")

    async def role_package_import_apply(self) -> Any:
        return await self._handle("role_package_import_apply")

    async def release_notes(self) -> Any:
        return await self._handle("release_notes")

    async def quick_setup_status(self) -> Any:
        return await self._handle("quick_setup_status")

    async def quick_setup_snapshot(self) -> Any:
        return await self._handle("quick_setup_snapshot")

    async def quick_setup_configure(self) -> Any:
        return await self._handle("quick_setup_configure")

    async def quick_setup_web_configure(self) -> Any:
        return await self._handle("quick_setup_web_configure")

    async def quick_setup_sticker_configure(self) -> Any:
        return await self._handle("quick_setup_sticker_configure")

    async def quick_setup_life_configure(self) -> Any:
        return await self._handle("quick_setup_life_configure")

    async def quick_setup_contact_configure(self) -> Any:
        return await self._handle("quick_setup_contact_configure")

    async def quick_setup_character_generate(self) -> Any:
        return await self._handle("quick_setup_character_generate")

    async def quick_setup_decision(self) -> Any:
        return await self._handle("quick_setup_decision")

    async def quick_setup_finish(self) -> Any:
        return await self._handle("quick_setup_finish")

    async def schema_recovery(self) -> Any:
        return await self._handle("schema_recovery")

    async def schema_recovery_action(self) -> Any:
        return await self._handle("schema_recovery_action")

    async def settings_snapshot(self) -> Any:
        return await self._handle("settings_snapshot")

    async def settings_section(self) -> Any:
        return await self._handle("settings_section")

    async def identity_annotations(self) -> Any:
        return await self._handle("identity_annotations")

    async def instance_workspace(self) -> Any:
        return await self._handle("instance_workspace")

    async def background_workspace(self) -> Any:
        return await self._handle("background_workspace")

    async def background_action(self) -> Any:
        return await self._handle("background_action")

    async def delivery_failure_acknowledge(self) -> Any:
        return await self._handle("delivery_failure_acknowledge")

    async def player_profile_snapshot(self) -> Any:
        return await self._handle("player_profile_snapshot")

    async def player_profile_entry(self) -> Any:
        return await self._handle("player_profile_entry")

    async def player_profile_action(self) -> Any:
        return await self._handle("player_profile_action")

    async def ai_work_records(self) -> Any:
        return await self._handle("ai_work_records")

    async def ai_work_record(self) -> Any:
        return await self._handle("ai_work_record")

    async def ai_work_attempt_debug(self) -> Any:
        return await self._handle("ai_work_attempt_debug")

    async def ai_work_attempt_raw(self) -> Any:
        return await self._handle("ai_work_attempt_raw")

    async def console_profile(self) -> Any:
        return await self._handle("save_console_profile")

    async def instance_contact_override(self) -> Any:
        return await self._handle("get_instance_contact_override")

    async def platform_contact_policy(self) -> Any:
        return await self._handle("get_platform_contact_policy")

    async def instances(self) -> Any:
        return await self._handle("instances")

    async def support_bundle(self) -> Any:
        return await self._handle("support_bundle")

    async def context_summary(self) -> Any:
        return await self._handle("context_summary")

    async def context_dry_run(self) -> Any:
        return await self._handle("context_dry_run")

    async def knowledge_snapshot(self) -> Any:
        return await self._handle("knowledge_snapshot")

    async def knowledge_form(self) -> Any:
        return await self._handle("knowledge_form")

    async def recall_probe(self) -> Any:
        return await self._handle("recall_probe")

    async def recall_benchmark(self) -> Any:
        return await self._handle("recall_benchmark")

    async def recall_configuration(self) -> Any:
        return await self._handle("recall_configuration")

    async def recall_configuration_update(self) -> Any:
        return await self._handle("recall_configuration_update")

    async def recall_rebuild(self) -> Any:
        return await self._handle("recall_rebuild")

    async def recall_integrity(self) -> Any:
        return await self._handle("recall_integrity")

    async def knowledge_record(self) -> Any:
        return await self._handle("knowledge_record")

    async def character_intent_action(self) -> Any:
        return await self._handle("character_intent_action")

    async def image_snapshot(self) -> Any:
        return await self._handle("image_snapshot")

    async def image_preview(self) -> Any:
        return await self._handle("image_preview")

    async def image_download(self) -> Any:
        return await self._handle_download("image_download")

    async def file_artifacts(self) -> Any:
        return await self._handle("file_artifacts")

    async def file_artifact_action(self) -> Any:
        return await self._handle("file_artifact_action")

    async def web_provider(self) -> Any:
        return await self._handle("web_provider")

    async def web_provider_credential(self) -> Any:
        return await self._handle("web_provider_credential")

    async def web_provider_probe(self) -> Any:
        return await self._handle("web_provider_probe")

    async def web_snapshot(self) -> Any:
        return await self._handle("web_snapshot")

    async def sticker_snapshot(self) -> Any:
        return await self._handle("sticker_snapshot")

    async def timer_lifecycle_snapshot(self) -> Any:
        return await self._handle("timer_lifecycle_snapshot")

    async def sticker_action(self) -> Any:
        return await self._handle("sticker_action")

    async def sticker_run(self) -> Any:
        return await self._handle("sticker_run")

    async def sticker_stop(self) -> Any:
        return await self._handle("sticker_stop")

    async def sticker_references(self) -> Any:
        return await self._handle("sticker_reference_action")

    async def sticker_intake(self) -> Any:
        return await self._handle("sticker_intake")

    async def sticker_intake_start(self) -> Any:
        return await self._handle("sticker_intake_start")

    async def sticker_intake_upload(self) -> Any:
        return await self._handle("sticker_intake_upload")

    async def sticker_intake_action(self) -> Any:
        return await self._handle("sticker_intake_action")

    async def sticker_intake_preview(self) -> Any:
        return await self._handle("sticker_intake_preview")

    async def ai_api_packages(self) -> Any:
        return await self._handle("ai_api_packages")

    async def ai_api_package(self) -> Any:
        return await self._handle("ai_api_package")

    async def ai_api_package_credential(self) -> Any:
        return await self._handle("ai_api_package_credential")

    async def ai_api_model(self) -> Any:
        return await self._handle("ai_api_model")

    async def ai_api_package_probe(self) -> Any:
        return await self._handle("ai_api_package_probe")

    async def ai_api_model_probe(self) -> Any:
        return await self._handle("ai_api_model_probe")

    async def reset_instance(self) -> Any:
        return await self._handle("reset_instance")

    async def _handle(self, action: str, *, web: Any | None = None) -> Any:
        web = web or _web_api()
        try:
            _reject_plugin_page_asset_token(web.request)
            payload = await _request_payload(web.request)
            result: Any = self.callbacks.call(action, payload)
            if inspect.isawaitable(result):
                result = await result
            return web.json_response(result)
        except ValueError as exc:
            status_code = 400
            if action in _BOOTSTRAP_READ_ACTIONS and not isinstance(exc, ConsoleValidationError):
                status_code = 500
                _page_logger().error(
                    f"[SoulCore] Page bootstrap read failed "
                    f"action={action} error={type(exc).__name__}",
                    exc_info=True,
                )
            return web.json_response(
                {
                    "ok": False,
                    "error": console_error_envelope(
                        exc,
                        action=action,
                        status_code=status_code,
                    ),
                }
            )
        except Exception as exc:
            _page_logger().error(
                f"[SoulCore] Page action failed action={action} error={type(exc).__name__}",
                exc_info=True,
            )
            return web.json_response(
                {
                    "ok": False,
                    "error": console_error_envelope(exc, action=action, status_code=500),
                }
            )

    async def _handle_download(self, action: str, *, web: Any | None = None) -> Any:
        web = web or _web_api()
        try:
            _reject_plugin_page_asset_token(web.request)
            payload = await _request_payload(web.request)
            result: Any = self.callbacks.download(action, payload)
            if inspect.isawaitable(result):
                result = await result
            response = web.file_response(
                result.path,
                filename=result.filename,
                content_type=result.content_type,
                headers=dict(result.headers),
            )
            if inspect.isawaitable(response):
                response = await response
            return response
        except ValueError as exc:
            return web.json_response(
                {
                    "ok": False,
                    "error": console_error_envelope(exc, action=action, status_code=400),
                },
                status_code=400,
                headers={"Cache-Control": "private, no-store"},
            )
        except Exception as exc:
            _page_logger().error(
                f"[SoulCore] Page download failed action={action} error={type(exc).__name__}",
                exc_info=True,
            )
            return web.json_response(
                {
                    "ok": False,
                    "error": console_error_envelope(exc, action=action, status_code=500),
                },
                status_code=500,
                headers={"Cache-Control": "private, no-store"},
            )


def _remove_previous_page_routes(context: Any) -> None:
    """Replace SoulCore's Page contract as one unit across plugin reloads.

    AstrBot replaces a registered Web API only when both its path and the
    complete methods list are equal. A route changed from ``GET+POST`` to
    ``GET`` would therefore leave the terminated plugin instance first in the
    registry and keep dispatching requests to its closed runtime. AstrBot has
    no public unregister API, so remove only SoulCore's own Page namespace
    before registering the current, sole contract.
    """

    registered = getattr(context, "registered_web_apis", None)
    if not isinstance(registered, list):
        return
    route_prefix = f"{PAGE_API_PREFIX}/"
    registered[:] = [
        item
        for item in registered
        if not (
            isinstance(item, tuple)
            and item
            and isinstance(item[0], str)
            and item[0].startswith(route_prefix)
        )
    ]


async def _request_payload(request: Any) -> dict[str, Any]:
    if getattr(request, "method", "GET").upper() in {"POST", "PUT", "PATCH"}:
        get_json = getattr(request, "get_json", None)
        if callable(get_json):
            value = await get_json(silent=True)
            if value is None:
                value = {}
        else:
            json_reader = getattr(request, "json", None)
            if not callable(json_reader):
                raise RuntimeError("AstrBot request JSON API is unavailable")
            value = await json_reader(default={})
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value
    query = getattr(request, "query", None)
    if query is None:
        query = getattr(request, "args", {})
    if hasattr(query, "to_dict"):
        return dict(query.to_dict())
    items = getattr(query, "items", None)
    if callable(items):
        return dict(items())
    if isinstance(query, Mapping):
        return dict(query)
    try:
        return dict(query)
    except (TypeError, ValueError):
        return {}


def _web_api() -> Any:
    try:
        return import_module("astrbot.api.web")
    except ModuleNotFoundError as exc:
        if exc.name != "astrbot.api.web":
            raise
        return _quart_web_api()


class _QuartWebApi:
    """Bridge AstrBot 4.24-4.25's Quart plugin routes to the Page contract."""

    def __init__(self, request: Any, jsonify: Any, send_file: Any) -> None:
        self.request = request
        self._jsonify = jsonify
        self._send_file = send_file

    def json_response(
        self,
        data: Any = None,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        response = self._jsonify({} if data is None else data)
        response.status_code = status_code
        if headers:
            response.headers.update(headers)
        return response

    async def file_response(
        self,
        path: Any,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        response = await self._send_file(
            str(path),
            mimetype=content_type,
            as_attachment=bool(filename),
            attachment_filename=filename,
        )
        if headers:
            response.headers.update(headers)
        return response


def _quart_web_api() -> _QuartWebApi:
    try:
        from quart import jsonify, request, send_file
    except ImportError as exc:
        raise RuntimeError("AstrBot web API is unavailable") from exc
    return _QuartWebApi(request, jsonify, send_file)


def _page_logger() -> Any:
    try:
        from astrbot.api import logger
    except ImportError as exc:
        raise RuntimeError("AstrBot logger is unavailable") from exc
    return logger


def _reject_plugin_page_asset_token(request: Any) -> None:
    """Reject AstrBot's short-lived page-asset JWT at the state-changing API.

    AstrBot authenticates the request before dispatching it to this handler. Its
    current scope resolver nevertheless treats a signed page-asset token as a
    full dashboard JWT. Inspecting the already-authenticated token is therefore
    a deliberately negative defense: it never grants access and only rejects
    the token type that is scoped to static page assets.
    """

    token = _request_bearer_or_dashboard_cookie(request)
    if not token:
        return
    payload = _unverified_jwt_payload(token)
    if payload is not None and payload.get("token_type") == "plugin_page_asset":
        raise ConsoleValidationError("页面资源令牌不能调用 SoulCore 管理接口")


def _request_bearer_or_dashboard_cookie(request: Any) -> str:
    headers = request.headers
    authorization = str(headers.get("Authorization", "") or "").strip()
    if authorization.startswith("Bearer "):
        return authorization.removeprefix("Bearer ").strip()
    cookies = request.cookies
    return str(cookies.get("astrbot_dashboard_jwt", "") or "").strip()


def _unverified_jwt_payload(token: str) -> Mapping[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    encoded = parts[1]
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None
