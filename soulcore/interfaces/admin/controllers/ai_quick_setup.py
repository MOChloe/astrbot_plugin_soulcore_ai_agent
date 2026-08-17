"""New-user model setup composed from the existing AI administration surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....features.ai.model_parameters import (
    DEFAULT_MODEL_MAX_CONTEXT_TOKENS,
    MINIMUM_MODEL_MAX_CONTEXT_TOKENS,
)
from .ai_quick_setup_shared import (
    FAST_CAPABILITIES,
    IMAGE_PROTOCOLS,
    MAIN_CAPABILITIES,
    SLOT_CAPABILITIES,
    SLOT_PROBE_CAPABILITY,
    TEXT_PROTOCOLS,
    ConfigurationPort,
    PoolWriter,
    ProbePort,
    RepositoryPort,
)
from .ai_quick_setup_sources import AIQuickSetupSourceMixin


class AIQuickSetupViewMixin:
    @staticmethod
    def _model_views(packages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for package in packages:
            for model in package.get("models") or ():
                if isinstance(model, Mapping):
                    result.append(_quick_setup_model_view(package, model))
        return result

    @classmethod
    def _slot_view(
        cls,
        slot: str,
        orders: Mapping[str, Any],
        by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        primary_capability = SLOT_CAPABILITIES[slot][0]
        backend_ids = cls._order_backend_ids(orders.get(primary_capability) or ())
        model = dict(by_id.get(backend_ids[0], {})) if backend_ids else None
        return {
            "configured": model is not None,
            "model": model,
            "backend_ids": backend_ids,
        }

    @classmethod
    def _model_payload(
        cls,
        model: Mapping[str, Any],
        package: Mapping[str, Any],
        capabilities: tuple[Any, ...],
        *,
        expected_version: Any = None,
    ) -> dict[str, Any]:
        config = dict(model.get("config") or {})
        return {
            "package_id": str(package["package_id"]),
            "backend_id": str(model["backend_id"]),
            "expected_version": (
                model.get("version") if expected_version is None else expected_version
            ),
            "model_key": str(model["model_key"]),
            "display_name": str(model.get("display_name") or model["model_key"]),
            "capabilities": cls._unique(capabilities),
            "priority": int(model.get("priority") or 1),
            "enabled": bool(model.get("enabled", True)),
            "config": config,
            "supports_vision": bool(config.get("supports_vision")),
            "max_context_tokens": int(
                config.get("max_context_tokens") or DEFAULT_MODEL_MAX_CONTEXT_TOKENS
            ),
            "image_generation_mode": str(config.get("image_generation_mode") or "IMAGES_API"),
        }

    @staticmethod
    def _order_backend_ids(values: Any) -> list[str]:
        result: list[str] = []
        for item in values or ():
            backend_id = (
                str(item.get("backend_id") or "") if isinstance(item, Mapping) else str(item or "")
            ).strip()
            if backend_id and backend_id not in result:
                result.append(backend_id)
        return result

    @staticmethod
    def _redact_probe(
        candidate: Mapping[str, Any], probe: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if probe is None:
            return None
        result = dict(probe)
        error = str(result.get("error") or "")
        for value in candidate.get("redaction_values") or ():
            secret = str(value or "")
            if secret:
                error = error.replace(secret, "[已隐藏]")
        if error:
            result["error"] = error
        return result

    @classmethod
    def _failed_result(
        cls,
        slot: str,
        candidate: Mapping[str, Any],
        *,
        probe: Mapping[str, Any] | None = None,
        exc: Exception | None = None,
    ) -> dict[str, Any]:
        safe_probe = cls._redact_probe(candidate, probe) or {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}" if exc is not None else "连接测试失败",
        }
        safe_probe = cls._redact_probe(candidate, safe_probe) or {"ok": False}
        return {
            "ok": True,
            "applied": False,
            "slot": slot,
            "probe": safe_probe,
            "can_skip": slot != "main",
        }

    @staticmethod
    def _slot(payload: Mapping[str, Any]) -> str:
        slot = str(payload.get("slot") or "").strip().lower()
        if slot not in SLOT_CAPABILITIES:
            raise ValueError("快速设置步骤无效")
        return slot

    @staticmethod
    def _unique(values: tuple[Any, ...]) -> list[str]:
        return list(
            dict.fromkeys(str(item).strip().lower() for item in values if str(item).strip())
        )


def _quick_setup_model_view(package: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    protocol = str(package.get("protocol") or "").upper()
    max_context_tokens = int(
        model.get("max_context_tokens")
        or dict(model.get("config") or {}).get("max_context_tokens")
        or DEFAULT_MODEL_MAX_CONTEXT_TOKENS
    )
    context_compatible = max_context_tokens >= MINIMUM_MODEL_MAX_CONTEXT_TOKENS
    usages = _quick_setup_model_usages(protocol, context_compatible, model)
    return {
        "backend_id": str(model.get("backend_id") or ""),
        "package_id": str(package.get("package_id") or ""),
        "display_name": str(model.get("display_name") or model.get("model_key") or "模型"),
        "model_key": str(model.get("model_key") or ""),
        "protocol": protocol,
        "supports_vision": bool(model.get("supports_vision")),
        "max_context_tokens": max_context_tokens,
        "context_compatible": context_compatible,
        "enabled": bool(model.get("enabled", True)) and bool(package.get("enabled", True)),
        "status": str(model.get("status") or "CONFIGURED"),
        "usages": usages,
    }


def _quick_setup_model_usages(
    protocol: str, context_compatible: bool, model: Mapping[str, Any]
) -> list[str]:
    text_ready = protocol in TEXT_PROTOCOLS and context_compatible
    usages = ["text"] if text_ready else []
    if text_ready and bool(model.get("supports_vision")):
        usages.append("vision")
    if protocol in IMAGE_PROTOCOLS:
        usages.append("image")
    return usages


class AIQuickSetupController(AIQuickSetupSourceMixin, AIQuickSetupViewMixin):
    """Offer one guided view while retaining the existing configuration as truth."""

    def __init__(
        self,
        repository: RepositoryPort,
        configuration: ConfigurationPort,
        probes: ProbePort,
        pool_writer: PoolWriter,
        runtime_context: Any | None,
    ) -> None:
        self.repository = repository
        self.configuration = configuration
        self.probes = probes
        self.pool_writer = pool_writer
        self.runtime_context = runtime_context

    async def snapshot(self, profile_id: str) -> dict[str, Any]:
        configuration = await self.configuration.snapshot(profile_id)
        packages = list(configuration.get("api_packages") or [])
        orders = dict(configuration.get("effective_orders") or {})
        models = self._model_views(packages)
        by_id = {str(item["backend_id"]): item for item in models}
        slots = {slot: self._slot_view(slot, orders, by_id) for slot in SLOT_CAPABILITIES}
        return {
            "returning": any(value["configured"] for value in slots.values()),
            "slots": slots,
            "existing_models": models,
            "astrbot_sources": self._astrbot_sources(),
        }

    async def configure(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        slot = self._slot(payload)
        action = str(payload.get("action") or "configure").strip().lower()
        action_result = await self._configuration_action(profile_id, slot, action, payload)
        if action_result is not None:
            return action_result
        before = await self.configuration.snapshot(profile_id)
        candidate, failure = await self._resolved_candidate_result(
            profile_id, slot, payload, before
        )
        if failure is not None:
            return failure
        assert candidate is not None
        model_before = await self.repository.get_ai_api_model(candidate["backend_id"])
        touched_orders = self._touched_pool_orders(before, slot, candidate, model_before)
        saved, probe, failure = await self._prepared_probe_result(
            profile_id,
            slot,
            payload,
            candidate,
            model_before,
            touched_orders,
        )
        if failure is not None:
            return failure
        assert saved is not None and probe is not None
        vision_probe, failure = await self._activation_result(
            profile_id,
            slot,
            candidate,
            model_before,
            saved,
            before,
            touched_orders,
        )
        if failure is not None:
            return failure
        return {
            "ok": True,
            "applied": True,
            "slot": slot,
            "backend_id": str(saved["backend_id"]),
            "probe": probe,
            "vision_probe": self._redact_probe(candidate, vision_probe),
        }

    async def _configuration_action(
        self,
        profile_id: str,
        slot: str,
        action: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if action == "configure":
            return None
        if action == "raise_context":
            if slot != "main":
                raise ValueError("只有主力模型可以在这里调整上下文上限")
            return await self._raise_main_context(profile_id, payload)
        if action == "disable":
            if slot == "main":
                raise ValueError("主力模型不能在快速设置中关闭")
            before = await self.configuration.snapshot(profile_id)
            await self._disable_slot(profile_id, slot, before)
            return {"ok": True, "applied": True, "slot": slot, "action": action}
        if action == "use_main":
            if slot != "fast":
                raise ValueError("只有快速模型可以直接改用主力模型")
            return await self._use_main(profile_id)
        raise ValueError("不支持的快速设置操作")

    async def _resolved_candidate_result(
        self,
        profile_id: str,
        slot: str,
        payload: Mapping[str, Any],
        before: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        try:
            return await self._resolve_candidate(profile_id, slot, payload, before), None
        except ValueError as exc:
            selection = payload.get("selection")
            secret = str(selection.get("secret") or "") if isinstance(selection, Mapping) else ""
            failure = self._failed_result(
                slot,
                {"redaction_values": [secret] if secret else []},
                probe={"ok": False, "error": str(exc)},
            )
            return None, failure

    async def _prepared_probe_result(
        self,
        profile_id: str,
        slot: str,
        payload: Mapping[str, Any],
        candidate: Mapping[str, Any],
        model_before: Mapping[str, Any] | None,
        touched_orders: Mapping[str, list[str]],
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        saved: dict[str, Any] | None = None
        try:
            saved = await self._prepare_model(profile_id, slot, candidate, model_before)
            # save_model seeds every declared pool; keep live routing unchanged
            # until the explicit backend probe succeeds.
            await self._restore_pool_orders(profile_id, touched_orders)
            probe = await self.probes.probe_model(
                str(saved["backend_id"]),
                profile_id,
                {
                    "capability": SLOT_PROBE_CAPABILITY[slot],
                    "confirm_cost": payload.get("confirm_cost") is True,
                },
            )
        except Exception as exc:
            await self._restore_candidate(
                profile_id, candidate, model_before, saved, touched_orders
            )
            return None, None, self._failed_result(slot, candidate, exc=exc)
        if not bool(probe.get("ok")):
            await self._restore_candidate(
                profile_id, candidate, model_before, saved, touched_orders
            )
            return None, None, self._failed_result(slot, candidate, probe=probe)
        return saved, probe, None

    async def _activation_result(
        self,
        profile_id: str,
        slot: str,
        candidate: Mapping[str, Any],
        model_before: Mapping[str, Any] | None,
        saved: Mapping[str, Any],
        before: Mapping[str, Any],
        touched_orders: Mapping[str, list[str]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        try:
            vision_probe, vision_ready = await self._probe_main_vision(
                profile_id, slot, candidate, model_before, saved
            )
            await self._restore_pool_orders(profile_id, touched_orders)
            extra = ("vision.describe",) if slot == "main" and vision_ready else ()
            await self._assign_slot(
                profile_id,
                slot,
                str(saved["backend_id"]),
                before,
                extra_capabilities=extra,
            )
            return vision_probe, None
        except Exception as exc:
            await self._restore_candidate(
                profile_id, candidate, model_before, saved, touched_orders
            )
            return None, self._failed_result(slot, candidate, exc=exc)

    async def _probe_main_vision(
        self,
        profile_id: str,
        slot: str,
        candidate: Mapping[str, Any],
        model_before: Mapping[str, Any] | None,
        saved: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, bool]:
        if slot != "main" or not bool(candidate.get("supports_vision")):
            return None, False
        backend_id = str(saved["backend_id"])
        probe = await self.probes.probe_model(
            backend_id, profile_id, {"capability": "vision.describe"}
        )
        ready = bool(probe.get("ok"))
        if not ready:
            await self._drop_failed_candidate_vision(profile_id, backend_id, model_before, saved)
        return probe, ready

    async def _drop_failed_candidate_vision(
        self,
        profile_id: str,
        backend_id: str,
        model_before: Mapping[str, Any] | None,
        saved: Mapping[str, Any],
    ) -> None:
        current = await self.repository.get_ai_api_model(backend_id)
        package = await self.repository.get_ai_api_package(
            str(saved["package_id"]), profile_id=profile_id
        )
        if current is None or package is None:
            return
        original_capabilities = {
            str(item) for item in (model_before or {}).get("capabilities") or ()
        }
        keep_declared_vision = bool(
            dict((model_before or {}).get("config") or {}).get("supports_vision")
        )
        capabilities = tuple(
            item
            for item in current.get("capabilities") or ()
            if str(item) != "vision.describe" or "vision.describe" in original_capabilities
        )
        payload = self._model_payload(current, package, capabilities)
        payload["supports_vision"] = keep_declared_vision
        await self.configuration.save_model(payload, profile_id)

    async def _raise_main_context(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        requested = _requested_main_context(payload.get("max_context_tokens"))
        backend_id, model, package, current = await self._main_context_model(profile_id)
        adjusted = max(current, requested)
        if adjusted > current:
            model_payload = self._model_payload(
                model, package, tuple(model.get("capabilities") or ())
            )
            model_payload["max_context_tokens"] = adjusted
            await self.configuration.save_model(model_payload, profile_id)
        return {
            "ok": True,
            "applied": adjusted > current,
            "slot": "main",
            "action": "raise_context",
            "backend_id": backend_id,
            "previous_max_context_tokens": current,
            "max_context_tokens": adjusted,
        }

    async def _main_context_model(
        self, profile_id: str
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any], int]:
        before = await self.configuration.snapshot(profile_id)
        main_ids = self._order_backend_ids(
            dict(before.get("effective_orders") or {}).get("chat.completion") or ()
        )
        if not main_ids:
            raise ValueError("请先完成主力模型设置")
        backend_id = main_ids[0]
        model = await self.repository.get_ai_api_model(backend_id)
        if model is None:
            raise ValueError("当前主力模型已经不存在")
        package = await self.repository.get_ai_api_package(
            str(model.get("package_id") or ""), profile_id=profile_id
        )
        if package is None:
            raise ValueError("当前主力模型不属于这个角色")
        current = int(
            dict(model.get("config") or {}).get("max_context_tokens")
            or DEFAULT_MODEL_MAX_CONTEXT_TOKENS
        )
        return backend_id, model, package, current

    async def _use_main(self, profile_id: str) -> dict[str, Any]:
        before = await self.configuration.snapshot(profile_id)
        main_ids = self._order_backend_ids(
            dict(before.get("effective_orders") or {}).get("chat.completion") or []
        )
        if not main_ids:
            raise ValueError("请先完成主力模型设置")
        backend_id = str(main_ids[0])
        model = await self.repository.get_ai_api_model(backend_id)
        if model is None:
            raise ValueError("当前主力模型已经不存在")
        package = await self.repository.get_ai_api_package(
            str(model["package_id"]), profile_id=profile_id
        )
        if package is None:
            raise ValueError("当前主力模型不属于这个角色")
        touched_orders = self._touched_pool_orders(
            before, "fast", {"supports_vision": False}, model
        )
        saved = await self.configuration.save_model(
            self._model_payload(
                model,
                package,
                (*model.get("capabilities", ()), *FAST_CAPABILITIES),
            ),
            profile_id,
        )
        try:
            await self._restore_pool_orders(profile_id, touched_orders)
            await self._assign_slot(profile_id, "fast", backend_id, before)
        except Exception:
            current = dict(saved.get("model") or {})
            await self.configuration.save_model(
                self._model_payload(
                    model,
                    package,
                    tuple(model.get("capabilities") or ()),
                    expected_version=current.get("version"),
                ),
                profile_id,
            )
            await self._restore_pool_orders(profile_id, touched_orders)
            raise
        return {
            "ok": True,
            "applied": True,
            "slot": "fast",
            "action": "use_main",
            "backend_id": backend_id,
        }

    async def _prepare_model(
        self,
        profile_id: str,
        slot: str,
        candidate: Mapping[str, Any],
        existing: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload = self._candidate_model_payload(slot, candidate, existing)
        result = await self.configuration.save_model(payload, profile_id)
        return {
            "backend_id": str(result["backend_id"]),
            "package_id": str(candidate["package_id"]),
            "model": dict(result.get("model") or {}),
        }

    def _candidate_model_payload(
        self,
        slot: str,
        candidate: Mapping[str, Any],
        existing: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        package = candidate["package"]
        slot_capabilities = _candidate_slot_capabilities(slot, candidate)
        payload = {
            "package_id": str(candidate["package_id"]),
            "backend_id": str(candidate["backend_id"]),
            "expected_version": (existing or {}).get("version"),
            "model_key": str(candidate["model_key"]),
            "display_name": str(candidate["display_name"]),
            "capabilities": self._unique(
                (*((existing or {}).get("capabilities") or ()), *slot_capabilities)
            ),
            "priority": int((existing or {}).get("priority") or 999),
            "enabled": True,
            "supports_vision": bool(candidate.get("supports_vision")),
            "max_context_tokens": int(
                candidate.get("max_context_tokens") or DEFAULT_MODEL_MAX_CONTEXT_TOKENS
            ),
            "image_generation_mode": str(candidate.get("image_generation_mode") or "IMAGES_API"),
            "config": dict((existing or {}).get("config") or {}),
        }
        _apply_candidate_model_options(payload, candidate, package)
        return payload

    async def _restore_after_failure(
        self,
        profile_id: str,
        saved: Mapping[str, Any],
        original: Mapping[str, Any] | None,
        candidate: Mapping[str, Any],
    ) -> None:
        # Manual/AstrBot candidates update their package before probing. Restore
        # that package first so an existing model is validated against its old protocol.
        await self._restore_package_and_credential(profile_id, candidate)
        current = dict(saved.get("model") or {})
        if original is None:
            await self.configuration.save_model(
                {
                    "action": "disable",
                    "package_id": str(saved["package_id"]),
                    "backend_id": str(saved["backend_id"]),
                    "expected_version": current.get("version"),
                },
                profile_id,
            )
            return
        package = await self.repository.get_ai_api_package(
            str(original["package_id"]), profile_id=profile_id
        )
        if package is not None:
            await self.configuration.save_model(
                self._model_payload(
                    original,
                    package,
                    tuple(original.get("capabilities") or ()),
                    expected_version=current.get("version"),
                ),
                profile_id,
            )

    async def _restore_candidate(
        self,
        profile_id: str,
        candidate: Mapping[str, Any],
        original: Mapping[str, Any] | None,
        saved: Mapping[str, Any] | None,
        touched_orders: Mapping[str, list[str]],
    ) -> None:
        current = await self.repository.get_ai_api_model(str(candidate["backend_id"]))
        effective_saved = saved
        if current is not None:
            effective_saved = {
                "backend_id": str(candidate["backend_id"]),
                "package_id": str(candidate["package_id"]),
                "model": dict(current),
            }
        if effective_saved is not None:
            await self._restore_after_failure(profile_id, effective_saved, original, candidate)
        else:
            await self._restore_package_and_credential(profile_id, candidate)
        await self._restore_pool_orders(profile_id, touched_orders)

    async def _assign_slot(
        self,
        profile_id: str,
        slot: str,
        backend_id: str,
        before: Mapping[str, Any],
        *,
        extra_capabilities: tuple[str, ...] = (),
    ) -> None:
        orders = dict(before.get("effective_orders") or {})
        capabilities = self._unique((*SLOT_CAPABILITIES[slot], *extra_capabilities))
        original_orders = {
            capability: self._order_backend_ids(orders.get(capability) or ())
            for capability in capabilities
        }
        try:
            for capability in capabilities:
                existing = original_orders[capability]
                prior = existing[0] if existing else ""
                remaining = [item for item in existing if item not in {backend_id, prior}]
                await self.pool_writer(
                    {"capability": capability, "backend_ids": [backend_id, *remaining]},
                    profile_id,
                )
        except Exception:
            await self._restore_pool_orders(profile_id, original_orders)
            raise

    async def _disable_slot(self, profile_id: str, slot: str, before: Mapping[str, Any]) -> None:
        orders = dict(before.get("effective_orders") or {})
        original_orders = {
            capability: self._order_backend_ids(orders.get(capability) or ())
            for capability in SLOT_CAPABILITIES[slot]
        }
        try:
            for capability in SLOT_CAPABILITIES[slot]:
                await self.pool_writer({"capability": capability, "backend_ids": []}, profile_id)
        except Exception:
            await self._restore_pool_orders(profile_id, original_orders)
            raise

    async def _restore_pool_orders(self, profile_id: str, orders: Mapping[str, list[str]]) -> None:
        for capability, backend_ids in orders.items():
            await self.pool_writer(
                {"capability": capability, "backend_ids": backend_ids}, profile_id
            )

    @classmethod
    def _touched_pool_orders(
        cls,
        snapshot: Mapping[str, Any],
        slot: str,
        candidate: Mapping[str, Any],
        original: Mapping[str, Any] | None,
    ) -> dict[str, list[str]]:
        capabilities = cls._unique(
            (
                *((original or {}).get("capabilities") or ()),
                *SLOT_CAPABILITIES[slot],
                *(
                    ("vision.describe",)
                    if slot == "main" and bool(candidate.get("supports_vision"))
                    else ()
                ),
            )
        )
        orders = dict(snapshot.get("effective_orders") or {})
        return {
            capability: cls._order_backend_ids(orders.get(capability) or ())
            for capability in capabilities
        }


def _candidate_slot_capabilities(slot: str, candidate: Mapping[str, Any]) -> tuple[str, ...]:
    capabilities = SLOT_CAPABILITIES[slot]
    if slot == "main" and bool(candidate.get("supports_vision")):
        return (*capabilities, "vision.describe")
    return capabilities


def _apply_candidate_model_options(
    payload: dict[str, Any],
    candidate: Mapping[str, Any],
    package: Mapping[str, Any],
) -> None:
    if "generation_parameters" in candidate:
        payload["generation_parameters"] = dict(candidate.get("generation_parameters") or {})
    if str(package.get("protocol") or "").upper() == "GEMINI":
        payload["supports_vision"] = False


def _requested_main_context(value: Any) -> int:
    text = str(value or "").strip()
    if not text.isascii() or not text.isdigit():
        raise ValueError("建议上下文上限必须是正整数")
    requested = int(text)
    if requested < MINIMUM_MODEL_MAX_CONTEXT_TOKENS or requested > 10_000_000:
        raise ValueError("建议上下文上限超出可配置范围")
    return requested


__all__ = [
    "AIQuickSetupController",
    "FAST_CAPABILITIES",
    "MAIN_CAPABILITIES",
    "SLOT_CAPABILITIES",
]
