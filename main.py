"""SoulCore AstrBot plugin entry point.

Only AstrBot-owned decorators and signatures live here. Runtime assembly and
all product behaviour are delegated to the bootstrap application host.
"""

from __future__ import annotations

import asyncio
from typing import Any

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.star import Context, Star

from .soulcore.bootstrap.entry_runtime import PluginRuntimeHost
from .soulcore.interfaces.admin import PageApiFacade
from .soulcore.interfaces.astrbot.qq_reference_ids import (
    bind_qq_reference_id_capture,
    install_qq_reference_id_capture,
    uninstall_qq_reference_id_capture,
)


class SoulCorePlugin(Star):
    def __init__(self, context: Context, config: Any) -> None:
        super().__init__(context)
        install_qq_reference_id_capture()
        bind_qq_reference_id_capture(context)
        self._qq_reference_bind_task: asyncio.Task[None] | None = None
        self.runtime = PluginRuntimeHost(context, config)
        self.page_api = PageApiFacade(self.runtime)
        if hasattr(context, "register_web_api"):
            self.page_api.register(context)

    async def initialize(self) -> None:
        bind_qq_reference_id_capture(self.context)
        await self.runtime.initialize()
        if not self.runtime.running:
            return
        if self._qq_reference_bind_task is not None and not self._qq_reference_bind_task.done():
            return
        self._qq_reference_bind_task = asyncio.create_task(
            self._bind_qq_reference_capture_after_platform_start(),
            name="soulcore-qq-reference-bind",
        )

    async def terminate(self) -> None:
        failures: list[Exception] = []
        cancellation: asyncio.CancelledError | None = None
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        if self._qq_reference_bind_task is not None:
            bind_task = self._qq_reference_bind_task
            bind_task.cancel()
            while True:
                try:
                    await asyncio.shield(bind_task)
                    break
                except asyncio.CancelledError as exc:
                    current_cancellations = current.cancelling()
                    if current_cancellations > observed_cancellations:
                        cancellation = cancellation or exc
                        observed_cancellations = current_cancellations
                        continue
                    if bind_task.done():
                        break
                except Exception as exc:
                    failures.append(exc)
                    break
            self._qq_reference_bind_task = None
        runtime_cancellation = await self._drain_runtime_termination(
            current,
            observed_cancellations,
            failures,
        )
        uninstall_qq_reference_id_capture(self.context)
        cancellation = cancellation or runtime_cancellation
        if cancellation is not None:
            if failures:
                raise cancellation from ExceptionGroup(
                    "SoulCore plugin termination failed", failures
                )
            raise cancellation
        if failures:
            raise ExceptionGroup("SoulCore plugin termination failed", failures)

    async def _drain_runtime_termination(
        self,
        current: asyncio.Task[Any],
        observed_cancellations: int,
        failures: list[Exception],
    ) -> asyncio.CancelledError | None:
        cancellation: asyncio.CancelledError | None = None
        runtime_cleanup = asyncio.create_task(
            self.runtime.terminate(),
            name="soulcore-plugin-runtime-terminate",
        )
        while True:
            try:
                await asyncio.shield(runtime_cleanup)
                break
            except asyncio.CancelledError as exc:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    cancellation = cancellation or exc
                    observed_cancellations = current_cancellations
                    continue
                failure = RuntimeError("SoulCore runtime termination task was cancelled")
                failure.__cause__ = exc
                failures.append(failure)
                break
            except Exception as exc:
                failures.append(exc)
                break
        return cancellation

    async def _bind_qq_reference_capture_after_platform_start(self) -> None:
        """Bind parser dictionaries created after plugins during a cold boot."""

        for _ in range(120):
            result = bind_qq_reference_id_capture(self.context)
            if result["states"] > 0:
                return
            await asyncio.sleep(0.25)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        return await self.runtime.on_message(event)

    @filter.command_group("soulcore")
    def soulcore(self):
        """SoulCore administrator diagnostics."""
        pass

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("help")
    async def cmd_help(self, event: AstrMessageEvent):
        async for result in self.runtime.commands.cmd_help(event):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("doctor")
    async def cmd_doctor(self, event: AstrMessageEvent):
        async for result in self.runtime.commands.cmd_doctor(event):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("profiles")
    async def cmd_profiles(self, event: AstrMessageEvent):
        async for result in self.runtime.commands.cmd_profiles(event):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("status")
    async def cmd_status(self, event: AstrMessageEvent, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_status(event, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("state")
    async def cmd_state(self, event: AstrMessageEvent, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_state(event, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("targets")
    async def cmd_targets(self, event: AstrMessageEvent, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_targets(event, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("routes")
    async def cmd_routes(self, event: AstrMessageEvent, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_routes(event, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("instances")
    async def cmd_instances(self, event: AstrMessageEvent, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_instances(event, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("schedules")
    async def cmd_schedules(self, event: AstrMessageEvent, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_schedules(event, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("runs")
    async def cmd_runs(self, event: AstrMessageEvent, profile_id: str = "", limit: int = 10):
        async for result in self.runtime.commands.cmd_runs(event, profile_id, limit):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("run")
    async def cmd_run(self, event: AstrMessageEvent, run_id: int):
        async for result in self.runtime.commands.cmd_run(event, run_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("context")
    async def cmd_context(self, event: AstrMessageEvent, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_context(event, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("summarize")
    async def cmd_summarize(self, event: AstrMessageEvent, profile_id: str = "", mode: str = "dry"):
        async for result in self.runtime.commands.cmd_summarize(event, profile_id, mode):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("knowledge")
    async def cmd_knowledge(
        self,
        event: AstrMessageEvent,
        action: str = "status",
        value: str = "",
        profile_id: str = "",
    ):
        async for result in self.runtime.commands.cmd_knowledge(event, action, value, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("memory")
    async def cmd_memory(self, event: AstrMessageEvent, memory_id: int, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_memory(event, memory_id, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("world_info")
    async def cmd_world_info(
        self, event: AstrMessageEvent, world_info_id: int, profile_id: str = ""
    ):
        async for result in self.runtime.commands.cmd_world_info(event, world_info_id, profile_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("outbox")
    async def cmd_outbox(self, event: AstrMessageEvent, profile_id: str = "", limit: int = 10):
        async for result in self.runtime.commands.cmd_outbox(event, profile_id, limit):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("ai")
    async def cmd_ai(
        self,
        event: AstrMessageEvent,
        action: str = "status",
        value: str = "",
        reason: str = "",
    ):
        async for result in self.runtime.commands.cmd_ai(event, action, value, reason):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("image")
    async def cmd_image(self, event: AstrMessageEvent, action: str = "status", value: str = ""):
        async for result in self.runtime.commands.cmd_image(event, action, value):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("sticker")
    async def cmd_sticker(self, event: AstrMessageEvent, action: str = "status", value: str = ""):
        async for result in self.runtime.commands.cmd_sticker(event, action, value):
            yield result

    @permission_type(PermissionType.ADMIN)
    @filter.command("sticker_status")
    async def cmd_sticker_status_shortcut(self, event: AstrMessageEvent):
        async for result in self.runtime.commands.cmd_sticker_status_shortcut(event):
            yield result

    @permission_type(PermissionType.ADMIN)
    @filter.command("sticker_collect")
    async def cmd_sticker_collect_shortcut(self, event: AstrMessageEvent, theme: str = ""):
        async for result in self.runtime.commands.cmd_sticker_collect_shortcut(event, theme):
            yield result

    @permission_type(PermissionType.ADMIN)
    @permission_type(PermissionType.ADMIN)
    @filter.command("sticker_reinforce")
    async def cmd_sticker_reinforce_shortcut(self, event: AstrMessageEvent, item_id: str = ""):
        async for result in self.runtime.commands.cmd_sticker_reinforce_shortcut(event, item_id):
            yield result

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("web")
    async def cmd_web(self, event: AstrMessageEvent, action: str = "status", value: str = ""):
        async for result in self.runtime.commands.cmd_web(event, action, value):
            yield result

    @permission_type(PermissionType.ADMIN)
    @permission_type(PermissionType.ADMIN)
    @soulcore.command("tick")
    async def cmd_tick(self, event: AstrMessageEvent, profile_id: str = "", mode: str = "dry"):
        async for result in self.runtime.commands.cmd_tick(event, profile_id, mode):
            yield result

    @filter.command("soulcore重置")
    async def cmd_reset(self, event: AstrMessageEvent):
        event.should_call_llm(False)
        try:
            if not event.is_private_chat():
                return
            async for result in self.runtime.commands.cmd_reset(event):
                yield result
        finally:
            event.stop_event()

    @filter.command("soulcore重置并清空表情包")
    async def cmd_reset_and_clear_stickers(self, event: AstrMessageEvent):
        event.should_call_llm(False)
        try:
            if not event.is_private_chat():
                return
            async for result in self.runtime.commands.cmd_reset_and_clear_stickers(event):
                yield result
        finally:
            event.stop_event()

    @permission_type(PermissionType.ADMIN)
    @soulcore.command("probe")
    async def cmd_probe(self, event: AstrMessageEvent, component: str, profile_id: str = ""):
        async for result in self.runtime.commands.cmd_probe(event, component, profile_id):
            yield result


__all__ = ["SoulCorePlugin"]
