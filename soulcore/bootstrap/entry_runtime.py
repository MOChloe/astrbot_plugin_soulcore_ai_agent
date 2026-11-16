"""Stable runtime host used by the AstrBot entry class."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..interfaces.admin.console_errors import ConsoleValidationError, PageRuntimeStopping
from ..interfaces.admin.controllers.role_package_uploads import DatabaseBackupUploadTransport
from ..interfaces.admin.downloads import PageFileDownload
from ..interfaces.astrbot.support import PERSISTENT_DATA_NAMESPACE
from ..shared.data_paths import DATABASE_FILENAME
from ..storage.sqlite.schema.current import SchemaRecoveryRequired
from ..storage.sqlite.schema_recovery import (
    CREATE_BACKUP,
    IMPORT_BACKUP,
    SchemaRecoveryCoordinator,
    SchemaRecoveryResult,
)
from ..version import VERSION
from .runtime import BootstrapRollbackError, BootstrapRollbackOwner, rollback_bootstrap
from .runtime_fence import RuntimeOwnershipError, RuntimeOwnershipFence

if TYPE_CHECKING:
    from astrbot.api.star import Context

    from .application import SoulCoreApplication
    from .foundation import Foundation
    from .workers import CoreRuntimeParts

    CoreRuntimeFactory = Callable[[Foundation], CoreRuntimeParts | Awaitable[CoreRuntimeParts]]
else:
    CoreRuntimeFactory = Callable[..., Any]


async def assemble_application(
    context: Context, config: Any, *, core_factory: CoreRuntimeFactory | None = None
) -> SoulCoreApplication:
    """Assemble explicit resources; rollback every initialized layer on failure."""

    from astrbot.api.star import StarTools

    from .application import SoulCoreApplication
    from .foundation import assemble_foundation
    from .interfaces import assemble_interfaces
    from .workers import assemble_workers

    data_dir = _persistent_data_dir(StarTools)
    foundation = await assemble_foundation(context, config, data_dir)
    try:
        factory = core_factory or _default_core_factory()
        produced = factory(foundation)
        core = await produced if inspect.isawaitable(produced) else produced
        workers = assemble_workers(foundation, core)
        application = SoulCoreApplication(
            repositories=foundation.repositories,
            runner=core.runner,
            scheduler=workers.scheduler,
            background_scheduler=workers.background_scheduler,
            expression_outbox_worker=workers.expression_outbox_worker,
            ai_tasks=foundation.ai_tasks,
            visual_service=foundation.visual_service,
            media_storage=foundation.media_storage,
            file_artifacts=foundation.file_artifacts,
            voice_artifacts=foundation.voice_artifacts,
            sticker_triggers=workers.sticker_triggers,
            turn_buffer_worker=foundation.turn_buffer_worker,
            group_flow_worker=foundation.group_flow_worker,
            recall_index_worker=foundation.recall_index_worker,
            inbound_recall_worker=foundation.inbound_recall_worker,
            timer_runtime=workers.timer_runtime,
            route_readiness=foundation.route_readiness,
            plugin_version=_plugin_version(),
            runtime_fence=foundation.runtime_fence,
            activation_gate=foundation.activation_gate,
        )
        interfaces = assemble_interfaces(
            foundation, core, workers, application, plugin_version=_plugin_version()
        )
        application.bind_interfaces(
            commands=interfaces.commands, page=interfaces.page, inbound=interfaces.inbound
        )
        return application
    except BaseException as exc:
        return await rollback_bootstrap(
            BootstrapRollbackOwner(foundation.container, foundation.runtime_fence),
            exc,
        )


def _default_core_factory() -> CoreRuntimeFactory:
    """Import the only pending factory once its explicit-port refactor is present."""

    from .core_runtime import build_core_runtime

    return build_core_runtime


def _plugin_version() -> str:
    return VERSION


def _persistent_data_dir(star_tools: Any) -> Path:
    return Path(star_tools.get_data_dir(PERSISTENT_DATA_NAMESPACE))


_PAGE_READY_TIMEOUT_SECONDS = 30.0
_DATABASE_UPLOAD_OWNER = {"profile_id": "database", "role_ref": "database"}


class ApplicationPort(Protocol):
    commands: Any

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def on_message(self, event: AstrMessageEvent) -> Any: ...
    async def call_page(self, method: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    async def call_page_download(self, method: str, payload: Mapping[str, Any]) -> Any: ...


class PluginRuntimeHost:
    """Own application readiness without leaking construction into ``main.py``."""

    def __init__(self, context: Context, config: Any) -> None:
        self.context = context
        self.config = config or {}
        self.application: ApplicationPort | None = None
        self.ready = asyncio.Event()
        self.init_error: BaseException | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._generation = 0
        self._termination_requested = False
        self._rollback_owner: BootstrapRollbackOwner | None = None
        self._recovery_action_lock = asyncio.Lock()
        self._schema_recovery: SchemaRecoveryCoordinator | None = None
        self._online_maintenance: SchemaRecoveryCoordinator | None = None
        self._database_uploads: DatabaseBackupUploadTransport | None = None
        self._maintaining_database = False
        self._page_actions = 0
        self._page_idle = asyncio.Event()
        self._page_idle.set()
        # AstrBot purges plugin modules after terminate(), including when it
        # failed. Keep unfinished cleanup reachable on the surviving Context;
        # a module global would disappear before the next host could drain it.
        self._runtime_owners: dict[str, Any] = vars(context).setdefault(
            "_soulcore_runtime_owners", {}
        )

    @property
    def commands(self) -> Any:
        return self._require_application().commands

    @property
    def running(self) -> bool:
        return (
            not self._termination_requested
            and self.application is not None
            and self.init_error is None
        )

    async def initialize(self) -> None:
        generation = self._generation
        async with self._lifecycle_lock:
            if self._lifecycle_changed(generation):
                return
            if self.application is not None and self.init_error is None:
                return
            await self._initialize_locked(generation)

    async def _initialize_locked(self, generation: int) -> None:
        self.ready.clear()
        self._schema_recovery = None
        application: ApplicationPort | None = None
        try:
            if not await self._claim_runtime_owner(generation):
                return
            if not await self._retry_retained_application():
                return
            if not await self._retry_rollback_owner():
                return
            if self._lifecycle_changed(generation):
                return
            self.init_error = None
            application = await assemble_application(self.context, self.config)
            if self._lifecycle_changed(generation):
                await self._close_owned_application(application)
                return
            await application.start()
            if self._lifecycle_changed(generation):
                await self._close_owned_application(application)
                return
            self.application = application
        except asyncio.CancelledError as exc:
            await self._handle_initialize_cancellation(exc, application)
            raise
        except Exception as exc:
            if isinstance(exc, BootstrapRollbackError):
                self._rollback_owner = exc.owner
            if self._lifecycle_changed(generation):
                if application is not None:
                    self.application = application
                raise
            await self._record_initialize_failure(exc, application, generation)
        finally:
            self._release_runtime_owner_if_closed()
            self.ready.set()

    async def _claim_runtime_owner(self, generation: int) -> bool:
        previous = self._runtime_owners.get(PERSISTENT_DATA_NAMESPACE)
        if previous is not None and previous is not self:
            # Only an explicitly terminated predecessor may be drained. A
            # duplicate installation must never stop an active peer to steal
            # its data directory; the OS fence also excludes other processes.
            if not previous._termination_requested:
                raise RuntimeOwnershipError("another live SoulCore runtime owns this context")
            logger.warning("[SoulCore] resuming unfinished cleanup before runtime replacement")
            try:
                await previous.terminate()
            except Exception as exc:
                raise RuntimeError("previous SoulCore runtime cleanup is incomplete") from exc
            if self._lifecycle_changed(generation):
                return False
            # Another concurrent initializer may have claimed the slot while
            # we awaited cleanup. Never overwrite that new owner.
            current = self._runtime_owners.get(PERSISTENT_DATA_NAMESPACE)
            if current is not None and current is not self:
                raise RuntimeOwnershipError("another live SoulCore runtime owns this context")
        self._runtime_owners[PERSISTENT_DATA_NAMESPACE] = self
        return True

    def _release_runtime_owner_if_closed(self) -> None:
        if self.application is not None or self._rollback_owner is not None:
            return
        if self._runtime_owners.get(PERSISTENT_DATA_NAMESPACE) is self:
            del self._runtime_owners[PERSISTENT_DATA_NAMESPACE]

    async def _retry_retained_application(self) -> bool:
        application = self.application
        if application is None:
            return True
        try:
            await self._close_owned_application(application)
        except Exception as exc:
            self.init_error = exc
            return False
        return True

    async def _retry_rollback_owner(self) -> bool:
        rollback_owner = self._rollback_owner
        if rollback_owner is None:
            return True
        try:
            await rollback_owner.close()
        except Exception as exc:
            self.init_error = exc
            return False
        self._rollback_owner = None
        return True

    async def _handle_initialize_cancellation(
        self,
        cancellation: asyncio.CancelledError,
        application: ApplicationPort | None,
    ) -> None:
        rollback_owner = getattr(cancellation, "rollback_owner", None)
        if isinstance(rollback_owner, BootstrapRollbackOwner):
            self._rollback_owner = rollback_owner
        if application is None:
            return
        try:
            await self._close_owned_application(application)
        except Exception as close_error:
            self.init_error = close_error
            logger.error(
                "[SoulCore] initialization cancellation cleanup failed: "
                f"{type(close_error).__name__}: {close_error}",
                exc_info=True,
            )

    async def _record_initialize_failure(
        self,
        startup_error: Exception,
        application: ApplicationPort | None,
        generation: int,
    ) -> None:
        failures = [startup_error]
        if application is not None:
            try:
                await self._close_owned_application(application)
            except Exception as close_error:
                failures.append(close_error)
        effective_error: Exception = (
            failures[0]
            if len(failures) == 1
            else ExceptionGroup("SoulCore initialization and rollback failed", failures)
        )
        if not self._lifecycle_changed(generation):
            self.init_error = effective_error
            if isinstance(startup_error, SchemaRecoveryRequired):
                self._schema_recovery = SchemaRecoveryCoordinator(startup_error)
        logger.error(
            f"[SoulCore] initialization failed: {type(startup_error).__name__}: {startup_error}",
            exc_info=True,
        )

    def _lifecycle_changed(self, generation: int) -> bool:
        return self._termination_requested or generation != self._generation

    async def _close_owned_application(self, application: ApplicationPort) -> None:
        self.application = application
        cleanup = asyncio.create_task(
            application.close(),
            name="soulcore-application-rollback",
        )
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        deferred_cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError as exc:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    deferred_cancellation = deferred_cancellation or exc
                    observed_cancellations = current_cancellations
                    continue
                failure = RuntimeError("SoulCore application rollback task was cancelled")
                failure.__cause__ = exc
                self.init_error = failure
                raise
            except Exception as exc:
                self.init_error = exc
                if deferred_cancellation is not None:
                    raise deferred_cancellation from exc
                raise
        if self.application is application:
            self.application = None
        if deferred_cancellation is not None:
            raise deferred_cancellation

    async def terminate(self) -> None:
        self._termination_requested = True
        self._generation += 1
        self.ready.set()
        cleanup = asyncio.create_task(
            self._terminate_locked(),
            name="soulcore-runtime-terminate",
        )
        cancellation: asyncio.CancelledError | None = None
        cleanup_error: Exception | None = None
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError as exc:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    cancellation = cancellation or exc
                    observed_cancellations = current_cancellations
                    continue
                cleanup_error = RuntimeError("SoulCore runtime cleanup task was cancelled")
                cleanup_error.__cause__ = exc
                break
            except Exception as exc:
                cleanup_error = exc
                break
        if cancellation is not None:
            if cleanup_error is not None:
                raise cancellation from cleanup_error
            raise cancellation
        if cleanup_error is not None:
            raise cleanup_error

    async def _terminate_locked(self) -> None:
        async with self._lifecycle_lock:
            application = self.application
            failures: list[Exception] = []
            try:
                if application is not None:
                    await application.close()
            except Exception as exc:
                failures.append(exc)
            else:
                self.application = None
                self._schema_recovery = None
            rollback_owner = self._rollback_owner
            try:
                if rollback_owner is not None:
                    await rollback_owner.close()
            except Exception as exc:
                failures.append(exc)
            else:
                self._rollback_owner = None
            self.ready.set()
            self._release_runtime_owner_if_closed()
            if failures:
                raise ExceptionGroup("SoulCore runtime termination failed", failures)

    async def on_message(self, event: AstrMessageEvent) -> Any:
        if self._termination_requested or self._maintaining_database or not self.ready.is_set():
            return None
        if self.application is None or self.init_error is not None:
            return None
        return await self.application.on_message(event)

    async def call(self, method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        await self._wait_for_page_runtime()
        if method == "schema_recovery":
            async with self._lifecycle_lock:
                return await asyncio.to_thread(self._schema_recovery_view)
        if method == "schema_recovery_action":
            return await self._finish_database_operation(self._run_schema_recovery(payload))
        if method == "schema_recovery_upload":
            return await self._finish_database_operation(self._run_database_upload(payload))
        return await self._application_page(method, payload)

    async def download(self, method: str, payload: Mapping[str, Any]) -> Any:
        await self._wait_for_page_runtime()
        if method == "schema_recovery_download":
            async with self._lifecycle_lock:
                return await asyncio.to_thread(self._download_database_backup, payload)
        return await self._application_page(method, payload, download=True)

    async def _application_page(
        self, method: str, payload: Mapping[str, Any], *, download: bool = False
    ) -> Any:
        application = self._require_application()
        self._page_actions += 1
        self._page_idle.clear()
        try:
            handler = application.call_page_download if download else application.call_page
            return await handler(method, payload)
        finally:
            self._page_actions -= 1
            if self._page_actions == 0:
                self._page_idle.set()

    async def _wait_for_page_runtime(self) -> None:
        if self._maintaining_database:
            raise PageRuntimeStopping("SoulCore is maintaining its database")
        if not self._termination_requested and not self.ready.is_set():
            try:
                await asyncio.wait_for(
                    self.ready.wait(),
                    timeout=_PAGE_READY_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                raise RuntimeError("SoulCore initialization is still in progress") from exc
        if self._termination_requested or self._maintaining_database:
            raise PageRuntimeStopping("SoulCore is terminating")

    def _maintenance_coordinator(self) -> SchemaRecoveryCoordinator | None:
        if self._schema_recovery is not None:
            return self._schema_recovery
        if self.application is None or self.init_error is not None:
            return None
        if self._online_maintenance is None:
            from astrbot.api.star import StarTools

            self._online_maintenance = SchemaRecoveryCoordinator(
                _persistent_data_dir(StarTools) / DATABASE_FILENAME
            )
        return self._online_maintenance

    def _schema_recovery_view(self) -> dict[str, Any]:
        coordinator = self._maintenance_coordinator()
        return coordinator.view() if coordinator is not None else {"required": False}

    def _download_database_backup(self, payload: Mapping[str, Any]) -> PageFileDownload:
        coordinator = self._maintenance_coordinator()
        if coordinator is None:
            raise ConsoleValidationError("当前无法读取数据库备份")
        try:
            path = coordinator.download_path(
                str(payload.get("source_id") or ""), str(payload.get("backup_name") or "")
            )
        except ValueError as exc:
            raise ConsoleValidationError(str(exc)) from exc
        return PageFileDownload(
            path=path,
            filename=path.name,
            content_type="application/zip" if path.suffix == ".zip" else "application/octet-stream",
            headers={"Cache-Control": "private, no-store"},
        )

    def _upload_transport(
        self, coordinator: SchemaRecoveryCoordinator
    ) -> DatabaseBackupUploadTransport:
        if self._database_uploads is None:
            self._database_uploads = DatabaseBackupUploadTransport(
                coordinator.data_dir / "backups" / "import-staging"
            )
        return self._database_uploads

    async def _run_database_upload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        async with self._lifecycle_lock:
            coordinator = self._maintenance_coordinator()
            if coordinator is None or not coordinator.can_replace:
                raise ConsoleValidationError("当前数据库无法执行导入")
            try:
                return await self._handle_database_upload(
                    self._upload_transport(coordinator), payload
                )
            except (ValueError, TypeError) as exc:
                raise ConsoleValidationError(str(exc)) from exc

    @staticmethod
    async def _handle_database_upload(
        uploader: DatabaseBackupUploadTransport, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        action = payload.get("action")
        upload_id = str(payload.get("upload_id") or "")
        if action == "upload_begin":
            return await uploader.begin(
                **_DATABASE_UPLOAD_OWNER,
                expected_size=int(payload.get("file_size") or 0),
                filename=str(payload.get("filename") or ""),
            )
        if action == "chunk":
            return await uploader.append(
                **_DATABASE_UPLOAD_OWNER,
                upload_id=upload_id,
                chunk_index=int(payload.get("chunk_index") or 0),
                encoded_chunk=str(payload.get("chunk_data") or ""),
                expected_chunk_size=int(payload.get("chunk_size") or 0),
                expected_chunk_crc32=str(payload.get("chunk_crc32") or ""),
            )
        if action == "abort":
            await uploader.abort(**_DATABASE_UPLOAD_OWNER, upload_id=upload_id)
            return {"ok": True}
        raise ConsoleValidationError("未知的数据库上传操作")

    async def _run_schema_recovery(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        async with self._recovery_action_lock, self._lifecycle_lock:
            coordinator = self._maintenance_coordinator()
            if coordinator is None:
                raise ValueError("当前没有等待处理的数据库恢复问题")
            if self._rollback_owner is not None or self._termination_requested:
                raise ValueError("运行资源尚未关闭，当前不能还原数据库")
            action = str(payload.get("action") or "").strip().lower()
            token = str(payload.get("recovery_token") or "")
            confirmation = str(payload.get("confirmation") or "")
            try:
                coordinator.validate(action, token, confirmation)
            except ValueError as exc:
                raise ConsoleValidationError(str(exc)) from exc
            self._maintaining_database = True
            self.ready.clear()
            upload: Path | None = None
            try:
                if action == IMPORT_BACKUP:
                    completed = await self._upload_transport(coordinator).finish(
                        **_DATABASE_UPLOAD_OWNER,
                        upload_id=str(payload.get("upload_id") or ""),
                        expected_crc32=str(payload.get("file_crc32") or ""),
                    )
                    upload = completed.path
                return await self._apply_database_action(
                    coordinator, action, token, confirmation, upload
                )
            except ValueError as exc:
                raise ConsoleValidationError(str(exc)) from exc
            finally:
                if upload is not None:
                    try:
                        upload.unlink(missing_ok=True)
                    except OSError:
                        logger.warning(
                            "[SoulCore] database upload staging cleanup failed", exc_info=True
                        )
                self._maintaining_database = False
                self.ready.set()

    async def _apply_database_action(
        self,
        coordinator: SchemaRecoveryCoordinator,
        action: str,
        token: str,
        confirmation: str,
        upload: Path | None,
    ) -> dict[str, Any]:
        restart = False
        try:
            if self.application is not None:
                await self._page_idle.wait()
                await self._close_owned_application(self.application)
                restart = True
            if self._termination_requested:
                raise PageRuntimeStopping("SoulCore is terminating")
            result = await asyncio.to_thread(
                self._execute_database_action, coordinator, action, token, confirmation, upload
            )
            restart = restart or action != CREATE_BACKUP
        finally:
            if restart and not self._termination_requested:
                await self._initialize_locked(self._generation)
        if restart and not self.running:
            raise RuntimeError("数据库操作已完成，但运行服务尚未恢复；请查看启动日志和已保存的备份")
        return {**result.public_view(), "reinitialized": restart}

    @staticmethod
    def _execute_database_action(
        coordinator: SchemaRecoveryCoordinator,
        action: str,
        token: str,
        confirmation: str,
        upload: Path | None,
    ) -> SchemaRecoveryResult:
        owner = RuntimeOwnershipFence.acquire(coordinator.data_dir)
        try:
            values = {} if upload is None else {"upload": upload}
            return coordinator.execute(
                action=action, recovery_token=token, confirmation=confirmation, **values
            )
        finally:
            owner.close()

    @staticmethod
    async def _finish_database_operation(operation: Coroutine[Any, Any, Any]) -> Any:
        # Disconnecting a Page must not release either fence during publication.
        task = asyncio.create_task(operation)
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
        result = task.result()
        if cancellation is not None:
            raise cancellation
        return result

    def _require_application(self) -> ApplicationPort:
        if self._termination_requested:
            raise RuntimeError("SoulCore is terminating")
        if self.init_error is not None:
            raise RuntimeError(
                "SoulCore initialization failed: "
                f"{type(self.init_error).__name__}: {self.init_error}"
            ) from self.init_error
        if self.application is None:
            raise RuntimeError("SoulCore is not initialized")
        return self.application


__all__ = ["ApplicationPort", "PluginRuntimeHost"]
