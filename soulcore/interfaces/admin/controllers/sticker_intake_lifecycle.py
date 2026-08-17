"""Lifecycle, review and finalization for explicit sticker intake batches."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from ....features.stickers.domain import (
    StickerIntakeEntryStatus,
    StickerIntakeKind,
    StickerSourceKind,
)
from ....features.stickers.policy import load_sticker_runtime_policy


class StickerIntakeLifecycleMixin:
    async def sticker_intake_snapshot(
        self,
        profile_id: str,
        instance_id: str,
        *,
        session_id: str = "",
    ) -> dict[str, Any]:
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise ValueError("unknown conversation instance")
        await self._expire_sticker_intakes()
        session = (
            await self.repository.get_sticker_intake_session(session_id)
            if session_id
            else await self.repository.get_active_sticker_intake_session(
                profile_id,
                str(instance.scope),
                instance_id,
            )
        )
        if session is None:
            return {"session": None}
        self._require_intake_owner(session, profile_id, str(instance.scope), instance_id)
        if str(session["status"]) == "RUNNING" and not int(session.get("task_id") or 0):
            session = await self._recover_unlinked_intake_task(session)
        if str(session["status"]) == "FINALIZING":
            await self._finalize_frozen_intake(session)
            session = await self.repository.get_sticker_intake_session(str(session["session_id"]))
            assert session is not None
        entries = await self.repository.list_sticker_intake_entries(str(session["session_id"]))
        return self._public_intake_snapshot(session, entries)

    async def start_sticker_intake(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        await self._expire_sticker_intakes()
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise ValueError("unknown conversation instance")
        kind = StickerIntakeKind(str(payload.get("kind") or "").strip().upper())
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles_repository,
            profile_id,
            instance_id=instance_id,
        )
        if kind is StickerIntakeKind.SEARCH:
            policy.require_source(StickerSourceKind.WEB)
            raw_target = payload.get("target_count")
            target = (
                50 if raw_target is None or raw_target == "" else max(1, min(50, int(raw_target)))
            )
            manifest: list[dict[str, Any]] = []
        else:
            policy.require_enabled()
            raw_manifest = payload.get("manifest")
            if not isinstance(raw_manifest, list) or not 1 <= len(raw_manifest) <= 50:
                raise ValueError("批量导入必须包含1至50个文件")
            manifest = [self._intake_manifest_entry(raw) for raw in raw_manifest]
            client_ids = [str(entry["client_entry_id"]) for entry in manifest]
            if len(set(client_ids)) != len(client_ids):
                raise ValueError("批量导入清单中的 client_entry_id 不得重复")
            target = len(manifest)
        session = await self.repository.create_sticker_intake_session(
            profile_id,
            str(instance.scope),
            instance_id,
            intake_kind=kind,
            target_count=target,
            expected_count=len(manifest),
            user_prompt=self._intake_user_prompt(payload.get("user_prompt")),
            manifest=manifest,
        )
        if kind is StickerIntakeKind.SEARCH:
            try:
                session = await self._recover_unlinked_intake_task(session)
            except Exception:
                with suppress(Exception):
                    await self.repository.freeze_sticker_intake_session(
                        str(session["session_id"]), cancelled=True
                    )
                    await self.repository.complete_sticker_intake_session(
                        str(session["session_id"]),
                        cancelled=True,
                        error="搜索任务创建失败",
                    )
                raise
        return await self.sticker_intake_snapshot(
            profile_id,
            instance_id,
            session_id=str(session["session_id"]),
        )

    async def sticker_intake_action(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        session = await self._intake_action_session(profile_id, instance_id, payload)
        session_id = str(session["session_id"])
        action = str(payload.get("action") or "").strip().lower()
        if action == "seal_uploads":
            await self._seal_sticker_intake_uploads(session)
        elif action in {"select", "exclude", "restore"}:
            await self._select_sticker_intake_entry(session_id, action, payload)
        elif action == "retry":
            await self._retry_sticker_intake_entry(
                session,
                str(payload.get("entry_id") or ""),
            )
        elif action == "remove_upload":
            await self._remove_sticker_intake_upload(
                session,
                str(payload.get("entry_id") or ""),
            )
        elif action in {"finish", "finish_early", "early_finish"}:
            await self._finish_sticker_intake(session, cancelled=False)
        elif action == "cancel":
            await self._finish_sticker_intake(session, cancelled=True)
        else:
            raise ValueError("unknown sticker intake action")
        return await self.sticker_intake_snapshot(
            profile_id,
            instance_id,
            session_id=session_id,
        )

    async def _intake_action_session(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        session_id = str(payload.get("session_id") or "").strip()
        if instance is None or not session_id:
            raise ValueError("session_id is required")
        session = await self.repository.get_sticker_intake_session(session_id)
        if session is None:
            raise ValueError("快速注入批次不存在")
        self._require_intake_owner(session, profile_id, str(instance.scope), instance_id)
        return session

    async def _select_sticker_intake_entry(
        self,
        session_id: str,
        action: str,
        payload: Mapping[str, Any],
    ) -> None:
        selected = bool(payload.get("selected")) if action == "select" else action == "restore"
        await self.repository.set_sticker_intake_entry_selected(
            session_id,
            str(payload.get("entry_id") or ""),
            selected,
        )

    async def _finish_sticker_intake(
        self,
        session: Mapping[str, Any],
        *,
        cancelled: bool,
    ) -> None:
        status = str(session["status"])
        terminal = "CANCELLED" if cancelled else "COMPLETED"
        if status == terminal:
            return
        if status != "FINALIZING":
            session = await self.repository.freeze_sticker_intake_session(
                str(session["session_id"]),
                cancelled=cancelled,
            )
            reason = "用户取消快速注入" if cancelled else "用户完成快速注入"
            await self._cancel_intake_task(session, reason=reason)
        await self._finalize_frozen_intake(session)

    async def sticker_intake_preview(
        self,
        profile_id: str,
        instance_id: str,
        *,
        session_id: str,
        entry_id: str,
    ) -> dict[str, Any]:
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        session = await self.repository.get_sticker_intake_session(session_id)
        if instance is None or session is None:
            raise ValueError("快速注入批次不存在")
        self._require_intake_owner(session, profile_id, str(instance.scope), instance_id)
        entry = await self.repository.get_sticker_intake_entry(session_id, entry_id)
        if entry is None or not str(entry.get("candidate_id") or ""):
            return {"entry_id": entry_id, "preview_data_url": ""}
        preview = await self.references.sticker_candidate_thumbnail(
            profile_id,
            str(session["instance_id"]),
            str(entry["candidate_id"]),
        )
        return {"entry_id": entry_id, "preview_data_url": preview}

    async def _seal_sticker_intake_uploads(self, session: Mapping[str, Any]) -> None:
        if str(session["intake_kind"]) != "UPLOAD" or str(session["status"]) != "UPLOADING":
            raise ValueError("当前上传批次已封口")
        session_id = str(session["session_id"])
        for entry in await self.repository.list_sticker_intake_entries(session_id):
            if str(entry["status"]) == "PENDING":
                await self.repository.settle_sticker_intake_entry(
                    session_id,
                    str(entry["entry_id"]),
                    status=StickerIntakeEntryStatus.ERROR,
                    reason_code="UPLOAD_NOT_RECEIVED",
                    error_message="浏览器没有完成该文件的上传",
                )
        sealed = await self.repository.seal_sticker_intake_upload_session(session_id)
        await self._recover_unlinked_intake_task(sealed)

    async def _retry_sticker_intake_entry(self, session: Mapping[str, Any], entry_id: str) -> None:
        if not entry_id:
            raise ValueError("entry_id is required")
        task = await self._create_sticker_intake_task(session)
        try:
            await self.repository.retry_sticker_intake_entry(
                str(session["session_id"]),
                entry_id,
                int(task["task_id"]),
            )
        except Exception:
            with suppress(Exception):
                await self.ai_tasks.cancel(
                    int(task["task_id"]),
                    actor_id="sticker-intake",
                    reason="单项重试未能启动",
                )
            raise

    async def _remove_sticker_intake_upload(
        self, session: Mapping[str, Any], entry_id: str
    ) -> None:
        if (
            str(session["intake_kind"]) != StickerIntakeKind.UPLOAD.value
            or str(session["status"]) != "UPLOADING"
        ):
            raise ValueError("只能移除尚未封口的上传失败项")
        entry = await self.repository.get_sticker_intake_entry(str(session["session_id"]), entry_id)
        if entry is None or str(entry["status"]) not in {"PENDING", "ERROR"}:
            raise ValueError("只能移除等待上传或上传失败的文件")
        await self.repository.settle_sticker_intake_entry(
            str(session["session_id"]),
            entry_id,
            status=StickerIntakeEntryStatus.CANCELLED,
            reason_code="REMOVED_BY_USER",
        )
        candidate_id = str(entry.get("candidate_id") or "")
        if candidate_id:
            await self._discard_intake_candidate(
                str(session["profile_id"]),
                str(session["instance_id"]),
                candidate_id,
            )

    async def _create_sticker_intake_task(self, session: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.ai_tasks is None:
            raise RuntimeError("持久AI任务服务不可用")
        return await self.ai_repository.create_ai_task(
            str(session["profile_id"]),
            "STICKER_INTAKE",
            instance_id=str(session["instance_id"]),
            task_class="BACKGROUND",
            capability="sticker.collect",
            priority=35,
            mutex_key=f"sticker-intake:{session['profile_id']}:{session['scope']}",
            idempotency_key=(f"sticker-intake:{session['session_id']}:{uuid.uuid4().hex}"),
            input_data={"session_id": str(session["session_id"])},
            recovery_policy="RESTART_SAFE",
            max_attempts=4,
        )

    async def _recover_unlinked_intake_task(self, session: Mapping[str, Any]) -> Mapping[str, Any]:
        """Close the crash window between session creation and task attachment."""

        task = await self._create_sticker_intake_task(session)
        try:
            return await self.repository.set_sticker_intake_task(
                str(session["session_id"]),
                int(task["task_id"]),
            )
        except Exception:
            with suppress(Exception):
                await self.ai_tasks.cancel(
                    int(task["task_id"]),
                    actor_id="sticker-intake",
                    reason="快速注入任务已由其他恢复请求接管",
                )
            current = await self.repository.get_sticker_intake_session(str(session["session_id"]))
            if current is not None and (
                str(current["status"]) != "RUNNING" or int(current.get("task_id") or 0)
            ):
                return current
            raise

    async def _cancel_intake_task(self, session: Mapping[str, Any], *, reason: str) -> None:
        task_id = int(session.get("task_id") or 0)
        if not task_id or self.ai_tasks is None:
            return
        with suppress(Exception):
            await self.ai_tasks.cancel(
                task_id,
                actor_id="sticker-intake",
                reason=reason,
            )

    async def _finalize_frozen_intake(self, session: Mapping[str, Any]) -> None:
        session_id = str(session["session_id"])
        lock = self._intake_finalize_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            current = await self.repository.get_sticker_intake_session(session_id)
            if current is None or str(current["status"]) != "FINALIZING":
                return
            cancelled = str(current.get("finalize_action") or "") == "CANCEL"
            entries = await self.repository.list_sticker_intake_entries(session_id)
            if not cancelled:
                await self._promote_selected_intake_entries(current, entries)
            entries = await self.repository.list_sticker_intake_entries(session_id)
            for entry in entries:
                await self._finalize_sticker_intake_entry(current, entry, cancelled=cancelled)
            await self.repository.complete_sticker_intake_session(
                session_id,
                cancelled=cancelled,
            )
            with suppress(Exception):
                await self.media_storage.release_pending(limit=100)

    async def _promote_selected_intake_entries(
        self,
        session: Mapping[str, Any],
        entries: list[Mapping[str, Any]],
    ) -> None:
        for entry in entries:
            if str(entry["status"]) != "READY" or not bool(entry["selected"]):
                continue
            await self._promote_selected_intake_entry(session, entry)

    async def _promote_selected_intake_entry(
        self,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
    ) -> None:
        try:
            _item, created = await self.sticker_collector.promote_staged_candidate(
                str(session["profile_id"]),
                str(session["instance_id"]),
                str(entry["candidate_id"]),
            )
            await self.repository.mark_sticker_intake_entry_imported(
                str(session["session_id"]),
                str(entry["entry_id"]),
                status=(
                    StickerIntakeEntryStatus.IMPORTED
                    if created
                    else StickerIntakeEntryStatus.DUPLICATE
                ),
                reason_code="" if created else "DUPLICATE_AT_COMMIT",
            )
        except Exception as exc:
            message = str(exc)
            await self.repository.mark_sticker_intake_entry_imported(
                str(session["session_id"]),
                str(entry["entry_id"]),
                status=(
                    StickerIntakeEntryStatus.DUPLICATE
                    if "DUPLICATE" in message.upper()
                    else StickerIntakeEntryStatus.ERROR
                ),
                reason_code=type(exc).__name__.upper()[:100],
                error_message=message[:500],
            )

    async def _finalize_sticker_intake_entry(
        self,
        session: Mapping[str, Any],
        entry: Mapping[str, Any],
        *,
        cancelled: bool,
    ) -> None:
        status = str(entry["status"])
        if cancelled or (status == "READY" and not bool(entry["selected"])):
            await self.repository.mark_sticker_intake_entry_imported(
                str(session["session_id"]),
                str(entry["entry_id"]),
                status=StickerIntakeEntryStatus.CANCELLED,
                reason_code="BATCH_CANCELLED" if cancelled else "NOT_SELECTED",
            )
        candidate_id = str(entry.get("candidate_id") or "")
        if candidate_id and status != "IMPORTED":
            await self._discard_intake_candidate(
                str(session["profile_id"]),
                str(session["instance_id"]),
                candidate_id,
            )
            return
        if candidate_id:
            return
        upload_asset_id = str(dict(entry.get("metadata") or {}).get("upload_asset_id") or "")
        if upload_asset_id:
            with suppress(Exception):
                await self.repository.mark_media_asset_release_pending(
                    upload_asset_id,
                    reason="sticker_intake_unattached_upload_finished",
                )

    async def _discard_intake_candidate(
        self, profile_id: str, instance_id: str, candidate_id: str
    ) -> None:
        try:
            await self.repository.delete_sticker_candidate(profile_id, instance_id, candidate_id)
        except (KeyError, ValueError):
            return

    async def _expire_sticker_intakes(self) -> None:
        expired = await self.repository.expire_sticker_intake_sessions(limit=20)
        if expired:
            with suppress(Exception):
                await self.media_storage.release_pending(limit=100)

    @staticmethod
    def _require_intake_owner(
        session: Mapping[str, Any],
        profile_id: str,
        scope: str,
        instance_id: str,
    ) -> None:
        if (
            str(session.get("profile_id") or "") != profile_id
            or str(session.get("scope") or "") != scope
            or str(session.get("instance_id") or "") != instance_id
        ):
            raise ValueError("快速注入批次不属于当前会话实例")

    @staticmethod
    def _public_intake_snapshot(
        session: Mapping[str, Any], entries: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        counts = {
            "total": len(entries),
            "uploading": 0,
            "uploaded": 0,
            "analyzing": 0,
            "ready": 0,
            "selected": 0,
            "duplicate": 0,
            "rejected": 0,
            "failed": 0,
            "cancelled": 0,
            "imported": 0,
        }
        status_keys = {
            "PENDING": "uploading",
            "UPLOADED": "uploaded",
            "ANALYZING": "analyzing",
            "READY": "ready",
            "DUPLICATE": "duplicate",
            "REJECTED": "rejected",
            "ERROR": "failed",
            "CANCELLED": "cancelled",
            "IMPORTED": "imported",
        }
        public_entries = []
        for entry in entries:
            status = str(entry["status"])
            key = status_keys.get(status)
            if key:
                counts[key] += 1
            if status == "READY" and bool(entry["selected"]):
                counts["selected"] += 1
            public_entries.append(
                {
                    "entry_id": str(entry["entry_id"]),
                    "display_name": str(entry["display_name"]),
                    "status": status,
                    "selected": bool(entry["selected"]),
                    "reason": str(entry["reason_code"]),
                    "error": str(entry["error_message"]),
                    "preview_available": bool(entry.get("candidate_id")),
                }
            )
        status = str(session["status"])
        return {
            "session": {
                "session_id": str(session["session_id"]),
                "kind": str(session["intake_kind"]),
                "status": status,
                "target_count": int(session["target_count"]),
                "raw_limit": int(session["raw_limit"]),
                "expected_count": int(session["expected_count"]),
                "user_prompt": str(session["user_prompt"]),
                "counts": counts,
                "entries": public_entries,
                "error": str(session["last_error"]),
                "created_at": str(session["created_at"]),
                "updated_at": str(session["updated_at"]),
                "expires_at": str(session["expires_at"]),
                "can_seal": status == "UPLOADING",
                "can_select": status in {"RUNNING", "REVIEW"},
                "can_finish": status in {"UPLOADING", "RUNNING", "REVIEW"},
                "can_cancel": status in {"UPLOADING", "RUNNING", "REVIEW"},
            }
        }


__all__ = ["StickerIntakeLifecycleMixin"]
