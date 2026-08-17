"""Durable upload/search intake execution with explicit review before promotion."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..character_context import (
    projection_diagnostic,
    require_character_run,
)
from ..character_model import ProjectionPurpose
from .domain import (
    StickerCheckVerdict,
    StickerConfig,
    StickerIntakeEntryStatus,
    StickerIntakeKind,
    StickerSourceKind,
)
from .policy import StickerRuntimeDisabled


class StickerIntakeMixin:
    @staticmethod
    def combined_intake_requirements(global_requirements: str, user_prompt: str) -> str:
        """Append the exact batch request without weakening the global rules."""

        global_text = str(global_requirements or "").strip()
        batch_text = str(user_prompt or "").strip()[:500]
        if not batch_text:
            return global_text
        return (
            "全局表情包要求（继续生效，不得被本批次覆盖）：\n"
            f"{global_text or '无额外全局要求'}\n\n"
            "本批次追加要求（保留原意，只追加约束）：\n"
            f"{batch_text}"
        )

    async def execute_sticker_intake_task(
        self,
        task: Mapping[str, Any],
        control: Any,
        *,
        instance: Any,
        config: StickerConfig,
    ) -> Mapping[str, Any]:
        payload = dict(task.get("input") or {})
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            raise ValueError("sticker intake session_id is required")
        session = await self.repository.get_sticker_intake_session(session_id)
        if session is None:
            raise ValueError("sticker intake session no longer exists")
        if str(session["profile_id"]) != str(task.get("profile_id") or "") or str(
            session["instance_id"]
        ) != str(task.get("instance_id") or ""):
            raise ValueError("sticker intake task owner mismatch")
        if str(session["status"]) != "RUNNING" or bool(session["stop_requested"]):
            return {"_task_status": "CANCELLED", "cancelled": True, "reason": "batch_frozen"}

        user_prompt = str(session.get("user_prompt") or "")
        requirements = self.combined_intake_requirements(config.requirements, user_prompt)
        kind = StickerIntakeKind(str(session["intake_kind"]))
        if kind is StickerIntakeKind.SEARCH:
            return await self._execute_search_intake(
                task,
                control,
                instance=instance,
                config=config,
                session=session,
                requirements=requirements,
                user_prompt=user_prompt,
            )
        return await self._execute_upload_intake(
            task,
            control,
            instance=instance,
            session=session,
            requirements=requirements,
        )

    async def _intake_persona(
        self,
        profile_id: str,
        instance_id: str,
        *,
        relevance_text: str,
        control: Any,
    ) -> str:
        projection = await require_character_run(profile_id).project(
            ProjectionPurpose.STICKER_PLANNING,
            relevance_text=str(relevance_text)[:4000],
        )
        await self._progress(
            control,
            "CHARACTER_MODEL",
            detail="冻结快速注入的表情包角色投影",
            character_model=projection_diagnostic(projection),
        )
        identity_context, identity_catalog = await self.identity.catalog(profile_id, instance_id)
        return self.identity.project_for_model(
            projection.rendered_text,
            identity_catalog,
            scope=str(identity_context.scope),
        )

    async def _execute_upload_intake(
        self,
        task: Mapping[str, Any],
        control: Any,
        *,
        instance: Any,
        session: Mapping[str, Any],
        requirements: str,
    ) -> Mapping[str, Any]:
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        (await self._runtime_policy(profile_id, instance_id)).require_enabled()
        persona = await self._intake_persona(
            profile_id,
            instance_id,
            relevance_text=requirements,
            control=control,
        )
        entries = await self.repository.list_sticker_intake_entries(str(session["session_id"]))
        candidates = [
            entry
            for entry in entries
            if str(entry["status"]) in {"UPLOADED", "ANALYZING"}
            and str(entry.get("candidate_id") or "")
        ]
        for index, entry in enumerate(candidates, start=1):
            await control.check_control()
            if not await self.repository.sticker_intake_accepts_results(str(session["session_id"])):
                break
            await self._progress(
                control,
                "INTAKE_CHECK",
                detail="逐张分析批量导入图片",
                current=index,
                total=len(candidates),
            )
            await self._check_intake_entry(
                profile_id,
                instance_id,
                str(session["session_id"]),
                entry,
                persona=persona,
                requirements=requirements,
                control=control,
            )
        if await self.repository.sticker_intake_accepts_results(str(session["session_id"])):
            await self.repository.set_sticker_intake_review(str(session["session_id"]))
        return await self._intake_task_result(str(session["session_id"]))

    async def _execute_search_intake(
        self,
        task: Mapping[str, Any],
        control: Any,
        *,
        instance: Any,
        config: StickerConfig,
        session: Mapping[str, Any],
        requirements: str,
        user_prompt: str,
    ) -> Mapping[str, Any]:
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        task_id = int(task.get("task_id") or 0)
        (await self._runtime_policy(profile_id, instance_id)).require_source(StickerSourceKind.WEB)
        persona, queries, planning_error = await self._search_intake_plan(
            profile_id,
            instance_id,
            task_id,
            instance=instance,
            config=config,
            requirements=requirements,
            user_prompt=user_prompt,
            control=control,
        )
        errors: list[str] = []
        for query in queries:
            counts, should_stop = await self._search_intake_state(session)
            if should_stop:
                break
            await control.check_control()
            assets, error = await self._collect_search_intake_assets(
                profile_id,
                instance_id,
                task_id,
                query,
                control=control,
                limit=min(12, int(session["raw_limit"]) - counts["TOTAL"]),
            )
            if error:
                errors.append(error)
                continue
            await self._consume_search_intake_assets(
                profile_id,
                instance_id,
                task_id,
                session,
                assets,
                query=query,
                persona=persona,
                requirements=requirements,
                control=control,
            )
        session_id = str(session["session_id"])
        if await self.repository.sticker_intake_accepts_results(session_id):
            await self.repository.set_sticker_intake_review(
                session_id,
                error="；".join(errors[-3:]) or planning_error,
            )
        return await self._intake_task_result(
            session_id,
            planning_fallback=bool(planning_error),
        )

    async def _search_intake_plan(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        *,
        instance: Any,
        config: StickerConfig,
        requirements: str,
        user_prompt: str,
        control: Any,
    ) -> tuple[str, list[str], str]:
        search_config = replace(
            config,
            requirements=requirements,
            generation_enabled=False,
            generated_daily_limit=0,
        )
        (
            _snapshot,
            persona,
            _scope,
            theme,
            _requirements,
            plan,
            planning_error,
        ) = await self._prepare_collection_plan(
            profile_id,
            instance_id,
            task_id,
            {"theme": user_prompt},
            "intake_search",
            instance,
            search_config,
            control,
        )
        tiers = self._web_query_tiers(
            plan=plan,
            persona=persona,
            requirements=requirements,
            requested_theme=theme,
        )
        queries = list(dict.fromkeys(query for tier in tiers for query in tier if query))
        return persona, queries, planning_error

    async def _search_intake_state(
        self,
        session: Mapping[str, Any],
    ) -> tuple[dict[str, int], bool]:
        session_id = str(session["session_id"])
        counts = await self._intake_counts(session_id)
        if not await self.repository.sticker_intake_accepts_results(session_id):
            return counts, True
        reached_target = counts["READY"] >= int(session["target_count"])
        reached_limit = counts["TOTAL"] >= int(session["raw_limit"])
        return counts, reached_target or reached_limit

    async def _collect_search_intake_assets(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        query: str,
        *,
        control: Any,
        limit: int,
    ) -> tuple[list[Any], str]:
        try:
            assets = await self._collect_web(
                profile_id,
                instance_id,
                task_id,
                [query],
                control,
                limit=limit,
            )
            return list(assets), ""
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            return [], f"{type(exc).__name__}: {str(exc)[:180]}"

    async def _consume_search_intake_assets(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        session: Mapping[str, Any],
        assets: list[Any],
        *,
        query: str,
        persona: str,
        requirements: str,
        control: Any,
    ) -> None:
        session_id = str(session["session_id"])
        for offset, source in enumerate(assets, start=1):
            asset_id = source.asset_id
            source_kind = source.source_kind
            if not await self.repository.sticker_intake_accepts_results(session_id):
                await self._release_source_asset(
                    str(asset_id),
                    reason="INTAKE_RESULT_AFTER_FREEZE",
                )
                continue
            counts = await self._intake_counts(session_id)
            reached_target = counts["READY"] >= int(session["target_count"])
            reached_limit = counts["TOTAL"] >= int(session["raw_limit"])
            if reached_target or reached_limit:
                await self._release_source_asset(
                    str(asset_id),
                    reason="INTAKE_SEARCH_TARGET_REACHED",
                )
                continue
            entry = await self._register_search_intake_asset(
                profile_id,
                instance_id,
                session_id,
                str(asset_id),
                source_kind,
                query=query,
                ordinal=counts["TOTAL"] + offset,
                persona=persona,
                task_id=task_id,
            )
            if str(entry["status"]) == StickerIntakeEntryStatus.UPLOADED.value:
                await self._check_intake_entry(
                    profile_id,
                    instance_id,
                    session_id,
                    entry,
                    persona=persona,
                    requirements=requirements,
                    control=control,
                )

    async def _register_search_intake_asset(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        asset_id: str,
        source_kind: Any,
        *,
        query: str,
        ordinal: int,
        persona: str,
        task_id: int,
    ) -> dict[str, Any]:
        asset = await self.media.get_media_asset(
            asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        if asset is None:
            return await self.repository.add_sticker_intake_entry(
                session_id,
                client_entry_id="search_" + uuid.uuid4().hex,
                display_name=f"搜索结果 {ordinal}",
                source_ref=query,
                status=StickerIntakeEntryStatus.ERROR,
                metadata={
                    "query": query,
                    "ordinal": ordinal,
                    "error": "MEDIA_UNAVAILABLE",
                },
            )
        duplicate = await self.repository.find_sticker_item_by_sha(
            profile_id, instance_id, asset.sha256
        )
        duplicate = duplicate or await self.repository.find_sticker_intake_entry_by_sha(
            session_id, asset.sha256
        )
        if duplicate is not None:
            await self._release_source_asset(asset_id, reason="INTAKE_EXACT_DUPLICATE")
            return await self.repository.add_sticker_intake_entry(
                session_id,
                client_entry_id="search_" + uuid.uuid4().hex,
                display_name=f"搜索结果 {ordinal}",
                source_ref=query,
                status=StickerIntakeEntryStatus.DUPLICATE,
                metadata={"query": query, "ordinal": ordinal},
            )
        candidate, created = await self.repository.create_sticker_candidate(
            profile_id,
            instance_id,
            asset_id,
            source_kind=source_kind,
            source_ref=query,
            persona_fingerprint=self.persona_fingerprint(persona),
            metadata={
                "collector_task_id": task_id,
                "intake_session_id": session_id,
                "intake_query": query,
                "intake_ordinal": ordinal,
            },
        )
        if not created:
            await self._release_source_asset(asset_id, reason="INTAKE_CANDIDATE_DUPLICATE")
            return await self.repository.add_sticker_intake_entry(
                session_id,
                client_entry_id="search_" + uuid.uuid4().hex,
                display_name=f"搜索结果 {ordinal}",
                source_ref=query,
                status=StickerIntakeEntryStatus.DUPLICATE,
                metadata={"query": query, "ordinal": ordinal},
            )
        try:
            return await self.repository.add_sticker_intake_entry(
                session_id,
                client_entry_id="search_" + uuid.uuid4().hex,
                display_name=f"搜索结果 {ordinal}",
                source_ref=query,
                candidate_id=candidate.candidate_id,
                status=StickerIntakeEntryStatus.UPLOADED,
                metadata={"query": query, "ordinal": ordinal},
            )
        except Exception:
            await self.repository.delete_sticker_candidate(
                profile_id, instance_id, candidate.candidate_id
            )
            raise

    async def _check_intake_entry(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        entry: Mapping[str, Any],
        *,
        persona: str,
        requirements: str,
        control: Any,
    ) -> None:
        entry_id = str(entry["entry_id"])
        candidate_id = str(entry.get("candidate_id") or "")
        if not await self._begin_intake_entry_check(session_id, entry):
            return
        if await self._stage_existing_intake_check(
            profile_id,
            instance_id,
            session_id,
            entry_id,
            candidate_id,
        ):
            return
        completed = await self._run_intake_candidate_check(
            profile_id,
            instance_id,
            session_id,
            entry_id,
            candidate_id,
            persona=persona,
            requirements=requirements,
            control=control,
        )
        if completed:
            await self._settle_intake_check_outcome(
                profile_id,
                instance_id,
                session_id,
                entry_id,
                candidate_id,
            )

    async def _begin_intake_entry_check(
        self,
        session_id: str,
        entry: Mapping[str, Any],
    ) -> bool:
        if str(entry["status"]) != StickerIntakeEntryStatus.UPLOADED.value:
            return True
        return await self.repository.mark_sticker_intake_entry_analyzing(
            session_id,
            str(entry["entry_id"]),
        )

    async def _stage_existing_intake_check(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        entry_id: str,
        candidate_id: str,
    ) -> bool:
        candidate = await self.repository.get_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
        )
        if candidate is None or str(candidate.status.value) != "CHECKING":
            return False
        checks = await self.repository.list_sticker_checks(
            profile_id,
            instance_id,
            candidate_id=candidate_id,
            limit=1,
        )
        if not checks or checks[0].verdict is not StickerCheckVerdict.ACCEPT:
            return False
        return await self.repository.stage_sticker_intake_candidate(
            session_id,
            entry_id,
            candidate_id,
        )

    async def _run_intake_candidate_check(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        entry_id: str,
        candidate_id: str,
        *,
        persona: str,
        requirements: str,
        control: Any,
    ) -> bool:
        try:
            await self.check_candidate(
                profile_id,
                instance_id,
                candidate_id,
                control=control,
                persona=persona,
                requirements=requirements,
                staged_session_id=session_id,
                staged_entry_id=entry_id,
            )
            return True
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            await self.repository.settle_sticker_intake_entry(
                session_id,
                entry_id,
                status=StickerIntakeEntryStatus.ERROR,
                reason_code=type(exc).__name__.upper()[:100],
                error_message=str(exc)[:500],
            )
            return False

    async def _settle_intake_check_outcome(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        entry_id: str,
        candidate_id: str,
    ) -> None:
        current_entry = await self.repository.get_sticker_intake_entry(
            session_id,
            entry_id,
        )
        if current_entry is None or str(current_entry["status"]) == "READY":
            return
        candidate = await self.repository.get_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
        )
        checks = await self.repository.list_sticker_checks(
            profile_id,
            instance_id,
            candidate_id=candidate_id,
            limit=1,
        )
        reason = self._intake_check_reason(candidate, checks)
        status = self._intake_check_failure_status(candidate, reason)
        error_message = ""
        if status is StickerIntakeEntryStatus.ERROR:
            error_message = str(getattr(candidate, "last_error", "") or "检查未完成")[:500]
        await self.repository.settle_sticker_intake_entry(
            session_id,
            entry_id,
            status=status,
            reason_code=(reason or str(getattr(candidate, "failure_stage", "") or "CHECK_FAILED"))[
                :100
            ],
            error_message=error_message,
        )

    @staticmethod
    def _intake_check_reason(candidate: Any, checks: list[Any]) -> str:
        if checks:
            return str(checks[0].reason)
        return str(getattr(candidate, "last_error", "") or "")

    @staticmethod
    def _intake_check_failure_status(
        candidate: Any,
        reason: str,
    ) -> StickerIntakeEntryStatus:
        if "DUPLICATE" in reason.upper():
            return StickerIntakeEntryStatus.DUPLICATE
        if candidate is not None and str(candidate.status.value) == "REJECTED":
            return StickerIntakeEntryStatus.REJECTED
        return StickerIntakeEntryStatus.ERROR

    async def _intake_counts(self, session_id: str) -> dict[str, int]:
        entries = await self.repository.list_sticker_intake_entries(session_id)
        counts = {status.value: 0 for status in StickerIntakeEntryStatus}
        for entry in entries:
            counts[str(entry["status"])] = counts.get(str(entry["status"]), 0) + 1
        counts["TOTAL"] = len(entries)
        return counts

    async def _intake_task_result(
        self, session_id: str, *, planning_fallback: bool = False
    ) -> Mapping[str, Any]:
        counts = await self._intake_counts(session_id)
        return {
            "session_id": session_id,
            "checked": counts["READY"] + counts["REJECTED"] + counts["DUPLICATE"] + counts["ERROR"],
            "ready": counts["READY"],
            "rejected": counts["REJECTED"],
            "duplicates": counts["DUPLICATE"],
            "failed": counts["ERROR"],
            "planning_fallback": planning_fallback,
        }


__all__ = ["StickerIntakeMixin"]
