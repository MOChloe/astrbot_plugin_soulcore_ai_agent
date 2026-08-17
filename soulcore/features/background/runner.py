"""Durable single-stage executor for the five independent background authors."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ...contracts.ai_models import (
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
)
from ...shared.prompt_document import xml_text
from ...shared.time_display import model_datetime
from ..ai import record_structured_acceptance
from ..character_context import CharacterRunContext, CharacterRunScope
from ..character_model import CharacterCustomPrompts, ProjectionPurpose
from .creator_session import CreationResult, CreatorSessionSpec, run_creator_session
from .domain import (
    BackgroundAuthorInput,
    BackgroundAuthorKind,
    BackgroundDisabled,
    BackgroundDraft,
    BackgroundDraftStale,
    BackgroundInitializationStep,
)
from .output_contract import BackgroundOutputError
from .ports import (
    BackgroundAuthorRepositoryPort,
    BackgroundIdentityPort,
    BackgroundModelGatewayPort,
    BackgroundTaskControl,
    CharacterModelReadPort,
)
from .prompt_budget import (
    BackgroundPromptBudgetExceeded,
    FrozenBackgroundProjection,
    freeze_background_projection,
)
from .prompt_projection import (
    background_relevance,
    filter_background_character_projection,
)

_INITIALIZATION_AUTHOR = {
    BackgroundInitializationStep.WORLD: BackgroundAuthorKind.WORLD,
    BackgroundInitializationStep.LIFE_DIRECTION: BackgroundAuthorKind.LIFE_DIRECTION,
    BackgroundInitializationStep.STORY_SOURCE: BackgroundAuthorKind.STORY_SOURCE,
    BackgroundInitializationStep.ORDINARY_CURRENT: BackgroundAuthorKind.ORDINARY,
}
_PROACTIVE_CLOCK_ROLLBACK_TOLERANCE = timedelta(seconds=5)


def _optional_aware_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("background task datetime must be timezone-aware")
    return parsed.astimezone(UTC)


def _background_creation_prompt_values(
    kind: BackgroundAuthorKind,
    custom_prompts: CharacterCustomPrompts,
) -> tuple[tuple[str, object], ...]:
    creation = custom_prompts.background_creation
    story_boundary = (("故事创作边界", creation.story_boundary),)
    if kind in {BackgroundAuthorKind.KEYFRAME, BackgroundAuthorKind.ORDINARY}:
        story_styles = custom_prompts.story_styles
        style_values = (
            ("故事介入倾向", story_styles.involvement),
            ("故事姿态", story_styles.stance),
        )
        return story_boundary + style_values
    if kind is BackgroundAuthorKind.WORLD:
        return story_boundary + (
            ("世界变化倾向", creation.world_change),
            ("想象尺度", creation.imagination),
            ("内容温度", creation.temperature),
        )
    if kind is BackgroundAuthorKind.STORY_SOURCE:
        return story_boundary + (
            ("想象尺度", creation.imagination),
            ("内容温度", creation.temperature),
        )
    return story_boundary


@dataclass(frozen=True, slots=True)
class _AuthorTask:
    profile_id: str
    instance_id: str
    kind: BackgroundAuthorKind
    generation: int
    initialization_step: BackgroundInitializationStep
    task_id: int
    attempt_no: int
    frame_end_at: datetime | None = None
    deadline_at: datetime | None = None
    preserve_schedule: bool = False
    original_next_due_at: datetime | None = None
    original_hard_due_at: datetime | None = None

    def model_stage_key(self, stage: str) -> str:
        return (
            f"background:{self.instance_id}:{self.kind.value}:{self.generation}"
            f":attempt:{self.attempt_no}:{stage}"
        )


class BackgroundAuthorRunner:
    """Run one author outside the publication transaction."""

    def __init__(
        self,
        *,
        repository: BackgroundAuthorRepositoryPort,
        model_gateway: BackgroundModelGatewayPort,
        character_models: CharacterModelReadPort,
        identity: BackgroundIdentityPort,
        operation_timeout_seconds: int = 300,
        prewarm_operation_timeout_seconds: int = 20 * 60,
        random_source: random.Random | random.SystemRandom | None = None,
    ) -> None:
        self.repository = repository
        self.model_gateway = model_gateway
        self.character_models = character_models
        self.identity = identity
        self.operation_timeout_seconds = max(1, int(operation_timeout_seconds))
        self.prewarm_operation_timeout_seconds = max(1, int(prewarm_operation_timeout_seconds))
        self.random = random_source or random.SystemRandom()

    async def execute_task(
        self,
        task: dict[str, Any],
        control: BackgroundTaskControl,
    ) -> dict[str, Any]:
        await control.check_control()
        context = self._task_context(task)
        started = await self.repository.start_author_task(
            context.profile_id,
            context.instance_id,
            context.kind,
            generation=context.generation,
            task_id=context.task_id,
        )
        if not started:
            return {
                "_task_status": "CANCELLED",
                "cancelled": True,
                "reason": "background_author_fence_changed",
            }
        try:
            if (
                context.frame_end_at is not None
                and context.frame_end_at > datetime.now(UTC) + _PROACTIVE_CLOCK_ROLLBACK_TOLERANCE
            ):
                return {
                    "_task_status": "CANCELLED",
                    "cancelled": True,
                    "reason": "proactive_frame_clock_moved_backwards",
                }
            if context.deadline_at is None:
                return await self._execute_started(context, control)
            remaining = min(
                float(self.prewarm_operation_timeout_seconds),
                (context.deadline_at - datetime.now(UTC)).total_seconds(),
            )
            if remaining <= 0:
                return {
                    "_task_status": "CANCELLED",
                    "cancelled": True,
                    "reason": "proactive_frame_deadline_elapsed",
                }
            try:
                return await asyncio.wait_for(
                    self._execute_started(context, control),
                    timeout=remaining,
                )
            except TimeoutError:
                return {
                    "_task_status": "CANCELLED",
                    "cancelled": True,
                    "reason": "proactive_frame_deadline_elapsed",
                }
        except BackgroundDisabled:
            return {
                "_task_status": "CANCELLED",
                "cancelled": True,
                "reason": "background_disabled",
            }
        except BackgroundDraftStale:
            # A durable task cannot make an old snapshot current by retrying
            # the same generation.  Settle it as cancelled so the slot can be
            # rematerialized with a fresh continuity fence.
            return {
                "_task_status": "CANCELLED",
                "cancelled": True,
                "reason": "background_author_input_changed",
            }
        except BackgroundOutputError as exc:
            await self._raise_contract_failure(context, exc)
        except BackgroundPromptBudgetExceeded as exc:
            await self._raise_context_budget_failure(context, exc)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._mark_failure(context, f"{type(exc).__name__}: {exc}")
            raise

    async def _execute_started(
        self,
        context: _AuthorTask,
        control: BackgroundTaskControl,
    ) -> dict[str, Any]:
        if context.frame_end_at is None:
            source = await self.repository.load_author_input(
                context.profile_id,
                context.instance_id,
                context.kind,
            )
        else:
            source = await self.repository.load_author_input(
                context.profile_id,
                context.instance_id,
                context.kind,
                frame_end_at=context.frame_end_at,
            )
        self._validate_lifecycle(context, source)
        frozen = freeze_background_projection(context.kind, source)
        preferred_backend_id = self._preferred_backend(source)
        character_projection = await self._character_projection(
            context.profile_id,
            context.kind,
            background_relevance(source),
        )
        await control.check_control()
        creation = await self._create(
            context,
            frozen=frozen,
            preferred_backend_id=preferred_backend_id,
            character_projection=character_projection,
            control=control,
        )
        await record_structured_acceptance(
            model_gateway=self.model_gateway,
            completion=creation.completion,
            round_no=creation.round_no,
            value=creation.creator,
            normalizations=creation.normalizations,
            extra_processing=(
                {"repair_kind": creation.repair_kind} if creation.repair_kind else None
            ),
        )
        await control.check_control()
        return await self._publish_draft(context, source, creation.draft)

    async def _create(
        self,
        context: _AuthorTask,
        *,
        frozen: FrozenBackgroundProjection,
        preferred_backend_id: str,
        character_projection: str,
        control: BackgroundTaskControl | None = None,
    ) -> CreationResult:
        return await run_creator_session(
            CreatorSessionSpec(
                profile_id=context.profile_id,
                instance_id=context.instance_id,
                author_kind=context.kind,
                task_id=context.task_id,
                logical_stage_key=context.model_stage_key("creator"),
                operation_timeout_seconds=self._operation_timeout(context),
                opening_keyframe=self._opening_keyframe(context.kind, frozen.source),
                authoritative_time=self._authoritative_time(context.kind, frozen.source),
            ),
            frozen=frozen,
            preferred_backend_id=preferred_backend_id,
            character_projection=character_projection,
            model_gateway=self.model_gateway,
            identity=self.identity,
            control=control,
        )

    @staticmethod
    def _opening_keyframe(
        kind: BackgroundAuthorKind,
        source: BackgroundAuthorInput,
    ) -> bool:
        return (
            kind is BackgroundAuthorKind.KEYFRAME
            and source.initialization_state == "INITIALIZING"
            and source.initialization_step is BackgroundInitializationStep.ORDINARY_CURRENT
            and not source.opening_keyframe_completed
        )

    @staticmethod
    def _authoritative_time(
        kind: BackgroundAuthorKind,
        source: BackgroundAuthorInput,
    ) -> str:
        if kind not in {BackgroundAuthorKind.KEYFRAME, BackgroundAuthorKind.ORDINARY}:
            return ""
        interval = (
            source.keyframe_frame_interval
            if kind is BackgroundAuthorKind.KEYFRAME
            else source.ordinary_frame_interval
        )
        value = interval.end_at if interval is not None else source.prompt_now
        return model_datetime(value, timezone_name=source.timezone_name)

    async def _publish_draft(
        self,
        context: _AuthorTask,
        source: BackgroundAuthorInput,
        draft: BackgroundDraft,
    ) -> dict[str, Any]:
        if (
            context.preserve_schedule
            and context.original_next_due_at is not None
            and context.original_hard_due_at is not None
        ):
            next_due = context.original_next_due_at
            hard_due = context.original_hard_due_at
        else:
            next_due, hard_due = self._next_schedule(context.kind, source)
        result = await self.repository.publish(
            context.profile_id,
            context.instance_id,
            context.kind,
            generation=context.generation,
            task_id=context.task_id,
            draft=draft,
            versions=source.versions,
            next_due_at=next_due,
            hard_due_at=hard_due,
            preserve_schedule=context.preserve_schedule,
        )
        return {
            "profile_id": context.profile_id,
            "instance_id": context.instance_id,
            "author_kind": context.kind.value,
            "publication_id": result.publication_id,
            "public_ref": result.public_ref,
            "generation": result.generation,
            "initialization_step": result.initialization_step.value,
            "timeline_event_ids": list(result.timeline_event_ids),
            "story_source_refs": list(result.story_source_refs),
            "foreground_message_cursor": result.foreground_message_cursor,
            "foreground_run_cursor": result.foreground_run_cursor,
        }

    @staticmethod
    def _task_context(task: dict[str, Any]) -> _AuthorTask:
        data = dict(task.get("input") or {})
        prewarm = dict(data.get("proactive_frame") or {})
        return _AuthorTask(
            profile_id=str(data.get("profile_id") or task.get("profile_id") or ""),
            instance_id=str(data.get("instance_id") or task.get("instance_id") or ""),
            kind=BackgroundAuthorKind(str(data.get("author_kind") or "").upper()),
            generation=int(data.get("generation") or 0),
            initialization_step=BackgroundInitializationStep(
                str(data["initialization_step"]).upper()
            ),
            task_id=int(task["task_id"]),
            attempt_no=max(1, int(task.get("attempts") or 1)),
            frame_end_at=_optional_aware_datetime(
                prewarm.get("frame_end_at") or data.get("frame_end_at")
            ),
            deadline_at=_optional_aware_datetime(prewarm.get("deadline_at")),
            preserve_schedule=bool(prewarm.get("preserve_schedule")),
            original_next_due_at=_optional_aware_datetime(prewarm.get("original_next_due_at")),
            original_hard_due_at=_optional_aware_datetime(prewarm.get("original_hard_due_at")),
        )

    def _operation_timeout(self, context: _AuthorTask) -> int:
        if context.deadline_at is None:
            return self.operation_timeout_seconds
        remaining = max(1, int((context.deadline_at - datetime.now(UTC)).total_seconds()))
        return min(self.prewarm_operation_timeout_seconds, remaining)

    @staticmethod
    def _validate_lifecycle(
        context: _AuthorTask,
        source: BackgroundAuthorInput,
    ) -> None:
        if source.profile_id != context.profile_id or source.instance_id != context.instance_id:
            raise BackgroundDraftStale("background author input belongs to another instance")
        if source.author_kind is not context.kind or source.generation != context.generation:
            raise BackgroundDraftStale("background author generation changed")
        if source.initialization_step is not context.initialization_step:
            raise BackgroundDraftStale("background initialization step changed")
        state = str(source.initialization_state or "").upper()
        if source.initialization_step is BackgroundInitializationStep.READY:
            if state != "READY":
                raise BackgroundDraftStale("background initialization state is inconsistent")
            return
        if source.initialization_step is BackgroundInitializationStep.ORDINARY_CURRENT:
            expected = (
                BackgroundAuthorKind.ORDINARY
                if source.opening_keyframe_completed
                else BackgroundAuthorKind.KEYFRAME
            )
            if state != "INITIALIZING" or context.kind is not expected:
                raise BackgroundDraftStale("background opening life author is inconsistent")
            return
        if (
            state != "INITIALIZING"
            or _INITIALIZATION_AUTHOR[source.initialization_step] is not context.kind
        ):
            raise BackgroundDraftStale("background initialization author is inconsistent")

    async def _character_projection(
        self,
        profile_id: str,
        kind: BackgroundAuthorKind,
        relevance: str,
    ) -> str:
        character = await CharacterRunContext.start(self.character_models, profile_id)
        with CharacterRunScope(character):
            projection = await character.project(
                ProjectionPurpose.BACKGROUND_AUTHOR,
                relevance_text=relevance,
            )
        rendered = filter_background_character_projection(kind, projection.rendered_text)
        additions = tuple(
            (label, xml_text(value))
            for label, value in _background_creation_prompt_values(kind, projection.custom_prompts)
            if str(value or "").strip()
        )
        if not additions:
            return rendered
        story_text = "\n\n".join(f"[{label}]\n{value}" for label, value in additions)
        return "\n\n".join(part for part in (rendered, story_text) if part)

    def _next_schedule(
        self,
        kind: BackgroundAuthorKind,
        source: BackgroundAuthorInput,
    ) -> tuple[datetime, datetime]:
        now = datetime.now(UTC)
        config = source.config
        if (
            kind is BackgroundAuthorKind.KEYFRAME
            and source.initialization_step is not BackgroundInitializationStep.READY
        ):
            due = now + timedelta(minutes=int(config["keyframe_max_minutes"]))
            return due, due
        fields = {
            BackgroundAuthorKind.ORDINARY: (
                "ordinary_min_minutes",
                "ordinary_max_minutes",
            ),
            BackgroundAuthorKind.KEYFRAME: (
                "ordinary_min_minutes",
                "ordinary_max_minutes",
            ),
            BackgroundAuthorKind.STORY_SOURCE: (
                "story_source_min_minutes",
                "story_source_max_minutes",
            ),
            BackgroundAuthorKind.LIFE_DIRECTION: (
                "life_direction_min_minutes",
                "life_direction_max_minutes",
            ),
            BackgroundAuthorKind.WORLD: ("world_min_minutes", "world_max_minutes"),
        }
        minimum_field, maximum_field = fields[kind]
        due = now + timedelta(
            minutes=self.random.randint(
                int(config[minimum_field]),
                int(config[maximum_field]),
            )
        )
        return due, due

    @staticmethod
    def _preferred_backend(source: BackgroundAuthorInput) -> str:
        return str(source.author_state.backend_id or source.config.get("default_backend_id") or "")

    async def _raise_contract_failure(
        self,
        context: _AuthorTask,
        exc: BackgroundOutputError,
    ) -> None:
        await self._mark_failure(context, str(exc))
        raise AIInvocationError(
            AIErrorInfo(AIErrorCode.OUTPUT_CONTRACT, str(exc), retryable=True),
            cause=exc,
        ) from exc

    async def _raise_context_budget_failure(
        self,
        context: _AuthorTask,
        exc: BackgroundPromptBudgetExceeded,
    ) -> None:
        await self._mark_failure(context, str(exc))
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.CONTEXT_BUDGET,
                str(exc),
                phase="prepare",
                details={"context_error_kind": exc.reason_code},
            ),
            cause=exc,
        ) from exc

    async def _mark_failure(self, context: _AuthorTask, error: str) -> None:
        await self.repository.mark_author_failure(
            context.profile_id,
            context.instance_id,
            context.kind,
            generation=context.generation,
            task_id=context.task_id,
            error=error,
        )


__all__ = ["BackgroundAuthorRunner"]
