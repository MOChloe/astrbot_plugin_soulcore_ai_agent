"""Human-oriented controller for the background-life workspace."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ....features.background.domain import AUTHOR_ORDER, BackgroundAuthorKind
from .profiles import ProfilesAdminController

_INTEGER_SETTING_FIELDS = (
    "ordinary_min_minutes",
    "ordinary_max_minutes",
    "keyframe_every_ordinary",
    "keyframe_max_minutes",
    "story_source_min_minutes",
    "story_source_max_minutes",
    "life_direction_min_minutes",
    "life_direction_max_minutes",
    "world_min_minutes",
    "world_max_minutes",
)


class BackgroundAdminController:
    """Project the background runtime into one bounded management view."""

    def __init__(
        self,
        repository: Any,
        seed_repository: Any,
        profiles: ProfilesAdminController,
        scheduler: Any,
        ai_repository: Any,
    ) -> None:
        self.repository = repository
        self.seed_repository = seed_repository
        self.profiles = profiles
        self.scheduler = scheduler
        self.ai_repository = ai_repository

    async def workspace(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        await self.profiles.require_role_instance(profile_id, instance_id)
        source = _mapping(
            await self.repository.load_background_workspace(profile_id, instance_id),
            "background workspace",
        )
        return _workspace_view(source)

    async def world_snapshot(self, profile_id: str) -> dict[str, Any]:
        """Profile-level seed editor used by the settings workspace."""

        await self.profiles.require_known_profile(profile_id)
        world = await self.seed_repository.get_world_definition(profile_id)
        lore = await self.seed_repository.list_world_lore_entries(
            profile_id, include_content=True, limit=500
        )
        boundaries = await self.seed_repository.list_creative_boundaries(
            profile_id, enabled_only=False
        )
        return {
            "profile_id": profile_id,
            "definition": _definition(world),
            "lore": [_lore(item) for item in lore],
            "boundaries": [_boundary(item) for item in boundaries],
        }

    async def quick_setup_life_snapshot(self, profile_id: str) -> dict[str, Any]:
        await self.profiles.require_known_profile(profile_id)
        return _mapping(
            await self.repository.quick_setup_life_snapshot(profile_id),
            "角色生活设置",
        )

    async def quick_setup_life_configure(
        self,
        profile_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        await self.profiles.require_known_profile(profile_id)
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("请选择是否让角色拥有自己的生活")
        direction = str(payload.get("initial_direction") or "").strip()
        if len(direction) > 500:
            raise ValueError("开始时的想法最多填写 500 个字")
        result = _mapping(
            await self.repository.quick_setup_configure_life(
                profile_id,
                enabled=enabled,
                initial_direction=direction,
                expected_version=_required_int(payload, "expected_version"),
                expected_world_revision=_required_int(payload, "expected_world_revision"),
            ),
            "角色生活设置",
        )
        self.scheduler.notify()
        return result

    async def world_action(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Save the retained user world seed without invoking an author."""

        await self.profiles.require_known_profile(profile_id)
        action = str(payload.get("action") or "").strip().lower()
        translated = {
            "save_definition": "seed_save_definition",
            "lore_create": "seed_lore_create",
            "lore_update": "seed_lore_update",
            "lore_delete": "seed_lore_delete",
            "boundary_create": "seed_boundary_create",
            "boundary_update": "seed_boundary_update",
            "boundary_delete": "seed_boundary_delete",
        }.get(action)
        if translated is None:
            raise ValueError("不支持的世界种子操作")
        return await self._seed_action(profile_id, translated, payload)

    async def action(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        action = str(payload.get("action") or "").strip().lower()
        if action.startswith("seed_"):
            await self.profiles.require_known_profile(profile_id)
            return await self._seed_action(profile_id, action, payload)
        await self.profiles.require_role_instance(profile_id, instance_id)
        expected = _required_int(payload, "expected_version")
        handlers = {
            "enable": self._toggle_background,
            "disable": self._toggle_background,
            "save_config": self._save_config,
            "wake": self._wake_authors,
            "wake_all": self._wake_authors,
            "reset": self._reset_background,
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError("不支持的背景推演操作")
        result = await handler(profile_id, instance_id, action, payload, expected)
        self.scheduler.notify()
        return result

    async def _toggle_background(
        self,
        profile_id: str,
        instance_id: str,
        action: str,
        payload: Mapping[str, Any],
        expected: int,
    ) -> dict[str, Any]:
        del payload
        version = await self.repository.set_background_enabled(
            profile_id,
            instance_id,
            enabled=action == "enable",
            expected_version=expected,
        )
        return {"ok": True, "version": version}

    async def _save_config(
        self,
        profile_id: str,
        instance_id: str,
        action: str,
        payload: Mapping[str, Any],
        expected: int,
    ) -> dict[str, Any]:
        del action
        version = await self.repository.save_background_config(
            profile_id,
            instance_id,
            patch=_mapping(payload.get("value"), "value", allow_empty=True),
            backend_overrides=_mapping(
                payload.get("backend_overrides"),
                "backend_overrides",
                allow_empty=True,
            ),
            expected_version=expected,
        )
        return {"ok": True, "version": version}

    async def _wake_authors(
        self,
        profile_id: str,
        instance_id: str,
        action: str,
        payload: Mapping[str, Any],
        expected: int,
    ) -> dict[str, Any]:
        kinds = (
            tuple(AUTHOR_ORDER) if action == "wake_all" else (_author(payload.get("author_kind")),)
        )
        forced = _mapping(
            await self.repository.force_background_authors(
                profile_id,
                instance_id,
                author_kinds=kinds,
                expected_version=expected,
            ),
            "forced background authors",
        )
        for task_id in _sequence(forced.get("active_task_ids"), "active_task_ids", allow_none=True):
            await self.ai_repository.expedite_ai_task(
                int(task_id),
                actor_id="background-manual-wake",
            )
        return {"ok": True, "version": int(forced["config_version"])}

    async def _reset_background(
        self,
        profile_id: str,
        instance_id: str,
        action: str,
        payload: Mapping[str, Any],
        expected: int,
    ) -> dict[str, Any]:
        del action
        if str(payload.get("confirmation") or "") != "RESET_BACKGROUND":
            raise ValueError("清空背景需要明确确认")
        version = await self.repository.reset_background(
            profile_id,
            instance_id,
            expected_version=expected,
        )
        return {"ok": True, "version": version}

    async def _seed_action(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = _mapping(payload.get("value"), "value", allow_empty=True)
        handlers = {
            "seed_save_definition": self._seed_save_definition,
            "seed_lore_create": self._seed_lore_create,
            "seed_lore_update": self._seed_lore_mutate,
            "seed_lore_delete": self._seed_lore_mutate,
            "seed_boundary_create": self._seed_boundary_create,
            "seed_boundary_update": self._seed_boundary_mutate,
            "seed_boundary_delete": self._seed_boundary_mutate,
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError("不支持的世界种子操作")
        result = await handler(profile_id, action, payload, value)
        self.scheduler.notify()
        return result

    async def _seed_save_definition(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        del action
        row = await self.seed_repository.update_world_definition(
            profile_id,
            value,
            expected_revision=_required_int(payload, "expected_revision"),
        )
        return {"ok": True, "definition": _definition(row)}

    async def _seed_lore_create(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        del action, payload
        row = await self.seed_repository.create_world_lore_entry(
            profile_id,
            title=str(value.get("title") or ""),
            content=str(value.get("content") or ""),
            aliases=_strings(value.get("aliases")),
            tags=_strings(value.get("tags")),
            importance=float(value.get("importance", 0.5)),
        )
        return {"ok": True, "lore": _lore(row)}

    async def _seed_lore_mutate(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        lore_id = _required_int(payload, "lore_id")
        revision = _required_int(payload, "expected_revision")
        if action == "seed_lore_update":
            row = await self.seed_repository.update_world_lore_entry(
                profile_id, lore_id, value, expected_revision=revision
            )
            return {"ok": True, "lore": _lore(row)}
        deleted = await self.seed_repository.delete_world_lore_entry(
            profile_id, lore_id, expected_revision=revision
        )
        if not deleted:
            raise ValueError("世界资料已变化，请刷新后重试")
        return {"ok": True}

    async def _seed_boundary_create(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        del action, payload
        row = await self.seed_repository.create_creative_boundary(
            profile_id,
            severity=str(value.get("severity") or "PREFERENCE"),
            category=str(value.get("category") or "CUSTOM"),
            rule_text=str(value.get("rule_text") or ""),
            positive_space=str(value.get("positive_space") or ""),
            enabled=bool(value.get("enabled", True)),
        )
        return {"ok": True, "boundary": _boundary(row)}

    async def _seed_boundary_mutate(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        boundary_id = _required_int(payload, "boundary_id")
        revision = _required_int(payload, "expected_revision")
        if action == "seed_boundary_update":
            row = await self.seed_repository.update_creative_boundary(
                profile_id,
                boundary_id,
                value,
                expected_revision=revision,
            )
            return {"ok": True, "boundary": _boundary(row)}
        deleted = await self.seed_repository.delete_creative_boundary(
            profile_id,
            boundary_id,
            expected_revision=revision,
        )
        if not deleted:
            raise ValueError("创作边界已变化，请刷新后重试")
        return {"ok": True}


def _workspace_view(source: Mapping[str, Any]) -> dict[str, Any]:
    instance = _mapping(source.get("instance"), "instance")
    authors = _author_views(source.get("authors"))
    story_sources = _story_source_views(source.get("story_sources"))
    timeline = _timeline_views(source.get("timeline"))
    current = source.get("current_view")
    return {
        "status": _workspace_status(instance),
        "current_role": _current_view(_mapping(current, "current_view", allow_empty=True)),
        "authors": authors,
        "story_sources": story_sources,
        "timeline": timeline,
        "settings": _workspace_settings(instance, authors),
        "problem_count": _problem_count(authors),
    }


def _author_views(value: Any) -> list[dict[str, Any]]:
    authors = [_author_view(_mapping(item, "author")) for item in _sequence(value, "authors")]
    authors.sort(key=lambda item: _author_index(str(item["kind"])))
    return authors


def _story_source_views(value: Any) -> list[dict[str, Any]]:
    return [
        _story_source_view(_mapping(item, "story source"))
        for item in _sequence(value, "story_sources")
    ]


def _timeline_views(value: Any) -> list[dict[str, Any]]:
    return [
        _timeline_view(_mapping(item, "timeline event")) for item in _sequence(value, "timeline")
    ]


def _workspace_status(instance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(instance.get("enabled")),
        "initialization_state": _text(instance.get("initialization_state")),
        "simulated_through_at": instance.get("simulated_through_at"),
        "last_foreground_at": instance.get("last_foreground_at"),
        "updated_at": instance.get("updated_at"),
    }


def _workspace_settings(
    instance: Mapping[str, Any],
    authors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    settings = {
        "version": int(instance.get("config_version") or 0),
        "initial_life_direction": _text(instance.get("initial_life_direction")),
        "default_backend_id": _text(instance.get("default_backend_id")),
        "proactive_frame_prewarm_enabled": bool(
            instance.get("proactive_frame_prewarm_enabled", True)
        ),
    }
    settings.update({field: int(instance.get(field) or 0) for field in _INTEGER_SETTING_FIELDS})
    settings["backend_overrides"] = {str(item["kind"]): str(item["backend_id"]) for item in authors}
    return settings


def _problem_count(authors: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for item in authors if item["failure"] is not None)


def _author_view(source: Mapping[str, Any]) -> dict[str, Any]:
    kind = _author(source.get("author_kind")).value
    failure_count = int(source.get("failure_count") or 0)
    error = _text(source.get("last_error"))
    return {
        "kind": kind,
        "status": _text(source.get("status")) or "IDLE",
        "content": _author_originals(kind, source.get("state")),
        "next_run_at": source.get("next_due_at"),
        "hard_deadline_at": source.get("hard_due_at"),
        "last_success_at": source.get("last_success_at"),
        "backend_id": _text(source.get("backend_id")),
        "failure": (
            {"count": failure_count, "message": error or "最近一次运行失败"}
            if failure_count or error
            else None
        ),
    }


def _story_source_view(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "module_text": _text(source.get("module_text")),
        "created_at": source.get("created_at"),
    }


def _timeline_view(source: Mapping[str, Any]) -> dict[str, Any]:
    event_source = _text(source.get("source")).upper()
    if event_source not in {"ORDINARY", "KEYFRAME"}:
        raise ValueError("timeline source must be ORDINARY or KEYFRAME")
    return {
        "source": event_source,
        "content": _text(source.get("content")),
        "frame_start_at": source.get("frame_start_at"),
        "frame_end_at": source.get("frame_end_at"),
        "leftover_text": _text(source.get("leftover_text")),
        "created_at": source.get("created_at"),
    }


def _current_view(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "narrative_time": _text(source.get("narrative_time")),
        "location": _text(source.get("location")),
        "doing": _text(source.get("doing")),
        "body_state": _text(source.get("body_state")),
        "mood": _text(source.get("mood")),
        "intention": _text(source.get("intention")),
        "current_concern": _text(source.get("current_concern")),
        "as_of": source.get("as_of"),
        "updated_at": source.get("updated_at"),
    }


def _author_originals(kind: str, value: Any) -> list[str]:
    """Extract only author-written prose; never manufacture titles or summaries."""

    if not isinstance(value, Mapping):
        return []
    if kind == BackgroundAuthorKind.WORLD.value:
        items = value.get("items")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            return []
        return [
            text for item in items if isinstance(item, Mapping) if (text := _text(item.get("body")))
        ]
    if kind == BackgroundAuthorKind.LIFE_DIRECTION.value:
        text = _text(value.get("text"))
        return [text] if text else []
    return []


def _author(value: Any) -> BackgroundAuthorKind:
    try:
        return BackgroundAuthorKind(str(value or "").strip().upper())
    except ValueError as exc:
        raise ValueError("请选择有效的创作层") from exc


def _author_index(value: str) -> int:
    return tuple(item.value for item in AUTHOR_ORDER).index(value)


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload or payload[key] is None:
        raise ValueError(f"{key} is required")
    return int(payload[key])


def _mapping(value: Any, label: str, *, allow_empty: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        if allow_empty and value is None:
            return {}
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str, *, allow_none: bool = False) -> tuple[Any, ...]:
    if value is None and allow_none:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _definition(row: Any) -> dict[str, Any]:
    return {
        "revision": row.revision,
        "world_brief": row.world_brief,
        "world_rules": row.world_rules,
        "life_direction": row.life_direction,
        "world_texture": row.world_texture,
        "expansion_policy": row.expansion_policy.value,
    }


def _lore(row: Any) -> dict[str, Any]:
    return {
        "lore_id": row.lore_id,
        "revision": row.revision,
        "title": row.title,
        "content": row.content,
        "aliases": list(row.aliases),
        "tags": list(row.tags),
        "importance": row.importance,
    }


def _boundary(row: Any) -> dict[str, Any]:
    return {
        "boundary_id": row.boundary_id,
        "revision": row.revision,
        "severity": row.severity.value,
        "category": row.category,
        "rule_text": row.rule_text,
        "positive_space": row.positive_space,
        "enabled": row.enabled,
    }


__all__ = ["BackgroundAdminController"]
