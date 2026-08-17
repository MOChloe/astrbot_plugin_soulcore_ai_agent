"""Player-facing read actions assembled from the existing durable sources."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ...features.timers.domain import TimerScope
from ...features.timers.main_core_views import schedule_summary
from ...features.timers.rules import next_occurrence
from .console_view_records import _outbox_view
from .delivery_attention import (
    delivery_failure_preference_key,
    parse_delivery_failure_acknowledgements,
)
from .player_views import (
    player_character_view,
    player_contact_ref,
    player_contact_view,
    player_current_life_view,
    player_intents_view,
    player_life_events_view,
    player_memories_view,
    player_people_view,
    player_portrait_view,
    player_release_notes_view,
    player_role_ref,
    player_role_view,
    player_world_view,
)
from .presentation import jsonable

_PLAYER_GUIDE_VERSION = 1
_PLAYER_GUIDE_PREFERENCE_KEY = "player.guide.seen_version"
_ADVANCED_GUIDE_VERSION = 1
_ADVANCED_GUIDE_PREFERENCE_KEY = "advanced.guide.seen_version"


def _delivery_failures_by_instance(value: object) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in jsonable(value) or []:
        if not isinstance(item, Mapping):
            continue
        instance_id = str(item.get("instance_id") or "")
        if instance_id:
            grouped.setdefault(instance_id, []).append(item)
    return grouped


def _delivery_acknowledgements_by_instance(
    preference_keys: Mapping[str, str],
    preference_values: Mapping[str, str],
) -> dict[str, frozenset[str]]:
    return {
        instance_id: frozenset(
            parse_delivery_failure_acknowledgements(preference_values.get(preference_key, ""))
        )
        for instance_id, preference_key in preference_keys.items()
    }


class PlayerPageActionsMixin:
    async def _player_bootstrap(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, selected = await self._player_roles(payload)
        guide = await self._player_guide_state()
        roles = [
            player_role_view(row, selected=self._row_profile_id(row) == selected) for row in rows
        ]
        if not selected:
            return {
                "version": self._plugin_version(),
                "roles": roles,
                "selected_role_ref": "",
                "selected_contact_ref": "",
                "contacts": [],
                "readiness": self._player_readiness(None),
                "problem_count": 0,
                "guide": guide,
            }
        contacts = await self._player_contact_list(selected)
        requested = str(payload.get("contact_ref") or "").strip()
        selected_contact = next(
            (item["contact_ref"] for item in contacts if item["contact_ref"] == requested),
            contacts[0]["contact_ref"] if contacts else "",
        )
        settings = await self._settings_snapshot(selected, "private")
        readiness = self._player_readiness(settings.get("readiness"))
        return {
            "version": self._plugin_version(),
            "roles": roles,
            "selected_role_ref": player_role_ref(selected),
            "selected_contact_ref": selected_contact,
            "contacts": contacts[:8],
            "readiness": readiness,
            "problem_count": int(not readiness["ready"])
            + sum(int(item["problem_count"]) for item in contacts),
            "guide": guide,
        }

    async def _player_contacts(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        page = max(1, int(payload.get("page") or 1))
        page_size = max(5, min(int(payload.get("page_size") or 20), 50))
        contacts = await self._player_contact_list(profile_id)
        start = (page - 1) * page_size
        items = contacts[start : start + page_size]
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": len(contacts),
            "has_more": start + page_size < len(contacts),
            "counts": {
                "private": sum(item["kind"] == "private" for item in contacts),
                "group": sum(item["kind"] == "group" for item in contacts),
            },
        }

    async def _player_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        contact = await self._player_contact(profile_id, payload, required=False)
        if contact is None:
            return {
                "contact": None,
                "current": player_current_life_view({}),
                "events": [],
                "plans": [],
                "arrangements": [],
                "problems": [],
            }
        instance_id = str(contact["instance_id"])
        background, detail, arrangements = await asyncio.gather(
            self.background.workspace(profile_id, instance_id),
            self._instance_detail(profile_id, instance_id, {"message_page_size": 5}),
            self._player_arrangements(profile_id, instance_id),
        )
        problems = await self._player_instance_problems(profile_id, instance_id, detail)
        if int(background.get("problem_count") or 0):
            problems.append(
                {
                    "code": "life_update_problem",
                    "title": "角色最近的生活没有继续更新",
                    "summary": "已经发生的经历仍然保留，可以在高级设置中查看原因。",
                    "action": "developer-background",
                }
            )
        return {
            "contact": player_contact_view(profile_id, contact),
            "current": player_current_life_view(background.get("current_role") or {}),
            "events": player_life_events_view(background.get("timeline")),
            "plans": player_intents_view(detail.get("character_intents")),
            "arrangements": arrangements,
            "problems": problems,
        }

    async def _player_relationship(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.player_profiles is None:
            raise RuntimeError("player profile administration is unavailable")
        profile_id = await self._player_profile_id(payload)
        contact = await self._player_contact(profile_id, payload)
        instance_id = str(contact["instance_id"])
        scope = str(contact["scope"])
        person_ref = str(payload.get("person_ref") or "").strip()
        portrait, knowledge, policies, detail, arrangements = await asyncio.gather(
            self.player_profiles.snapshot(
                profile_id,
                instance_id,
                scope,
                {"page": 1, "page_size": 50, "person_ref": person_ref},
            ),
            self.knowledge.knowledge_snapshot(profile_id, instance_id),
            self.profile_settings.instance_contact_override_snapshot(profile_id, instance_id),
            self._instance_detail(profile_id, instance_id, {"message_page_size": 5}),
            self._player_arrangements(profile_id, instance_id),
        )
        effective = dict(policies.get("effective") or {})
        chat_policy = dict(policies.get("chat_policy") or {})
        problems = await self._player_instance_problems(profile_id, instance_id, detail)
        contact_projection = dict(contact)
        if scope != "group":
            observed_name = str(portrait.get("selected_display_name") or "").strip()
            if observed_name:
                contact_projection["display_name"] = observed_name
        return {
            "contact": player_contact_view(
                profile_id,
                contact_projection,
                latest_at=(detail.get("message_stats") or {}).get("latest_at"),
                problem_count=len(problems),
            ),
            "people": player_people_view(portrait.get("people")),
            "selected_person_ref": str(portrait.get("selected_person_ref") or ""),
            "selected_display_name": str(portrait.get("selected_display_name") or ""),
            "portrait": player_portrait_view(portrait.get("entries")),
            "memories": player_memories_view(knowledge.get("memories")),
            "arrangements": arrangements,
            "contact_preferences": {
                "can_reply": bool(chat_policy.get("soulcore_enabled", True)),
                "can_send_images": bool(chat_policy.get("image_send_enabled", True)),
                "proactive_enabled": bool(effective.get("proactive_enabled", True)),
                "quiet_enabled": bool(effective.get("quiet_enabled", True)),
                "quiet_start": str(effective.get("quiet_start") or "23:00"),
                "quiet_end": str(effective.get("quiet_end") or "08:00"),
                "daily_limit_mode": str(effective.get("daily_limit_mode") or "LIMITED"),
                "daily_limit": effective.get("daily_success_limit"),
            },
            "problems": problems,
        }

    async def _player_about(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        character, world = await asyncio.gather(
            self.character_models.snapshot(profile_id), self.background.world_snapshot(profile_id)
        )
        return {
            "character": player_character_view(character),
            "world": player_world_view(world),
        }

    async def _release_notes(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return player_release_notes_view()

    async def _player_guide_acknowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        await self.profiles_repository.set_console_preference(
            _PLAYER_GUIDE_PREFERENCE_KEY,
            str(_PLAYER_GUIDE_VERSION),
        )
        return {"ok": True, "version": _PLAYER_GUIDE_VERSION, "seen": True}

    async def _player_guide_state(self) -> dict[str, Any]:
        seen_version = await self.profiles_repository.get_console_preference(
            _PLAYER_GUIDE_PREFERENCE_KEY
        )
        return {
            "version": _PLAYER_GUIDE_VERSION,
            "seen": seen_version == str(_PLAYER_GUIDE_VERSION),
        }

    async def _advanced_guide_acknowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        await self.profiles_repository.set_console_preference(
            _ADVANCED_GUIDE_PREFERENCE_KEY,
            str(_ADVANCED_GUIDE_VERSION),
        )
        return {"ok": True, "version": _ADVANCED_GUIDE_VERSION, "seen": True}

    async def _advanced_guide_state(self) -> dict[str, Any]:
        seen_version = await self.profiles_repository.get_console_preference(
            _ADVANCED_GUIDE_PREFERENCE_KEY
        )
        return {
            "version": _ADVANCED_GUIDE_VERSION,
            "seen": seen_version == str(_ADVANCED_GUIDE_VERSION),
        }

    async def _player_roles(self, payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
        values = [
            value
            for value in (jsonable(item) for item in await self.profiles.sync_profiles())
            if isinstance(value, dict)
        ]
        requested_ref = str(payload.get("role_ref") or "").strip()
        for row in values:
            profile_id = self._row_profile_id(row)
            if requested_ref and player_role_ref(profile_id) == requested_ref:
                return values, profile_id
        preferred = str(
            await self.profiles_repository.get_console_preference(
                "role_settings.selected_profile_id"
            )
            or ""
        )
        known = {self._row_profile_id(row) for row in values}
        selected = (
            preferred if preferred in known else (self._row_profile_id(values[0]) if values else "")
        )
        return values, selected

    async def _player_profile_id(self, payload: Mapping[str, Any]) -> str:
        rows, selected = await self._player_roles(payload)
        if not selected or not rows:
            raise ValueError("还没有可使用的角色")
        return selected

    async def _player_contact(
        self,
        profile_id: str,
        payload: Mapping[str, Any],
        *,
        required: bool = True,
    ) -> dict[str, Any] | None:
        snapshot = await self.profiles.role_instances_snapshot(profile_id)
        instances = list(snapshot.get("instances") or [])
        requested = str(payload.get("contact_ref") or "").strip()
        if requested:
            for item in instances:
                if player_contact_ref(profile_id, str(item.get("instance_id") or "")) == requested:
                    return dict(item)
            raise ValueError("联系人已经变化，请返回联系人页面重新选择")
        if instances:
            return dict(instances[0])
        if required:
            raise ValueError("还没有可以查看的联系人")
        return None

    async def _player_contact_list(self, profile_id: str) -> list[dict[str, Any]]:
        snapshot = await self.profiles.role_instances_snapshot(profile_id)
        instances = list(snapshot.get("instances") or [])
        instance_ids = tuple(str(item.get("instance_id") or "") for item in instances)
        activity = await self.timeline.conversation_repository.list_instance_message_activity(
            profile_id, instance_ids
        )
        recent_failures = await self.timeline.delivery_repository.list_profile_recent_failed_outbox(
            profile_id,
            limit_per_instance=20,
        )
        chat_policies = await asyncio.gather(
            *(
                self.profiles_repository.get_instance_chat_policy(profile_id, instance_id)
                for instance_id in instance_ids
            )
        )
        chat_policy_by_instance = {str(policy.instance_id): policy for policy in chat_policies}
        failures_by_instance = _delivery_failures_by_instance(recent_failures)
        preference_keys = {
            instance_id: delivery_failure_preference_key(profile_id, instance_id)
            for instance_id in failures_by_instance
        }
        preference_values = await self.profiles_repository.get_console_preferences(
            tuple(preference_keys.values())
        )
        acknowledged_by_instance = _delivery_acknowledgements_by_instance(
            preference_keys,
            preference_values,
        )
        rows = [
            self._player_contact_summary(
                profile_id,
                item,
                activity.get(str(item.get("instance_id") or ""), {}),
                chat_policy_by_instance.get(str(item.get("instance_id") or "")),
                failures_by_instance.get(str(item.get("instance_id") or ""), []),
                acknowledged_by_instance.get(str(item.get("instance_id") or ""), frozenset()),
            )
            for item in instances
        ]
        rows.sort(
            key=lambda item: (
                str(item.get("latest_at") or ""),
                str(item.get("display_name") or ""),
            ),
            reverse=True,
        )
        rows.sort(key=lambda item: item["kind"] == "group")
        return rows

    @staticmethod
    def _player_contact_summary(
        profile_id: str,
        instance: Mapping[str, Any],
        activity: Mapping[str, Any],
        chat_policy: Any,
        recent_failures: list[Mapping[str, Any]],
        acknowledged: frozenset[str],
    ) -> dict[str, Any]:
        projected_instance = dict(instance)
        if str(instance.get("scope") or "").lower() != "group":
            configured_name = str(
                getattr(chat_policy, "private_fallback_player_name", "") or ""
            ).strip()
            if (
                bool(getattr(chat_policy, "private_name_override_enabled", False))
                and configured_name
            ):
                projected_instance["display_name"] = configured_name
            else:
                observed_name = str(activity.get("latest_sender_name") or "").strip()
                if observed_name:
                    projected_instance["display_name"] = observed_name
        failures = sum(
            bool(_outbox_view(item, acknowledged_failures=acknowledged)["requires_attention"])
            for item in recent_failures
        )
        return player_contact_view(
            profile_id,
            projected_instance,
            latest_at=activity.get("latest_at"),
            problem_count=failures,
        )

    async def _player_instance_problems(
        self, profile_id: str, instance_id: str, detail: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        acknowledged = await self._player_acknowledged(profile_id, instance_id)
        problems = []
        for item in detail.get("outbox") or []:
            view = _outbox_view(item, acknowledged_failures=acknowledged)
            if not view["requires_attention"]:
                continue
            problems.append(
                {
                    "code": "qq_delivery_failed",
                    "title": "有一条回复没有送到 QQ",
                    "summary": view["last_error"] or "回复已经保留，可以在高级设置中查看。",
                    "occurred_at": view["not_before_at"],
                    "occurrence_id": view["occurrence_id"],
                    "action": "developer-contact",
                }
            )
        return problems

    async def _player_acknowledged(self, profile_id: str, instance_id: str) -> frozenset[str]:
        return parse_delivery_failure_acknowledgements(
            await self.profiles_repository.get_console_preference(
                delivery_failure_preference_key(profile_id, instance_id)
            )
        )

    async def _player_arrangements(self, profile_id: str, instance_id: str) -> list[dict[str, Any]]:
        if self.timer_repository is None:
            return []
        scope = TimerScope(profile_id, instance_id)
        page = await self.timer_repository.list_rules(scope, limit=20)
        now = datetime.now(UTC)
        result = []
        for rule in page.items:
            if str(rule.status.value) != "ACTIVE":
                continue
            due = next_occurrence(rule.schedule, after=now)
            timezone = str(rule.timezone or "UTC")
            result.append(
                {
                    "summary": str(rule.prompt),
                    "when": schedule_summary(rule.schedule, due, timezone, timezone),
                    "due_at": due.isoformat() if due is not None else None,
                }
            )
        return result

    @staticmethod
    def _player_readiness(value: Any) -> dict[str, Any]:
        readiness = dict(value or {})
        if bool(readiness.get("ready")):
            return {"ready": True, "problem": None}
        issues = list(readiness.get("issues") or [])
        model_issue = next(
            (item for item in issues if str(item.get("code") or "").startswith("main_model")),
            None,
        )
        return {
            "ready": False,
            "problem": {
                "code": "thinking_unavailable" if model_issue else "role_paused",
                "title": "当前角色还不能思考" if model_issue else "当前角色没有接收新消息",
                "summary": (
                    "还没有可用的思考服务。连接完成后，角色才能在 QQ 回答。"
                    if model_issue
                    else "角色资料会继续保留，可以在高级设置中重新启用。"
                ),
                "action": "developer-models" if model_issue else "developer-settings",
            },
        }

    @staticmethod
    def _row_profile_id(row: Mapping[str, Any]) -> str:
        return str(row.get("profile_id") or row.get("id") or "")

    @staticmethod
    def _plugin_version() -> str:
        from ...version import VERSION

        return f"v{VERSION}"


__all__ = ["PlayerPageActionsMixin"]
