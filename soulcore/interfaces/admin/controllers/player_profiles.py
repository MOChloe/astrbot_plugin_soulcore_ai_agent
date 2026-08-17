"""Human-oriented administrator controller for structured player profiles."""

from __future__ import annotations

import hashlib
from datetime import datetime
from secrets import token_urlsafe
from typing import Any

from ....features.player_profiles import (
    AddProfileEntry,
    PlayerProfileEntry,
    PlayerProfileScope,
    ProfileAdminEvidence,
    ProfileCategory,
    ProfileEntryDraft,
    ProfileEntryStatus,
    ProfileLayer,
    ProfileSensitivity,
    ProfileSourceType,
    RestoreProfileEntry,
    ReviseProfileEntry,
    WithdrawProfileEntry,
)

PRIVATE_SUBJECT_KEY = "private-counterpart"
ADMIN_ACTOR = "administrator"
PURGE_CONFIRMATION = "PERMANENTLY_DELETE_PROFILE_ENTRY"

_CATEGORY_LABELS = {
    ProfileCategory.SELF_DESCRIPTION: "自我描述",
    ProfileCategory.LIKE: "喜欢",
    ProfileCategory.DISLIKE: "不喜欢",
    ProfileCategory.INTEREST: "兴趣",
    ProfileCategory.HABIT: "习惯",
    ProfileCategory.COMMUNICATION_PREFERENCE: "交流偏好",
    ProfileCategory.BOUNDARY: "边界",
    ProfileCategory.AVOID_TOPIC: "避免话题",
    ProfileCategory.RELATIONSHIP_NAME: "关系称呼",
    ProfileCategory.ALIAS: "别名",
    ProfileCategory.INSTANCE_ROLE: "会话身份",
    ProfileCategory.LITERARY_IMPRESSION: "整体印象",
    ProfileCategory.OTHER: "其他",
}


class PlayerProfilesAdminController:
    def __init__(self, repository: Any, identity: Any) -> None:
        self.repository = repository
        self.identity = identity

    async def snapshot(
        self,
        profile_id: str,
        instance_id: str,
        scope: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        context = await self.identity.context(profile_id, instance_id)
        subjects = await self._subjects(profile_id, instance_id, scope, context)
        page = max(1, int(payload.get("page") or 1))
        page_size = max(5, min(int(payload.get("page_size") or 20), 50))
        start = (page - 1) * page_size
        selected = self._selected_subject(
            subjects,
            str(payload.get("person_ref") or ""),
            profile_id,
            instance_id,
        )
        people = []
        for subject_key, label in subjects[start : start + page_size]:
            snapshot = await self.repository.load_player_profile(
                PlayerProfileScope(profile_id, instance_id, subject_key)
            )
            people.append(
                {
                    "person_ref": _person_ref(profile_id, instance_id, subject_key),
                    "display_name": label,
                    "active_count": len(snapshot.effective_entries),
                    "removed_count": len(snapshot.entries) - len(snapshot.effective_entries),
                    "selected": selected is not None and subject_key == selected[0],
                }
            )
        if selected is None:
            return {
                "people": people,
                "pagination": _pagination(page, page_size, len(subjects)),
                "selected_person_ref": "",
                "profile_version": 0,
                "entries": [],
            }
        subject_key, label = selected
        snapshot = await self.repository.load_player_profile(
            PlayerProfileScope(profile_id, instance_id, subject_key)
        )
        entries = sorted(
            snapshot.entries,
            key=lambda item: (item.status is ProfileEntryStatus.ACTIVE, item.updated_at),
            reverse=True,
        )
        return {
            "people": people,
            "pagination": _pagination(page, page_size, len(subjects)),
            "selected_person_ref": _person_ref(profile_id, instance_id, subject_key),
            "selected_display_name": label,
            "profile_version": snapshot.version,
            "entries": [self._entry_view(item, context) for item in entries],
            "options": {
                "categories": [
                    {"value": item.value, "label": _CATEGORY_LABELS[item]}
                    for item in ProfileCategory
                ],
                "sensitivities": [
                    {"value": "NORMAL", "label": "普通"},
                    {"value": "PRIVATE", "label": "私密"},
                    {"value": "SENSITIVE", "label": "敏感"},
                ],
            },
        }

    async def entry_detail(
        self,
        profile_id: str,
        instance_id: str,
        scope: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        context = await self.identity.context(profile_id, instance_id)
        subjects = await self._subjects(profile_id, instance_id, scope, context)
        subject_key, _label = self._require_subject(
            subjects,
            str(payload.get("person_ref") or ""),
            profile_id,
            instance_id,
        )
        profile_scope = PlayerProfileScope(profile_id, instance_id, subject_key)
        snapshot = await self.repository.load_player_profile(profile_scope)
        entry = self._require_entry(
            snapshot.entries, profile_scope, str(payload.get("entry_ref") or "")
        )
        revisions = await self.repository.list_entry_revisions(profile_scope, entry.entry_id)
        return {
            "profile_version": snapshot.version,
            "entry": self._entry_view(entry, context),
            "history": [self._revision_view(item, context) for item in revisions],
        }

    async def action(
        self,
        profile_id: str,
        instance_id: str,
        scope: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        action, reason = _validated_action(payload)
        context = await self.identity.context(profile_id, instance_id)
        subjects = await self._subjects(profile_id, instance_id, scope, context)
        subject_key, _label = self._require_subject(
            subjects,
            str(payload.get("person_ref") or ""),
            profile_id,
            instance_id,
        )
        profile_scope = PlayerProfileScope(profile_id, instance_id, subject_key)
        snapshot = await self.repository.load_player_profile(profile_scope)
        expected_profile_version = _integer(payload, "expected_profile_version", minimum=0)
        if expected_profile_version != snapshot.version:
            raise ValueError("人物肖像已经发生变化，请刷新后重试")
        now = datetime.now().astimezone()
        evidence = (ProfileAdminEvidence(profile_scope, ADMIN_ACTOR, reason),)
        if action == "create":
            record = _mapping(payload.get("record"), "record")
            command = AddProfileEntry(
                idempotency_key=_idempotency_key(),
                scope=profile_scope,
                expected_profile_version=snapshot.version,
                entry_id=f"profile-entry:admin:{token_urlsafe(18)}",
                draft=_draft_from_record(record, evidence, now),
            )
            result = await self.repository.commit_admin_command(command, now=now)
            return {"ok": True, "profile_version": result.profile_version}
        entry = self._require_entry(
            snapshot.entries,
            profile_scope,
            str(payload.get("entry_ref") or ""),
        )
        expected_entry_version = _integer(payload, "expected_entry_version", minimum=1)
        if expected_entry_version != entry.version:
            raise ValueError("这条人物肖像已经发生变化，请刷新后重试")
        if action == "purge":
            entry_ref = _entry_ref(profile_scope, entry.entry_id)
            if str(payload.get("confirm_entry_ref") or "") != entry_ref:
                raise ValueError("永久删除确认对象与当前条目不一致")
            if str(payload.get("confirm") or "") != PURGE_CONFIRMATION:
                raise ValueError("永久删除需要明确的服务端确认口令")
            version = await self.repository.purge_entry(
                profile_scope,
                entry.entry_id,
                expected_profile_version=snapshot.version,
                expected_entry_version=entry.version,
                now=now,
            )
            return {"ok": True, "profile_version": version, "purged": True}
        if action == "update":
            if entry.status is not ProfileEntryStatus.ACTIVE:
                raise ValueError("已移除的条目需要先恢复再编辑")
            patch = _mapping(payload.get("patch"), "patch")
            command = ReviseProfileEntry(
                idempotency_key=_idempotency_key(),
                scope=profile_scope,
                expected_profile_version=snapshot.version,
                entry_id=entry.entry_id,
                expected_entry_version=entry.version,
                draft=_draft_from_patch(entry, patch, evidence, now),
            )
        elif action == "withdraw":
            command = WithdrawProfileEntry(
                _idempotency_key(),
                profile_scope,
                snapshot.version,
                entry.entry_id,
                entry.version,
                evidence,
            )
        else:
            command = RestoreProfileEntry(
                _idempotency_key(),
                profile_scope,
                snapshot.version,
                entry.entry_id,
                entry.version,
                evidence,
            )
        result = await self.repository.commit_admin_command(command, now=now)
        return {"ok": True, "profile_version": result.profile_version}

    async def _subjects(
        self,
        profile_id: str,
        instance_id: str,
        scope: str,
        context: Any,
    ) -> list[tuple[str, str]]:
        if scope == "private":
            return [(PRIVATE_SUBJECT_KEY, context.private_display_name or "当前对方")]
        rows = [
            (str(item.participant_id), str(item.display_name or "一位群成员"))
            for item in context.participants
            if str(item.participant_id).strip()
        ]
        known = {subject_key for subject_key, _label in rows}
        for subject_key in await self.repository.list_subject_keys(profile_id, instance_id):
            if subject_key not in known:
                rows.append((subject_key, "一位群成员"))
        return rows

    @staticmethod
    def _selected_subject(
        subjects: list[tuple[str, str]],
        requested_ref: str,
        profile_id: str,
        instance_id: str,
    ) -> tuple[str, str] | None:
        if not subjects:
            return None
        if not requested_ref:
            return subjects[0]
        return next(
            (
                item
                for item in subjects
                if _person_ref(profile_id, instance_id, item[0]) == requested_ref
            ),
            None,
        )

    @staticmethod
    def _require_subject(
        subjects: list[tuple[str, str]],
        requested_ref: str,
        profile_id: str,
        instance_id: str,
    ) -> tuple[str, str]:
        if not requested_ref:
            raise ValueError("person_ref is required")
        for item in subjects:
            if requested_ref == _person_ref(profile_id, instance_id, item[0]):
                return item
        raise ValueError("人物引用已失效，请刷新页面")

    def _require_entry(
        self,
        entries: tuple[PlayerProfileEntry, ...],
        scope: PlayerProfileScope,
        requested_ref: str,
    ) -> PlayerProfileEntry:
        if not requested_ref:
            raise ValueError("entry_ref is required")
        entry = next(
            (item for item in entries if _entry_ref(scope, item.entry_id) == requested_ref),
            None,
        )
        if entry is None:
            raise ValueError("人物肖像条目引用已失效，请刷新页面")
        return entry

    def _entry_view(self, entry: PlayerProfileEntry, context: Any) -> dict[str, Any]:
        return {
            "entry_ref": _entry_ref(entry.scope, entry.entry_id),
            "entry_version": entry.version,
            "layer": entry.layer.value,
            "layer_label": "明确事实" if entry.layer is ProfileLayer.PLAYER_FACT else "角色观察",
            "category": entry.category.value,
            "category_label": _CATEGORY_LABELS[entry.category],
            "text": self.identity.render(entry.text, context),
            "confidence": entry.confidence,
            "sensitivity": entry.sensitivity.value,
            "sensitivity_label": {
                ProfileSensitivity.NORMAL: "普通",
                ProfileSensitivity.PRIVATE: "私密",
                ProfileSensitivity.SENSITIVE: "敏感",
            }[entry.sensitivity],
            "status": entry.status.value,
            "status_label": "有效" if entry.status is ProfileEntryStatus.ACTIVE else "已移除",
            "updated_at": entry.updated_at.isoformat(),
        }

    def _revision_view(self, entry: PlayerProfileEntry, context: Any) -> dict[str, Any]:
        reasons = [
            item.reason
            for item in (*entry.evidence, *entry.withdrawal_evidence)
            if isinstance(item, ProfileAdminEvidence)
        ]
        return {
            **self._entry_view(entry, context),
            "reason": reasons[-1] if reasons else "由对话证据形成",
        }


def _draft_from_record(
    record: dict[str, Any], evidence: tuple[ProfileAdminEvidence, ...], now: datetime
) -> ProfileEntryDraft:
    layer = ProfileLayer(str(record.get("layer") or ""))
    confidence = record.get("confidence")
    return ProfileEntryDraft(
        layer=layer,
        category=ProfileCategory(str(record.get("category") or "")),
        text=str(record.get("text") or ""),
        source_type=(
            ProfileSourceType.PLAYER_CORRECTION
            if layer is ProfileLayer.PLAYER_FACT
            else ProfileSourceType.AI_OBSERVATION
        ),
        evidence=evidence,
        confirmed_at=now,
        confidence=(float(confidence) if confidence is not None else None),
        sensitivity=ProfileSensitivity(str(record.get("sensitivity") or "NORMAL")),
    )


def _draft_from_patch(
    entry: PlayerProfileEntry,
    patch: dict[str, Any],
    evidence: tuple[ProfileAdminEvidence, ...],
    now: datetime,
) -> ProfileEntryDraft:
    allowed = {"layer", "category", "text", "confidence", "sensitivity"}
    supplied = allowed.intersection(patch)
    if not supplied:
        raise ValueError("修改时至少需要提交一个变化字段")
    if set(patch) - allowed:
        raise ValueError("修改内容包含不支持的字段")
    layer = ProfileLayer(str(patch.get("layer") or entry.layer.value))
    confidence = patch.get(
        "confidence",
        entry.confidence if layer is entry.layer else None,
    )
    if layer is ProfileLayer.PLAYER_FACT:
        if confidence is not None:
            raise ValueError("明确事实不能设置可信度")
        source_type = ProfileSourceType.PLAYER_CORRECTION
    else:
        if confidence is None:
            raise ValueError("角色观察必须设置可信度")
        source_type = ProfileSourceType.AI_OBSERVATION
    return ProfileEntryDraft(
        layer=layer,
        category=ProfileCategory(str(patch.get("category") or entry.category.value)),
        text=str(patch.get("text") or entry.text),
        source_type=source_type,
        evidence=evidence,
        confirmed_at=now,
        confidence=float(confidence) if confidence is not None else None,
        sensitivity=ProfileSensitivity(str(patch.get("sensitivity") or entry.sensitivity.value)),
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _validated_action(payload: dict[str, Any]) -> tuple[str, str]:
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"create", "update", "withdraw", "restore", "purge"}:
        raise ValueError("不支持的人物肖像操作")
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        raise ValueError("请填写本次操作原因")
    return action, reason


def _integer(payload: dict[str, Any], name: str, *, minimum: int) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _idempotency_key() -> str:
    return f"admin-profile:{token_urlsafe(18)}"


def _person_ref(
    profile_id: str,
    instance_id: str,
    subject_key: str,
) -> str:
    source = f"{profile_id}\x1f{instance_id}\x1f{subject_key}"
    return "person-" + hashlib.sha256(source.encode()).hexdigest()[:24]


def _entry_ref(scope: PlayerProfileScope, entry_id: str) -> str:
    source = "\x1f".join((*scope.persistence_key, entry_id))
    return "portrait-" + hashlib.sha256(source.encode()).hexdigest()[:24]


def _pagination(page: int, page_size: int, total: int) -> dict[str, Any]:
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": page * page_size < total,
    }


__all__ = ["PURGE_CONFIRMATION", "PlayerProfilesAdminController"]
