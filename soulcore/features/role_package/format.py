"""Strict version-one ``.soulcore-role`` ZIP container implementation."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from ..character_model import (
    MAX_LIST_ITEMS,
    MAX_TRIGGER_KEYS,
    MAX_TRIGGER_LOOKBACK_TURNS,
    MAX_TRIGGER_RULES,
    MIN_TRIGGER_LOOKBACK_TURNS,
)
from ..media import MAX_IMAGE_BYTES
from .domain import (
    MAX_ARCHIVE_BYTES,
    MAX_EXPANDED_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PACKAGE_FILES,
    MAX_ROLE_JSON_BYTES,
    ROLE_PACKAGE_CONTENT_MODE,
    ROLE_PACKAGE_FORMAT,
    ROLE_PACKAGE_FORMAT_VERSION,
    PackageAsset,
    ParsedRolePackage,
    RolePackageError,
)
from .model_patch import (
    CHARACTER_ROOT_FIELDS,
    CHARACTER_SECTION_FIELDS,
    PORTRAIT_SCOPES,
    PROMPT_GROUP_FIELDS,
    WORLD_DEFINITION_FIELDS,
    WORLD_ROOT_FIELDS,
    require_portable_identities,
)

_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "content_mode",
        "title",
        "generator_version",
        "role_file",
        "assets",
    }
)
_FILE_FIELDS = frozenset({"path", "byte_size", "sha256"})
_ASSET_FIELDS = frozenset({"scope", "path", "mime_type", "byte_size", "sha256"})
_STRING_LIST_FIELDS: dict[str, frozenset[str]] = {
    "identity": frozenset({"aliases", "facts"}),
    "personality": CHARACTER_SECTION_FIELDS["personality"],
    "social": CHARACTER_SECTION_FIELDS["social"],
    "preferences": CHARACTER_SECTION_FIELDS["preferences"],
    "language": CHARACTER_SECTION_FIELDS["language"],
    "visual": CHARACTER_SECTION_FIELDS["visual"],
    "capabilities": CHARACTER_SECTION_FIELDS["capabilities"],
}
_NESTED_ARCHIVE_SUFFIXES = frozenset(
    {".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz", ".soulcore-role"}
)


def build_role_package(
    target: str | Path,
    *,
    title: str,
    generator_version: str,
    role_document: Mapping[str, Any],
    portrait_assets: Mapping[str, tuple[bytes, str, str]],
) -> Path:
    """Write one deterministic package and return its resolved path."""

    clean_title = _bounded_string(title, "角色标题", maximum=200)
    clean_version = _bounded_string(generator_version, "生成器版本", maximum=80)
    document, assets = _export_role_document(role_document, portrait_assets)
    role_bytes = _json_bytes(document)
    if len(role_bytes) > MAX_ROLE_JSON_BYTES:
        raise RolePackageError("角色内容超过 128 MiB，无法导出。")
    manifest_bytes = _export_manifest_bytes(clean_title, clean_version, role_bytes, assets)
    return _write_role_package_archive(target, manifest_bytes, role_bytes, assets)


def _export_role_document(
    role_document: Mapping[str, Any],
    portrait_assets: Mapping[str, tuple[bytes, str, str]],
) -> tuple[dict[str, Any], list[PackageAsset]]:
    document = json.loads(json.dumps(dict(role_document), ensure_ascii=False))
    assets: list[PackageAsset] = []
    existing_portraits = document.get("portraits")
    portraits: dict[str, Any] = (
        dict(existing_portraits) if isinstance(existing_portraits, Mapping) else {}
    )
    for scope in PORTRAIT_SCOPES:
        item = portrait_assets.get(scope)
        if item is None:
            continue
        asset, portrait = _export_portrait(scope, item)
        assets.append(asset)
        portraits[scope] = portrait
    if portraits:
        document["portraits"] = portraits
    _validate_role_document(document)
    require_portable_identities(document)
    return document, assets


def _export_portrait(
    scope: str, item: tuple[bytes, str, str]
) -> tuple[PackageAsset, dict[str, str]]:
    data, mime_type, label = item
    mime = str(mime_type or "").strip().lower()
    extension = _MIME_EXTENSIONS.get(mime)
    if extension is None:
        raise RolePackageError(f"{_scope_label(scope)}立绘格式不受支持。")
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise RolePackageError(f"{_scope_label(scope)}立绘大小不符合限制。")
    asset_path = f"assets/identity/{scope}{extension}"
    digest = hashlib.sha256(data).hexdigest()
    asset = PackageAsset(scope, asset_path, mime, digest, len(data), bytes(data))
    return asset, {"asset": asset_path, "label": str(label or "").strip()[:80]}


def _export_manifest_bytes(
    title: str,
    generator_version: str,
    role_bytes: bytes,
    assets: Sequence[PackageAsset],
) -> bytes:
    manifest = {
        "format": ROLE_PACKAGE_FORMAT,
        "format_version": ROLE_PACKAGE_FORMAT_VERSION,
        "content_mode": ROLE_PACKAGE_CONTENT_MODE,
        "title": title,
        "generator_version": generator_version,
        "role_file": {
            "path": "role.json",
            "byte_size": len(role_bytes),
            "sha256": hashlib.sha256(role_bytes).hexdigest(),
        },
        "assets": [
            {
                "scope": item.scope,
                "path": item.path,
                "mime_type": item.mime_type,
                "byte_size": item.byte_size,
                "sha256": item.sha256,
            }
            for item in assets
        ],
    }
    manifest_bytes = _json_bytes(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise RolePackageError("角色包清单超过 64 KiB，无法导出。")
    return manifest_bytes


def _write_role_package_archive(
    target: str | Path,
    manifest_bytes: bytes,
    role_bytes: bytes,
    assets: Sequence[PackageAsset],
) -> Path:
    destination = Path(target).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            destination,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            _write_member(archive, "manifest.json", manifest_bytes)
            _write_member(archive, "role.json", role_bytes)
            for item in assets:
                _write_member(archive, item.path, item.data)
        if destination.stat().st_size > MAX_ARCHIVE_BYTES:
            raise RolePackageError("角色包压缩后超过 128 MiB，无法导出。")
        return destination
    except Exception:
        with suppress(FileNotFoundError):
            destination.unlink()
        raise


def read_role_package(path: str | Path) -> ParsedRolePackage:
    """Fully validate a local package without following any external reference."""

    source = Path(path).resolve(strict=False)
    if not source.is_file():
        raise RolePackageError("上传的角色包临时文件已经失效，请重新选择。")
    size = source.stat().st_size
    if size < 1 or size > MAX_ARCHIVE_BYTES:
        raise RolePackageError("角色包必须小于 128 MiB。")
    archive_digest = _file_sha256(source)
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            members = _validated_members(archive)
            manifest_bytes = _read_member(
                archive, members["manifest.json"], maximum=MAX_MANIFEST_BYTES
            )
            manifest = _json_object(manifest_bytes, "manifest.json")
            role_descriptor, asset_descriptors = _validate_manifest(manifest)
            expected_paths = {
                "manifest.json",
                "role.json",
                *(str(item["path"]) for item in asset_descriptors),
            }
            if set(members) != expected_paths:
                raise RolePackageError("角色包包含未声明文件或缺少清单中的文件。")

            role_bytes = _read_member(archive, members["role.json"], maximum=MAX_ROLE_JSON_BYTES)
            _verify_descriptor(role_descriptor, role_bytes, expected_path="role.json")
            role = _json_object(role_bytes, "role.json")
            _validate_role_document(role)
            require_portable_identities(role)

            assets: dict[str, PackageAsset] = {}
            for descriptor in asset_descriptors:
                scope = str(descriptor["scope"])
                asset_path = str(descriptor["path"])
                raw = _read_member(archive, members[asset_path], maximum=MAX_IMAGE_BYTES)
                _verify_descriptor(descriptor, raw, expected_path=asset_path)
                assets[scope] = PackageAsset(
                    scope=scope,
                    path=asset_path,
                    mime_type=str(descriptor["mime_type"]),
                    sha256=str(descriptor["sha256"]),
                    byte_size=int(descriptor["byte_size"]),
                    data=raw,
                )
            _validate_asset_references(role, assets)
    except RolePackageError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise RolePackageError("角色包已损坏或不是 SoulCore 生成的有效文件。") from exc
    return ParsedRolePackage(
        title=str(manifest["title"]),
        generator_version=str(manifest["generator_version"]),
        role=role,
        assets=assets,
        archive_sha256=archive_digest,
        archive_path=source,
    )


def _validated_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not 2 <= len(infos) <= MAX_PACKAGE_FILES:
        raise RolePackageError("角色包只能包含清单、角色内容和最多两张立绘。")
    result: dict[str, zipfile.ZipInfo] = {}
    casefolded: set[str] = set()
    expanded = 0
    for info in infos:
        name, folded = _validated_member_info(info)
        if name in result or folded in casefolded:
            raise RolePackageError("角色包包含重复或大小写冲突的文件路径。")
        expanded += int(info.file_size)
        if expanded > MAX_EXPANDED_BYTES:
            raise RolePackageError("角色包展开后超过 192 MiB。")
        result[name] = info
        casefolded.add(folded)
    if "manifest.json" not in result or "role.json" not in result:
        raise RolePackageError("角色包缺少 manifest.json 或 role.json。")
    if result["manifest.json"].file_size > MAX_MANIFEST_BYTES:
        raise RolePackageError("manifest.json 超过 64 KiB。")
    if result["role.json"].file_size > MAX_ROLE_JSON_BYTES:
        raise RolePackageError("role.json 超过 128 MiB。")
    return result


def _validated_member_info(info: zipfile.ZipInfo) -> tuple[str, str]:
    name = info.filename
    _validate_member_path(name)
    if info.flag_bits & 0x1:
        raise RolePackageError("角色包不能使用 ZIP 加密。")
    if info.is_dir():
        raise RolePackageError("角色包不能包含目录项。")
    mode = (info.external_attr >> 16) & 0xFFFF
    if info.create_system != 3 or not stat.S_ISREG(mode):
        raise RolePackageError("角色包包含符号链接或特殊文件。")
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise RolePackageError("角色包使用了不支持的压缩方式。")
    if info.file_size < 0 or info.compress_size < 0:
        raise RolePackageError("角色包文件大小无效。")
    return name, name.casefold()


def _validate_member_path(name: str) -> None:
    if not name or "\\" in name or "\x00" in name:
        raise RolePackageError("角色包包含不安全的文件路径。")
    parts = name.split("/")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or (parts and ":" in parts[0])
    ):
        raise RolePackageError("角色包包含绝对路径或路径穿越。")
    if path.suffix.casefold() in _NESTED_ARCHIVE_SUFFIXES:
        raise RolePackageError("角色包不能嵌套其他压缩包。")


def _validate_manifest(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_manifest_header(value)
    role_descriptor = _manifest_role_descriptor(value["role_file"])
    descriptors = _manifest_asset_descriptors(value["assets"])
    return role_descriptor, descriptors


def _validate_manifest_header(value: Mapping[str, Any]) -> None:
    _exact_fields(value, _MANIFEST_FIELDS, "manifest.json")
    if value["format"] != ROLE_PACKAGE_FORMAT:
        raise RolePackageError("这不是 SoulCore 角色包。")
    if type(value["format_version"]) is not int:
        raise RolePackageError("角色包格式版本必须是整数。")
    if int(value["format_version"]) != ROLE_PACKAGE_FORMAT_VERSION:
        raise RolePackageError("角色包格式版本不受支持，请升级 SoulCore 后重试。")
    if value["content_mode"] != ROLE_PACKAGE_CONTENT_MODE:
        raise RolePackageError("角色包内容模式不受支持。")
    _bounded_string(value["title"], "角色标题", maximum=200)
    _bounded_string(value["generator_version"], "生成器版本", maximum=80)


def _manifest_role_descriptor(value: Any) -> dict[str, Any]:
    role_file = _mapping(value, "manifest.role_file")
    _exact_fields(role_file, _FILE_FIELDS, "manifest.role_file")
    role_descriptor = dict(role_file)
    _validate_descriptor(role_descriptor, "manifest.role_file")
    if role_descriptor["path"] != "role.json":
        raise RolePackageError("角色内容路径必须是 role.json。")
    return role_descriptor


def _manifest_asset_descriptors(value: Any) -> list[dict[str, Any]]:
    raw_assets = _sequence(value, "manifest.assets")
    if len(raw_assets) > 2:
        raise RolePackageError("角色包最多包含私聊和群聊两张立绘。")
    descriptors: list[dict[str, Any]] = []
    seen_scopes: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_assets):
        field = f"manifest.assets[{index}]"
        descriptor = _manifest_asset_descriptor(raw, field)
        scope = descriptor["scope"]
        if scope not in PORTRAIT_SCOPES or scope in seen_scopes:
            raise RolePackageError("立绘清单的范围重复或无效。")
        if descriptor["path"] in seen_paths:
            raise RolePackageError("立绘清单包含重复路径。")
        seen_scopes.add(str(scope))
        seen_paths.add(str(descriptor["path"]))
        descriptors.append(descriptor)
    return descriptors


def _manifest_asset_descriptor(value: Any, field: str) -> dict[str, Any]:
    item = _mapping(value, field)
    _exact_fields(item, _ASSET_FIELDS, field)
    descriptor = dict(item)
    _validate_descriptor(descriptor, field)
    scope = descriptor["scope"]
    mime = descriptor["mime_type"]
    extension = _MIME_EXTENSIONS.get(mime) if isinstance(mime, str) else None
    expected_path = f"assets/identity/{scope}{extension or ''}"
    if extension is None or descriptor["path"] != expected_path:
        raise RolePackageError(f"{_scope_label(str(scope))}立绘路径或格式不匹配。")
    if int(descriptor["byte_size"]) > MAX_IMAGE_BYTES:
        raise RolePackageError(f"{_scope_label(str(scope))}立绘超过 20 MiB。")
    return descriptor


def _validate_role_document(value: Mapping[str, Any]) -> None:
    _reject_unknown(value, {"character", "world", "portraits"}, "role")
    if "character" in value:
        _validate_character(_mapping(value["character"], "role.character"))
    if "world" in value:
        _validate_world(_mapping(value["world"], "role.world"))
    if "portraits" in value:
        _validate_portraits(_mapping(value["portraits"], "role.portraits"))


def _validate_character(value: Mapping[str, Any]) -> None:
    _reject_unknown(value, set(CHARACTER_ROOT_FIELDS), "role.character")
    _validate_character_sections(value)
    if "dialogue_reference" in value and not isinstance(value["dialogue_reference"], str):
        raise RolePackageError("role.character.dialogue_reference 必须是字符串。")
    if "custom_prompts" in value:
        _validate_custom_prompts(value["custom_prompts"])
    if "trigger_rules" in value:
        _validate_trigger_rules(value["trigger_rules"])


def _validate_character_sections(value: Mapping[str, Any]) -> None:
    for section, fields in CHARACTER_SECTION_FIELDS.items():
        if section not in value:
            continue
        content = _mapping(value[section], f"role.character.{section}")
        _reject_unknown(content, set(fields), f"role.character.{section}")
        list_fields = _STRING_LIST_FIELDS.get(section, frozenset())
        for field, raw in content.items():
            if field in list_fields:
                _string_list(raw, f"role.character.{section}.{field}")
            elif not isinstance(raw, str):
                raise RolePackageError(f"role.character.{section}.{field} 必须是字符串。")


def _validate_custom_prompts(raw: Any) -> None:
    prompts = _mapping(raw, "role.character.custom_prompts")
    _reject_unknown(prompts, set(PROMPT_GROUP_FIELDS), "role.character.custom_prompts")
    for group, raw_fields in prompts.items():
        fields = _mapping(raw_fields, f"role.character.custom_prompts.{group}")
        _reject_unknown(
            fields,
            set(PROMPT_GROUP_FIELDS[str(group)]),
            f"role.character.custom_prompts.{group}",
        )
        if any(not isinstance(item, str) for item in fields.values()):
            raise RolePackageError(f"role.character.custom_prompts.{group} 只能包含字符串。")


def _validate_trigger_rules(raw: Any) -> None:
    rules = _sequence(raw, "role.character.trigger_rules")
    if len(rules) > MAX_TRIGGER_RULES:
        raise RolePackageError("触发规则数量超过当前 SoulCore 限制。")
    for index, raw_rule in enumerate(rules):
        field = f"role.character.trigger_rules[{index}]"
        rule = _mapping(raw_rule, field)
        _exact_fields(rule, {"keys", "lookback_turns", "content"}, field)
        keys = _string_list(rule["keys"], f"{field}.keys")
        if not keys or len(keys) > MAX_TRIGGER_KEYS:
            raise RolePackageError(f"{field}.keys 数量无效。")
        lookback = rule["lookback_turns"]
        if type(lookback) is not int or not (
            MIN_TRIGGER_LOOKBACK_TURNS <= lookback <= MAX_TRIGGER_LOOKBACK_TURNS
        ):
            raise RolePackageError(f"{field}.lookback_turns 超出允许范围。")
        if not isinstance(rule["content"], str):
            raise RolePackageError(f"{field}.content 必须是字符串。")


def _validate_world(value: Mapping[str, Any]) -> None:
    _reject_unknown(value, set(WORLD_ROOT_FIELDS), "role.world")
    if "definition" in value:
        _validate_world_definition(value["definition"])
    if "lore" in value:
        _validate_world_lore(value["lore"])
    if "boundaries" in value:
        _validate_world_boundaries(value["boundaries"])


def _validate_world_definition(raw: Any) -> None:
    definition = _mapping(raw, "role.world.definition")
    _reject_unknown(definition, set(WORLD_DEFINITION_FIELDS), "role.world.definition")
    for field, value in definition.items():
        if field == "expansion_policy":
            if value not in {"OPEN", "CANON_GUARDED"}:
                raise RolePackageError("世界扩展方式无效。")
        elif not isinstance(value, str):
            raise RolePackageError(f"role.world.definition.{field} 必须是字符串。")


def _validate_world_lore(raw: Any) -> None:
    lore = _sequence(raw, "role.world.lore")
    if len(lore) > 500:
        raise RolePackageError("世界资料最多包含 500 条。")
    titles: set[str] = set()
    for index, raw_item in enumerate(lore):
        field = f"role.world.lore[{index}]"
        item = _mapping(raw_item, field)
        _exact_fields(item, {"title", "aliases", "tags", "content", "importance"}, field)
        if not isinstance(item["title"], str) or not isinstance(item["content"], str):
            raise RolePackageError(f"{field} 的标题和内容必须是字符串。")
        _string_list(item["aliases"], f"{field}.aliases")
        _string_list(item["tags"], f"{field}.tags")
        score = item["importance"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise RolePackageError(f"{field}.importance 必须在 0 到 1 之间。")
        title = str(item["title"]).strip()
        if not title or title in titles:
            raise RolePackageError("世界资料标题不能为空或重复。")
        titles.add(title)


def _validate_world_boundaries(raw: Any) -> None:
    boundaries = _sequence(raw, "role.world.boundaries")
    if len(boundaries) > 500:
        raise RolePackageError("创作边界最多包含 500 条。")
    for index, raw_item in enumerate(boundaries):
        field = f"role.world.boundaries[{index}]"
        item = _mapping(raw_item, field)
        _exact_fields(
            item,
            {"severity", "category", "rule_text", "positive_space", "enabled"},
            field,
        )
        if item["severity"] not in {"HARD", "PREFERENCE"}:
            raise RolePackageError(f"{field}.severity 无效。")
        text_fields = ("category", "rule_text", "positive_space")
        if any(not isinstance(item[name], str) for name in text_fields):
            raise RolePackageError(f"{field} 包含非法类型。")
        if type(item["enabled"]) is not bool:
            raise RolePackageError(f"{field} 包含非法类型。")


def _validate_portraits(value: Mapping[str, Any]) -> None:
    _reject_unknown(value, set(PORTRAIT_SCOPES), "role.portraits")
    for scope, raw in value.items():
        field = f"role.portraits.{scope}"
        item = _mapping(raw, field)
        if item.get("clear") is True:
            _exact_fields(item, {"clear"}, field)
            continue
        _exact_fields(item, {"asset", "label"}, field)
        if not isinstance(item["asset"], str) or not isinstance(item["label"], str):
            raise RolePackageError(f"{field} 的资产路径和名称必须是字符串。")
        if len(item["label"].strip()) > 80:
            raise RolePackageError(f"{field}.label 不能超过 80 个字符。")


def _validate_asset_references(role: Mapping[str, Any], assets: Mapping[str, PackageAsset]) -> None:
    portraits = role.get("portraits")
    mapping = dict(portraits) if isinstance(portraits, Mapping) else {}
    referenced: set[str] = set()
    for scope, raw in mapping.items():
        if not isinstance(raw, Mapping) or raw.get("clear") is True:
            continue
        asset = assets.get(str(scope))
        if asset is None or raw.get("asset") != asset.path:
            raise RolePackageError(f"{_scope_label(str(scope))}立绘引用未在清单中声明。")
        referenced.add(str(scope))
    if referenced != set(assets):
        raise RolePackageError("角色包清单包含未被角色内容引用的立绘。")


def _validate_descriptor(value: Mapping[str, Any], field: str) -> None:
    if not isinstance(value.get("path"), str):
        raise RolePackageError(f"{field}.path 必须是字符串。")
    size = value.get("byte_size")
    if type(size) is not int or size < 0:
        raise RolePackageError(f"{field}.byte_size 必须是非负整数。")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise RolePackageError(f"{field}.sha256 无效。")


def _verify_descriptor(descriptor: Mapping[str, Any], data: bytes, *, expected_path: str) -> None:
    if descriptor["path"] != expected_path:
        raise RolePackageError("角色包清单中的文件路径不一致。")
    if int(descriptor["byte_size"]) != len(data):
        raise RolePackageError("角色包文件大小校验失败。")
    if str(descriptor["sha256"]) != hashlib.sha256(data).hexdigest():
        raise RolePackageError("角色包文件 SHA-256 校验失败。")


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, maximum: int) -> bytes:
    if info.file_size > maximum:
        raise RolePackageError("角色包中的文件超过安全限制。")
    chunks: list[bytes] = []
    total = 0
    with archive.open(info, mode="r") as handle:
        while True:
            chunk = handle.read(min(1024 * 1024, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RolePackageError("角色包展开内容超过安全限制。")
            chunks.append(chunk)
    if total != info.file_size:
        raise RolePackageError("角色包文件长度与 ZIP 清单不一致。")
    return b"".join(chunks)


def _json_object(data: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except RolePackageError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RolePackageError(f"{filename} 不是有效的 UTF-8 JSON。") from exc
    if not isinstance(value, dict):
        raise RolePackageError(f"{filename} 顶层必须是对象。")
    _reject_null(value, filename)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RolePackageError("角色包 JSON 包含重复字段。")
        result[key] = value
    return result


def _reject_null(value: Any, field: str, *, depth: int = 0) -> None:
    if depth > 64:
        raise RolePackageError("角色包 JSON 嵌套层级过深。")
    if value is None:
        raise RolePackageError(f"{field} 不能使用 null。")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_null(item, f"{field}.{key}", depth=depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_null(item, f"{field}[{index}]", depth=depth + 1)


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RolePackageError(f"{field} 必须是对象。")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RolePackageError(f"{field} 必须是数组。")
    return value


def _string_list(value: Any, field: str) -> Sequence[str]:
    result = _sequence(value, field)
    if len(result) > MAX_LIST_ITEMS or any(not isinstance(item, str) for item in result):
        raise RolePackageError(f"{field} 必须是数量受限的字符串数组。")
    return result  # type: ignore[return-value]


def _exact_fields(value: Mapping[str, Any], fields: set[str] | frozenset[str], field: str) -> None:
    actual = set(value)
    unknown = sorted(str(item) for item in actual - set(fields))
    missing = sorted(str(item) for item in set(fields) - actual)
    if unknown:
        raise RolePackageError(f"{field} 包含未知字段：{', '.join(unknown)}。")
    if missing:
        raise RolePackageError(f"{field} 缺少字段：{', '.join(missing)}。")


def _reject_unknown(value: Mapping[str, Any], fields: set[str], field: str) -> None:
    unknown = sorted(str(item) for item in set(value) - fields)
    if unknown:
        raise RolePackageError(f"{field} 包含未知字段：{', '.join(unknown)}。")


def _bounded_string(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise RolePackageError(f"{field} 必须是字符串。")
    text = value.strip()
    if not text or len(text) > maximum or "\x00" in text:
        raise RolePackageError(f"{field} 必须包含 1 到 {maximum} 个字符。")
    return text


def _scope_label(scope: str) -> str:
    return "私聊" if scope == "private" else "群聊"


__all__ = ["build_role_package", "read_role_package"]
