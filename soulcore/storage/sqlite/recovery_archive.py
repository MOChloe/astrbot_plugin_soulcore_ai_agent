"""Verify the existing recovery ZIP format without trusting archive paths."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from ...shared.private_paths import create_private_file, ensure_private_directory

MANIFEST = "RECOVERY_MANIFEST.json"
MAX_FILES = 100_000
MAX_BYTES = 16 * 1024**3
MAX_MANIFEST_BYTES = 32 * 1024**2


def _safe_name(name: str) -> None:
    parts = PurePosixPath(name).parts
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or ":" in name
        or any(part in {".", ".."} or part.endswith((" ", ".")) for part in parts)
        or "/".join(parts) != name
        or any(
            part.split(".")[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *[f"COM{i}" for i in range(10)],
                *[f"LPT{i}" for i in range(10)],
            }
            for part in parts
        )
    ):
        raise ValueError("恢复包包含不安全的文件名")


def _archive_names(archive: zipfile.ZipFile) -> set[str]:
    infos = archive.infolist()
    if len(infos) > MAX_FILES + 1:
        raise ValueError("恢复包文件数量超出限制")
    names = [info.filename for info in infos]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("恢复包包含重复文件名")
    for info in infos:
        _safe_name(info.filename)
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 1:
            raise ValueError("恢复包包含目录、链接或加密成员")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError("恢复包包含不支持的压缩格式")
    if MANIFEST not in names or archive.getinfo(MANIFEST).file_size > MAX_MANIFEST_BYTES:
        raise ValueError("恢复包缺少校验清单或清单过大")
    return set(names) - {MANIFEST}


def _record_name(record: dict) -> str:
    name, size, digest = (record.get(key) for key in ("name", "size", "sha256"))
    if not isinstance(name, str) or name == MANIFEST:
        raise ValueError("恢复包文件名无效")
    _safe_name(name)
    if type(size) is not int or size < 0 or size > MAX_BYTES:
        raise ValueError("恢复包文件大小无效")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("恢复包文件校验信息无效")
    return name


def _manifest_members(archive: zipfile.ZipFile) -> dict[str, dict]:
    manifest = json.loads(archive.read(MANIFEST))
    if not isinstance(manifest, dict) or manifest.get("format") != "soulcore-sqlite-recovery-v1":
        raise ValueError("不支持的 SoulCore 恢复包格式")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) > MAX_FILES:
        raise ValueError("恢复包文件清单无效")
    members: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("恢复包文件清单无效")
        name = _record_name(record)
        if name in members:
            raise ValueError("恢复包清单包含重复文件名")
        members[name] = record
    return members


def archive_members(archive: zipfile.ZipFile) -> dict[str, dict]:
    """Check the manifest and ZIP directory before creating any files."""
    names = _archive_names(archive)
    members = _manifest_members(archive)
    if set(members) != names:
        raise ValueError("恢复包内容与校验清单不一致")
    if sum(item["size"] for item in members.values()) > MAX_BYTES:
        raise ValueError("恢复包展开大小超出限制")
    for name, record in members.items():
        if archive.getinfo(name).file_size != record["size"]:
            raise ValueError("恢复包文件大小与校验清单不一致")
    return members


def extract_verified(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        members = archive_members(archive)
        for name, record in members.items():
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.resolve(strict=False).relative_to(destination.resolve())
            ensure_private_directory(target.parent)
            create_private_file(target)
            digest = hashlib.sha256()
            total = 0
            with archive.open(name) as incoming, target.open("wb") as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    total += len(chunk)
                    if total > record["size"]:
                        raise ValueError("恢复包文件超出清单大小")
                    digest.update(chunk)
                    outgoing.write(chunk)
            if total != record["size"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("恢复包校验失败，原数据未被替换")


def copy_private(source: Path, destination: Path) -> None:
    """Copy a known regular file to an exclusively created private file."""
    from .backup import SQLiteBackupManager

    SQLiteBackupManager._reject_link(source)
    ensure_private_directory(destination.parent)
    create_private_file(destination)
    shutil.copyfile(source, destination)
