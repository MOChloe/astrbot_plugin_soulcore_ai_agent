"""Validated public release notes shared by the UI and release tooling."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CHANGELOG = PLUGIN_ROOT / "CHANGELOG.md"
_RELEASE_HEADING = re.compile(
    r"^##\s+v(?P<version>\d+\.\d+\.\d+)\s+-\s+(?P<date>\d{4}-\d{2}-\d{2})\s*$"
)
_TITLE_HEADING = re.compile(r"^###\s+(?P<title>\S(?:.*\S)?)\s*$")


def parse_public_release_notes(markdown: str) -> list[dict[str, Any]]:
    """Parse the intentionally small public CHANGELOG format.

    Each release must contain one ``## vX.Y.Z - YYYY-MM-DD`` heading,
    one ``###`` title and at least one bullet. Other release prose is rejected
    so AstrBot Cloud and the installed UI cannot silently render different
    interpretations of the same file.
    """

    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line_number, raw_line in enumerate(markdown.splitlines(), start=1):
        line = raw_line.rstrip()
        release_match = _RELEASE_HEADING.fullmatch(line)
        if release_match:
            if current is not None:
                _finish_release(items, current)
            current = _start_release(release_match, line_number)
            continue

        if line.startswith("## "):
            raise ValueError(f"invalid public release heading at line {line_number}")
        if current is None or not line.strip() or line.startswith("# "):
            continue

        _append_release_content(current, line, line_number)

    if current is not None:
        _finish_release(items, current)
    if not items:
        raise ValueError("public CHANGELOG contains no releases")

    versions = [_version_tuple(str(item["version"])) for item in items]
    if len(set(versions)) != len(versions):
        raise ValueError("public CHANGELOG contains duplicate versions")
    if versions != sorted(versions, reverse=True):
        raise ValueError("public CHANGELOG releases must be newest first")
    return items


def _start_release(match: re.Match[str], line_number: int) -> dict[str, Any]:
    released_at = match.group("date")
    try:
        date.fromisoformat(released_at)
    except ValueError as exc:
        raise ValueError(
            f"invalid public release date at line {line_number}: {released_at}"
        ) from exc
    return {
        "version": f"v{match.group('version')}",
        "title": "",
        "released_at": released_at,
        "changes": [],
    }


def _append_release_content(current: dict[str, Any], line: str, line_number: int) -> None:
    title_match = _TITLE_HEADING.fullmatch(line)
    if title_match:
        if current["title"]:
            raise ValueError(f"duplicate public release title at line {line_number}")
        current["title"] = title_match.group("title")
        return
    if line.startswith("- "):
        change = line[2:].strip()
        if not change:
            raise ValueError(f"empty public release change at line {line_number}")
        current["changes"].append(change)
        return
    raise ValueError(f"unsupported public release content at line {line_number}")


def load_public_release_notes(path: Path = PUBLIC_CHANGELOG) -> list[dict[str, Any]]:
    return parse_public_release_notes(path.read_text(encoding="utf-8"))


def validate_public_release_notes(path: Path, expected_version: str) -> list[dict[str, Any]]:
    items = load_public_release_notes(path)
    newest_public_version = _version_tuple(str(items[0]["version"]))
    runtime_version = _version_tuple(expected_version)
    if newest_public_version > runtime_version:
        raise ValueError(
            "public CHANGELOG latest version is newer than the release version: "
            f"{items[0]['version']} > v{expected_version}"
        )
    return items


def _finish_release(items: list[dict[str, Any]], current: dict[str, Any]) -> None:
    if not current["title"]:
        raise ValueError(f"public release {current['version']} is missing a title")
    if not current["changes"]:
        raise ValueError(f"public release {current['version']} has no changes")
    items.append(current)


def _version_tuple(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.removeprefix("v").split("."))  # type: ignore[return-value]


__all__ = [
    "PUBLIC_CHANGELOG",
    "load_public_release_notes",
    "parse_public_release_notes",
    "validate_public_release_notes",
]
