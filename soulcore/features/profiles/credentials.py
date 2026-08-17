"""Small file-backed credential vault with redacted diagnostics.

The file is intentionally simple because SoulCore cannot assume an OS keyring
inside containers.  It is restricted to the owning account (0600), written
atomically, and never exposes secret values through repr/list/describe APIs.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...shared.private_paths import (
    ensure_private_directory,
    restrict_private_path,
    sync_directory,
)


@dataclass(frozen=True, slots=True)
class CredentialInfo:
    credential_id: str
    source: str
    last4: str
    configured: bool
    reference: str = ""


class CredentialVault:
    def __init__(
        self, data_dir: str | os.PathLike[str], filename: str = "credentials.json"
    ) -> None:
        self.path = Path(data_dir) / filename
        self._lock = threading.RLock()
        self._items: dict[str, dict[str, str]] = {}
        self._load()

    def set_secret(self, credential_id: str, value: str) -> CredentialInfo:
        key = self._key(credential_id)
        secret = str(value or "")
        if not secret:
            raise ValueError("credential value must not be empty")
        with self._lock:
            updated = {**self._items, key: {"source": "file", "value": secret}}
            self._save(updated)
            self._items = updated
        return self.describe(key)

    def set_env_reference(self, credential_id: str, environment_variable: str) -> CredentialInfo:
        key = self._key(credential_id)
        reference = str(environment_variable or "").strip()
        if not reference:
            raise ValueError("environment variable name must not be empty")
        with self._lock:
            updated = {**self._items, key: {"source": "env", "reference": reference}}
            self._save(updated)
            self._items = updated
        return self.describe(key)

    def resolve(self, credential_id: str) -> str:
        key = self._key(credential_id)
        with self._lock:
            item = dict(self._items.get(key) or {})
        if not item:
            raise KeyError(key)
        if item.get("source") == "env":
            value = os.environ.get(str(item.get("reference") or ""), "")
        else:
            value = str(item.get("value") or "")
        if not value:
            raise RuntimeError(f"credential is not configured: {key}")
        return value

    def describe(self, credential_id: str) -> CredentialInfo:
        key = self._key(credential_id)
        with self._lock:
            item = dict(self._items.get(key) or {})
        if not item:
            return CredentialInfo(key, "missing", "", False)
        source = str(item.get("source") or "file")
        reference = str(item.get("reference") or "") if source == "env" else ""
        try:
            value = self.resolve(key)
        except (KeyError, RuntimeError):
            value = ""
        return CredentialInfo(
            credential_id=key,
            source=source,
            last4=value[-4:] if value else "",
            configured=bool(value),
            reference=reference,
        )

    def list_credentials(self) -> list[CredentialInfo]:
        with self._lock:
            keys = sorted(self._items)
        return [self.describe(key) for key in keys]

    def delete(self, credential_id: str) -> bool:
        key = self._key(credential_id)
        with self._lock:
            existed = key in self._items
            if not existed:
                return False
            updated = dict(self._items)
            del updated[key]
            self._save(updated)
            self._items = updated
        return existed

    @staticmethod
    def _key(value: str) -> str:
        key = str(value or "").strip()
        if not key or any(char in key for char in "\\/\0"):
            raise ValueError("invalid credential_id")
        return key

    def _load(self) -> None:
        with self._lock:
            if not self.path.is_file():
                return
            try:
                value: Any = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError("credential vault exists but cannot be read safely") from exc
            self._items = _validated_items(value)
            restrict_private_path(self.path.parent, directory=True)
            restrict_private_path(self.path, directory=False)

    def _save(self, items: dict[str, dict[str, str]]) -> None:
        ensure_private_directory(self.path.parent)
        payload = json.dumps(items, ensure_ascii=False, indent=2, sort_keys=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            os.chmod(temporary, 0o600)
            restrict_private_path(temporary, directory=False)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            restrict_private_path(self.path, directory=False)
            sync_directory(self.path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()


def _validated_items(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise RuntimeError("credential vault root must be an object")
    result: dict[str, dict[str, str]] = {}
    for raw_key, raw_item in value.items():
        key = CredentialVault._key(str(raw_key))
        if not isinstance(raw_item, dict):
            raise RuntimeError("credential vault entry must be an object")
        source = str(raw_item.get("source") or "")
        if source == "file":
            secret = raw_item.get("value")
            if not isinstance(secret, str) or not secret:
                raise RuntimeError("credential vault file entry is invalid")
            result[key] = {"source": "file", "value": secret}
        elif source == "env":
            reference = raw_item.get("reference")
            if not isinstance(reference, str) or not reference.strip():
                raise RuntimeError("credential vault environment entry is invalid")
            result[key] = {"source": "env", "reference": reference.strip()}
        else:
            raise RuntimeError("credential vault entry source is invalid")
    return result


__all__ = ["CredentialInfo", "CredentialVault"]
