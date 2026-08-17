"""Controlled Markdown, text, and PDF artifact generation.

The service owns paths and rendering. Models only provide a bounded body and
display name; no model-controlled host path ever crosses this boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .markdown_pdf import (
    document_title,
    effective_document_style,
    parse_markdown_blocks,
    parse_markdown_nodes,
    pdf_inline,
    pdf_table_widths,
)
from .pdf_renderer import PDFRenderer

ALLOWED_FILE_FORMATS = {"MD", "TXT", "PDF"}
FILE_MIME_TYPES = {
    "MD": "text/markdown; charset=utf-8",
    "TXT": "text/plain; charset=utf-8",
    "PDF": "application/pdf",
}
FILE_EXTENSIONS = {"MD": ".md", "TXT": ".txt", "PDF": ".pdf"}
MAX_FILE_CONTENT_CHARS = 100_000
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 40
MAX_PDF_IMAGES = 5
PDF_DOCUMENT_STYLES = {"AUTO", "REPORT", "EDITORIAL", "DATA_BRIEF"}
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class GeneratedFileArtifact:
    storage_relpath: str
    display_name: str
    file_format: str
    mime_type: str
    sha256: str
    byte_size: int
    char_count: int
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PDFImageAsset:
    """One already-owned image made available under a document-local alias."""

    path: Path
    description: str = ""


class FileArtifactService:
    """Write immutable, instance-owned document artifacts under one root."""

    _effective_document_style = staticmethod(effective_document_style)
    _document_title = staticmethod(document_title)
    _pdf_inline = staticmethod(pdf_inline)
    _parse_markdown_nodes = staticmethod(parse_markdown_nodes)
    _parse_markdown_blocks = staticmethod(parse_markdown_blocks)
    _pdf_table_widths = staticmethod(pdf_table_widths)

    def __init__(self, root: str | Path, *, font_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.font_path = (
            Path(font_path).resolve()
            if font_path
            else (
                Path(__file__).resolve().parents[2]
                / "assets"
                / "fonts"
                / "SoulCoreSansSC-Regular.ttf"
            )
        )

    @staticmethod
    def normalize_format(value: str) -> str:
        file_format = str(value or "").strip().upper().lstrip(".")
        if file_format not in ALLOWED_FILE_FORMATS:
            raise ValueError("file format must be MD, TXT or PDF")
        return file_format

    @classmethod
    def safe_display_name(cls, value: str, file_format: str) -> str:
        file_format = cls.normalize_format(file_format)
        raw = unicodedata.normalize("NFKC", str(value or "").strip())
        if not raw:
            raw = "SoulCore文件"
        if Path(raw).name != raw or "/" in raw or "\\" in raw:
            raise ValueError("display filename must be a basename")
        raw = _INVALID_FILENAME.sub("_", raw).rstrip(" .")
        extension = FILE_EXTENSIONS[file_format]
        stem = Path(raw).stem if Path(raw).suffix else raw
        stem = stem.replace(".", "_").strip().rstrip(" .")[:80]
        if not stem:
            stem = "SoulCore文件"
        if stem.upper() in _WINDOWS_RESERVED:
            stem = f"_{stem}"
        return f"{stem}{extension}"

    @staticmethod
    def _scope_component(value: str) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]

    def _relative_path(
        self,
        profile_id: str,
        instance_id: str,
        job_id: str,
        display_name: str,
    ) -> Path:
        return Path(
            self._scope_component(profile_id),
            self._scope_component(instance_id),
            self._scope_component(job_id),
            display_name,
        )

    def resolve_path(self, storage_relpath: str) -> Path:
        path = self._controlled_path(storage_relpath)
        if not path.is_file():
            raise FileNotFoundError(path.name)
        return path

    def _controlled_path(self, storage_relpath: str) -> Path:
        relative = Path(str(storage_relpath or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid controlled file path")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("controlled file escaped its root") from exc
        return path

    def release(self, storage_relpath: str) -> bool:
        """Delete one controlled artifact without accepting an arbitrary host path."""

        path = self._controlled_path(storage_relpath)
        existed = path.exists()
        if existed and not path.is_file():
            raise ValueError("controlled artifact path is not a file")
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        return existed

    def generate(
        self,
        *,
        profile_id: str,
        instance_id: str,
        job_id: str,
        file_format: str,
        display_name: str,
        content: str,
        document_style: str = "AUTO",
        image_assets: Mapping[str, PDFImageAsset] | None = None,
    ) -> GeneratedFileArtifact:
        normalized_format = self.normalize_format(file_format)
        normalized_name = self.safe_display_name(display_name, normalized_format)
        body = str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not body:
            raise ValueError("generated file content cannot be empty")
        if len(body) > MAX_FILE_CONTENT_CHARS:
            raise ValueError("generated file content exceeds the character limit")
        normalized_style = self.normalize_document_style(document_style)
        controlled_images = self._validate_pdf_images(
            normalized_format,
            image_assets or {},
        )
        relative = self._relative_path(profile_id, instance_id, job_id, normalized_name)
        destination = (self.root / relative).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        page_count = 0
        try:
            if normalized_format in {"MD", "TXT"}:
                temporary.write_text(body + "\n", encoding="utf-8", newline="\n")
            else:
                page_count, rendered_images, effective_style = self._render_pdf(
                    body,
                    temporary,
                    document_style=normalized_style,
                    image_assets=controlled_images,
                )
            with temporary.open("r+b") as durable:
                os.fsync(durable.fileno())
            size = int(temporary.stat().st_size)
            if size < 1 or size > MAX_FILE_BYTES:
                raise ValueError("generated file size is outside the allowed range")
            digest = self._sha256(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
        return GeneratedFileArtifact(
            storage_relpath=relative.as_posix(),
            display_name=normalized_name,
            file_format=normalized_format,
            mime_type=FILE_MIME_TYPES[normalized_format],
            sha256=digest,
            byte_size=size,
            char_count=len(body),
            page_count=page_count,
            metadata={
                "document_style": effective_style if normalized_format == "PDF" else "PLAIN",
                "image_count": rendered_images if normalized_format == "PDF" else 0,
            },
        )

    @staticmethod
    def normalize_document_style(value: str) -> str:
        style = str(value or "AUTO").strip().upper().replace("-", "_")
        if style not in PDF_DOCUMENT_STYLES:
            raise ValueError("document style must be AUTO, REPORT, EDITORIAL or DATA_BRIEF")
        return style

    @staticmethod
    def _validate_pdf_images(
        file_format: str,
        image_assets: Mapping[str, PDFImageAsset],
    ) -> dict[str, PDFImageAsset]:
        if image_assets and file_format != "PDF":
            raise ValueError("controlled images are supported only for PDF artifacts")
        if len(image_assets) > MAX_PDF_IMAGES:
            raise ValueError("a PDF may contain at most five controlled images")
        validated: dict[str, PDFImageAsset] = {}
        for raw_ref, asset in image_assets.items():
            reference = str(raw_ref or "").strip().lower()
            if not re.fullmatch(r"i[1-5]", reference):
                raise ValueError("invalid controlled PDF image reference")
            if not isinstance(asset, PDFImageAsset):
                raise TypeError("PDF image entries must be PDFImageAsset values")
            path = Path(asset.path).resolve()
            if not path.is_file():
                raise FileNotFoundError("controlled PDF image is unavailable")
            validated[reference] = PDFImageAsset(
                path=path,
                description=str(asset.description or "").strip()[:500],
            )
        return validated

    def _render_pdf(
        self,
        content: str,
        destination: Path,
        *,
        document_style: str,
        image_assets: Mapping[str, PDFImageAsset],
    ) -> tuple[int, int, str]:
        return PDFRenderer(self.font_path, max_pages=MAX_PDF_PAGES).render(
            content,
            destination,
            document_style=document_style,
            image_assets=image_assets,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


def verify_artifact(path: Path, expected_size: int, expected_sha256: str) -> bool:
    """Verify one controlled artifact before handing it to delivery."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size != int(expected_size):
        return False
    return FileArtifactService._sha256(candidate) == str(expected_sha256).strip().lower()


__all__ = [
    "ALLOWED_FILE_FORMATS",
    "FILE_EXTENSIONS",
    "FILE_MIME_TYPES",
    "MAX_PDF_IMAGES",
    "PDF_DOCUMENT_STYLES",
    "FileArtifactService",
    "GeneratedFileArtifact",
    "PDFImageAsset",
    "verify_artifact",
]
