"""Bounded ReportLab renderer used by the file artifact service."""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .markdown_pdf import (
    MarkdownNode,
    document_title,
    effective_document_style,
    parse_markdown_nodes,
    pdf_inline,
    pdf_table_widths,
)

try:  # Kept optional until a PDF is actually requested.
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
        XPreformatted,
    )
    from reportlab.platypus import (
        Image as ReportLabImage,
    )

    _REPORTLAB_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - declared plugin dependency
    _REPORTLAB_ERROR = exc


def _require_reportlab() -> None:
    if _REPORTLAB_ERROR is not None:
        raise RuntimeError("ReportLab is required for PDF generation") from _REPORTLAB_ERROR


class PDFRenderer:
    def __init__(self, font_path: Path, *, max_pages: int) -> None:
        self.font_path = Path(font_path)
        self.max_pages = int(max_pages)

    def render(
        self,
        content: str,
        destination: Path,
        *,
        document_style: str,
        image_assets: Mapping[str, Any],
    ) -> tuple[int, int, str]:
        _require_reportlab()
        if not self.font_path.is_file():
            raise RuntimeError("bundled Chinese PDF font is missing")
        font_name = self._register_font()
        style = effective_document_style(document_style, content, len(image_assets))
        palette = _palette(style)
        styles = _styles(font_name, style, palette)
        title = document_title(content)
        document = self._document(destination, title)
        rendering = _Rendering(
            document=document,
            title=title,
            font_name=font_name,
            effective_style=style,
            palette=palette,
            styles=styles,
            image_assets=image_assets,
        )
        rendering.append_nodes(parse_markdown_nodes(content))
        rendering.append_unreferenced_images()
        if not rendering.story:
            raise ValueError("generated PDF has no renderable content")
        document.build(
            rendering.story,
            onFirstPage=rendering.decorate_page,
            onLaterPages=rendering.decorate_page,
        )
        self._validate(destination, document.page)
        return int(document.page), len(rendering.rendered_image_refs), style

    def _register_font(self) -> str:
        font_name = "SoulCoreNotoSansSC"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font_name, str(self.font_path)))
            pdfmetrics.registerFontFamily(
                font_name,
                normal=font_name,
                bold=font_name,
                italic=font_name,
                boldItalic=font_name,
            )
        return font_name

    def _document(self, destination: Path, title: str) -> Any:
        max_pages = self.max_pages

        class BoundedDocument(SimpleDocTemplate):
            def afterPage(document) -> None:  # type: ignore[no-untyped-def]
                if document.page > max_pages:
                    raise ValueError("generated PDF exceeds the page limit")

        return BoundedDocument(
            str(destination),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=22 * mm,
            bottomMargin=18 * mm,
            title=title,
            author="SoulCore",
        )

    def _validate(self, destination: Path, page_count: int) -> None:
        rendered = destination.read_bytes()
        if not rendered.startswith(b"%PDF-") or b"%%EOF" not in rendered[-2048:]:
            raise ValueError("generated PDF signature validation failed")
        if page_count < 1 or page_count > self.max_pages:
            raise ValueError("generated PDF page validation failed")


class _Rendering:
    def __init__(
        self,
        *,
        document: Any,
        title: str,
        font_name: str,
        effective_style: str,
        palette: dict[str, Any],
        styles: dict[str, Any],
        image_assets: Mapping[str, Any],
    ) -> None:
        self.document = document
        self.title = title
        self.font_name = font_name
        self.effective_style = effective_style
        self.palette = palette
        self.styles = styles
        self.image_assets = image_assets
        self.story: list[Any] = []
        self.rendered_image_refs: set[str] = set()

    def decorate_page(self, canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setTitle(self.title)
        canvas.setAuthor("SoulCore")
        canvas.setStrokeColor(self.palette["line"])
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.setFont(self.font_name, 8)
        canvas.setFillColor(self.palette["muted"])
        canvas.drawString(18 * mm, 9 * mm, self.title[:42])
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    def append_nodes(
        self,
        nodes: list[MarkdownNode],
        depth: int = 0,
        quoted: bool = False,
    ) -> None:
        for node in nodes:
            if node.kind in {"h1", "h2", "h3"}:
                self.story.append(Paragraph(pdf_inline(node.text), self.styles[node.kind]))
            elif node.kind == "paragraph":
                style = self.styles["quote"] if quoted else self.styles["body"]
                self.story.append(Paragraph(pdf_inline(node.text), style))
            elif node.kind == "quote":
                self.append_nodes(node.children, depth, True)
            elif node.kind == "code":
                self.story.append(XPreformatted(html.escape(node.text), self.styles["code"]))
            elif node.kind == "rule":
                self._append_rule()
            elif node.kind in {"ul", "ol"}:
                self._append_list(node, depth)
            elif node.kind == "table":
                self._append_table(node.rows)
            elif node.kind == "image":
                self._append_image(node.target, node.text)

    def append_unreferenced_images(self) -> None:
        missing = [
            reference
            for reference in self.image_assets
            if reference not in self.rendered_image_refs
        ]
        if not missing:
            return
        self.story.append(Paragraph("相关配图", self.styles["h2"]))
        for reference in missing:
            self._append_image(reference, self.image_assets[reference].description)

    def _append_rule(self) -> None:
        self.story.extend(
            (
                Spacer(1, 1.5 * mm),
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=self.palette["line"],
                    spaceBefore=2 * mm,
                    spaceAfter=4 * mm,
                ),
            )
        )

    def _append_image(self, reference: str, caption: str = "") -> None:
        normalized = str(reference or "").strip().lower()
        asset = self.image_assets.get(normalized)
        if asset is None:
            safe_caption = str(caption or "外部图片")[:200]
            self.story.append(
                Paragraph(
                    f"[未嵌入图片：{html.escape(safe_caption)}]",
                    self.styles["quote"],
                )
            )
            return
        if normalized in self.rendered_image_refs:
            return
        width_px, height_px = self._image_dimensions(asset.path)
        max_height = 118 * mm if self.effective_style == "EDITORIAL" else 105 * mm
        scale = min(
            self.document.width / float(width_px),
            max_height / float(height_px),
        )
        image = ReportLabImage(
            str(asset.path),
            width=max(1.0, float(width_px) * scale),
            height=max(1.0, float(height_px) * scale),
        )
        image.hAlign = "CENTER"
        label = str(caption or asset.description or "配图").strip()[:300]
        elements: list[Any] = [Spacer(1, 2 * mm), image]
        elements.append(
            Paragraph(pdf_inline(label), self.styles["caption"]) if label else Spacer(1, 4 * mm)
        )
        self.story.append(KeepTogether(elements))
        self.rendered_image_refs.add(normalized)

    @staticmethod
    def _image_dimensions(path: Path) -> tuple[int, int]:
        try:
            width, height = ImageReader(str(path)).getSize()
            if width <= 0 or height <= 0:
                raise ValueError("invalid image dimensions")
            if int(width) * int(height) > 100_000_000:
                raise ValueError("image pixel budget exceeded")
            return int(width), int(height)
        except Exception as exc:
            raise ValueError("controlled PDF image could not be decoded") from exc

    def _append_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        column_count = max(len(row) for row in rows)
        normalized = [row + [""] * (column_count - len(row)) for row in rows]
        cells = self._table_cells(normalized)
        table = Table(
            cells,
            colWidths=pdf_table_widths(normalized, self.document.width),
            repeatRows=1,
            splitByRow=1,
            splitInRow=1,
            hAlign="LEFT",
        )
        commands = self._table_commands(cells)
        table.setStyle(TableStyle(commands))
        self.story.extend((table, Spacer(1, 5 * mm)))

    def _table_cells(self, rows: list[list[str]]) -> list[list[Any]]:
        return [
            [
                Paragraph(
                    pdf_inline(cell),
                    self.styles["table_head"] if index == 0 else self.styles["small"],
                )
                for cell in row
            ]
            for index, row in enumerate(rows)
        ]

    def _table_commands(self, cells: list[list[Any]]) -> list[tuple[Any, ...]]:
        commands: list[tuple[Any, ...]] = [
            ("BACKGROUND", (0, 0), (-1, 0), self.palette["table_head"]),
            ("TEXTCOLOR", (0, 0), (-1, -1), self.palette["ink"]),
            ("FONTNAME", (0, 0), (-1, -1), self.font_name),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.45, self.palette["line"]),
            ("LINEBELOW", (0, 0), (-1, 0), 1, self.palette["accent"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        for row_index in range(2, len(cells), 2):
            commands.append(
                (
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    self.palette["table_alt"],
                )
            )
        return commands

    def _append_list(self, node: MarkdownNode, depth: int) -> None:
        item_style, continuation_style = self._list_styles(depth)
        for offset, item in enumerate(node.children):
            marker = f"{node.start + offset}." if node.kind == "ol" else "•"
            first = True
            for child in item.children:
                if child.kind == "paragraph":
                    self.story.append(
                        Paragraph(
                            pdf_inline(child.text),
                            item_style if first else continuation_style,
                            bulletText=marker if first else None,
                        )
                    )
                    first = False
                elif child.kind in {"ol", "ul"}:
                    self._append_list(child, depth + 1)
                else:
                    self.append_nodes([child], depth + 1)

    def _list_styles(self, depth: int) -> tuple[Any, Any]:
        left = (8 + min(depth, 6) * 6) * mm
        bullet = (2 + min(depth, 6) * 6) * mm
        item = ParagraphStyle(
            f"SoulCoreList{depth}",
            parent=self.styles["body"],
            leftIndent=left,
            firstLineIndent=0,
            bulletIndent=bullet,
            bulletFontName=self.font_name,
            bulletFontSize=9,
            bulletColor=self.palette["accent"],
            spaceAfter=1.8 * mm,
        )
        continuation = ParagraphStyle(
            f"SoulCoreListContinuation{depth}",
            parent=item,
            bulletIndent=left,
            spaceBefore=0,
        )
        return item, continuation


def _palette(style: str) -> dict[str, Any]:
    accents = {
        "REPORT": ("#2563EB", "#EFF6FF", "#E8EEF8"),
        "EDITORIAL": ("#B45309", "#FFF7ED", "#FDE7D3"),
        "DATA_BRIEF": ("#0F766E", "#ECFDF5", "#DDF4EE"),
    }
    accent, accent_soft, table_head = accents[style]
    return {
        "ink": colors.HexColor("#172033"),
        "muted": colors.HexColor("#667085"),
        "accent": colors.HexColor(accent),
        "accent_soft": colors.HexColor(accent_soft),
        "line": colors.HexColor("#D0D5DD"),
        "table_head": colors.HexColor(table_head),
        "table_alt": colors.HexColor("#F8FAFC"),
        "code": colors.HexColor("#F2F4F7"),
    }


def _styles(font_name: str, style: str, palette: dict[str, Any]) -> dict[str, Any]:
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "SoulCoreTitle",
            parent=base["Title"],
            fontName=font_name,
            fontSize=27 if style == "EDITORIAL" else 23,
            leading=36 if style == "EDITORIAL" else 32,
            textColor=palette["ink"],
            alignment=TA_LEFT,
            spaceAfter=12 * mm,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "SoulCoreH2",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=16,
            leading=23,
            textColor=palette["ink"],
            spaceBefore=7 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "SoulCoreH3",
            parent=base["Heading3"],
            fontName=font_name,
            fontSize=13,
            leading=19,
            textColor=palette["ink"],
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "SoulCoreBody",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=17,
            textColor=palette["ink"],
            spaceAfter=3.5 * mm,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "SoulCoreSmall",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=13,
            textColor=palette["ink"],
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "SoulCoreCaption",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=13,
            textColor=palette["muted"],
            alignment=TA_CENTER,
            spaceBefore=2 * mm,
            spaceAfter=5 * mm,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "SoulCoreTableHead",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=13,
            textColor=palette["ink"],
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "SoulCoreQuote",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=16,
            textColor=palette["muted"],
            leftIndent=6 * mm,
            rightIndent=3 * mm,
            borderColor=palette["accent"],
            borderWidth=0,
            borderLeft=2,
            borderPadding=4 * mm,
            backColor=palette["accent_soft"],
            spaceAfter=4 * mm,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "SoulCoreCode",
            parent=base["Code"],
            fontName=font_name,
            fontSize=8.5,
            leading=13,
            textColor=palette["ink"],
            leftIndent=3 * mm,
            rightIndent=3 * mm,
            borderPadding=3 * mm,
            backColor=palette["code"],
            spaceAfter=4 * mm,
            wordWrap="CJK",
        ),
    }


__all__ = ["PDFRenderer"]
