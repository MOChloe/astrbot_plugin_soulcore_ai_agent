from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from .assets import AssetImages
from .drawing import (
    FontBook,
    cover_image,
    draw_android_status_bar,
    draw_bottom_gesture,
    draw_disclosure,
    draw_lines,
    line_height,
    rounded_avatar,
    wrap_text,
)
from .dto import (
    EntryKind,
    ParticipantSide,
    SceneMode,
    SnapshotEntry,
    SnapshotParticipant,
    SnapshotTheme,
    SocialSnapshotScene,
)
from .errors import SocialSnapshotError, SocialSnapshotErrorCode


@dataclass(frozen=True, slots=True)
class ChatStyle:
    background: str
    header: str
    accent: str
    self_bubble: str
    other_bubble: str
    self_text: str
    other_text: str
    name_color: str


STYLES = {
    SnapshotTheme.MOBILE_CHAT: ChatStyle(
        "#f3f4f6", "#f7f7f7", "#168cff", "#168cff", "#ffffff", "#ffffff", "#1f1f1f", "#777777"
    ),
    SnapshotTheme.WECHAT: ChatStyle(
        "#ededed", "#ededed", "#07c160", "#95ec69", "#ffffff", "#111111", "#111111", "#7b7b7b"
    ),
    SnapshotTheme.DINGTALK: ChatStyle(
        "#f2f5f8", "#ffffff", "#1677ff", "#d9ebff", "#ffffff", "#18212b", "#18212b", "#66717d"
    ),
}


@dataclass(frozen=True, slots=True)
class ChatBlock:
    entry: SnapshotEntry
    top: int
    height: int
    text_lines: tuple[str, ...]
    quote_lines: tuple[str, ...]
    image_size: tuple[int, int] | None


class ChatRenderer:
    def __init__(self, fonts: FontBook, assets: AssetImages) -> None:
        self._fonts = fonts
        self._assets = assets

    def render(self, scene: SocialSnapshotScene) -> Image.Image:
        style = STYLES[scene.theme]
        width = scene.ui.width
        scale = width / 873
        body_font = self._fonts.get(max(20, round(30 * scale)))
        meta_font = self._fonts.get(max(16, round(22 * scale)))
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        blocks, content_bottom = self._layout(scene, measure, body_font, meta_font, scale)
        composer_height = round(150 * scale)
        minimum = scene.ui.height or round(700 * scale)
        natural_height = content_bottom + composer_height + round(70 * scale)
        height = max(minimum, natural_height) if scene.ui.height is None else minimum
        if scene.ui.height is not None and natural_height > height:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.LIMIT_EXCEEDED, "chat content exceeds fixed canvas"
            )
        image = Image.new("RGB", (width, height), style.background)
        self._draw_chrome(image, scene, style, scale)
        self._draw_blocks(image, scene, blocks, style, body_font, meta_font, scale)
        self._draw_composer(image, scene, style, body_font, scale)
        draw_bottom_gesture(image, background=style.header)
        return image

    def _layout(
        self,
        scene: SocialSnapshotScene,
        draw: ImageDraw.ImageDraw,
        body_font: ImageFont.FreeTypeFont,
        meta_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> tuple[tuple[ChatBlock, ...], int]:
        y = round(220 * scale)
        blocks: list[ChatBlock] = []
        bubble_width = round(540 * scale)
        for entry in scene.entries:
            if entry.kind is EntryKind.TIMESTAMP:
                height = round(62 * scale)
                blocks.append(ChatBlock(entry, y, height, (), (), None))
                y += height + round(14 * scale)
                continue
            text_lines = (
                wrap_text(draw, entry.text, body_font, bubble_width - round(50 * scale))
                if entry.text
                else ()
            )
            quote_text = ""
            if entry.quote is not None:
                quote_text = f"{entry.quote.sender}：{entry.quote.text or entry.quote.media_label}"
            quote_lines = (
                wrap_text(draw, quote_text, meta_font, bubble_width - round(56 * scale))
                if quote_text
                else ()
            )
            image_size = (round(390 * scale), round(245 * scale)) if entry.media_ref else None
            body_height = len(text_lines) * line_height(body_font)
            quote_height = len(quote_lines) * line_height(meta_font, 5)
            if quote_lines:
                quote_height += round(22 * scale)
            image_height = (image_size[1] + round(18 * scale)) if image_size else 0
            file_height = round(78 * scale) if entry.kind is EntryKind.FILE else 0
            metadata = round(36 * scale) if self._show_sender_meta(scene, entry) else 0
            height = max(
                round(96 * scale),
                body_height
                + quote_height
                + image_height
                + file_height
                + metadata
                + round(44 * scale),
            )
            blocks.append(ChatBlock(entry, y, height, text_lines, quote_lines, image_size))
            y += height + round(24 * scale)
        return tuple(blocks), y

    @staticmethod
    def _show_sender_meta(scene: SocialSnapshotScene, entry: SnapshotEntry) -> bool:
        if scene.mode is not SceneMode.GROUP_CHAT or entry.author_id is None:
            return False
        return scene.participant(entry.author_id).side is ParticipantSide.LEFT

    def _draw_chrome(
        self, image: Image.Image, scene: SocialSnapshotScene, style: ChatStyle, scale: float
    ) -> None:
        draw_android_status_bar(
            image,
            clock=scene.ui.clock,
            battery_percent=scene.ui.battery_percent,
            charging=scene.ui.battery_charging,
            fonts=self._fonts,
            background=style.header,
        )
        draw = ImageDraw.Draw(image)
        header_top = round(84 * scale)
        header_bottom = round(184 * scale)
        draw.rectangle((0, header_top, image.width, header_bottom), fill=style.header)
        title_font = self._fonts.get(max(22, round(34 * scale)))
        subtitle_font = self._fonts.get(max(15, round(20 * scale)))
        draw.text((round(72 * scale), round(102 * scale)), "‹", font=title_font, fill="#202020")
        title_box = draw.textbbox((0, 0), scene.title, font=title_font)
        title_x = (image.width - (title_box[2] - title_box[0])) // 2
        draw.text((title_x, round(101 * scale)), scene.title, font=title_font, fill="#171717")
        if scene.ui.subtitle:
            subtitle_box = draw.textbbox((0, 0), scene.ui.subtitle, font=subtitle_font)
            draw.text(
                ((image.width - (subtitle_box[2] - subtitle_box[0])) // 2, round(145 * scale)),
                scene.ui.subtitle,
                font=subtitle_font,
                fill="#707070",
            )
        draw_disclosure(draw, width=image.width, y=round(190 * scale), fonts=self._fonts)

    def _draw_blocks(
        self,
        image: Image.Image,
        scene: SocialSnapshotScene,
        blocks: tuple[ChatBlock, ...],
        style: ChatStyle,
        body_font: ImageFont.FreeTypeFont,
        meta_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> None:
        draw = ImageDraw.Draw(image)
        for block in blocks:
            if block.entry.kind is EntryKind.TIMESTAMP:
                label = block.entry.time or block.entry.text
                box = draw.textbbox((0, 0), label, font=meta_font)
                x = (image.width - (box[2] - box[0])) // 2
                draw.text((x, block.top + round(12 * scale)), label, font=meta_font, fill="#9a9a9a")
                continue
            self._draw_message(image, scene, block, style, body_font, meta_font, scale)

    def _draw_message(
        self,
        image: Image.Image,
        scene: SocialSnapshotScene,
        block: ChatBlock,
        style: ChatStyle,
        body_font: ImageFont.FreeTypeFont,
        meta_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> None:
        entry = block.entry
        assert entry.author_id is not None
        participant = scene.participant(entry.author_id)
        avatar_size = round(74 * scale)
        margin = round(28 * scale)
        avatar_x = (
            margin
            if participant.side is ParticipantSide.LEFT
            else image.width - margin - avatar_size
        )
        avatar = rounded_avatar(
            self._assets.get(participant.avatar_ref),
            avatar_size,
            color=participant.color,
            label=participant.display_name,
            fonts=self._fonts,
        )
        image.paste(avatar, (avatar_x, block.top), avatar)
        bubble_width = round(540 * scale)
        bubble_left = (
            avatar_x + avatar_size + round(18 * scale)
            if participant.side is ParticipantSide.LEFT
            else avatar_x - round(18 * scale) - bubble_width
        )
        bubble_top = block.top
        if self._show_sender_meta(scene, entry):
            self._draw_sender_meta(
                image, participant, bubble_left, bubble_top, style, meta_font, scale
            )
            bubble_top += round(34 * scale)
        bubble_bottom = block.top + block.height
        bubble_color = (
            style.self_bubble if participant.side is ParticipantSide.RIGHT else style.other_bubble
        )
        text_color = (
            style.self_text if participant.side is ParticipantSide.RIGHT else style.other_text
        )
        draw = ImageDraw.Draw(image)
        quote_height = len(block.quote_lines) * line_height(meta_font, 5) + round(18 * scale)
        wechat_quote = scene.theme is SnapshotTheme.WECHAT and bool(block.quote_lines)
        main_bottom = (
            bubble_bottom - quote_height - round(10 * scale) if wechat_quote else bubble_bottom
        )
        draw.rounded_rectangle(
            (bubble_left, bubble_top, bubble_left + bubble_width, main_bottom),
            radius=round(18 * scale),
            fill=bubble_color,
        )
        self._draw_message_content(
            image,
            draw,
            scene,
            block,
            bubble_left,
            bubble_top,
            main_bottom,
            bubble_width,
            body_font,
            meta_font,
            text_color,
            wechat_quote,
            scale,
        )

    def _draw_message_content(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        scene: SocialSnapshotScene,
        block: ChatBlock,
        bubble_left: int,
        bubble_top: int,
        main_bottom: int,
        bubble_width: int,
        body_font: ImageFont.FreeTypeFont,
        meta_font: ImageFont.FreeTypeFont,
        text_color: str,
        wechat_quote: bool,
        scale: float,
    ) -> None:
        entry = block.entry
        x = bubble_left + round(24 * scale)
        y = bubble_top + round(20 * scale)
        if block.quote_lines and not wechat_quote:
            y = self._draw_quote(draw, scene.theme, block, x, y, bubble_width, meta_font, scale)
        if entry.kind is EntryKind.FILE:
            y = self._draw_file_card(draw, block, x, y, bubble_width, body_font, scale)
        if block.text_lines and entry.kind is not EntryKind.FILE:
            y = draw_lines(draw, block.text_lines, (x, y), body_font, fill=text_color)
            y += round(8 * scale)
        if block.image_size and entry.media_ref:
            source = self._assets.get(entry.media_ref)
            assert source is not None
            attachment = cover_image(source, block.image_size)
            image.paste(attachment, (x, y))
        if wechat_quote:
            self._draw_quote(
                draw,
                scene.theme,
                block,
                x,
                main_bottom + round(8 * scale),
                bubble_width,
                meta_font,
                scale,
            )

    @staticmethod
    def _draw_file_card(
        draw: ImageDraw.ImageDraw,
        block: ChatBlock,
        x: int,
        y: int,
        bubble_width: int,
        body_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> int:
        card_height = round(94 * scale)
        card_right = x + bubble_width - round(48 * scale)
        draw.rounded_rectangle(
            (x, y, card_right, y + card_height),
            radius=round(10 * scale),
            fill="#ffffff",
        )
        icon = round(60 * scale)
        icon_left = x + round(12 * scale)
        icon_top = y + round(16 * scale)
        draw.rounded_rectangle(
            (icon_left, icon_top, icon_left + icon, icon_top + icon),
            radius=round(8 * scale),
            fill="#1677ff",
        )
        draw.text(
            (icon_left + round(18 * scale), icon_top + round(6 * scale)),
            "文",
            font=body_font,
            fill="#ffffff",
        )
        draw.text(
            (x + round(86 * scale), y + round(24 * scale)),
            block.entry.text[:36],
            font=body_font,
            fill="#26323f",
        )
        return y + card_height + round(8 * scale)

    @staticmethod
    def _draw_sender_meta(
        image: Image.Image,
        participant: SnapshotParticipant,
        x: int,
        y: int,
        style: ChatStyle,
        meta_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> None:
        draw = ImageDraw.Draw(image)
        badge = f"{participant.badge} " if participant.badge else ""
        draw.text((x, y), badge + participant.display_name, font=meta_font, fill=style.name_color)

    @staticmethod
    def _draw_quote(
        draw: ImageDraw.ImageDraw,
        theme: SnapshotTheme,
        block: ChatBlock,
        x: int,
        y: int,
        bubble_width: int,
        meta_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> int:
        height = len(block.quote_lines) * line_height(meta_font, 5) + round(18 * scale)
        if theme is SnapshotTheme.WECHAT:
            fill, text_fill = "#d8d8d8", "#656565"
        elif theme is SnapshotTheme.DINGTALK:
            fill, text_fill = "#edf3fa", "#56616f"
        else:
            fill, text_fill = "#e9eef5", "#6f7884"
        draw.rounded_rectangle(
            (x, y, x + bubble_width - round(48 * scale), y + height),
            radius=round(9 * scale),
            fill=fill,
        )
        draw_lines(
            draw,
            block.quote_lines,
            (x + round(12 * scale), y + round(8 * scale)),
            meta_font,
            fill=text_fill,
            spacing=5,
        )
        return y + height + round(10 * scale)

    def _draw_composer(
        self,
        image: Image.Image,
        scene: SocialSnapshotScene,
        style: ChatStyle,
        body_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> None:
        draw = ImageDraw.Draw(image)
        top = image.height - round(150 * scale)
        draw.rectangle((0, top, image.width, image.height), fill=style.header)
        left = round(75 * scale)
        right = image.width - round(110 * scale)
        draw.rounded_rectangle(
            (left, top + round(22 * scale), right, top + round(94 * scale)),
            radius=round(14 * scale),
            fill="#ffffff",
        )
        draft = scene.draft or "输入消息"
        color = "#272727" if scene.draft else "#a1a1a1"
        clipped = draft[:80]
        draw.text(
            (left + round(20 * scale), top + round(37 * scale)), clipped, font=body_font, fill=color
        )
        draw.text(
            (image.width - round(82 * scale), top + round(38 * scale)),
            "+",
            font=body_font,
            fill=style.accent,
        )
