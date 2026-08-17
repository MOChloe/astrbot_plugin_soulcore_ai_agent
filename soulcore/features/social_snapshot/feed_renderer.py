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
from .dto import EntryKind, SnapshotEntry, SnapshotTheme, SocialSnapshotScene
from .errors import SocialSnapshotError, SocialSnapshotErrorCode


@dataclass(frozen=True, slots=True)
class FeedStyle:
    accent: str
    background: str
    card: str
    divider: str
    title: str
    subtitle: str
    action_labels: tuple[str, ...]


STYLES = {
    SnapshotTheme.WEIBO_FEED: FeedStyle(
        "#ff8200", "#f2f2f2", "#ffffff", "#ececec", "微博", "推荐", ("转发", "评论", "赞")
    ),
    SnapshotTheme.X: FeedStyle(
        "#111111",
        "#ffffff",
        "#ffffff",
        "#e6e9ec",
        "X",
        "为你推荐",
        ("回复", "转帖", "喜欢", "分享"),
    ),
    SnapshotTheme.XIAOHONGSHU: FeedStyle(
        "#ff2442", "#ffffff", "#ffffff", "#eeeeee", "小红书", "发现", ("评论", "收藏", "赞")
    ),
}


@dataclass(frozen=True, slots=True)
class FeedBlock:
    entry: SnapshotEntry
    top: int
    height: int
    lines: tuple[str, ...]
    image_size: tuple[int, int] | None


class FeedRenderer:
    def __init__(self, fonts: FontBook, assets: AssetImages) -> None:
        self._fonts = fonts
        self._assets = assets

    def render(self, scene: SocialSnapshotScene) -> Image.Image:
        style = STYLES[scene.theme]
        width = scene.ui.width
        height = scene.ui.height or 1920
        scale = width / 873
        body_font = self._fonts.get(max(20, round(29 * scale)))
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        blocks, bottom = self._layout(scene, measure, body_font, scale)
        footer_top = height - round(105 * scale)
        if bottom > footer_top:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.LIMIT_EXCEEDED, "feed content exceeds fixed canvas"
            )
        image = Image.new("RGB", (width, height), style.background)
        self._draw_chrome(image, scene, style, scale)
        for block in blocks:
            self._draw_block(image, scene, block, style, body_font, scale)
        self._draw_footer(image, scene, style, body_font, scale)
        draw_bottom_gesture(image)
        return image

    def _layout(
        self,
        scene: SocialSnapshotScene,
        draw: ImageDraw.ImageDraw,
        body_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> tuple[tuple[FeedBlock, ...], int]:
        y = round(225 * scale)
        blocks: list[FeedBlock] = []
        max_text_width = round(725 * scale)
        for entry in scene.entries:
            lines = wrap_text(draw, entry.text, body_font, max_text_width) if entry.text else ()
            image_size = None
            if entry.media_ref:
                image_size = (round(710 * scale), round(330 * scale))
            text_height = len(lines) * line_height(body_font)
            image_height = image_size[1] + round(18 * scale) if image_size else 0
            base = round(118 * scale) if entry.kind is EntryKind.COMMENT else round(150 * scale)
            actions = 0 if entry.kind is EntryKind.COMMENT else round(54 * scale)
            height = base + text_height + image_height + actions
            blocks.append(FeedBlock(entry, y, height, lines, image_size))
            y += height + round(10 * scale)
        return tuple(blocks), y

    def _draw_chrome(
        self, image: Image.Image, scene: SocialSnapshotScene, style: FeedStyle, scale: float
    ) -> None:
        draw_android_status_bar(
            image,
            clock=scene.ui.clock,
            battery_percent=scene.ui.battery_percent,
            charging=scene.ui.battery_charging,
            fonts=self._fonts,
        )
        draw = ImageDraw.Draw(image)
        header_font = self._fonts.get(max(24, round(38 * scale)))
        tab_font = self._fonts.get(max(18, round(25 * scale)))
        draw.rectangle((0, round(84 * scale), image.width, round(188 * scale)), fill="#ffffff")
        title = scene.title or style.title
        draw.text((round(36 * scale), round(108 * scale)), title, font=header_font, fill="#151515")
        draw.text(
            (round(355 * scale), round(121 * scale)),
            style.subtitle,
            font=tab_font,
            fill=style.accent,
        )
        draw.rectangle(
            (round(350 * scale), round(174 * scale), round(480 * scale), round(180 * scale)),
            fill=style.accent,
        )
        draw_disclosure(draw, width=image.width, y=round(190 * scale), fonts=self._fonts)

    def _draw_block(
        self,
        image: Image.Image,
        scene: SocialSnapshotScene,
        block: FeedBlock,
        style: FeedStyle,
        body_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> None:
        entry = block.entry
        assert entry.author_id is not None
        participant = scene.participant(entry.author_id)
        draw = ImageDraw.Draw(image)
        left = round(24 * scale) if entry.kind is not EntryKind.COMMENT else round(95 * scale)
        right = image.width - round(24 * scale)
        draw.rectangle((left, block.top, right, block.top + block.height), fill=style.card)
        avatar_size = round((64 if entry.kind is EntryKind.COMMENT else 76) * scale)
        avatar = rounded_avatar(
            self._assets.get(participant.avatar_ref),
            avatar_size,
            color=participant.color,
            label=participant.display_name,
            fonts=self._fonts,
        )
        avatar_x = left + round(18 * scale)
        avatar_y = block.top + round(20 * scale)
        image.paste(avatar, (avatar_x, avatar_y), avatar)
        name_font = self._fonts.get(max(18, round(25 * scale)))
        meta_font = self._fonts.get(max(15, round(20 * scale)))
        text_x = avatar_x + avatar_size + round(18 * scale)
        draw.text((text_x, avatar_y), participant.display_name, font=name_font, fill="#191919")
        meta = entry.time or ("评论" if entry.kind is EntryKind.COMMENT else "刚刚")
        draw.text((text_x, avatar_y + round(39 * scale)), meta, font=meta_font, fill="#8d8d8d")
        y = avatar_y + avatar_size + round(15 * scale)
        if entry.kind is EntryKind.REPOST:
            draw.rounded_rectangle(
                (
                    text_x,
                    y,
                    right - round(20 * scale),
                    y + round(18 * scale) + len(block.lines) * line_height(body_font),
                ),
                radius=round(10 * scale),
                fill="#f1f2f3",
            )
            y += round(10 * scale)
        y = draw_lines(draw, block.lines, (text_x, y), body_font, fill="#202020")
        if block.image_size and entry.media_ref:
            y += round(12 * scale)
            source = self._assets.get(entry.media_ref)
            assert source is not None
            image.paste(cover_image(source, block.image_size), (text_x, y))
            y += block.image_size[1]
        if entry.kind is not EntryKind.COMMENT:
            self._draw_actions(draw, style, text_x, y + round(14 * scale), right, meta_font)
        draw.line(
            (left, block.top + block.height, right, block.top + block.height),
            fill=style.divider,
            width=max(1, round(2 * scale)),
        )

    @staticmethod
    def _draw_actions(
        draw: ImageDraw.ImageDraw,
        style: FeedStyle,
        left: int,
        y: int,
        right: int,
        action_font: ImageFont.FreeTypeFont,
    ) -> None:
        width = max(1, right - left)
        for index, label in enumerate(style.action_labels):
            x = left + round(width * index / len(style.action_labels))
            draw.text((x, y), f"○ {label}", font=action_font, fill="#737373")

    def _draw_footer(
        self,
        image: Image.Image,
        scene: SocialSnapshotScene,
        style: FeedStyle,
        body_font: ImageFont.FreeTypeFont,
        scale: float,
    ) -> None:
        draw = ImageDraw.Draw(image)
        top = image.height - round(110 * scale)
        draw.rectangle((0, top, image.width, image.height), fill="#ffffff")
        if scene.theme is SnapshotTheme.XIAOHONGSHU and scene.draft:
            draw.rounded_rectangle(
                (
                    round(34 * scale),
                    top + round(14 * scale),
                    image.width - round(34 * scale),
                    top + round(68 * scale),
                ),
                radius=round(26 * scale),
                fill="#f4f4f4",
            )
            draw.text(
                (round(58 * scale), top + round(20 * scale)),
                scene.draft[:80],
                font=body_font,
                fill="#4b4b4b",
            )
        else:
            labels = ("首页", "发现", "+", "消息", "我")
            for index, label in enumerate(labels):
                x = round((index + 0.5) * image.width / len(labels))
                draw.text(
                    (x - round(24 * scale), top + round(28 * scale)),
                    label,
                    font=body_font,
                    fill=style.accent if index == 0 else "#777777",
                )
