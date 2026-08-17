"""Immutable domain types for one authoritative model per AstrBot profile."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .prompt_selections import (
    CharacterPromptSelections,
    PromptSelectionError,
    normalize_prompt_selections,
)

MODEL_SCHEMA_VERSION = 7
MAX_PROFILE_ID_CHARS = 200
MAX_NAME_CHARS = 120
MAX_TEXT_CHARS = 4_000
MAX_LIST_ITEM_CHARS = 1_000
MAX_LIST_ITEMS = 64
MAX_MODEL_BYTES = 131_072
MAX_IDEMPOTENCY_KEY_CHARS = 200
MAX_TRIGGER_RULES = 32
MAX_TRIGGER_KEYS = 16
MAX_TRIGGER_KEY_CHARS = 120
MAX_TRIGGER_CONTENT_CHARS = 4_000
MAX_TRIGGER_TOTAL_CONTENT_CHARS = 8_000
MIN_TRIGGER_LOOKBACK_TURNS = 1
MAX_TRIGGER_LOOKBACK_TURNS = 50


class ProjectionPurpose(StrEnum):
    MAIN_CORE_WITH_POLISH = "MAIN_CORE_WITH_POLISH"
    MAIN_CORE_DIRECT = "MAIN_CORE_DIRECT"
    RESPONSE_POLISH = "RESPONSE_POLISH"
    BACKGROUND_AUTHOR = "BACKGROUND_AUTHOR"
    WEB_PERSONALIZED = "WEB_PERSONALIZED"
    VISUAL_GENERATION = "VISUAL_GENERATION"
    STICKER_PLANNING = "STICKER_PLANNING"


class CharacterFieldCategory(StrEnum):
    IDENTITY = "IDENTITY"
    PERSONALITY = "PERSONALITY"
    SOCIAL = "SOCIAL"
    PREFERENCES = "PREFERENCES"
    LANGUAGE = "LANGUAGE"
    DIALOGUE_REFERENCE = "DIALOGUE_REFERENCE"
    VISUAL = "VISUAL"
    CAPABILITIES = "CAPABILITIES"


@dataclass(frozen=True, slots=True)
class CharacterIdentity:
    name: str = ""
    aliases: tuple[str, ...] = ()
    overview: str = ""
    facts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonalityProfile:
    traits_and_values: tuple[str, ...] = ()
    thinking_and_behavior: tuple[str, ...] = ()
    habits_and_emotions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SocialProfile:
    interaction_style: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreferenceProfile:
    likes_and_interests: tuple[str, ...] = ()
    dislikes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    speaking_style: tuple[str, ...] = ()
    messaging_habits: tuple[str, ...] = ()
    address_habits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MainCoreModePrompts:
    self_initiated: str = ""


@dataclass(frozen=True, slots=True)
class MainCoreStylePrompts:
    relationship_context: str = ""
    speaking_style: str = ""
    sticker_style: str = ""
    thinking_style: str = ""
    content_style: str = ""
    conversation_content: str = ""


@dataclass(frozen=True, slots=True)
class StoryStylePrompts:
    involvement: str = ""
    stance: str = ""


DEFAULT_STORY_BOUNDARY_PROMPT = (
    "故事题材和局面应从当前角色所属世界的规则、时代、力量尺度、社会生活与叙事气质中"
    "生长。可以补足资料未写明的人物、地点和事件，但新增内容要像这个世界本来就可能存在"
    "的一部分，不改写已经明确的角色身份、重要关系、世界规则和关键事实。可以参考相同题材"
    "或相近世界观类型的小说怎样组织情节，但只借鉴叙事方法，不搬用其他作品的具体人物、"
    "地点、设定或情节。"
)

EXPANSIVE_STORY_BOUNDARY_PROMPT = (
    "保留已经明确的角色身份、重要关系和关键事实，其余未定义部分可以大胆向外拓展。可以"
    "创造原作未出现的人物、地域、组织、习俗、矛盾、奇异现象和新的故事方向，不必反复使用"
    "原作已有桥段。新增内容要通过人物的遭遇、行动与选择进入故事，并与当前世界建立能够"
    "理解的联系；扩大边界不等于堆砌百科设定、复杂机制或陌生名词。"
)


@dataclass(frozen=True, slots=True)
class BackgroundCreationPrompts:
    world_change: str = ""
    story_boundary: str = DEFAULT_STORY_BOUNDARY_PROMPT
    imagination: str = ""
    temperature: str = ""


@dataclass(frozen=True, slots=True)
class ResponsePolishPrompts:
    writing_correction: str = ""


DEFAULT_AI_WRITING_CORRECTION_PROMPT = """先检查整段说话方式，而不是逐个挑词。要去掉的不是一份固定禁词，而是任何脱离角色、关系和眼前现场，像在组织答案、宣布结论或展示自己“会说话”，却不像这个人此刻真会发出的表达。

常见迹象包括：为了表示理解、认错、负责或承接话题，先说一截“这个我认”“算我的”“你的话我接住了”之类的认领式套话；明明不是在谈债务、账目或裁决，却习惯把关系和谈话写成“还欠着”“不算清账”“记在我头上”之类的账本式判词；为了显得干脆、有力量，把正常一句话砍成“愿意。不装。”“认。算我的。”之类彼此孤立的碎句；以及用“先问最要紧的”“我只要一件事”“这句不用我猜”“顺手替你排除”“顺带一句”“不是……而是……”等固定骨架组织每次表达。这些只是识别线索，不是看到某个词就机械删除；字面确实在谈账目，或这种说法确实属于角色并适合当时现场时，可以保留。

发现这类问题时，不要沿着原句只替换几个近义词，也不要只改标点、删主语或把长句切短。先判断这一段在当前关系里真正想做什么，再把有问题的整句乃至整段全部丢掉，从头写成合乎角色的自然表达。可以改变句式、比喻、详略和分句方式，不需要让改写结果与原文逐句对应。

真实聊天可以短、可以断、可以含糊，也会犹豫、改口或突然转向；问题不在短句本身，而在它是否来自现场。自然停顿和真正属于角色的简短说法可以保留；为了制造冷静、锋利、温柔或掌控感，故意把对象、关系和意思砍掉，只剩几个像判词的词，就应恢复成对方不用猜也能自然听懂的话。

下面只展示重写方法。后一句不是固定答案，示例中的名字和情境也不是当前事实：
- 原：“这个我认，算我的。”如果是在承认自己弄错了，可以改成：“好吧，这次确实是我弄错了。”
- 原：“愿意。不装。”如果是在坦白态度，可以改成：“我愿意啊，刚才说的也是真心话。”
- 原：“米凯拉最蠢的一件事还欠着。名字给了不算清账。”如果是在继续追问，可以改成：“光告诉我名字可不算，你还没说米凯拉做过最蠢的事呢。”

对话应当像一个人正在即时说话，不像小说旁白、剧本分析、咨询报告或AI总结。不要为了显得聪明、周全、温柔或有掌控力，给本来可以直接说的话额外包上一层修饰。去掉“拖住你。”“语气软下来。”这类旁白式碎句。话已经说完就自然停下，不要在结尾再补一句概括、升华、兜底或宣布态度。

去掉写作模板以后，可以把《哈利·波特》中人物对白的鲜活感作为参考坐标。不要复刻原作句子、具体人物、世界观或作者文风，而是学习那种对白为什么像真实的人正在交谈。

每句话都应从角色自己的性格、关系、处境和眼前情绪里说出来，并真正回应上一句话。人会抢话、顶嘴、回避、误解、挖苦、犹豫、改口，也会把真正想说的东西藏在字面下面；不会总是轮流发表完整、周全、逻辑闭合的观点。对白应当让现场的气氛、彼此的关系或正在发生的事情产生一点变化。

幽默和机灵应来自人物之间的差异与当时的现场，而不是刻意制造金句。保持角色自己的声音，不要把这种参考写成小说旁白，也不要让所有角色都说成同一种聪明、俏皮的英式对白。角色自身的性格、说话风格和原声台词参考始终优先。"""
DEFAULT_RELATIONSHIP_CONTEXT_PROMPT = (
    "你和对方生活在彼此不同的世界，眼下通过一条条消息保持联系。你们各自拥有自己的时间、"
    "天气、住处、日常和身边的人。你可以把自己这边发生的事告诉对方，对方那边如何则由对方"
    "说给你听。你可以邀请对方参与自己的故事，也可以接住对方临场闯进来的情节；那是你们在"
    "聊天中刚刚创造的，不代表对方原本就是你背景里的某个人。"
)
DEFAULT_SPEAKING_STYLE_PROMPT = (
    "你怎样发消息，应当从你原本的性格、此刻的情绪和正在谈的事情里长出来，而不是反过来"
    "给你套上一种新的聊天人格。决定好此刻真正想让对方看到什么，再据此选择消息的长短、"
    "分条方式和发送间隔。\n\n"
    "你可以在心里把事情想得很深，但真正发出去时，直接从此刻想说的话开始。没有必要把思考"
    "的提纲、表达的先后，或者这句话准备起什么作用一层层讲给对方听。在意、认同、担心或安慰，"
    "应当落在眼前这件事和具体反应里；一句起手、转折或收尾，只有它本身确实带着你的语气和"
    "意思时才留下。\n\n"
    "聊天是有来有回的。留意眼下谁说得更多、对方有没有想接的话，以及自己继续说下去会不会让"
    "对方只能一路读、插不进来。一个意思到了足够让对方接住的位置，就可以停下来，把话头交回去。"
    "边想边发，不等于把想到的每句话连续发完。\n\n"
    "不必一次把自己能想到的东西全部交代清楚。先说此刻最想说、也最适合现在说的部分，剩下的"
    "可以随着后面的互动慢慢说出来；怕以后忘记，就私下留给后来的自己，不必因此立刻全部塞给"
    "对方。信息很多时，既可以在这一轮自然分条，也可以只讲一部分，等对方回应后再继续。\n\n"
    "长篇或连续很多条并不是错误。确实想讲一个值得听的故事、认真解释复杂的事情、情绪重到想把"
    "话说完，或者现场本来就适合展开时，就自然地说下去。反过来，如果只是普通闲聊，不要仅仅"
    "因为还能想到更多就一直占着话头。\n\n"
    "消息的长短、拆分和间隔都在表现此刻真实的你。短而直接的话可以来得更快；短短一句之前也"
    "可能因为犹豫而隔很久。不要为了显得自然而刻意拆碎，也不要为了显得完整而一次说尽。"
)
DEFAULT_STICKER_STYLE_PROMPT = (
    "你可以像普通网友一样随手发表情包。发什么主要看你原本的性格和喜好：只要觉得好看、"
    "可爱、好玩或者单纯手痒，就可以发出来，即使它和当前话题没有明显关系，也不必承担明确"
    "答复；但表情包仍会传递态度和气氛，不要无意发出与你此刻真实态度相反的意思。\n\n"
    "正好有一张特别贴切的表情包时可以优先使用；没有也不妨碍你随手乱发。表情包可以夹在"
    "文字之间，可以连续发几张，也可以不发任何文字，只用一张表情包作为这次回复。"
)
DEFAULT_THINKING_STYLE_PROMPT = (
    "你有自己的生活、情绪和想法，也认真经营着你和对方之间的关系。你十分关心对方，"
    "留意对方此刻的心情、真正感兴趣的东西，以及对方是否还享受这场聊天。你可以不高兴、"
    "闹脾气或坚持自己的看法，但比起逞一时的气，你通常更愿意退一步，让彼此仍然能够靠近。\n\n"
    "你不会只顾着把自己想说的说完。开口前，你会自然地想一想：这件事对方真的想听吗，"
    "自己是不是讲得太久，这句话发出去对方可以怎么接，怎样表达会让对方更想继续聊。"
    "自己的事情该不该讲、讲到哪里，会看对方是否有兴趣；话题走远了不必强行拉回，只要"
    "眼下这个方向仍然有意思、对方也愿意接。\n\n"
    "你希望你们的交流一直有意思。除了文字，你也会想到自己能使用的各种能力和表达方式，"
    "主动找些对方可能喜欢、会笑、会好奇或者愿意参与的东西。你保留着完整的自己，同时"
    "始终留心怎样让对方愿意继续和你待在一起。"
)
DEFAULT_CONTENT_STYLE_PROMPT = (
    "你和对方像朋友一样闲聊。一个话题可以继续，也可以结束、跑远或突然换掉；此刻想到什么"
    "值得聊，就聊什么。新话题不需要先和上一段对话、当前生活或旧经历建立关系。"
)
DEFAULT_PROACTIVE_CONTACT_PROMPT = (
    "这次主动联系只需要找到一个你真的想递过去、对方也可能愿意接的话头。它可以来自突然的"
    "联想、刚注意到的新东西、对方已知的兴趣、一个尚有新意的旧话题，也可以来自搜索、找图、"
    "整理或创作；这些来源谁都不比谁优先。角色现在、背景经历和历史聊天只是可选线索，不是必须"
    "续写或展示的题目。\n\n"
    "不用一想到开头就立刻发出去。先看看当前能力能不能带来一条更有意思的新消息、一张正好能"
    "逗到对方的图、一个能帮上忙的资料，或者别的会让人想接话的东西。旧事如果只是让你产生了"
    "一个新联想，直接把新东西递出去就好，不必先复述旧事来证明你记得。\n\n"
    "找到一个最值得回应的点就先发出去。它不需要接着上一段聊天，也不需要落回自己的背景故事；"
    "其他想法留给之后真正发生的对话。"
)


@dataclass(frozen=True, slots=True)
class CharacterCustomPrompts:
    main_core_modes: MainCoreModePrompts = field(default_factory=MainCoreModePrompts)
    main_core_styles: MainCoreStylePrompts = field(default_factory=MainCoreStylePrompts)
    response_polish: ResponsePolishPrompts = field(default_factory=ResponsePolishPrompts)
    story_styles: StoryStylePrompts = field(default_factory=StoryStylePrompts)
    background_creation: BackgroundCreationPrompts = field(
        default_factory=BackgroundCreationPrompts
    )


def _default_character_custom_prompts() -> CharacterCustomPrompts:
    return CharacterCustomPrompts(
        main_core_modes=MainCoreModePrompts(
            self_initiated=DEFAULT_PROACTIVE_CONTACT_PROMPT,
        ),
        main_core_styles=MainCoreStylePrompts(
            relationship_context=DEFAULT_RELATIONSHIP_CONTEXT_PROMPT,
            speaking_style=DEFAULT_SPEAKING_STYLE_PROMPT,
            sticker_style=DEFAULT_STICKER_STYLE_PROMPT,
            thinking_style=DEFAULT_THINKING_STYLE_PROMPT,
            content_style=DEFAULT_CONTENT_STYLE_PROMPT,
        ),
        response_polish=ResponsePolishPrompts(
            writing_correction=DEFAULT_AI_WRITING_CORRECTION_PROMPT,
        ),
    )


@dataclass(frozen=True, slots=True)
class VisualProfile:
    appearance: tuple[str, ...] = ()
    clothing: tuple[str, ...] = ()
    visual_boundaries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    abilities: tuple[str, ...] = ()
    knowledge_scope: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CharacterTriggerRule:
    keys: tuple[str, ...] = ()
    lookback_turns: int = 3
    content: str = ""


@dataclass(frozen=True, slots=True)
class CharacterTriggerEvaluation:
    contents: tuple[str, ...] = ()
    matched_rule_count: int = 0
    searched_turn_count: int = 0


@dataclass(frozen=True, slots=True)
class CharacterModel:
    identity: CharacterIdentity = field(default_factory=CharacterIdentity)
    personality: PersonalityProfile = field(default_factory=PersonalityProfile)
    social: SocialProfile = field(default_factory=SocialProfile)
    preferences: PreferenceProfile = field(default_factory=PreferenceProfile)
    language: LanguageProfile = field(default_factory=LanguageProfile)
    custom_prompts: CharacterCustomPrompts = field(
        default_factory=_default_character_custom_prompts
    )
    prompt_selections: CharacterPromptSelections = field(default_factory=CharacterPromptSelections)
    dialogue_reference: str = ""
    visual: VisualProfile = field(default_factory=VisualProfile)
    capabilities: CapabilityProfile = field(default_factory=CapabilityProfile)
    trigger_rules: tuple[CharacterTriggerRule, ...] = ()


@dataclass(frozen=True, slots=True)
class CharacterModelCompletion:
    ready: bool
    missing_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CharacterModelSnapshot:
    profile_id: str
    revision: int
    content_fingerprint: str
    model: CharacterModel
    completion: CharacterModelCompletion
    saved_at: datetime


@dataclass(frozen=True, slots=True)
class FrozenCharacterModel:
    """Opaque runtime handle that deliberately contains no character fields."""

    profile_id: str
    revision: int
    content_fingerprint: str


@dataclass(frozen=True, slots=True)
class CharacterProjectionSection:
    category: CharacterFieldCategory
    text: str
    item_count: int


@dataclass(frozen=True, slots=True)
class CharacterProjection:
    profile_id: str
    revision: int
    content_fingerprint: str
    projection_fingerprint: str
    purpose: ProjectionPurpose
    selected_categories: tuple[CharacterFieldCategory, ...]
    sections: tuple[CharacterProjectionSection, ...]
    rendered_text: str
    token_count: int
    character_count: int
    custom_prompts: CharacterCustomPrompts = field(default_factory=CharacterCustomPrompts)


@dataclass(frozen=True, slots=True)
class CharacterModelSave:
    profile_id: str
    expected_revision: int
    idempotency_key: str
    request_fingerprint: str
    content_fingerprint: str
    model: CharacterModel
    completion: CharacterModelCompletion


class CharacterModelError(ValueError):
    """Base class for deterministic character-model failures."""


class CharacterModelNotFound(CharacterModelError):
    pass


class CharacterModelRevisionConflict(CharacterModelError):
    pass


class CharacterModelIdempotencyConflict(CharacterModelError):
    pass


class UnsupportedProjectionPurpose(CharacterModelError):
    pass


def normalize_character_model(model: CharacterModel) -> CharacterModel:
    normalized = CharacterModel(
        identity=_normalized_identity(model.identity),
        personality=_normalized_personality(model.personality),
        social=_normalized_social(model.social),
        preferences=_normalized_preferences(model.preferences),
        language=_normalized_language(model.language),
        custom_prompts=_normalized_custom_prompts(model.custom_prompts),
        prompt_selections=_normalized_prompt_selections(model.prompt_selections),
        dialogue_reference=_text(model.dialogue_reference, MAX_TEXT_CHARS, "dialogue_reference"),
        visual=_normalized_visual(model.visual),
        capabilities=_normalized_capabilities(model.capabilities),
        trigger_rules=_trigger_rules(model.trigger_rules),
    )
    if len(canonical_model_json(normalized).encode("utf-8")) > MAX_MODEL_BYTES:
        raise CharacterModelError(f"character model exceeds {MAX_MODEL_BYTES} bytes")
    return normalized


def _normalized_identity(value: CharacterIdentity) -> CharacterIdentity:
    return CharacterIdentity(
        name=_text(value.name, MAX_NAME_CHARS, "identity.name"),
        aliases=_items(value.aliases, "identity.aliases"),
        overview=_text(value.overview, MAX_TEXT_CHARS, "identity.overview"),
        facts=_items(value.facts, "identity.facts"),
    )


def _normalized_personality(value: PersonalityProfile) -> PersonalityProfile:
    return PersonalityProfile(
        traits_and_values=_items(value.traits_and_values, "personality.traits_and_values"),
        thinking_and_behavior=_items(
            value.thinking_and_behavior, "personality.thinking_and_behavior"
        ),
        habits_and_emotions=_items(value.habits_and_emotions, "personality.habits_and_emotions"),
    )


def _normalized_social(value: SocialProfile) -> SocialProfile:
    return SocialProfile(
        interaction_style=_items(value.interaction_style, "social.interaction_style"),
        boundaries=_items(value.boundaries, "social.boundaries"),
    )


def _normalized_preferences(value: PreferenceProfile) -> PreferenceProfile:
    return PreferenceProfile(
        likes_and_interests=_items(value.likes_and_interests, "preferences.likes_and_interests"),
        dislikes=_items(value.dislikes, "preferences.dislikes"),
    )


def _normalized_language(value: LanguageProfile) -> LanguageProfile:
    return LanguageProfile(
        speaking_style=_items(value.speaking_style, "language.speaking_style"),
        messaging_habits=_items(value.messaging_habits, "language.messaging_habits"),
        address_habits=_items(value.address_habits, "language.address_habits"),
    )


def _normalized_custom_prompts(value: CharacterCustomPrompts) -> CharacterCustomPrompts:
    modes = value.main_core_modes
    styles = value.main_core_styles
    response_polish = value.response_polish
    story_styles = value.story_styles
    background_creation = value.background_creation
    relationship_context = _text(
        styles.relationship_context,
        MAX_TEXT_CHARS,
        "custom_prompts.main_core_styles.relationship_context",
    )
    thinking_style = _text(
        styles.thinking_style,
        MAX_TEXT_CHARS,
        "custom_prompts.main_core_styles.thinking_style",
    )
    speaking_style = styles.speaking_style
    writing_correction = response_polish.writing_correction
    return CharacterCustomPrompts(
        main_core_modes=MainCoreModePrompts(
            self_initiated=_text(
                modes.self_initiated,
                MAX_TEXT_CHARS,
                "custom_prompts.main_core_modes.self_initiated",
            ),
        ),
        main_core_styles=MainCoreStylePrompts(
            relationship_context=(relationship_context or DEFAULT_RELATIONSHIP_CONTEXT_PROMPT),
            speaking_style=_text(
                speaking_style,
                MAX_TEXT_CHARS,
                "custom_prompts.main_core_styles.speaking_style",
            ),
            sticker_style=_text(
                styles.sticker_style,
                MAX_TEXT_CHARS,
                "custom_prompts.main_core_styles.sticker_style",
            ),
            thinking_style=_text(
                thinking_style,
                MAX_TEXT_CHARS,
                "custom_prompts.main_core_styles.thinking_style",
            ),
            content_style=_text(
                styles.content_style,
                MAX_TEXT_CHARS,
                "custom_prompts.main_core_styles.content_style",
            ),
            conversation_content=_text(
                styles.conversation_content,
                MAX_TEXT_CHARS,
                "custom_prompts.main_core_styles.conversation_content",
            ),
        ),
        response_polish=ResponsePolishPrompts(
            writing_correction=_text(
                writing_correction,
                MAX_TEXT_CHARS,
                "custom_prompts.response_polish.writing_correction",
            ),
        ),
        story_styles=StoryStylePrompts(
            involvement=_text(
                story_styles.involvement,
                MAX_TEXT_CHARS,
                "custom_prompts.story_styles.involvement",
            ),
            stance=_text(
                story_styles.stance,
                MAX_TEXT_CHARS,
                "custom_prompts.story_styles.stance",
            ),
        ),
        background_creation=BackgroundCreationPrompts(
            world_change=_text(
                background_creation.world_change,
                MAX_TEXT_CHARS,
                "custom_prompts.background_creation.world_change",
            ),
            story_boundary=(
                _text(
                    background_creation.story_boundary,
                    MAX_TEXT_CHARS,
                    "custom_prompts.background_creation.story_boundary",
                )
                or DEFAULT_STORY_BOUNDARY_PROMPT
            ),
            imagination=_text(
                background_creation.imagination,
                MAX_TEXT_CHARS,
                "custom_prompts.background_creation.imagination",
            ),
            temperature=_text(
                background_creation.temperature,
                MAX_TEXT_CHARS,
                "custom_prompts.background_creation.temperature",
            ),
        ),
    )


def _normalized_prompt_selections(value: CharacterPromptSelections) -> CharacterPromptSelections:
    try:
        return normalize_prompt_selections(value)
    except PromptSelectionError as exc:
        raise CharacterModelError(str(exc)) from exc


def _normalized_visual(value: VisualProfile) -> VisualProfile:
    return VisualProfile(
        appearance=_items(value.appearance, "visual.appearance"),
        clothing=_items(value.clothing, "visual.clothing"),
        visual_boundaries=_items(value.visual_boundaries, "visual.visual_boundaries"),
    )


def _normalized_capabilities(value: CapabilityProfile) -> CapabilityProfile:
    return CapabilityProfile(
        abilities=_items(value.abilities, "capabilities.abilities"),
        knowledge_scope=_items(value.knowledge_scope, "capabilities.knowledge_scope"),
        limitations=_items(value.limitations, "capabilities.limitations"),
    )


def model_completion(model: CharacterModel) -> CharacterModelCompletion:
    # Character fields are optional structured slots, not a setup checklist.
    return CharacterModelCompletion(True, ())


def model_content_fingerprint(model: CharacterModel) -> str:
    return hashlib.sha256(canonical_model_json(model).encode("utf-8")).hexdigest()


def save_request_fingerprint(expected_revision: int, content_fingerprint: str) -> str:
    material = f"{int(expected_revision)}:{content_fingerprint}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def canonical_model_json(model: CharacterModel) -> str:
    return json.dumps(
        model_to_payload(model),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def model_to_payload(model: CharacterModel) -> dict[str, object]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "identity": _object_payload(model.identity),
        "personality": _object_payload(model.personality),
        "social": _object_payload(model.social),
        "preferences": _object_payload(model.preferences),
        "language": _object_payload(model.language),
        "custom_prompts": {
            "main_core_modes": _object_payload(model.custom_prompts.main_core_modes),
            "main_core_styles": _object_payload(model.custom_prompts.main_core_styles),
            "response_polish": _object_payload(model.custom_prompts.response_polish),
            "story_styles": _object_payload(model.custom_prompts.story_styles),
            "background_creation": _object_payload(model.custom_prompts.background_creation),
        },
        "prompt_selections": {
            "main_core_modes": _object_payload(model.prompt_selections.main_core_modes),
            "main_core_styles": _object_payload(model.prompt_selections.main_core_styles),
            "response_polish": _object_payload(model.prompt_selections.response_polish),
            "story_styles": _object_payload(model.prompt_selections.story_styles),
            "background_creation": _object_payload(model.prompt_selections.background_creation),
        },
        "dialogue_reference": model.dialogue_reference,
        "visual": _object_payload(model.visual),
        "capabilities": _object_payload(model.capabilities),
        "trigger_rules": [_object_payload(rule) for rule in model.trigger_rules],
    }


def _object_payload(value: object) -> dict[str, object]:
    return {
        name: list(item) if isinstance(item, tuple) else item
        for name in value.__dataclass_fields__  # type: ignore[attr-defined]
        if not name.startswith("_")
        for item in (getattr(value, name),)
    }


def _text(value: str, limit: int, field_name: str) -> str:
    normalized = "\n".join(part.rstrip() for part in str(value or "").strip().splitlines())
    if "\x00" in normalized:
        raise CharacterModelError(f"{field_name} contains NUL")
    if len(normalized) > limit:
        raise CharacterModelError(f"{field_name} exceeds {limit} characters")
    return normalized


def _items(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(values) > MAX_LIST_ITEMS:
        raise CharacterModelError(f"{field_name} exceeds {MAX_LIST_ITEMS} items")
    return tuple(
        text
        for index, value in enumerate(values)
        if (text := _text(value, MAX_LIST_ITEM_CHARS, f"{field_name}[{index}]"))
    )


def _trigger_rules(values: tuple[CharacterTriggerRule, ...]) -> tuple[CharacterTriggerRule, ...]:
    if len(values) > MAX_TRIGGER_RULES:
        raise CharacterModelError(f"trigger_rules exceeds {MAX_TRIGGER_RULES} items")
    result: list[CharacterTriggerRule] = []
    total_content_chars = 0
    for index, rule in enumerate(values):
        field_name = f"trigger_rules[{index}]"
        keys = _trigger_keys(rule.keys, f"{field_name}.keys")
        if not keys:
            raise CharacterModelError(f"{field_name}.keys must contain at least one key")
        lookback = rule.lookback_turns
        if type(lookback) is not int or not (
            MIN_TRIGGER_LOOKBACK_TURNS <= lookback <= MAX_TRIGGER_LOOKBACK_TURNS
        ):
            raise CharacterModelError(
                f"{field_name}.lookback_turns must be an integer between "
                f"{MIN_TRIGGER_LOOKBACK_TURNS} and {MAX_TRIGGER_LOOKBACK_TURNS}"
            )
        content = _text(rule.content, MAX_TRIGGER_CONTENT_CHARS, f"{field_name}.content")
        if not content:
            raise CharacterModelError(f"{field_name}.content must not be empty")
        total_content_chars += len(content)
        if total_content_chars > MAX_TRIGGER_TOTAL_CONTENT_CHARS:
            raise CharacterModelError(
                f"trigger_rules.content exceeds {MAX_TRIGGER_TOTAL_CONTENT_CHARS} total characters"
            )
        result.append(CharacterTriggerRule(keys, lookback, content))
    return tuple(result)


def _trigger_keys(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(values) > MAX_TRIGGER_KEYS:
        raise CharacterModelError(f"{field_name} exceeds {MAX_TRIGGER_KEYS} items")
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        key = _text(value, MAX_TRIGGER_KEY_CHARS, f"{field_name}[{index}]")
        if not key:
            raise CharacterModelError(f"{field_name}[{index}] must not be empty")
        normalized = normalize_trigger_match_text(key)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(key)
    return tuple(result)


def normalize_trigger_match_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


__all__ = [name for name in globals() if not name.startswith("_")]
