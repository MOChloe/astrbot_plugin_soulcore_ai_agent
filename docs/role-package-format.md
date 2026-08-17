# SoulCore 角色包格式

本文定义 SoulCore `.soulcore-role` 格式版本 1。版本 1 只接受 SoulCore 自己生成的包，不声明 CHARX、其他角色卡或通用 ZIP 兼容性。

## 容器

文件扩展名为 `.soulcore-role`，内容是未加密 ZIP。容器最多四个普通文件，名称和用途固定：

```text
manifest.json
role.json
assets/identity/private.<png|jpg|webp|gif>   # 可选
assets/identity/group.<png|jpg|webp|gif>     # 可选
```

目录项、符号链接、特殊文件、重复路径、大小写冲突路径、绝对路径、反斜杠、空路径段、`.`、`..`、驱动器路径、嵌套压缩包和未在清单声明的文件全部非法。只允许 ZIP `stored` 与 `deflated` 压缩方法。

安全上限：

- ZIP 文件：128 MiB；
- 全部条目声明的展开总量：192 MiB；
- `manifest.json`：64 KiB；
- `role.json`：128 MiB；
- 每张立绘：20 MiB，并继续服从 SoulCore 当前图片像素、帧数、播放时长与累计解码像素限制。

JSON 必须是 UTF-8 对象，不能包含重复键、`null`、未知字段、非法类型或超过 64 层嵌套。读取器不会跟随 URL，也不会把任意字段解释为远程资源。

## `manifest.json`

版本 1 的清单必须且只能包含以下字段：

```json
{
  "format": "soulcore-role-package",
  "format_version": 1,
  "content_mode": "sparse_patch",
  "title": "角色可读标题",
  "generator_version": "SoulCore 版本",
  "role_file": {
    "path": "role.json",
    "byte_size": 1234,
    "sha256": "64 位小写十六进制"
  },
  "assets": [
    {
      "scope": "private",
      "path": "assets/identity/private.png",
      "mime_type": "image/png",
      "byte_size": 5678,
      "sha256": "64 位小写十六进制"
    }
  ]
}
```

`title` 只用于下载文件名和预览，不重命名目标角色。`generator_version` 用于诊断格式来源，不决定 Prompt 预设。每个文件的实际字节数与 SHA-256 必须和清单完全一致。资产 `scope` 只能是 `private` 或 `group`，每种最多一个；路径扩展名必须与声明 MIME 精确匹配：PNG `.png`、JPEG `.jpg`、WebP `.webp`、GIF `.gif`。

## `role.json`

顶层字段均可缺失，但出现时只能是 `character`、`world` 和 `portraits`：

```json
{
  "character": {
    "identity": {
      "name": "阿澈",
      "aliases": ["小澈"],
      "overview": "……",
      "facts": ["……"]
    },
    "personality": {
      "traits_and_values": ["……"],
      "thinking_and_behavior": ["……"],
      "habits_and_emotions": ["……"]
    },
    "social": {
      "interaction_style": ["……"],
      "boundaries": ["……"]
    },
    "preferences": {
      "likes_and_interests": ["……"],
      "dislikes": ["……"]
    },
    "language": {
      "speaking_style": ["……"],
      "messaging_habits": ["……"],
      "address_habits": ["……"]
    },
    "dialogue_reference": "……",
    "visual": {
      "appearance": ["……"],
      "clothing": ["……"],
      "visual_boundaries": ["……"]
    },
    "capabilities": {
      "abilities": ["……"],
      "knowledge_scope": ["……"],
      "limitations": ["……"]
    },
    "custom_prompts": {
      "main_core_modes": { "self_initiated": "……" },
      "main_core_styles": {
        "relationship_context": "……",
        "speaking_style": "……",
        "sticker_style": "……",
        "thinking_style": "……",
        "content_style": "……",
        "conversation_content": "……"
      },
      "response_polish": { "writing_correction": "……" },
      "story_styles": { "involvement": "……", "stance": "……" },
      "background_creation": {
        "world_change": "……",
        "story_boundary": "……",
        "imagination": "……",
        "temperature": "……"
      }
    },
    "trigger_rules": [
      { "keys": ["潮汐"], "lookback_turns": 3, "content": "联系海边经验" }
    ]
  },
  "world": {
    "definition": {
      "world_brief": "……",
      "world_rules": "……",
      "life_direction": "……",
      "world_texture": "……",
      "expansion_policy": "OPEN"
    },
    "lore": [
      {
        "title": "旧灯塔",
        "aliases": ["灯塔"],
        "tags": ["地点"],
        "content": "……",
        "importance": 0.8
      }
    ],
    "boundaries": [
      {
        "severity": "HARD",
        "category": "CANON",
        "rule_text": "……",
        "positive_space": "……",
        "enabled": true
      }
    ]
  },
  "portraits": {
    "private": { "asset": "assets/identity/private.png", "label": "私聊立绘" },
    "group": { "clear": true }
  }
}
```

`world.definition.expansion_policy` 只能是 `OPEN` 或 `CANON_GUARDED`。世界资料标题必须非空且在包内唯一；重要度为 0 到 1。创作边界 `severity` 只能是 `HARD` 或 `PREFERENCE`。触发规则数量、Key 数量与回看轮数继续服从当前 SoulCore 角色模型上限。

立绘引用必须和 `manifest.assets` 一一对应。`clear: true` 不能与其他字段共存；资产引用必须同时带 `asset` 与 `label`。清单中未被 `role.json` 引用的资产也非法。

## 稀疏导出与导入语义

导出以当前版本 `CharacterModel()` 及空世界为基准，省略空白和内置默认内容。Prompt 预设 ID 不写入包；非默认 Prompt 只导出当前实际文字。

导入规则：

1. 字段缺失：保留目标当前值。
2. 对象：递归合并。
3. 普通数组：字段一旦出现便整体替换；`[]` 表示清空。
4. 允许为空的字符串：`""` 表示清空。
5. 必需 Prompt `custom_prompts.main_core_styles.relationship_context` 与 `custom_prompts.background_creation.story_boundary` 收到 `""` 时恢复导入版本的系统默认，并在预览中标注。
6. 任何导入的非默认 Prompt 都标记为 `custom`，不根据预设 ID 或字面相等关系重新推断。
7. `world.lore` 与 `world.boundaries` 缺失时保留，出现时整体替换，空数组清空。
8. 私聊与群聊立绘独立处理：缺失保留，资产替换，只有 `clear: true` 删除。
9. 包标题不创建、不选择、不重命名角色；应用目标始终是确认窗口中锁定的现有角色。

`{[character]}` 与 `{[User]}` 是唯一跨安装身份模板。发现 `{[User:…]}` 或其他绑定具体平台参与者的内部身份标记时，导出和导入都会失败；错误只报告字段路径，不回显标记内容。

## 预览、并发与幂等

上传完成不直接写入角色。服务端先完整校验包并生成按角色资料、自定义行为、触发规则、世界资料和立绘分组的预览。确认令牌绑定：

- 不可逆目标 `role_ref`；
- 角色修订与角色内容指纹；
- 世界修订；
- 私聊、群聊立绘指纹；
- 整个 ZIP 的 SHA-256。

确认时任一绑定值变化都会拒绝应用并要求重新预览。应用使用调用方提供的幂等键，并在角色修订表中以整包请求指纹锚定；同一请求重复提交返回原结果，幂等键被另一份包或另一版目标复用时失败关闭。

角色、世界、世界资料、创作边界和立绘记录在同一 `BEGIN IMMEDIATE` 事务中写入。新立绘文件在事务前登记耐久清理保护；事务失败自动回收，事务成功才解除保护并清理旧立绘。应用后发布数据库备份并使背景世界种子失效。角色包不会修改模型配置、快速设置决定、SoulCore 开关或任何实例数据。
