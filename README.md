<p align="center">
  <img src="https://raw.githubusercontent.com/MOChloe/astrbot_plugin_soulcore_ai_agent/main/pages/role-settings/readme/soulcore-hero.svg" alt="SoulCore——让角色在每一次交流里，都更像自己。">
</p>

<h1 align="center">SoulCore AI Agent</h1>

<p align="center">
  面向 AstrBot 长期角色交流场景的独立运行系统。完成设置后继续正常聊天，角色需要的设定、记忆、关系与当前生活由 SoulCore 整理并延续。
</p>

<p align="center">
  <a href="https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/getting-started.md">快速开始</a> ·
  <a href="https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/features.md">能力说明</a> ·
  <a href="https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/troubleshooting.md">故障排查</a> ·
  <a href="https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/CHANGELOG.md">更新日志</a>
</p>

## SoulCore 是什么

SoulCore 适合希望角色在较长时间里保持连续、又不想手动整理上下文的 AstrBot 用户。它不只是把更多聊天记录塞给模型，而是围绕角色当前真正需要的内容，分别管理角色资料、长期记忆、关系状态、生活进展和行动结果。

完成快速设置后，日常使用仍然发生在原来的聊天窗口。独立界面用于查看和调整角色状态，不要求用户为了继续对话反复搬运、压缩或选择历史消息。

## 实际界面与核心体验

| 快速设置 | 角色当前状态 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/MOChloe/astrbot_plugin_soulcore_ai_agent/main/docs/assets/quick-setup.png" alt="SoulCore 快速设置欢迎页"> | <img src="https://raw.githubusercontent.com/MOChloe/astrbot_plugin_soulcore_ai_agent/main/docs/assets/player-now.png" alt="SoulCore 玩家页面"> |

界面展示的是已经保存的角色状态，而不是聊天窗口的实时镜像。角色仍通过 AstrBot 原有会话收发消息，SoulCore 在后台组织与这次交流有关的信息。

## 能做什么

| 能力 | 使用中的表现 |
| :--- | :--- |
| 角色连续性 | 统一延续角色设定、说话方式、关系边界与当前处境。 |
| 长期记忆与关系 | 从真实交流中整理值得保留的经历，并按不同联系人分别维护关系状态。 |
| 生活与主动联系 | 角色可以拥有聊天之外的生活进展，并在满足条件时主动联系。 |
| 多轮行动 | 查询、生成等任务先取得真实结果，再由角色形成最终表达；计划和草稿不会直接发进聊天。 |
| 图片、语音与表情 | 根据模型、适配器和已启用能力理解或发送媒体，让表达方式更贴合当前情境。 |
| 可查看、可调整 | 在独立界面查看当前生活、联系人、相处状态和角色资料，并按需修改设置。 |

更完整的能力边界与依赖见[能力说明](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/features.md)。

## 四步开始使用

1. 在 AstrBot 插件市场安装 SoulCore AI Agent，或使用本仓库地址安装。
2. 进入 `插件管理 → SoulCore AI Agent → 打开插件 UI 界面`。
3. 按快速设置完成模型、角色和所需能力的配置。
4. 回到聊天窗口正常交流；新会话首次消息完成初始化后，再发送下一条消息开始聊天。

详细步骤见[快速开始](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/getting-started.md)。

<p align="center">
  <img src="https://raw.githubusercontent.com/MOChloe/astrbot_plugin_soulcore_ai_agent/main/pages/role-settings/readme/soulcore-entry.svg" alt="SoulCore 插件界面入口与聊天方式说明">
</p>

## 模型与可选能力

| 项目 | 是否必需 | 说明 |
| :--- | :---: | :--- |
| 主力文字模型 | 必需 | 负责角色在普通对话中的判断、行动与最终回复。快速设置会用真实请求验证连接。 |
| 快速文字模型 | 可选 | 可复用主力模型，也可单独配置以承担轻量任务。 |
| 视觉理解 | 可选 | 用于理解对话中的图片。 |
| 联网查询 | 可选 | 用于需要实时信息的查询与网页读取。 |
| 图片生成 | 可选 | 用于角色按情境生成并发送图片；测试可能产生实际调用费用。 |
| 回复润色 | 可选 | 在不改变角色决策的前提下优化最终表达。 |

## 与其他插件共存

SoulCore 支持与仅通过 **AstrBot 标准注册指令** 工作的插件共同使用。这类指令及其插件回复不会写入 SoulCore，可正常调用。

会读取、改写对话上下文，或同时介入普通对话链路的插件不在兼容范围内，例如另一套记忆、上下文管理或自动回复系统。多个插件同时接管同一段普通对话时，无法保证各自拿到完整且一致的输入。

## 常见问题

**安装后为什么第一条消息没有正常回复？**<br>
新会话第一次收到普通消息时会先完成 SoulCore 初始化，并明确提示“初始化完成”。看到提示后再发送一条消息即可。

**必须把所有可选模型都配好吗？**<br>
不需要。主力文字模型是唯一必需项；视觉、联网、生图、润色等按实际需要启用。

**为什么独立界面里的内容没有立刻变化？**<br>
界面展示已保存状态。刷新对应页面可读取最新结果；仍不一致时参照[故障排查](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/troubleshooting.md)。

## 文档

- [文档入口](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/README.md)
- [快速开始](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/getting-started.md)
- [能力说明](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/features.md)
- [故障排查](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/troubleshooting.md)
- [架构说明](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/architecture.md)
- [数据边界](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/data-boundaries.md)
- [角色包格式](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/role-package-format.md)
- [开发说明](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/docs/development.md)
- [公开更新日志](https://github.com/MOChloe/astrbot_plugin_soulcore_ai_agent/blob/main/CHANGELOG.md)

## 交流与反馈

- 使用交流、配置疑问和一般报错：[加入 SoulCore 交流报错群](https://qm.qq.com/q/EJTchyUL5e)，群号 `1038479108`。
- 能稳定复现的问题：请使用仓库的 Issues，并提供 AstrBot 版本、SoulCore 版本、适配器、复现步骤和必要日志。
- 公开反馈前请移除 API Key、Token、Cookie、真实私聊内容及其他敏感信息。

<p align="center">
  <a href="https://qm.qq.com/q/EJTchyUL5e"><img src="https://raw.githubusercontent.com/MOChloe/astrbot_plugin_soulcore_ai_agent/main/pages/role-settings/readme/soulcore-qq.svg" alt="SoulCore 交流报错群，群号 1038479108"></a>
</p>
