# 开发与检查

SoulCore 要求 Python 3.11 或更高版本。使用 `requirements-dev.txt` 安装运行与开发依赖。

在 PowerShell 7 中运行唯一的本地检查入口：

```powershell
./scripts/check.ps1 -PythonPath <python.exe>
```

检查入口会把 Pytest、Python 临时文件统一写入项目内的 `.tmp/`。可通过
`-PytestBaseTemp <名称>` 指定 `.tmp/pytest/` 下的独立子目录；绝对路径也必须位于该
受管目录内。脚本和测试入口都会拒绝盘符根目录、用户目录或其他项目外路径，避免测试
产物污染外围文件系统。

使用 `-Full` 可额外执行浏览器冒烟测试和发布包验证。Node 只用于插件高级设置页面的开发与测试，插件运行不依赖 Node。

当前 mypy 门禁覆盖跨域 `contracts`、`shared` 公共内核，以及所有 feature 的
`domain.py` / `ports.py`。这些文件定义模块之间必须稳定的类型边界；实现层不得用
`ignore_errors` 掩盖问题，并继续接受 Ruff、架构契约和全量行为测试约束。类型覆盖范围
应随实现整理逐步扩大。

单独构建可直装 ZIP：

```powershell
<python.exe> scripts/build_release.py
```
