# hermes-plugin/ — Hermes Agent 记忆插件

把 V5 长期记忆作为 Hermes Agent 的 **Memory Provider** 接入，实现：

- 上下文压缩前自动注入相关记忆（情感状态 + 向量/关键词检索 + 记忆统计）
- 每轮对话后台写回 V5
- 会话结束时触发 consolidate（把对话提炼为事实）
- 会话开始注入身份/记忆能力说明

## 两种部署方式

### A. 自包含（推荐，零配置）

把整个 `ikaros_v5/` 目录（含同级的 `v5/` 包）放进 Hermes Agent 的插件目录：

```
<hermes-agent>/plugins/memory/ikaros_v5/
├── __init__.py
└── v5/                # V5 记忆引擎包（由 install.py --hermes-agent 自动复制）
```

Hermes 启动时会自动发现 `ikaros_v5` 作为 memory provider，无需任何环境变量。

### B. 仅放插件 + 环境变量

只复制 `__init__.py`，并设置：

```
V5_MEMORY_ROOT=<v5-memory 仓库路径>     # 其下需有 v5/ 包
```

插件会在 `V5_MEMORY_ROOT/v5/` 找到引擎。

## 与 MCP 服务的关系

- 本插件负责**记忆生命周期闭环**（注入/写回/压缩），直接 import `v5` 包。
- 另有一个 **MCP 服务器**（`v5/mcp_server.py`，41 个 `v5_*` 工具）提供给 agent
  按需主动调用记忆工具。两者互补：MCP 在 `hermes-mcp-v5.json` 中注册。

通常由 `install.py` 一次性完成两件事：生成 `hermes-mcp-v5.json`（MCP 配置片段），
并在 `--hermes-agent` 时把自包含插件部署到位。
