# ikaros-memory-v5

**Ikaros V5 长期记忆架构 —— 独立记忆插件。**

把 V5 的记忆引擎抽离成一个**不依赖 Ikaros 主工程**的独立仓库，便于在全新下载的
干净 [Hermes Agent](https://www.codebuddy.cn) 上通过 **MCP 服务器 + Agent Provider 插件** 两种方式接入长期记忆。

- 对外唯一接口：`v5/mcp_server.py`（FastMCP，41 个 `v5_*` 工具）
- 生命周期闭环：`hermes-plugin/ikaros_v5/`（Hermes `MemoryProvider`）
- 依赖极小：仅 `chromadb` + `httpx` + `mcp` + `python-dotenv`

---

## 一、V5 架构的优点（为何值得独立成插件）

V5 不是简单的"把对话塞进向量库"，而是一套**受控、可解释、可持久**的记忆架构：

1. **三路融合召回（Fused Recall）** — SQLite FTS5 关键词 + ChromaDB 向量语义 + 时间范围，按
   `fused_score`（向量 0.7 / 关键词 0.3）融合。既抓"意思相近"也抓"字面命中"，召回比单路向量更稳。
2. **结构化内容守卫（V5-0109）** — `validation.py` 的 `StructuredContentGuard` 拦截 LLM 旁白、
   裸 JSON、markdown fence、超长 dump 写入记忆。记忆库不会被垃圾污染，检索质量长期稳定。
3. **逐轮轻写 + 后台整合双轨** — 每轮对话即时落库（轻、不阻塞），后台 `reflect` 流水线把对话
   提炼为事实 / 情感 / 自我模型更新。写入快、思考深，互不拖累。
4. **实体图谱扩散激活** — `entity_graph.py` 用图结构做"相关记忆扩散"，一次查询能带出一整片
   关联上下文，而不是孤立碎片。
5. **PAD + TLS 情感模型** — 用 Pleasure-Arousal-Dominance 三维 + 信任/亲密等关系量，
   情绪与关系是有状态、可衰减、可注入 prompt 的，而非一次性分类标签。
6. **双后端可移植** — 认知任务走云端 DeepSeek，本地 `:8080` llama-server 作为可选懒加载兜底。
   无 GPU 也能跑（纯云端），有 GPU 可完全离线。
7. **受控键空间 + 耐久安全** — 记忆键（type/tags/weight）受控、可审计；数据落在本地 SQLite+Chroma，
   用户完全拥有，不依赖任何第三方服务。
8. **即插即用 MCP 接口** — 全部能力通过标准 MCP 暴露，任何兼容 MCP 的 Agent 都能直接调用，
   零改造接入。

---

## 二、项目结构

```
v5-memory/
├── v5/                     # V5 记忆引擎包（核心，可移植）
│   ├── mcp_server.py       # MCP 服务器入口（41 个 v5_* 工具）
│   ├── cli.py              # 控制台入口 ikaros-mem-v5
│   ├── store.py            # SQLite FTS5 存储
│   ├── search.py           # 三路融合检索 + 向量索引
│   ├── memory_retrieval.py # 对外检索 API
│   ├── validation.py       # 结构化内容守卫
│   ├── affect.py / self_model.py / profile.py / relationship.py ...
│   ├── reflect/            # 后台整合流水线（consolidate/distill/reflect）
│   ├── tools/              # MCP 工具实现
│   └── llama_launcher.py   # standalone 本地 LLM 热载入（替代 Ikaros 看门狗）
├── models/                 # 模型目录（.gguf 不入库）
│   ├── model_config.py     # 本地 LLM 加载配置
│   ├── download_models.py  # 从 HuggingFace 下载 GGUF
│   └── serve_embeddings.py # 自托管 nomic embedding 服务 (:8587)
├── hermes-plugin/
│   └── ikaros_v5/__init__.py  # Hermes Agent MemoryProvider（自包含部署）
├── install.py / install.sh    # 一键安装脚本
├── requirements.txt
├── pyproject.toml
├── hermes-mcp-v5.json         # 安装后生成的 MCP 配置片段
└── README.md
```

运行时数据在 `<repo>/data/v5/`（SQLite + Chroma + 各类 JSON 状态），**已 gitignore，不入库**。

---

## 三、快速开始

需要 Python ≥ 3.12。

```bash
# 1) 创建 venv 并安装依赖 + 生成 MCP 配置片段
python install.py

# 2)（可选）下载模型：聊天模型 + embedding 模型
python install.py --download-models

# 3)（可选）自托管 embedding 服务（语义检索需要）
python models/serve_embeddings.py
```

装好后把生成的 `hermes-mcp-v5.json` 里的 `mcp_servers.ikaros-v5` 段粘到你的 Hermes Agent
配置文件（`mcp_servers` 节点）下，重启 Hermes 即可。

---

## 四、两种集成方式

### A. MCP 服务器（主动调用记忆工具）

`hermes-mcp-v5.json` 形如：

```json
{
  "mcp_servers": {
    "ikaros-v5": {
      "command": "<repo>/.venv/Scripts/python.exe",
      "args": ["<repo>/v5/mcp_server.py"],
      "env": {
        "V5_MEMORY_ROOT": "<repo>",
        "PYTHONPATH": "<repo>",
        "IKAROS_EMBED_URL": "http://127.0.0.1:8587/embedding"
      }
    }
  }
}
```

Agent 于是能调用 `v5_memory_search` / `v5_memory_store` / `v5_self_model` / `v5_emotion_status` 等工具。

### B. Hermes Provider 插件（记忆生命周期闭环）

```bash
python install.py --hermes-agent "C:/path/to/hermes-agent"
```

会把自包含插件（`__init__.py` + 打包的 `v5/`）放进
`<hermes-agent>/plugins/memory/ikaros_v5/`，Hermes 自动发现为 `ikaros-v5` provider：
压缩前注入记忆、每轮写回、会话结束 consolidate。

两种方式互补：MCP 负责"按需主动调工具"，Plugin 负责"自动闭环"。通常两者都开。

---

## 五、环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IKAROS_EMBED_URL` | `http://127.0.0.1:8587/embedding` | 语义检索的 embedding 端点 |
| `V5_MEMORY_ROOT` | 自动推断 | 指向仓库根（含 `v5/` 包），用于 Plugin / 路径解析 |
| `IKAROS_LLAMA_SERVER` | PATH 查找 | `llama-server` 二进制路径（本地 LLM 懒加载） |
| `DEEPSEEK_API_KEY` | 空 | 云端认知任务（后台整合）密钥；缺省时整合优雅降级 |
| `HERMES_HOME` | 空 | Hermes Agent 家目录（加载其 `.env` 取密钥） |
| `IKAROS_ROOT` | 空 | 向后兼容：若指向 Ikaros 主工程，可启用 Ikaros 专属子特性 |

模型选择：`models/model_config.json`（`model_config.py` 首跑自动生成，手动改 `initial_model` 切换聊天模型）。

---

## 六、注意事项

- **模型文件不入库**：`.gguf` 体积数 GB，请用 `models/download_models.py` 下载或自行放入。
- **embedding 端点是语义检索的前提**：不配置 `IKAROS_EMBED_URL` 时，向量召回会降级（仅关键词/FTS5 命中），不影响基础记忆读写。
- **本地 LLM 是可选的**：仅当使用 `provider="local"` 路径时才需要 `llama-server` + 聊天模型；默认认知/整合走云端（设 `DEEPSEEK_API_KEY`）。
- **纯 Python 标准库存储**：SQLite + Chroma 都在本地，数据归用户所有。

---

## 七、与原 Ikaros 工程的关系

本仓库是从 Ikaros 主工程 `core/v5/` 抽离出的**干净副本**，所有 `E:\Ikaros` / `core/v5` /
`hermes-agent` 等硬编码路径已改为相对路径或环境变量驱动，可独立部署在任意机器。
Ikaros 主工程本身的 `core/v5/` 不受影响。
