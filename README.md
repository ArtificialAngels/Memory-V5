# ikaros-memory-v5

**面向 Agent 的长期记忆系统** —— 通过 MCP 协议为任意兼容 MCP 的 Agent 提供记忆存储、语义检索、情绪建模、自我反思、关系关怀、项目与技能沉淀等能力。

对外唯一接口：`v5/mcp_server.py`（FastMCP，48 个 `v5_*` 工具）。也提供 Hermes Agent 的 `MemoryProvider` 插件，实现记忆生命周期自动闭环。

依赖极小（Python 包）：仅 `chromadb` + `httpx` + `mcp` + `python-dotenv` + `psutil`。
运行时二进制依赖：语义检索（embedding）与可选本地聊天 LLM 依赖 `llama.cpp` 的 `llama-server`；纯云端 + 英文/数字关键词模式可跳过，详见「七、启用 embedding 语义检索」。

---

## 一、核心能力

| 能力 | 说明 |
|------|------|
| **记忆存储与双路融合检索** | `memory_store` 写入带 PAD 情绪指纹的记忆；`memory_search` 融合 SQLite FTS5 关键词（0.3）+ ChromaDB 向量语义（0.7）双路召回，既抓"字面命中"也抓"意思相近"。 |
| **结构化内容守卫** | `validation.py` 拦截 LLM 旁白、裸 JSON、markdown fence、超长 dump 写入记忆，记忆库不被垃圾污染。 |
| **实体图谱扩散激活** | `entity_graph.py` 用图结构做相关记忆扩散，一次查询带出一整片关联上下文，而非孤立碎片。 |
| **情感模型（PAD）** | Pleasure-Arousal-Dominance 三维 + 信任/亲密等关系量，情绪与关系有状态、可衰减、可注入 prompt。 |
| **自我模型与后台反思** | `self_model` 持久"她是谁"；`reflect/` 流水线把对话提炼为事实/情感/自我更新，`memory_promote`/`temporal_extract` 让短期对话沉淀为长期事实。 |
| **时序矛盾消解（temporal supersede）** | 新事实与旧记忆冲突时，旧事实按时间线 `valid_to` 失效、权重降级，记忆库随对话演化不堆积矛盾。 |
| **统一检索路由** | `memory_retrieval.unified_retrieve` 一路入口覆盖 `auto/semantic/lexical/graph/tree/temporal`；语义不足自动补图扩散，排序带频率/反馈/新鲜度加权。 |
| **对话树** | 树形对话结构（fork / merge / conclude / abandon），主线/分支脉络可检索，长对话有组织地生长。 |
| **关怀与精力** | `care_*` 监测是否需要主动关怀（休息/喝水/睡眠）；`vitality_*` 建模精力消耗与恢复。 |
| **关系建模** | `relationship_*` 记录每次互动强度，维护与用户的亲密度。 |
| **防重复** | `anti_repeat_*` 把已发回应记入语料，对候选回应做重复风险检测与惩罚提示。 |
| **用户指令** | `directive_*` 增/列/停用用户指令（禁谈话题、偏好等）。 |
| **项目轨** | `project_*` 按 decision/pitfall/convention/idea 沉淀并检索项目决策与教训，跨会话连续。 |
| **技能轨** | `skill_*` 以 Markdown 文件沉淀可复用工作流，支持渐进式检索（窄命中→取全文）。 |
| **活动感知** | `activity_status` 实时感知前台窗口与 LLM 推理状态；`context_compression_stats` 供面板查看上下文压缩状态。 |
| **叙事生成** | `narrative_generate` 从近期记忆生成月度自述。 |

---

## 二、MCP 工具清单（48 个 `v5_*` 工具）

按功能分组，Agent 调用对应工具名即可。

### 记忆读写
| 工具 | 功能 |
|------|------|
| `v5_memory_store` | 存一条记忆（事实/偏好/教训/情感事件，带 PAD 情绪指纹） |
| `v5_memory_search` | 检索长期记忆（FTS5 关键词 + 向量语义双路融合） |
| `v5_memory_get` | 按 ID 取单条记忆 |
| `v5_memory_delete` | 按 ID 删除记忆 |
| `v5_memory_stats` | 存储统计（总数/长期数/均权重/类型分布） |

### 情绪与自我
| 工具 | 功能 |
|------|------|
| `v5_analyze_emotion` | 从文本更新 PAD 情绪状态 |
| `v5_emotion_status` | 返回当前 PAD 情绪状态 |
| `v5_emotion_label` | 返回 1–2 个情绪标签 |
| `v5_self_model` | 返回持久自我模型（"她是谁"） |
| `v5_self_reflect` | 跑一个 metacog 循环 |
| `v5_latest_thought` | 返回最近一次内心独白 |
| `v5_curiosity_check` | 返回好奇驱动状态 |
| `v5_subconscious` | 返回最近潜意识低语 |

### 关怀 / 精力 / 关系
| 工具 | 功能 |
|------|------|
| `v5_care_check` | 检查是否需要主动关怀（休息/喝水/睡眠） |
| `v5_care_status` | 关怀监控累计活动计数 |
| `v5_vitality` | 返回当前精力状态 |
| `v5_vitality_tick` | 推进精力模型一步 |
| `v5_relationship` | 返回与用户的亲密度 |
| `v5_relationship_tick` | 记录一次互动（强度 0–1）并返回更新后状态 |

### 叙事 / 矛盾 / 主动 / 反思
| 工具 | 功能 |
|------|------|
| `v5_narrative_generate` | 从近期记忆生成月度自述 |
| `v5_dissonance_check` | 检测内容是否与已有记忆矛盾 |
| `v5_proactive_check` | 决策此刻是否主动开口 |
| `v5_self_discover` | 跑一次自我架构发现（读项目文件） |
| `v5_reflect_run_op` | 跑一个或多个注册反思 op |
| `v5_reflection_synthesize` | 从事实合成一条反思 |
| `v5_reflection_read` | 从库读取反思 |
| `v5_reflection_apply_evidence` | 对反思施加证据信号（强化/反驳） |
| `v5_reflection_promote` | 把反思提升为人格（self_model） |
| `v5_reflection_stats` | 反思统计 |

### 防重复
| 工具 | 功能 |
|------|------|
| `v5_anti_repeat_record` | 把回应记入防重复语料 |
| `v5_anti_repeat_check` | 检测候选回应重复风险 |
| `v5_anti_repeat_penalty` | 风险高时返回惩罚提示 |
| `v5_anti_repeat_clear` | 清空防重复语料 |
| `v5_anti_repeat_stats` | 防重复语料统计 |

### 用户指令
| 工具 | 功能 |
|------|------|
| `v5_directive_add` | 新增用户指令（禁谈话题/偏好等） |
| `v5_directive_list` | 列出生效中的指令 |
| `v5_directive_deactivate` | 按 ID 停用指令 |
| `v5_directive_stats` | 指令统计 |

### 活动感知
| 工具 | 功能 |
|------|------|
| `v5_activity_status` | 实时活动感知（前台窗口 + LLM 推理） |
| `v5_context_compression_stats` | 上下文压缩引擎状态（供面板） |

### 项目轨
| 工具 | 功能 |
|------|------|
| `v5_project_note` | 存一条项目记忆（decision/pitfall/convention/idea） |
| `v5_project_retrieve` | 检索项目记忆（决策/坑/约定/想法） |
| `v5_project_stats` | 按类型汇总项目记忆数量 |

### 技能轨
| 工具 | 功能 |
|------|------|
| `v5_skill_write` | 新建/更新可复用技能（Markdown，kebab-case 命名） |
| `v5_skill_list` | 列出全部技能（名称/描述/路径，不含正文） |
| `v5_skill_get` | 按名读取技能全文（渐进检索的"宽"层） |
| `v5_skill_search` | 搜索技能（仅窄命中：名称/描述/路径/分数） |
| `v5_skill_remove` | 按名删除技能（幂等） |

---

## 三、项目结构

```
v5-memory/
├── v5/                     # V5 记忆引擎包（核心，可移植）
│   ├── mcp_server.py       # MCP 服务器入口（48 个 v5_* 工具）
│   ├── cli.py              # 控制台入口 ikaros-mem-v5
│   ├── store.py            # SQLite FTS5 存储
│   ├── search.py           # 三路融合检索 + 向量索引
│   ├── memory_retrieval.py # 统一检索路由（auto/semantic/lexical/graph/tree/temporal）
│   ├── validation.py       # 结构化内容守卫
│   ├── conversation_tree.py # 树形对话结构（fork/merge/conclude/abandon）
│   ├── affect.py / self_model.py / profile.py / relationship.py ...
│   ├── context_anchor.py / graph_rank.py / importance.py / project_edges.py / skill_store.py / lifecycle.py  # 引擎依赖
│   ├── extensions/         # tree_adapter / temporal_graph / ontology_align（图谱树适配 / 时序矛盾消解 / 本体对齐）
│   ├── reflect/            # 后台整合流水线（consolidate/distill + memory_promote/temporal_extract）
│   ├── tools/              # MCP 工具实现
│   └── llama_launcher.py   # 本地 LLM 热载入
├── models/                 # 模型目录（.gguf 不入库）
│   ├── model_config.py     # 本地 LLM 加载配置
│   ├── download_models.py  # 从 HuggingFace 下载 GGUF
│   └── serve_embeddings.py # 自托管 embedding 服务 (:8587)
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

## 四、快速开始

需要 Python ≥ 3.12。

> **⚠️ 前置：安装 llama.cpp（启用语义检索必需）**
> 中文（含向量）语义检索依赖 embedding 端点，由 `llama.cpp` 的 `llama-server` 二进制提供。
> 请先安装 llama.cpp（见「七、启用 embedding 语义检索」），将其 `llama-server` 放入 `PATH`，
> 或设置 `IKAROS_LLAMA_SERVER` 指向它。仅用纯云端 + 英文/数字关键词检索可跳过此步。

```bash
# 1) 创建 venv、安装依赖、生成 MCP 配置片段
python install.py

# 2)（可选）下载模型：聊天模型 + embedding 模型
python install.py --download-models

# 3)（可选）自托管 embedding 服务（语义检索需要）
python models/serve_embeddings.py
```

装好后把生成的 `hermes-mcp-v5.json` 里的 `mcp_servers.ikaros-v5` 段粘到 Agent 配置文件
（`mcp_servers` 节点）下，重启 Agent 即可调用 `v5_*` 工具。

---

## 五、两种集成方式

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

## 六、环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IKAROS_EMBED_URL` | `http://127.0.0.1:8587/embedding` | 语义检索的 embedding 端点 |
| `V5_MEMORY_ROOT` | 自动推断 | 指向仓库根（含 `v5/` 包），用于 Plugin / 路径解析 |
| `IKAROS_LLAMA_SERVER` | PATH 查找 | `llama-server` 二进制路径（本地 LLM 懒加载） |
| `DEEPSEEK_API_KEY` | 空 | 云端认知任务（后台整合）密钥；缺省时整合优雅降级 |
| `HERMES_HOME` | 空 | Hermes Agent 家目录（加载其 `.env` 取密钥） |
| `V5_MCP_TOOL_GROUPS` | 空（全量） | 按组过滤注册的 `v5_*` 工具，如 `memory,self,emotion` |

模型选择：`models/model_config.json`（`model_config.py` 首跑自动生成，手动改 `initial_model` 切换聊天模型）。

---

## 七、启用 embedding 语义检索（安装 llama.cpp）

V5 的语义检索（向量召回）需要一个 `/embedding` 端点。本仓库自带自托管方案，依赖 **llama.cpp** 的 `llama-server` 二进制。其他用户部署时请先安装它：

1. **下载 llama.cpp 预编译二进制**
   - Windows：到 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载含 `llama-server` 的包（如 `llama-server-bin-win-cuda-cu12.x` 带 GPU 加速，或 CPU 版），解压得到 `llama-server.exe`。
   - Linux / macOS：`brew install llama.cpp`，或下载 release 中的 `llama-server`，或 `cmake` 自行编译。
2. **放入 PATH 或设环境变量**
   - 把 `llama-server`（或 `llama-server.exe`）所在目录加入系统 `PATH`；或
   - 设置 `IKAROS_LLAMA_SERVER` 指向其完整路径：
     - Windows：`set IKAROS_LLAMA_SERVER=C:\tools\llama.cpp\llama-server.exe`
     - Linux/macOS：`export IKAROS_LLAMA_SERVER=/usr/local/bin/llama-server`
3. **下载 embedding 模型并启动服务**
   ```bash
   python models/download_models.py --only embed   # 取得 embedding 模型
   python models/serve_embeddings.py                # 监听 127.0.0.1:8587/embedding
   ```
4. **让 V5 指向该端点**（`install.py` 生成的配置已默认指向 `http://127.0.0.1:8587/embedding`）：
   设置 `IKAROS_EMBED_URL=http://127.0.0.1:8587/embedding`。

> 同一份 `llama-server` 也用于**可选**的本地聊天 LLM（懒加载）。纯云端部署（认知/整合走 DeepSeek）可只装 embedding 用的 `llama-server`，不启本地聊天模型。

---

## 八、注意事项

- **模型文件不入库**：`.gguf` 体积数 GB，请用 `models/download_models.py` 下载或自行放入。
- **embedding 端点是语义检索的前提**：不配置 `IKAROS_EMBED_URL` 时，向量召回降级（仅关键词/FTS5 命中），不影响基础记忆读写。
- **中文语义检索建议开启 embedding 服务**：FTS5 默认使用 `unicode61` 分词器，对中文按整段而非分词处理，纯关键词路径对中文召回较弱；`models/serve_embeddings.py`（`:8587`）提供向量语义召回，能正确匹配"意思相近"的中文记忆。英文 / 数字（如 `Rust`）关键词检索不受影响。
- **本地 LLM 是可选的**：仅当使用 `provider="local"` 路径时才需要 `llama-server` + 聊天模型；默认认知/整合走云端（设 `DEEPSEEK_API_KEY`）。
- **纯 Python 标准库存储**：SQLite + Chroma 都在本地，数据归用户所有。
