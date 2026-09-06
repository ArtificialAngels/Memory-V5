# Memory V5

> **Ikaros 本地记忆系统** — SQLite (FTS5) + Chroma 向量存储，三路融合召回，全链路本地运行。为 AI Agent 提供长期记忆、情感建模、认知锚点和反思能力。

[![Python](https://img.shields.io/badge/python-3.12.10-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 🎯 目标

构建一个**完全本地化**的长期记忆系统，让 AI Agent 具备：
- 跨会话的记忆保持与召回
- 情感状态建模与关系推进
- 自我认知与反思成长
- 实体关系图谱与知识关联

所有记忆数据存储在本地，不上传任何云端服务。

## 💡 理念

- **全本地** — 记忆写入、召回、embedding 全部本地运行，隐私优先
- **三路融合** — 关键词 (FTS5) + 向量语义 (Chroma) + 时间范围，`min_fused_score` 融合评分
- **主动记忆** — 不只是被动存储，通过标准记忆循环主动沉淀、召回、反思
- **情感驱动** — PAD 情感模型驱动记忆重要性和关系推进
- **认知锚点** — 5D 认知注入（时间/设备/地理/情绪/上下文）让记忆有情境

## 🏗️ 框架

```
                    ┌── 标准记忆循环 (loop.py) ──┐
                    │  pre-step 召回 → post 沉淀 → 维护反思 │
                    └───────────────┬───────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
   │   存储层     │          │   召回层     │          │   认知层     │
   │ store.py    │          │ memory_     │          │ cogno_5d    │
   │ memory_api  │          │ retrieval   │          │ self_model  │
   │ file_store  │          │ search.py   │          │ context_    │
   │             │          │ recall_     │          │ anchor      │
   │ SQLite+FTS5 │          │ budget/ledger│         │ metacog     │
   │ Chroma 向量 │          │             │          │ reflections │
   └─────────────┘          └─────────────┘          └─────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
   │  情感关系层  │          │  实体图谱层  │          │  叙事摘要层  │
   │ emotional_  │          │ entity_     │          │ narrative   │
   │ memory      │          │ graph       │          │ summary     │
   │ affect      │          │ graph_rank  │          │ conversation│
   │ relationship│          │ project_    │          │ _tree       │
   │ care        │          │ edges       │          │             │
   │ vitality    │          │             │          │             │
   └─────────────┘          └─────────────┘          └─────────────┘
```

| 层 | 核心模块 | 说明 |
|---|---|---|
| 存储 | `store.py` `memory_api.py` `file_store.py` | SQLite + FTS5 + Chroma，统一写入/读取 |
| 召回 | `memory_retrieval.py` `search.py` `recall_budget.py` | 三路融合召回，预算控制，召回账本 |
| 循环 | `loop.py` | 标准记忆循环：pre-step 召回 / post 沉淀 / 6h 维护反思 |
| 重要性 | `importance.py` | 统一记忆重要性评分，驱动保留/遗忘 |
| 反思 | `reflections.py` `metacog.py` | 记忆归约、元认知、自我反思 |
| 情感 | `emotional_memory.py` `affect.py` `vitality.py` | PAD 情感模型，精力/关系推进 |
| 关系 | `relationship.py` `care.py` `dissonance.py` | 人际关系建模，关怀，认知失调 |
| 图谱 | `entity_graph.py` `graph_rank.py` | 实体关系图谱，图排序 |
| 认知 | `cogno_5d.py` `self_model.py` `context_anchor.py` | 5D 认知锚点，自我模型，情境锚定 |
| 叙事 | `narrative.py` `summary.py` | 记忆叙事化，会话摘要 |
| MCP | `mcp_server.py` | 49 个 v5_* 工具，通过 MCP 协议暴露给外部 Agent |

## ⚠️ 项目状态

**这是一个半成品。** 核心链路可运行，但存在大量未优化的逻辑：

- 召回融合策略（关键词/向量/时间）的权重和阈值尚在调优，召回质量不稳定
- 50+ 模块之间存在功能重叠和耦合，缺少清晰的分层边界
- 情感模型 (PAD) 和关系推进的逻辑较为粗糙，缺少验证
- 反思和归约依赖云端 LLM，本地无 LLM 时降级不完整
- 测试覆盖率低，多数模块只有 smoke test
- 性能未优化，大规模记忆下召回延迟可能较高
- 配置分散在多个 yaml/py 文件，缺少统一的配置管理

欢迎基于此框架继续开发，但请勿将当前状态视为生产可用。

## 🔌 dsh 插件

Memory V5 以 dsh 插件形式集成到 DeepSeek Harness，提供自动记忆工程层 + embedding 模型管理：

```bash
dsh plugin --profile web add github:ArtificialAngels/Ikaros#main:core/ikaros-dsh/plugins/ikaros-memory
```

插件能力：
- **自动记忆循环** — pre-step 召回注入 / turn-stopping 沉淀写回 / compaction 捕获 / 6h 维护
- **Embedding 管理** — 本地 bge-m3 模型启动/切换/下载/重建向量（HTTP RPC :19001）
- **dsh 设置面板** — embedding 模型管理 UI
- **MCP 工具** — 49 个 v5_* 记忆操作工具

## 🚀 快速开始

```bash
# 克隆
git clone https://github.com/ArtificialAngels/Memory-V5.git
cd Memory-V5

# 安装依赖 (便携 Python 3.12)
pip install -r requirements.txt

# 启动 MCP 服务器 (供 dsh 等外部 Agent 调用)
python mcp_server.py

# 或直接使用 Python API
python -c "from memory_api import MemoryAPI; api = MemoryAPI(); api.store('hello', scope='test')"
```

## 📁 目录

```
memory_v5/
├── store.py              # 核心存储 (SQLite + Chroma)
├── memory_api.py         # 统一 API 入口
├── memory_retrieval.py   # 三路融合召回
├── loop.py               # 标准记忆循环引擎
├── mcp_server.py         # MCP 服务器 (49 工具)
├── importance.py         # 记忆重要性评分
├── reflections.py        # 记忆归约与反思
├── entity_graph.py       # 实体关系图谱
├── emotional_memory.py   # PAD 情感记忆
├── cogno_5d.py           # 5D 认知锚点
├── self_model.py         # 自我模型
├── relationship.py       # 人际关系
├── narrative.py          # 记忆叙事化
├── search.py             # 搜索
├── data/                 # 运行时数据 (v5.db + Chroma)
├── models/               # 本地模型权重 (bge-m3 等)
├── extensions/           # 扩展模块
├── reflect/              # 反思相关
├── scripts/              # 工具脚本
├── services/             # 服务层
├── tests/                # 测试
└── tools/                # 工具层
```

## 📜 协议

MIT。
