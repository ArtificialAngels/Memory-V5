# V5 记忆/上下文增强骨架 (EXPERIMENTAL)

本目录是上一轮 GitHub 调研（Hermes 历史回放 / TencentDB / LLMLingua / Graphiti
对比）落地的**实现骨架**，目标是补齐 V5 相对这几个方案的三个硬缺口：

| 缺口 | 对应方案 | 骨架文件 |
|------|----------|----------|
| token 级压缩（委派 llmlingua 现成库 + 离线规则回退） | LLMLingua | `token_compressor.py` |
| 分层检索门控（默认注高层、按需下钻） | TencentDB L0-L3 | `gated_retrieval.py` |
| 时序图谱 + 矛盾即更替 | Graphiti | `temporal_graph.py` |
| **树模式定向适配（chat tree 适配记忆/压缩）** | —（Ikaros 自有） | `tree_adapter.py` |

**状态**：`token_compressor` **已接入主链路并集成测试通过**（2026-07-30，guard + 异常回退，
替换 hermes 插件 `on_pre_compress` 的 `text[:150]` 硬截断）；`gated_retrieval` / `temporal_graph`
/`tree_adapter` 仍为骨架，未在主链路启用。`tree_adapter` 已有 8 项模块测试通过
（`tests/test_tree_adapter_module.py`）；其 **CORE_HOOKS（存储打标 + memory_ids 持久化）已于
2026-07-31 落地**（`tests/test_tree_core_hooks.py` 4 项通过），使 A/B 真正生效、修掉
`_node_memories` 重启丢失 bug。所有模块均 `py_compile` 通过。

---

## 1. token_compressor.py —— token 级压缩预通道

**解决的问题**：`preprocess_config.yaml:58` 的 `token_budget`(min800/max1200/char_x)
至今**未被任何代码消费**（grep 确认 `memory_retrieval.py` 不引用）。V5 缩减上下文
只靠 LLM 摘要 + 硬截断，缺 LLMLingua 式 token 级压缩。

**提供**：
- `est_tokens(text)` —— 复用 `char_x` 的粗略 token 估算（中文~1 token/字）。
- `rule_compress(text, ratio)` —— 确定性零 LLM 瘦身（折叠空白/重复标点、删重复行、
  删语气 filler、超目标则保头尾截中段，缓解 lost-in-the-middle）。**作为 llmlingua
  不可用时的离线回退**。
- `llmlingua_compress(text, target_token, rate)` —— **委派微软现成库 `llmlingua`
  (`PromptCompressor.compress_prompt`)** 做真实 token 级压缩（README 实测 11x+）。
  导入守护：`try: from llmlingua import PromptCompressor`，未装/离线/模型下载失败
  都返回 None 触发回退（U 盘便携环境不强制装，避免首次运行拉 HF 模型破坏离线性）。
- `compress_text(text, quality="auto")` —— 统一入口：`auto`=llmlingua→规则
  （零 LLM、离线安全），`llm`=本地 :8080，`rule`=仅规则。
- `llm_compress(text, quality="auto")` —— 高质量压缩链：llmlingua→本地 :8080→规则。
- `compress_old_rounds` / `compress_retrieval_block` —— 已改调 `compress_text`，
  环境装了 llmlingua 自动走现成库，否则规则回退（**不改动 V5 主链路**）。
- `enforce_budget(texts/blocks, budget_tokens)` —— 按 score/顺序截到预算内。

**启用 llmlingua（可选）**：在 Ikaros venv 里 `pip install llmlingua`，首次调用会
自动下载 HF 模型（需联网一次）；之后离线也能用缓存。不装则全程规则回退，功能不降级。

**接入点（已落地）**：
- hermes 插件 `on_pre_compress`（`data/hermes-agent/plugins/ikaros_v5/memory_provider.py`）在
  `self._v5_search(...)` 返回后调用 `compress_retrieval_block(max_chars_per_item=150)`，
  **已替换**原 `text[:150]` 硬截断；整段 `try/except` 包裹，压缩器异常时自动回退原始硬截断。
- :8080 本地小模型构建 system/记忆前缀时跑 `enforce_budget`（尚未接线，待评估）。

**未做**：真实中文分词级压缩、与 Hermes `ContextCompressor` 头尾保护协同的全局预算分配、
`llm_compress` 的旁白守卫（本地 Qwen3-1.7B 不听指令，需复用 V5 已有的
`is_clean_structured_content` 守卫）。

---

## 2. gated_retrieval.py —— 分层检索门控

**解决的问题**：`retrieve()` 三路融合后直接相似注入 top_k=5，缺 TencentDB 式
"默认注高层、按需下钻低层"的分层门控，默认 token 效率偏低、易塞无关旧记忆。

**提供**：
- `gated_retrieve(query, ...)` —— 永远注入高层（self_model 提示 + 最近
  distill/reflect/identity/axiom 记忆 top 3），仅在 query 实质化（`_is_substantive`
  过滤寒暄/短句）且高层预算有余量时才下钻 `retrieve()` 拉低层事实，用 `drill_min_fused`
  门控，命中不足则跳过。

**接入点**：替换 hermes 插件 `on_pre_compress` 里的 `self._v5_search(query_text, top_k=5)`
调用；或直接作为 memory-context 组装入口，`high_layer` 文本 + `low_memories` 一起拼进
`<memory-context>`。

**未做**：高层记忆也应有 token 上限与去重（当前 `_high_layer_memories` 取固定 top 3，
没算进 `low_budget_tokens` 的总预算）；distill/reflect 类型与 self_model 可能重复，
需合并策略；下钻触发阈值（`_is_substantive` 的 len<4 / 寒暄词表）需按真实语料校准。

---

## 3. temporal_graph.py —— 时序图谱 + dissonance supersede

**解决的问题**（上一轮确认）：
- V5 缺 fact 级时效窗口；`eg_activations.expires_at` 只记"最近访问"，检索
  `spreading_activation_search` 用 `ORDER BY importance DESC, created_at DESC` 没用时序过滤。
- `dissonance._record_dissonance` 只记事件、**不废旧事实** → "用户住 X"变"住 Y"后
  两条共存，旧值可能被捞回（Graphiti 用 valid_to 解决，LongMemEval 领先 Mem0 主因）。

**提供**：
- `apply_migration()` —— 给 `memory` + `eg_entities/eg_edges/eg_episodic` 加
  `valid_from/valid_to`（幂等 ALTER）。
- `supersede_memory(old_id, now)` —— 把旧记忆 `valid_to=now` 作废（仅当仍生效）。
- `supersede_entity_attribute(entity_id, now)` —— Graphiti 式粗粒度失效（**注意**：
  V5 的 `eg_edges` 无 relation_type，无法精确区分被推翻的属性，生产化需先加列 + 让
  consolidate 填关系类型。TODO）。
- `resolve_dissonance_supersede(new_content, conflicts, now)` —— 在
  `dissonance._record_dissonance` 之后调用，把每条冲突旧事实作废。
- `retrieve_temporal(...)` / `filter_expired_episodic(...)` —— 时效感知检索（过滤过期）。
- 文件末尾给出**直接改 SQL** 的推荐补丁（entity_graph.py:445 加
  `AND (em.valid_to IS NULL OR em.valid_to > ?)` + 排序让生效事实沉底）。

**接入点**：
- 启动时（V5 init / setup-native）跑一次 `apply_migration()`。
- `dissonance.py:91` 的 `if conflicts: _record_dissonance(...)` 之后加
  `resolve_dissonance_supersede(content, conflicts)`。
- 替换/包裹 `memory_retrieval.retrieve` 为 `retrieve_temporal`；
  `entity_graph.spreading_activation_search` 结果过 `filter_expired_episodic`
  （或直接应用 SQL 补丁，更高效）。

**未做**：`eg_edges` 关系类型建模（让 supersede 精确到"某属性"而非整实体出边）；
矛盾检测的误判兜底（supersede 是破坏性操作，建议先灰度：记 supersede 候选、人工/二次
NLI 确认后再写 valid_to）；失效事实的"历史仍可查"视图（审计/反思可能需要看旧值）。

---

## 4. tree_adapter.py —— 树模式（chat tree）定向适配层

**解决的问题**：Ikaros 对话是**树**而非线性流，但 V5 记忆/压缩按"线性 transcript"假设设计。
`tree_adapter` 补足四个具体缺口（均非侵入：包裹/新增，便于回滚）：

| 缺口 | 适配 | API |
|------|------|-----|
| 存储不标节点/分支 → 全局检索无法树域过滤 | 存储打标 | `tag_for_node(node_id, branch_label)` |
| 全局 V5 检索不感知树（hermes `on_pre_compress` 裸 `_v5_search`） | 树域检索重排+过滤 | `tree_scoped_retrieve(tree, node_id, query, top_k)` |
| 压缩是线性的（把树当扁平列表压中段） | 按节点边界压缩 | `TreePathCompressor.compress(meta)` |
| `build_context_v2` 盲截 `messages[-50:]` 不保结构 | 树感知上下文组装 | `build_tree_aware_context(tree, node_id)` |

**`TreePathCompressor` 的核心思想**：压缩单元是**节点**而非消息序号。head(根+早期)/
tail(近期) 完整保留，middle(远端祖先) 用节点自带 `summary` 替换（树拓扑即压缩结构），
fork/merge 锚点（`branch_label` 非空）始终可见。这天然具备线性 `ContextCompressor`
需特判的"last-user/last-assistant 锚点"与"lost-in-middle"防护，且**绝不跨节点切分
一个 turn**。

**A/B 钩子已落地（2026-07-31，见 `conversation_tree.py` CORE_HOOKS）**：
1. ✅ `add_turn` / `fork_branch` / `init` 三处 `_store_fn(..., tags=...)` 改为在生成
   node id **之后**追加 `_tree_tag(node_id, branch_label)`（= `node:<id>` / `branch:<label>`），
   对话内容记忆带树域标签。`_tree_tag` 惰性导入 `tag_for_node`，不可用时内联构造相同格式，
   零硬依赖。
2. ✅ `MemoryRetriever.add_memory` 同样给 fact 打 `_tree_tag`；并给 `ConvNode` 新增
   `memory_ids: List[int]` 字段（to_dict/from_dict 同步），add_memory 时**双写**内存
   `_node_memories` + 持久化 `node.memory_ids` 并 `persist()`，**修掉 `_node_memories`
   重启丢失的已知 bug**。`retrieve()` 路径分支现在**优先读持久化 `memory_ids`**，回退内存映射。
   验证：`tests/test_tree_core_hooks.py`（4 项全过：标签打标、id 契约、持久化、重启存活+检索）。

**接入点（已落地，2026-07-31）**：
- ✅ `build_tree_aware_context` 已接入 `core/conversation-tree/server.py` 的 `/api/chat`
  （替换原 `build_context_v2` 的线性 `[-50:]` 截断），获得树感知压缩 + L1 兄弟 + L2 合并结论。
  新增 `system_prompt` / `extra_memory` 两参数，分别注入 Ikaros 人格与 V5 记忆块。
- ✅ `tree_scoped_retrieve` 已接入 `/api/chat` 的 `build_v5_memory_block`（按用户消息语义检索 +
  树域加权，依赖 CORE_HOOKS 打的 `node:/branch:` 标签做路径/分支优先级）。这是 chat 真正把
  V5 当作**记忆引擎**的入口。
- ✅ Ikaros 人格注入：`build_ikaros_persona()` 组装 axiom.md + SOUL.md(标题白名单抽取) +
  self_model.json 动态心绪，替换原写死的 "Explore" system prompt。全部 fail-open。
- 验证：`tests/test_chat_v5_integration.py`（4 项：人格/记忆块 fail-open/完整组装/回退）。
- ⚠️ **尚未做**：`/api/full_context` 调试端点仍走旧 `build_context_v2`（未升级，仅检视用）；
  `tree_scoped_retrieve` 喂 hermes 插件 `on_pre_compress` 的选项仍待定（chat tree 当前不经 hermes 插件）。

---

## 通用风险
- 模块都 import `v5.*`，需在 `sys.path` 含 `E:/Ikaros/core` 的环境下运行
  （V5 主进程已满足）。`tree_adapter` 的 `tree_scoped_retrieve` 与 `build_tree_aware_context`
  通过 duck-typed `tree`（`get_path`/`get_node`/`get_sibling_nodes`/`get_context_with_meta`）
  解耦，测试用 FakeTree 即可离线跑。
- 都是**增量增强**，不修改现有 `retrieve` / `spreading_activation_search` /
  `dissonance` / `build_context_v2` 函数体，通过"包裹/新增调用"接入，便于回滚。
- 上线前务必在沙箱跑一遍：树域检索不误排相关全局记忆、树压缩不丢 fork 锚点、
  持久化 node→memory 绑定后重启路径检索不空。
