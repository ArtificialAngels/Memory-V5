# 详细说明见 docs/scripts/core/v5/v5/memory_retrieval.md
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("ikaros.v5.memory_retrieval")

# 检索结果短 TTL 缓存 (哥哥优化项): 同 query(含 top_k/time_range/exclude) 20s 内直接返回,
# 跳过 embedding + chroma 全程. 聊天里"继续/好的/然后呢"等高频短句命中率高, 体感明显.
_RET_CACHE: dict = {}
_RET_CACHE_LOCK = threading.Lock()


def _cache_cfg() -> dict:
    try:
        from v5 import preprocess_config as pc
        return pc.cfg().get("cache", {})
    except Exception:
        return {}


def _retrieve_ttl() -> float:
    try:
        c = _cache_cfg()
        if not c.get("retrieve_ttl_enabled", True):
            return 0.0
        return float(c.get("retrieve_ttl_seconds", 20))
    except Exception:
        return 20.0


def _defaults() -> dict:
    return {
        "vector_weight": 0.7, "fts_weight": 0.3,
        "time_decay_per_day": 0.05, "min_fused_score": 0.6, "top_k": 5,
        "type_boost": {"emotion": 1.2, "fact": 1.1, "conversation": 0.8, "default": 1.0},
        # Phase 4 全套加权默认值 (与 preprocess_config.yaml 一致; 无配置时兜底)
        "base_weight_factor": 0.5,
        "merge_reinforce_increment": 0.05,
        "type_decay": {
            "conversation": {"per_day": 0.05, "floor": 0.2},
            "fact": {"per_day": 0.03, "floor": 0.4},
            "user_trait": {"per_day": 0.01, "floor": 0.6},
            "identity": {"per_day": 0.005, "floor": 0.7},
            "preference": {"per_day": 0.02, "floor": 0.5},
            "decision": {"per_day": 0.005, "floor": 0.7},
            "lesson": {"per_day": 0.01, "floor": 0.6},
            "default": {"per_day": 0.05, "floor": 0.2},
        },
        "situational": {
            "enabled": True,
            "project_activity_boost": 0.10,
            "hour_match_boost": 0.05,
        },
    }


# ─── 意图检测 (mnemon 借鉴: WHY/WHEN/ENTITY/GENERAL) ──────────────
# 查询意图决定检索时哪类记忆加权更重 (问 why → 决策/教训; 问 when → 时间;
# 问 X → 实体图扩散)。纯 regex, 零 LLM 成本, fail-safe 默认 GENERAL。

_INTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("WHY", ("为什么", "为何", "原因", "理由", "动机", "怎么解决", "如何解决",
             "怎么修", "如何修", "why", "because", "reason", "cause", "导致")),
    ("WHEN", ("什么时候", "何时", "时间", "多久", "最近", "上次", "哪天", "几月",
              "几号", "when", "before", "after", "timeline", "时间线")),
    ("ENTITY", ("什么是", "是谁", "关于", "介绍", "哪个", "哪些", "what is",
                "who is", "tell me about", "讲讲")),
]
_INTENTS = ("WHY", "WHEN", "ENTITY", "GENERAL")


def detect_intent(query: str) -> str:
    """检测查询意图 (mnemon 式意图识别, 纯规则). 返回 WHY/WHEN/ENTITY/GENERAL."""
    q = (query or "").strip().lower()
    if not q:
        return "GENERAL"
    for intent, kws in _INTENT_PATTERNS:
        for kw in kws:
            if kw in q:
                return intent
    return "GENERAL"


def _score_items(
    merged: dict[str, dict],
    mr: dict,
    *,
    now: float | None = None,
    sit_ctx: dict | None = None,
    coding_activity: bool = False,
    excl: list[str] | None = None,
    min_fused: float = 0.3,
    tk: int = 5,
    intent: str = "GENERAL",
) -> list[dict]:
    """融合评分 (Phase 4 全套加权; 纯函数, 可单测).

    fused = 语义相关分(raw)
          × 基础权重因子(A: bwf + (1-bwf)×weight)
          × 类型化衰减(B+E: 每类 per_day/floor, 人格/项目保值)
          × 类型 boost
          × (1 + 频率/强化/新鲜度/长期)
          × 情境(D: 写代码→v5_project 加分; 时段联想→created 小时≈now 加分)
    返回按 score 降序、≥min_fused 的 top tk 列表。
    """
    import time as _t
    now = _t.time() if now is None else now
    excl = [e for e in (excl or []) if e]
    boosts = mr.get("type_boost", {}) or {}
    bwf = float(mr.get("base_weight_factor", 1.0))
    td_cfg = mr.get("type_decay", {}) or {}
    fw_cfg = float(mr.get("frequency_weight", 0.0))
    rw_cfg = float(mr.get("reinforcement_weight", 0.0))
    frs_cfg = float(mr.get("freshness_weight", 0.0))
    lt_cfg = float(mr.get("long_term_boost", 0.0))
    sit_cfg = mr.get("situational", {}) or {}
    proj_boost = float(sit_cfg.get("project_activity_boost", 0.0))
    hour_boost = float(sit_cfg.get("hour_match_boost", 0.0))
    sit_enabled = bool(sit_cfg.get("enabled", True))
    # E: 意图驱动加权 (mnemon 借鉴): 按查询意图调类型 boost; enabled=false 不加
    intent_cfg = mr.get("intent", {}) or {}
    intent_enabled = bool(intent_cfg.get("enabled", False))
    intent_boosts = (intent_cfg.get(intent.lower(), {}) or {}) if intent_enabled else {}

    out: list[dict] = []
    for item in merged.values():
        fused = float(item["raw"])
        itype = item.get("type", "default")
        base_factor = 1.0
        decay_factor = 1.0
        type_boost_factor = 1.0
        freq_amount = 0.0
        sit_amount = 0.0
        # A: 写侧基础权重进评分 (bwf=1.0 时完全忽略, 保持旧行为)
        if bwf < 1.0:
            base_factor = bwf + (1.0 - bwf) * float(item.get("weight", 0.5))
            fused *= base_factor
        # B+E: 类型化衰减 (conversation 快衰减, user_trait/identity/decision 保值)
        if item.get("source") != "time" and item.get("created"):
            days = (now - float(item["created"])) / 86400.0
            if days > 0:
                tdc = td_cfg.get(itype, td_cfg.get("default", {"per_day": 0.05, "floor": 0.2}))
                pd = float(tdc.get("per_day", 0.05))
                fl = float(tdc.get("floor", 0.2))
                decay_factor = max(fl, 1.0 - pd * days)
                fused *= decay_factor
        b = boosts.get(itype, boosts.get("default", 1.0))
        if intent_boosts:
            b *= float(intent_boosts.get(itype, intent_boosts.get("default", 1.0)))
        type_boost_factor = b
        fused *= b
        # 频率/反馈/新鲜度/永久库 boost
        if fused > 0 and (fw_cfg or rw_cfg or frs_cfg or lt_cfg):
            boost = 0.0
            ac = int(item.get("access_count", 0))
            if fw_cfg and ac > 0:
                boost += min(0.25, (ac + 1).bit_length() * fw_cfg)
            if rw_cfg and item.get("reinforcement", 0.0) > 0:
                boost += min(0.15, float(item["reinforcement"]) * rw_cfg)
            if frs_cfg and item.get("last_accessed", 0.0) > 0 and \
                    (now - float(item["last_accessed"])) < 7 * 86400.0:
                boost += frs_cfg
            if lt_cfg and item.get("long_term"):
                boost += lt_cfg
            if boost:
                freq_amount = boost
                fused *= (1.0 + boost)
        # D: 情境加权 (enabled=false 或未提供上下文 → 不加)
        if fused > 0 and sit_enabled and sit_ctx is not None:
            sit_boost = 0.0
            if proj_boost and coding_activity and "v5_project" in (item.get("tags") or ""):
                sit_boost += proj_boost
            if hour_boost and item.get("created"):
                try:
                    import datetime as _dt
                    ch = _dt.datetime.fromtimestamp(float(item["created"])).hour
                    nh = _dt.datetime.fromtimestamp(now).hour
                    if abs(ch - nh) <= 1 or abs(ch - nh) >= 23:  # ±1h 或跨午夜
                        sit_boost += hour_boost
                except Exception:
                    pass
            if sit_boost:
                sit_amount = sit_boost
                fused *= (1.0 + sit_boost)
        item["score"] = fused
        item["intent"] = intent
        # P2 统一重要性: EI 透出 (与 lifecycle/upsert 同一口径 importance.effective_importance)
        try:
            from v5.importance import effective_importance as _ei_fn
            _ei_val = round(_ei_fn(item.get("weight", 0.5), item.get("access_count", 0),
                                   item.get("last_accessed", 0.0), now,
                                   item.get("reinforcement", 0.0)), 4)
        except Exception:
            _ei_val = 0.0
        # 信号透明 (mnemon 借鉴): 暴露评分各分量, 供上层/LLM 自主重排与解释
        item["signals"] = {
            "fts": round(float(item.get("fts_raw", 0.0)), 4),
            "vector": round(float(item.get("vec_raw", 0.0)), 4),
            "time": round(float(item.get("time_raw", 0.0)), 4),
            "base_weight": round(base_factor, 4),
            "type_decay": round(decay_factor, 4),
            "type_boost": round(type_boost_factor, 4),
            "frequency": round(freq_amount, 4),
            "situational": round(sit_amount, 4),
            "ei": _ei_val,
        }
        # 去重已知信息 (子串重叠)
        if excl:
            for ex in excl:
                if ex and (ex in item["content"] or item["content"] in ex):
                    item["score"] = -1.0
                    break
        out.append(item)

    out = [x for x in out if x["score"] >= min_fused]
    out.sort(key=lambda x: -x["score"])
    return out[:tk]


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    time_range: tuple[float, float] | None = None,
    exclude: list[str] | None = None,
    min_weight: float = 0.0,
    character: str = '',
    intent: str | None = None,
) -> list[dict]:
    """三路融合检索, 返按 fused_score 降序的 list[dict].

    返回字段: id, content, type, weight, tags, created, pad_p, pad_a, source, score

    V5.2: 新增 character 参数, 过滤指定角色的记忆.
    """
    if not query or not query.strip():
        return []

    # 意图检测 (mnemon 借鉴; 显式传入则跳过自动检测)
    intent = detect_intent(query) if intent is None else intent

    # 检索结果短 TTL 缓存
    ttl = _retrieve_ttl()
    cache_key = (query, top_k, time_range, tuple(exclude or []), character, intent)
    if ttl > 0:
        with _RET_CACHE_LOCK:
            hit = _RET_CACHE.get(cache_key)
            if hit is not None and (time.time() - hit[0]) < ttl:
                return hit[1]

    # 阈值 (fail-open)
    try:
        from v5 import preprocess_config as pc
        mr = pc.cfg()["memory_retrieval"]
    except Exception:
        mr = _defaults()
    vw = float(mr["vector_weight"])
    fw = float(mr["fts_weight"])
    decay = float(mr["time_decay_per_day"])
    min_fused = float(mr["min_fused_score"])
    tk = int(top_k or mr["top_k"])
    boosts = mr["type_boost"]
    # Phase 4 全套加权参数由 _score_items 内部读取 mr (纯函数, 可单测)

    # ── ① FTS5 关键词 ──
    fts_list: list = []
    try:
        from v5 import store
        fts_list = store.search(query, top_k=max(tk * 2, 6), min_weight=min_weight,
                                character=character)
    except Exception as e:
        logger.debug("FTS5 search failed: %s", e)

    # ── ② 向量语义 ──
    vec_list: list = []
    try:
        from v5.search import get_vector_index
        vec_list = get_vector_index().search(query, top_k=max(tk * 2, 6))
        if not vec_list:
            # 兜底: 单例可能持有旧快照, 或新记忆向量还在 Chroma WAL 未应用
            # (compaction 无 vision 模型被跳过). 强制刷新重建实例(从持久化
            # 重放)后重查一次, 消除新记忆 30s 语义不可见窗口。失败由 except
            # 兜住, 不影响 FTS 路。
            vec_list = get_vector_index(refresh=True).search(
                query, top_k=max(tk * 2, 6))
    except Exception as e:
        logger.debug("vector search failed: %s", e)

    # ── ③ 时间范围 ──
    time_list: list = []
    if time_range:
        try:
            from v5 import store
            start, end = time_range
            time_list = store.search_by_time_range(start, end, limit=max(tk * 2, 6))
        except Exception as e:
            logger.debug("time-range search failed: %s", e)

    # ── 去重合并 (按 id) ──
    merged: dict[str, dict] = {}

    def _add(mid, content, mtype, weight, created, pad_p, pad_a, source, raw, **extra):
        key = str(mid)
        if key in merged:
            # 同一记忆多路命中 → 累加分量 (0.7 向量分量 + 0.3 FTS5 分量 = 融合分)
            merged[key]["raw"] += raw
            if source in ("fts", "vec", "time"):
                merged[key][f"{source}_raw"] = float(merged[key].get(f"{source}_raw", 0.0)) + float(raw)
            return
        merged[key] = {
            "id": key, "content": content, "type": mtype, "weight": weight,
            # Phase 4: tags 透传 (情境加权需要 v5_project 判断)
            "tags": extra.get("tags", "") or "",
            "created": created, "pad_p": pad_p, "pad_a": pad_a,
            "source": source, "raw": raw,
            # 各路原始贡献 (信号透明: 哪些路径命中、各贡献多少)
            "fts_raw": 0.0, "vec_raw": 0.0, "time_raw": 0.0,
            # 阶段 4: 频率/反馈字段透传 (排序用; 未暴露时默认 0)
            "access_count": int(extra.get("access_count", 0)),
            "reinforcement": float(extra.get("reinforcement", 0.0)),
            "last_accessed": float(extra.get("last_accessed", 0.0)),
            "long_term": bool(extra.get("long_term", False)),
        }
        if source in ("fts", "vec", "time"):
            merged[key][f"{source}_raw"] = float(raw)

    for i, m in enumerate(fts_list):
        _add(m.id, m.content, m.type, m.weight, m.created,
             getattr(m, "pad_p", 0.0), getattr(m, "pad_a", 0.0), "fts", fw * (1.0 / (i + 1)),
             access_count=getattr(m, "access_count", 0),
             reinforcement=getattr(m, "reinforcement", 0.0),
             last_accessed=getattr(m, "last_accessed", 0.0),
             long_term=getattr(m, "long_term", False),
             tags=getattr(m, "tags", ""))
    for r in vec_list:
        _add(r.get("id"), r.get("content", ""), r.get("type", "fact"),
             r.get("weight", 0.5), r.get("created", 0.0),
             r.get("pad_p", 0.0), r.get("pad_a", 0.0), "vec", vw * float(r.get("score", 0.0)),
             access_count=r.get("access_count", 0),
             reinforcement=r.get("reinforcement", 0.0),
             last_accessed=r.get("last_accessed", 0.0),
             long_term=r.get("long_term", False),
             tags=r.get("tags", ""))
    for m in time_list:
        # 时间指代命中是用户明确信号, 给强初始分确保过 min_fused 阈值
        _add(m.id, m.content, m.type, m.weight, m.created,
             getattr(m, "pad_p", 0.0), getattr(m, "pad_a", 0.0), "time", 1.0,
             access_count=getattr(m, "access_count", 0),
             reinforcement=getattr(m, "reinforcement", 0.0),
             last_accessed=getattr(m, "last_accessed", 0.0),
             long_term=getattr(m, "long_term", False),
             tags=getattr(m, "tags", ""))

    # ── 融合分: 基础权重 + 类型化衰减 + 类型 boost + 频率/反馈 + 情境 ──
    now = time.time()

    # Phase 4 D: 情境上下文 (每轮检索取一次; 失败则跳过情境加权)
    sit_ctx = None
    coding_activity = False
    sit_cfg = mr.get("situational", {}) or {}
    if bool(sit_cfg.get("enabled", True)) and (
            float(sit_cfg.get("project_activity_boost", 0.0))
            or float(sit_cfg.get("hour_match_boost", 0.0))):
        try:
            from v5.context_anchor import now_context
            sit_ctx = now_context()
            _act = f"{sit_ctx.get('activity') or ''} {sit_ctx.get('window') or ''}"
            coding_activity = any(k in _act for k in (
                "写代码", "终端", "IDE", "VS Code", "PyCharm", "IntelliJ", "Code", "开发"))
        except Exception:
            sit_ctx = None

    excl = [e for e in (exclude or []) if e]
    result = _score_items(merged, mr, now=now, sit_ctx=sit_ctx,
                          coding_activity=coding_activity,
                          excl=excl, min_fused=min_fused, tk=tk, intent=intent)

    # ── 关键词兜底: 长句/混合 query FTS+向量双 miss 时, 拆词逐词 FTS 重查 ──
    # 实测 (2026-08-10): "memU 调研学到了什么" 整句 0 命中, 拆成 "memU"/"调研"
    # 后各能命中。这是 memU progressive_retrieve 文档里 "reword query" 场景的
    # 自动版 —— agent 不会每次手动重写 query, 检索层自己兜。
    # 触发条件: 结果不足 top_k 即补足 (2026-08-10 从 <3 放宽, 避免 3-4 条
    # 低相关历史记忆占位时特异性 token 命中永远进不来)
    if len(result) < tk:
        result = _keyword_fallback(query, result, tk, min_weight, character)

    # ── Vault fallback: 本体检索不足时, 去 ThirdSpace Vault 搜 ──
    if len(result) < 3:
        vault_hits = _vault_search(query, limit=tk - len(result))
        # dedup by content
        seen = {r["content"] for r in result}
        for v in vault_hits:
            if v["content"] not in seen:
                seen.add(v["content"])
                result.append(v)

    if ttl > 0:
        with _RET_CACHE_LOCK:
            _RET_CACHE[cache_key] = (time.time(), result)
            # 防膨胀: 超过 200 条清最旧 50 条
            if len(_RET_CACHE) > 200:
                oldest = sorted(_RET_CACHE.items(), key=lambda kv: kv[1][0])[:50]
                for k, _ in oldest:
                    _RET_CACHE.pop(k, None)
    return result


# ─────────────────────────────────────────────────────────────────────────
# 统一检索路由层 (unified_retrieve) —— 借鉴 cognee recall 的 auto scope 蓝图.
# V5 已有 5 个独立检索器 (semantic 三路融合 / lexical FTS / graph 扩散 /
# tree 树域加权 / vault 关键词), 这里提供唯一入口 + 自动路由 + 空则回退,
# 调用方 (memory_api / conversation-tree / 未来 agent 检索) 不再各自拼装.
# 全部 fail-open: 任一路异常静默跳过, 不阻塞; 返回统一归一化 dict.
# ─────────────────────────────────────────────────────────────────────────
SCOPES = ("auto", "semantic", "lexical", "graph", "tree", "temporal")

# 统一归一化字段 (与 retrieve() 返回一致 + source 标记来源)
_REQ_FIELDS = ("id", "content", "type", "weight", "tags", "created",
               "pad_p", "pad_a", "pad_d", "source", "score")


def _val(d, key, default=None):
    """兼容 dict / sqlite3.Row / store.Memory 的字段访问 (P6 归一化收敛)."""
    try:
        return d[key]
    except Exception:
        try:
            return getattr(d, key, default)
        except Exception:
            return default


def _norm(d) -> dict:
    """统一归一化 (dict / sqlite3.Row / store.Memory → 统一结果字典).

    P6 收敛 (2026-08-14): 唯一结果形状定义; memory_api._row_to_dict 委托本函数,
    结构化检索/语义检索/图检索全部输出同一字段集。
    """
    return {
        "id": _val(d, "id", ""),
        "content": _val(d, "content", "") or "",
        "type": _val(d, "type", "fact"),
        "weight": float(_val(d, "weight", 0.5) or 0.5),
        "tags": _val(d, "tags", "") or "",
        "created": float(_val(d, "created", 0.0) or 0.0),
        "pad_p": float(_val(d, "pad_p", 0.0) or 0.0),
        "pad_a": float(_val(d, "pad_a", 0.0) or 0.0),
        "pad_d": float(_val(d, "pad_d", 0.0) or 0.0),
        "source": _val(d, "source", "semantic"),
        "score": float(_val(d, "score", 0.0) or 0.0),
        "access_count": int(_val(d, "access_count", 0) or 0),
        "reinforcement": float(_val(d, "reinforcement", 0.0) or 0.0),
        "last_accessed": float(_val(d, "last_accessed", 0.0) or 0.0),
        "long_term": bool(_val(d, "long_term", False)),
        "intent": _val(d, "intent", "GENERAL"),
        "signals": _val(d, "signals") or {},
        "relation": _val(d, "relation", ""),
        "kind": _val(d, "kind", ""),
    }


def explain_result(item: dict) -> str:
    """P8 可观测性 (2026-08-14): 从 signals/intent/relation 生成"为什么召回这条"的可读说明.

    供 MCP 工具 (v5_memory_search / v5_project_retrieve) 给每条结果附 `why` 字段,
    让 pi/Hermes 能看到召回依据 (语义各路径分量 / 图 relation / 意图加权 / EI)。
    纯函数, 不依赖外部。
    """
    parts: list[str] = []
    sg = item.get("signals") or {}
    intent = item.get("intent") or "GENERAL"
    src = item.get("source") or "semantic"

    if src == "semantic":
        path = []
        if float(sg.get("vector", 0.0)) > 0:
            path.append(f"向量{sg['vector']:.2f}")
        if float(sg.get("fts", 0.0)) > 0:
            path.append(f"关键词{sg['fts']:.2f}")
        if float(sg.get("time", 0.0)) > 0:
            path.append(f"时间{sg['time']:.2f}")
        parts.append("语义融合(" + "+".join(path) + ")" if path else "语义")
    elif src in ("graph", "project_graph"):
        rel = item.get("relation")
        parts.append(f"图扩散(relation={rel})" if rel and rel != "self" else "图扩散")
    elif src == "lexical":
        parts.append("关键词命中")
    elif src == "structured":
        parts.append("精确标签命中")
    elif src == "kw":
        parts.append("关键词兜底")
    elif src in ("tree", "temporal", "vault"):
        parts.append(f"{src}路径")

    if intent != "GENERAL":
        parts.append(f"意图{intent}")
    tb = float(sg.get("type_boost", 1.0) or 1.0)
    if tb and tb != 1.0:
        parts.append(f"类型加权×{tb}")
    ei = float(sg.get("ei", 0.0) or 0.0)
    if ei > 0:
        parts.append(f"EI={ei:.2f}")
    if item.get("kind"):
        parts.append(f"kind={item['kind']}")

    return "，".join(parts) if parts else f"来源{src}"


def _route_cfg() -> dict:
    try:
        from v5 import preprocess_config as pc
        return pc.cfg().get("memory_retrieval", {})
    except Exception:
        return {}


def _graph_retrieve(query: str, tk: int, graph_min: float = 0.0) -> list[dict]:
    """P3 图收敛 (2026-08-14): 统一图检索 = 实体图 + 项目知识图, 一致性收集.

    取代 graph scope / auto 补路里对 entity_graph_search / project_graph_search
    的重复 OR 拼装。同一张"V5 图"的两个边类型: eg_edges (实体共现) + project_edges
    (笔记类型化边)。低分 graph 结果按 graph_min 过滤。
    """
    out: dict[str, dict] = {}
    try:
        from v5.search import entity_graph_search
        for x in entity_graph_search(query, top_k=tk * 2):
            if float(x.get("score", 0.0)) >= graph_min:
                out.setdefault(str(x.get("id")), x)
    except Exception as e:
        logger.debug("_graph_retrieve entity failed: %s", e)
    try:
        from v5.project_edges import project_graph_search
        for x in project_graph_search(query, top_k=tk):
            if float(x.get("score", 0.0)) >= graph_min:
                out.setdefault(str(x.get("id")), x)
    except Exception as e:
        logger.debug("_graph_retrieve project failed: %s", e)
    return list(out.values())


def unified_retrieve(
    query: str,
    *,
    top_k: int | None = None,
    scope: str = "auto",
    node_id: str | None = None,
    tree=None,
    character: str = "",
    time_range: tuple[float, float] | None = None,
    exclude: list[str] | None = None,
    min_weight: float = 0.0,
) -> list[dict]:
    """统一检索入口 (对应 cognee recall). scope 自动路由, 空则回退语义.

    scope:
      - "auto"     (默认): 语义三路融合 → 结果 <3 时补图扩散路 → 仍不足走 Vault (retrieve 内置)
      - "semantic": 等价现有 retrieve() (三路融合 + Vault fallback)
      - "lexical" : 仅 FTS5 关键词, 空则回退 semantic
      - "graph"   : 仅实体图扩散激活, 空则回退 semantic
      - "tree"    : 树域加权检索 (需 node_id + tree 对象; tree 缺失自动降级 auto)
      - "temporal": 时间过滤检索 (阶段 5 接线后可用; 当前降级 semantic)
    返回: 按 score 降序的统一归一化 list[dict], source ∈ semantic/lexical/graph/tree/vault.
    """
    if not query or not query.strip():
        return []
    if scope not in SCOPES:
        scope = "auto"
    mr = _route_cfg()
    auto_route = bool(mr.get("auto_route", True))
    tk = int(top_k or mr.get("top_k", 5) or 5)
    graph_min = float(mr.get("graph_min_score", 0.2) or 0.2)
    intent = detect_intent(query)
    intent_enabled = bool((mr.get("intent", {}) or {}).get("enabled", False))
    merged: dict[str, dict] = {}
    used: list[str] = []

    def _merge(items: list[dict], src: str) -> None:
        if not items:
            return
        for it in items:
            d = _norm(it)
            d["source"] = src
            d["intent"] = intent  # 统一带查询意图 (信号透明)
            key = d["id"]
            if not key:
                continue
            if key in merged:
                # 多路命中: 取最高分 (不累加, 避免 graph 低分稀释 semantic 高分)
                if d["score"] > merged[key]["score"]:
                    merged[key] = d
            else:
                merged[key] = d

    # ── 显式 scope 优先 ──
    if scope == "lexical":
        try:
            from v5 import store
            hits = store.search(query, top_k=tk, min_weight=min_weight, character=character)
            for i, m in enumerate(hits):
                _merge([_norm({
                    "id": str(m.id), "content": m.content, "type": m.type,
                    "weight": m.weight, "tags": getattr(m, "tags", ""),
                    "created": m.created, "pad_p": getattr(m, "pad_p", 0.0),
                    "pad_a": getattr(m, "pad_a", 0.0),
                    "score": 0.3 * (1.0 / (i + 1)),
                })], "lexical")
            used.append("lexical")
        except Exception as e:
            logger.debug("unified lexical failed: %s", e)
        if not merged:
            return unified_retrieve(query, top_k=tk, scope="semantic",
                                    character=character, time_range=time_range,
                                    exclude=exclude, min_weight=min_weight)

    elif scope == "graph":
        # P3 图收敛: 统一图检索 (实体图 + 项目知识图)
        try:
            _merge(_graph_retrieve(query, tk), "graph")
            used.append("graph")
        except Exception as e:
            logger.debug("unified graph failed: %s", e)
        if not merged:
            return unified_retrieve(query, top_k=tk, scope="semantic",
                                    character=character, time_range=time_range,
                                    exclude=exclude, min_weight=min_weight)

    elif scope == "tree":
        if tree is not None and node_id:
            try:
                from v5.extensions.tree_adapter import tree_scoped_retrieve
                _merge(tree_scoped_retrieve(tree, node_id, query, top_k=tk,
                                            character=character), "tree")
                used.append("tree")
            except Exception as e:
                logger.debug("unified tree failed: %s", e)
            if merged:
                return _finish(merged, tk)
        # tree 不可用 → 降级 auto (保持树端调用行为不崩)
        return unified_retrieve(query, top_k=tk, scope="auto", character=character,
                                time_range=time_range, exclude=exclude,
                                min_weight=min_weight)

    elif scope == "temporal":
        # 阶段 5: retrieve_temporal —— 过滤 valid_to 已失效的事实 (时效图谱)
        try:
            _merge(retrieve_temporal(query, top_k=tk * 2, character=character,
                                     time_range=time_range, exclude=exclude,
                                     min_weight=min_weight), "temporal")
            used.append("temporal")
        except Exception as e:
            logger.debug("unified temporal failed: %s", e)
        if merged:
            return _finish(merged, tk)
        return unified_retrieve(query, top_k=tk, scope="semantic", character=character,
                                time_range=time_range, exclude=exclude,
                                min_weight=min_weight)

    # ── auto / semantic ──
    sem: list = []
    try:
        sem = retrieve(query, top_k=tk, time_range=time_range, exclude=exclude,
                       min_weight=min_weight, character=character, intent=intent)
    except Exception as e:
        logger.debug("unified semantic failed: %s", e)
    _merge(sem, "semantic")
    used.append("semantic")
    if scope == "semantic" or not auto_route:
        return _finish(merged, tk)

    # ── auto 补路: semantic 不足时补图扩散 (低分 graph 不过 threshold) ──
    # 意图为 ENTITY (问"什么是/关于/是谁") 时总是补实体图扩散 (即使 semantic 已足),
    # 与 mnemon 的 "ENTITY 意图 → 实体边加权" 一致。
    if len(merged) < 3 or (intent_enabled and intent == "ENTITY"):
        try:
            # P3 图收敛: 统一图检索 (实体图 + 项目知识图 + graph_min 过滤)
            _merge(_graph_retrieve(query, tk, graph_min=graph_min), "graph")
            used.append("graph")
        except Exception as e:
            logger.debug("unified auto graph fallback failed: %s", e)
    return _finish(merged, tk)


def _finish(merged: dict[str, dict], tk: int) -> list[dict]:
    """排序截断 (fail-open: merged 可能为空).

    Phase 3 (2026-08-14): 时间锚定检索 ——
      - now 用 context_anchor.now_epoch() 统一时间锚
      - 默认排除已失效事实 (memory.valid_to < now), 与 temporal_graph
        "检索永远取当前值" 设计意图一致; 列不存在/迁移未跑/查询失败 → fail-open
        不过滤 (不阻塞检索)。
    """
    out = [v for v in merged.values() if v["score"] > 0]
    if not out:
        return []
    try:
        from v5.store import valid_to_map
        from v5.context_anchor import now_epoch
        ids = [str(v["id"]) for v in out]
        vt = valid_to_map(ids, "memory", "id")
        now = now_epoch()
        out = [v for v in out
               if vt.get(str(v["id"])) is None
               or float(vt[str(v["id"])]) > now]
    except Exception as exc:
        logger.debug("_finish: temporal filter skipped (%s)", exc)
    out.sort(key=lambda x: -x["score"])
    return out[:tk]


def retrieve_temporal(query: str, *, now: float | None = None,
                      top_k: int = 5, **kw) -> list[dict]:
    """时效感知检索: 包裹 retrieve, 过滤 valid_to 已失效(valid_to < now)的事实。

    原位于 extensions.temporal_graph (2026-08-14 迁移至此以解开
    temporal_graph ↔ memory_retrieval 循环依赖)。过期事实被直接剔除
    (而非降权) —— 失效意味着"该值已被新事实取代", 召回它就是错误。
    """
    import time as _t
    from v5.store import valid_to_map
    now = _t.time() if now is None else now
    results = retrieve(query, top_k=top_k, **kw)
    if not results:
        return results
    ids = [str(r.get("id")) for r in results if r.get("id")]
    vt = valid_to_map(ids, "memory", "id")
    kept = [r for r in results
            if vt.get(str(r.get("id"))) is None or vt[str(r.get("id"))] > now]
    return kept


# ─── 关键词兜底 (长句拆词重查) ─────────────────────────────────────
# 复用 skill_store 的分词器 (ASCII 词原样 + 中文 2-gram, 支持中英混排),
# 保证两处检索的拆词行为一致。


def _keyword_tokens(query: str) -> list[str]:
    try:
        from v5.skill_store import _tokens
        return _tokens(query)
    except Exception:
        return []


def _keyword_fallback(
    query: str,
    result: list[dict],
    tk: int,
    min_weight: float,
    character: str,
) -> list[dict]:
    """整句检索 miss 时, 把长 query 拆成关键词逐词 FTS 重查, 补足结果.

    只补足 (append), 不改动已命中的排序; 每条兜底命中标记 source='kw'.
    关键词命中是弱信号, score 用固定小值 (0.45, 略低于融合阈值 0.6 的
    常见命中, 保证不喧宾夺主, 但能进 top_k)。
    """
    if len(result) >= tk:
        return result
    tokens = _keyword_tokens(query)
    if len(tokens) < 2:
        return result
    try:
        from v5 import store
    except Exception:
        return result

    seen_ids = {str(r["id"]) for r in result}
    # 稀有 token 优先: 常见词("哥哥"/"什么")LIKE 命中噪音多, 先查特异性 token
    # (2026-08-10 实测: '文艺' 1 条命中 vs '哥哥' 50+ 条; 顺序错则金丝雀被挤掉)
    try:
        scored = []
        for tok in tokens:
            try:
                n = store.count_like(tok, min_weight=min_weight, character=character)
            except Exception:
                n = 999
            scored.append((n, tok))
        scored.sort(key=lambda x: x[0])
        ordered = [t for _, t in scored]
    except Exception:
        ordered = tokens
    for tok in ordered:
        if len(result) >= tk:
            break
        try:
            # FTS5 unicode61 对中文 2-gram MATCH 无效(整串分词), 走 LIKE 子串
            # 查询 (2026-08-10 实测: MATCH '主力' 0 命中 vs LIKE %主力% 命中)
            hits = store.search_like(tok, top_k=3, min_weight=min_weight,
                                     character=character)
        except Exception as e:
            logger.debug("keyword fallback failed for %r: %s", tok, e)
            continue
        for m in hits:
            mid = str(m.id)
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            result.append({
                "id": mid, "content": m.content, "type": m.type,
                "weight": m.weight, "tags": getattr(m, "tags", ""),
                "created": m.created,
                "pad_p": getattr(m, "pad_p", 0.0),
                "pad_a": getattr(m, "pad_a", 0.0),
                "source": "kw",
                "score": 0.45,
                "access_count": int(getattr(m, "access_count", 0)),
                "reinforcement": float(getattr(m, "reinforcement", 0.0)),
                "last_accessed": float(getattr(m, "last_accessed", 0.0)),
                "long_term": bool(getattr(m, "long_term", False)),
            })
    return result


# ─── ThirdSpace Vault fallback ─────────────────────────────────────
# 当 V5 本体检索命中不足时, 轻量搜 vault 的 03-知识/ 和 02-日记/。
# 不建索引, 不依赖 :8587, 纯关键词匹配（维护成本为零）。

_VAULT_ROOT = None
_VAULT_INITED = False


def _get_vault_root() -> str | None:
    global _VAULT_ROOT, _VAULT_INITED
    if _VAULT_INITED:
        return _VAULT_ROOT
    _VAULT_INITED = True
    import os
    env = os.environ.get("THIRDSPACE_VAULT", "")
    if env:
        _VAULT_ROOT = env
        return _VAULT_ROOT
    # fallback: 从 Ikaros 项目根推断
    try:
        from pathlib import Path
        p = Path(__file__).resolve().parent
        candidate = p / "data" / "thirdspace-vault"
        if candidate.is_dir():
            _VAULT_ROOT = str(candidate)
    except Exception:
        pass
    return _VAULT_ROOT


def _vault_search(query: str, limit: int = 3) -> list[dict]:
    """在 vault 的 03-知识/ 和 02-日记/ 下做轻量关键词搜索。

    Returns:
        list[dict]: 跟 retrieve() 返回格式兼容，type='vault_note'。
    """
    root = _get_vault_root()
    if not root:
        return []

    import os
    import re

    tokens = [t for t in re.split(r"[\\s,，。、]+", query.strip().lower()) if len(t) >= 2]
    if not tokens:
        return []

    hits: list[tuple[float, str, str]] = []  # (score, content, path)

    for dirname in ("03-知识", "02-日记"):
        base = os.path.join(root, dirname)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            # 跳过 00-系统 层级（vault 内不嵌套, 但防手滑）
            if "00-系统" in dirpath:
                continue
            for fn in filenames:
                if not fn.endswith(".md"):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    text = open(fp, "r", encoding="utf-8").read()
                except Exception:
                    continue
                # 去掉 frontmatter (---...---)
                body = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)
                # 去掉 frontmatter 后空行
                body = body.strip()
                if not body:
                    body = text
                # 关键词匹配评分
                score = 0.0
                lower = body.lower()
                for t in tokens:
                    if t in lower:
                        score += 1.0
                if score > 0:
                    # 取前 300 字作为摘要
                    preview = re.sub(r"\s+", " ", body[:300]).strip()
                    rel = os.path.relpath(fp, root)
                    title = re.sub(r"\.md$", "", fn)
                    content = f"[vault] {title}: {preview}"
                    # 再加分：标题命中
                    if any(t in title.lower() for t in tokens):
                        score += 2.0
                    hits.append((score, content, rel))

    hits.sort(key=lambda x: -x[0])
    return [
        {
            "id": f"vault:{h[2]}",
            "content": h[1][:400],
            "type": "vault_note",
            "weight": min(0.8, h[0] * 0.15),
            "tags": "vault",
            "score": min(1.0, h[0] * 0.12),
            "created": 0.0,
            "pad_p": 0.0,
            "pad_a": 0.0,
            "source": "vault",
        }
        for h in hits[:limit]
    ]


if __name__ == "__main__":
    import json
    for q in ["哥哥喜欢简洁", "CUDA 升级"]:
        print(f"## {q}")
        print(json.dumps(retrieve(q), ensure_ascii=False, indent=2)[:800])
