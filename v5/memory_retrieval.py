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
    }


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    time_range: tuple[float, float] | None = None,
    exclude: list[str] | None = None,
    min_weight: float = 0.0,
    character: str = '',
) -> list[dict]:
    """三路融合检索, 返按 fused_score 降序的 list[dict].

    返回字段: id, content, type, weight, tags, created, pad_p, pad_a, source, score

    V5.2: 新增 character 参数, 过滤指定角色的记忆.
    """
    if not query or not query.strip():
        return []

    # 检索结果短 TTL 缓存
    ttl = _retrieve_ttl()
    cache_key = (query, top_k, time_range, tuple(exclude or []), character)
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

    def _add(mid, content, mtype, weight, created, pad_p, pad_a, source, raw):
        key = str(mid)
        if key in merged:
            # 同一记忆多路命中 → 累加分量 (0.7 向量分量 + 0.3 FTS5 分量 = 融合分)
            merged[key]["raw"] += raw
            return
        merged[key] = {
            "id": key, "content": content, "type": mtype, "weight": weight,
            "tags": "", "created": created, "pad_p": pad_p, "pad_a": pad_a,
            "source": source, "raw": raw,
        }

    for i, m in enumerate(fts_list):
        _add(m.id, m.content, m.type, m.weight, m.created,
             getattr(m, "pad_p", 0.0), getattr(m, "pad_a", 0.0), "fts", fw * (1.0 / (i + 1)))
    for r in vec_list:
        _add(r.get("id"), r.get("content", ""), r.get("type", "fact"),
             r.get("weight", 0.5), r.get("created", 0.0),
             r.get("pad_p", 0.0), r.get("pad_a", 0.0), "vec", vw * float(r.get("score", 0.0)))
    for m in time_list:
        # 时间指代命中是用户明确信号, 给强初始分确保过 min_fused 阈值
        _add(m.id, m.content, m.type, m.weight, m.created,
             getattr(m, "pad_p", 0.0), getattr(m, "pad_a", 0.0), "time", 1.0)

    # ── 融合分: 时间衰减 + 类型 boost ──
    now = time.time()
    excl = [e for e in (exclude or []) if e]
    out: list[dict] = []
    for item in merged.values():
        fused = float(item["raw"])
        # 时间衰减: 仅作用于 fts/vec 来源 (spec: 衰减不能太激进, 旧偏好仍有效)
        # 时间指代命中 (source=='time') 本身是用户明确信号, 不叠加衰减
        if item["source"] != "time" and item["created"]:
            days = (now - float(item["created"])) / 86400.0
            if days > 0:
                fused *= max(0.2, 1.0 - decay * days)  # 下限 0.2, 不归零
        b = boosts.get(item["type"], boosts.get("default", 1.0))
        fused *= b
        item["score"] = fused
        # 去重已知信息 (子串重叠)
        if excl:
            for ex in excl:
                if ex and (ex in item["content"] or item["content"] in ex):
                    item["score"] = -1.0
                    break
        out.append(item)

    out = [x for x in out if x["score"] >= min_fused]
    out.sort(key=lambda x: -x["score"])
    result = out[:tk]

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
    # fallback: 仅在仓库同级的 data/thirdspace-vault 查找 (不依赖 Ikaros)
    try:
        from pathlib import Path as _P
        candidate = _P(__file__).resolve().parent.parent / "data" / "thirdspace-vault"
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
