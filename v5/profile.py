# 详细说明见 docs/scripts/core/v5/v5/profile.md
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.profile")

V5_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = V5_ROOT / "data" / "v5"
PROFILE_PATH = DATA_DIR / "profile.json"

# ─── 旧版兼容: 简单偏好/讨厌注入 (cloud_chat 使用) ──────────────

_CONF_GATE = 0.7
_MAX_INJECT = 2
_MAX_PREF_LEN = 80  # 偏好不应是长篇哲学反思，超过此长度不注入


def _read(kind: str, limit: int = 50) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    try:
        from v5 import store
        mems = store.list_all(type_filter=kind, limit=limit)
        for m in mems:
            # 长度过滤: 超过 _MAX_PREF_LEN 的不是有效偏好/讨厌
            if len(m.content) > _MAX_PREF_LEN:
                continue
            out.append((m.content, float(m.weight)))
    except Exception as e:
        logger.debug("profile._read(%s) failed: %s", kind, e)
    return out


def load_dislikes() -> list[str]:
    return [c for c, w in _read("dislike") if w >= _CONF_GATE]


def load_preferences() -> list[str]:
    return [c for c, w in _read("preference") if w >= _CONF_GATE]


def record(kind: str, content: str, weight: float = 0.8) -> Optional[int]:
    """写一条画像记忆到 v5.db (供 cloud_chat._self_review 调用). 返 memory id."""
    try:
        from v5 import store
        return store.store(content=content, type=kind, weight=weight, tags="profile")
    except Exception as e:
        logger.warning("profile.record failed: %s", e)
        return None


def build_profile_block() -> str:
    """返回注入句 (空字符串 = 跳过). 负面优先, 最多 _MAX_INJECT 条.

    防御性去前缀: v5.db 里存的用户原话可能已含 "哥哥偏好/偏好/不喜欢" 等,
    这里统一剥掉, 避免拼成 "哥哥偏好哥哥偏好…" 的重复前缀。
    """
    def _clean(s: str) -> str:
        s = (s or "").strip()
        for pre in ("哥哥偏好", "哥哥", "偏好", "不喜欢", "讨厌"):
            if s.startswith(pre):
                s = s[len(pre):].strip()
        return s

    dislikes = load_dislikes()
    prefs = load_preferences()
    parts: list[str] = []
    for d in dislikes[:_MAX_INJECT]:
        cd = _clean(d)
        if cd:
            parts.append(f"不喜欢{cd}")
    remaining = _MAX_INJECT - len(parts)
    for p in prefs[:remaining]:
        cp = _clean(p)
        if cp:
            parts.append(f"偏好{cp}")
    if not parts:
        return ""
    line = "哥哥" + "、".join(parts) + "。"
    return f"\n---\n{line}"


# ─── V5 新增: 哥哥画像聚合模块 (User Profile Aggregation) ──────

# trait 分类标签
TRAIT_CATEGORIES = {
    "communication_style": ["语气", "长度", "风格", "直接", "委婉", "短句", "详细"],
    "topic_preference": ["话题", "兴趣", "关注", "喜欢", "关心"],
    "response_pattern": ["回应", "沉默", "继续", "结束", "接话"],
    "emotional_tone": ["开心", "烦躁", "焦虑", "沉默", "高兴", "低落", "生气"],
    "work_style": ["工作", "代码", "项目", "调试", "技术"],
}


@dataclass
class Trait:
    """单个沟通特征的聚合结果."""
    category: str              # 所属大类
    content: str               # 具体的特征描述
    confidence: float          # 聚合置信度 0.0-1.0
    observation_count: int     # 观察到这个特征的次数
    last_observed: float       # 最近一次观察到的时间戳
    evidence: list[str] = field(default_factory=list)  # 原文引用 (最多3条)


@dataclass
class UserProfile:
    """哥哥的完整画像."""
    traits: list[Trait] = field(default_factory=list)
    aggregated_at: float = 0.0
    trait_count: int = 0


def _categorize(content: str) -> str:
    """根据内容关键词分类 trait."""
    for category, keywords in TRAIT_CATEGORIES.items():
        for kw in keywords:
            if kw in content:
                return category
    return "uncategorized"


def aggregate() -> UserProfile:
    """从 v5.db 读取所有 user_trait 记忆, 按 category + 语义相似度聚合.

    Returns:
        UserProfile: 聚合后的画像
    """
    from v5 import store

    try:
        rows = store.list_all(200)
        traits_raw = [m for m in rows if m.type == "user_trait"]
    except Exception as e:
        logger.warning("profile: 读取 user_trait 失败 %s", e)
        return UserProfile()

    if not traits_raw:
        logger.info("profile: 无 user_trait 数据")
        return UserProfile()

    # 按 category 分组
    by_category: dict[str, list] = {}
    for m in traits_raw:
        cat = _categorize(m.content)
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m)

    # 每组内合并相似 trait
    traits: list[Trait] = []
    for cat, items in by_category.items():
        # 简单策略: 同一 category 内的 trait 按 confidence 降序取 top 3
        # (更复杂的语义合并需要 LLM 调用，留给后续优化)
        sorted_items = sorted(
            items,
            key=lambda x: x.weight * (1 + x.reinforcement - x.disputation),
            reverse=True
        )
        seen = set()
        for m in sorted_items[:5]:
            key = m.content[:30]  # 用前缀去重
            if key in seen:
                continue
            seen.add(key)
            confidence = min(0.95, m.weight + m.reinforcement * 0.1)
            traits.append(Trait(
                category=cat,
                content=m.content[:200],
                confidence=round(confidence, 2),
                observation_count=min(50, m.access_count + 1),
                last_observed=m.created,
                evidence=[m.content[:100]] if m.content else [],
            ))

    profile = UserProfile(
        traits=traits,
        aggregated_at=time.time(),
        trait_count=len(traits_raw),
    )
    return profile


def save_profile(profile: UserProfile) -> None:
    """保存画像到 data/v5/profile.json.

    原子写: 写 tmp + rename, 防中途崩溃丢数据.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_PATH.with_suffix(PROFILE_PATH.suffix + ".tmp")
    tmp.write_text(
        json.dumps(asdict(profile), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(PROFILE_PATH)
    logger.info("profile: 已保存 %d 条 trait", len(profile.traits))


def load_profile() -> UserProfile:
    """加载画像. 文件不存在或损坏时返空."""
    if not PROFILE_PATH.exists():
        return UserProfile()
    try:
        data = json.loads(PROFILE_PATH.read_text("utf-8"))
        return UserProfile(
            traits=[Trait(**t) for t in data.get("traits", [])],
            aggregated_at=data.get("aggregated_at", 0.0),
            trait_count=data.get("trait_count", 0),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("profile: 加载失败 %s, 重新聚合", e)
        return UserProfile()


def run_sync() -> dict:
    """触发一次画像聚合 + 保存。供 scheduler 调用。

    Returns:
        dict: {traits: int, raw_sources: int, elapsed_sec: float}
    """
    t0 = time.time()
    profile = aggregate()
    save_profile(profile)
    elapsed = time.time() - t0
    logger.info("profile sync: %d traits from %d raw sources, %.2fs",
                len(profile.traits), profile.trait_count, elapsed)
    return {
        "traits": len(profile.traits),
        "raw_sources": profile.trait_count,
        "elapsed_sec": round(elapsed, 2),
    }


def get_snapshot() -> dict:
    """获取当前画像摘要 (给检索/注入用).

    Returns:
        dict: 含 top traits 和聚合时间. 如果没有数据返空 dict.
    """
    profile = load_profile()
    if not profile.traits:
        return {}
    return {
        "aggregated_at": profile.aggregated_at,
        "top_traits": [
            {"category": t.category, "content": t.content, "confidence": t.confidence}
            for t in sorted(profile.traits, key=lambda x: -x.confidence)[:5]
        ],
    }


if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    # 如果传 --sync 参数则跑聚合, 否则默认跑 build_profile_block
    if "--sync" in _sys.argv:
        result = run_sync()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif "--snapshot" in _sys.argv:
        snap = get_snapshot()
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    else:
        print(build_profile_block())
