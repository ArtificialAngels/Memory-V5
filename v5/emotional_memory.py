# 详细说明见 docs/scripts/core/v5/v5/emotional_memory.md

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("ikaros.v5.emotional_memory")

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

# PAD 变化阈值 (低于此值不记录)
_DELTA_THRESHOLD = 0.12  # |ΔP|+|ΔA|+|ΔD| 总和阈值

_CAUSAL_PROMPT = """你是伊卡洛斯的情绪解释器。从对话中推断"为什么情绪变了"。

规则:
- 输入: 最近的对话 (哥哥说了什么) + 旧情绪→新情绪的变化
- 输出: 一句简洁的第一人称因果陈述
- 格式: "因为哥哥 <做了什么>, 我感到 <情绪词>"
- 如果情绪不是因哥哥的话而变 (自然衰减等), 输出: "没特别原因, 情绪自然平复了"
- 只输出一句话, 不要多余解释

示例:
  哥哥说"你真棒,帮了大忙" + 情绪从平和变愉悦
  → 因为哥哥夸了我, 我感到很开心

  哥哥说"又报错了烦死了" + 情绪从平和变低落
  → 因为哥哥遇到了麻烦, 我有点担心"""


def maybe_record_emotion(
    old_pad: tuple[float, float, float],
    new_pad: tuple[float, float, float],
    user_text: str,
    prev_user_text: str = "",
) -> dict | None:
    """检测 PAD 变化是否够大, 够大则生成因果记忆并写入 V5。

    Args:
        old_pad: 变更前的 (p, a, d)
        new_pad: 变更后的 (p, a, d)
        user_text: 当前轮哥哥说的话
        prev_user_text: 上一轮哥哥说的话 (提供更多上下文)

    Returns:
        dict | None: 生成的因果记忆, 如果变化不够大则 None
    """
    op, oa, od = old_pad
    np, na, nd = new_pad
    delta = abs(np - op) + abs(na - oa) + abs(nd - od)

    if delta < _DELTA_THRESHOLD:
        return None

    intensity = min(1.0, delta / 0.6)  # 归一化 0~1

    # 生成因果句
    causal_text = _generate_causal(user_text, prev_user_text, old_pad, new_pad)
    if not causal_text:
        return None

    # 写入 V5
    try:
        from v5 import store as store
        mid = store.store(
            content=causal_text,
            type="emotional_event",
            weight=min(0.95, 0.5 + intensity * 0.45),
            tags=f"v5,causal,intensity:{intensity:.2f}",
            pad_p=np, pad_a=na, pad_d=nd,
        )
        logger.info("emotional_memory: recorded id=%d [i=%.2f] %s",
                    mid, intensity, causal_text[:80])
        return {
            "id": mid, "content": causal_text,
            "intensity": round(intensity, 3),
            "delta": round(delta, 4),
        }
    except Exception as exc:
        logger.debug("emotional_memory: v4 store failed (%s)", exc)
        return None


def _generate_causal(
    user_text: str,
    prev_text: str,
    old_pad: tuple[float, float, float],
    new_pad: tuple[float, float, float],
) -> str | None:
    """用云端 LLM (DeepSeek) 推断情感变化的因果."""
    try:
        from v5.reflect.llm_client import call_llm_auto
    except Exception as exc:
        logger.debug("emotional_memory: LLM unavailable (%s)", exc)
        return _rule_based_causal(user_text, old_pad, new_pad)

    # PAD 变化描述
    op, oa, od = old_pad
    np, na, nd = new_pad

    def _p_label(v: float, dim: str) -> str:
        if dim == "p":
            return "愉悦" if v > 0.1 else ("低落" if v < -0.1 else "平和")
        if dim == "a":
            return "兴奋" if v > 0.1 else ("困倦" if v < -0.1 else "平静")
        return "自信" if v > 0.1 else ("乖巧" if v < -0.1 else "中立")

    old_desc = f"{_p_label(op,'p')}-{_p_label(oa,'a')}-{_p_label(od,'d')}"
    new_desc = f"{_p_label(np,'p')}-{_p_label(na,'a')}-{_p_label(nd,'d')}"

    context = f"哥哥刚才说: \"{user_text[:200]}\""
    if prev_text:
        context += f"\n哥哥上一句话: \"{prev_text[:200]}\""
    context += f"\n\n情绪从 [{old_desc}] 变成了 [{new_desc}]"

    try:
        result = call_llm_auto(
            _CAUSAL_PROMPT,
            context,
            max_tokens=128,
            temperature=0.3,
            timeout=45,
        )
        text = result.content.strip()
        if len(text) < 4 or len(text) > 300:
            return None
        return text
    except Exception as exc:
        logger.debug("emotional_memory: LLM call failed (%s)", exc)
        return _rule_based_causal(user_text, old_pad, new_pad)


def _rule_based_causal(
    user_text: str,
    old_pad: tuple[float, float, float],
    new_pad: tuple[float, float, float],
) -> str | None:
    """降级: 基于规则推断因果 (无 LLM 时)."""
    op, _oa, _od = old_pad
    np, _na, _nd = new_pad
    dp = np - op

    # 简易规则: 只看 pleasure 方向
    snippet = user_text[:50].replace("\n", " ")
    if dp > 0.1:
        return f"因为哥哥说了\"{snippet}\", 我感到更开心了"
    elif dp < -0.1:
        return f"因为哥哥说了\"{snippet}\", 我有点难过"
    else:
        return f"和哥哥说了\"{snippet}\"之后, 我的心情有些微妙的变化"


# 内联说明见 docs/scripts/core/v5/v5/emotional_memory.md（见“内联注释摘录”）

import json as _json
from pathlib import Path as _Path
from typing import Optional

_EMOTION_STATE_PATH = _Path(__file__).resolve().parent / "data" / "v5" / "emotion_state.json"
_LABEL_STATE_PATH = _Path(__file__).resolve().parent / "data" / "v5" / "emotion_labels.json"

# 模块级 5s 去重 (spec 4.1: 每个模块维护 _last_injected 时间戳)
_LAST_DIFF_INJECT = 0.0

# 情感关键词 → 注入召回触发 (用户显式提到情绪时, 用 search_by_emotion 拉一条旧记忆)
_EMOTION_LEXICON = [
    "开心", "开心", "难过", "伤心", "低落", "焦虑", "紧张", "生气", "愤怒",
    "平静", "放松", "专注", "期待", "兴奋", "温柔", "害羞", "满足", "困惑",
    "迷茫", "怀旧", "想念", "无聊", "烦", "累", "高兴", "愉快",
]


def _emotion_cfg(*keys, default=None):
    try:
        from v5.preprocess_config import get
        return get("emotion", *keys, default=default)
    except Exception:
        return default


def _mood_from_pad(p: float, a: float, d: float) -> str:
    """PAD → 粗粒度心情标签 (用于差异比较)."""
    if p > 0.15 and a > 0.1:
        return "开心"
    if p > 0.15:
        return "愉悦"
    if p < -0.15:
        return "低落"
    if a > 0.25:
        return "专注"
    return "平静"


def _classify_tags_from_pad(p: float, a: float, d: float) -> list[str]:
    """规则兜底: PAD → 1-2 个标签 (LLM 不可用)."""
    max_tags = int(_emotion_cfg("max_tags", default=2))
    tags: list[str] = []
    if p > 0.15:
        tags.append("开心" if a > 0.1 else "愉悦")
    elif p < -0.15:
        tags.append("难过" if a > -0.05 else "低落")
    if a > 0.25:
        tags.append("专注")
    if not tags:
        tags.append("平静")
    return tags[:max_tags]


def label_emotion(user_text: str, new_pad: tuple[float, float, float]) -> list[str]:
    """为当前轮对话打 1-2 个情感标签 (云端 LLM). 失败回退规则.

    Args:
        user_text: 当前轮哥哥说的话
        new_pad: 变更后的 (p, a, d)

    Returns:
        list[str]: 1-2 个来自受控词表的标签
    """
    max_tags = int(_emotion_cfg("max_tags", default=2))
    vocab = _emotion_cfg("tag_vocab", default=["开心", "平静", "低落", "专注"])
    try:
        from v5.reflect.llm_client import call_llm
        prompt = (
            "你是伊卡洛斯的情绪标注器。从下面的对话里挑 1-2 个最贴切的情绪标签。\n"
            "只能从这份固定词表里选, 用顿号隔开, 不要解释, 不要其他字:\n"
            + "、".join(vocab) + "\n\n对话: " + (user_text or "")[:200]
        )
        resp = call_llm(
            prompt, "",
            provider=os.environ.get("IKAROS_LABEL_EMOTION_PROVIDER", "deepseek"),
            max_tokens=16, temperature=0.1,
            timeout=int(_emotion_cfg("timeout_s", default=5)),
        )
        text = (resp.content or "").strip()
        picked = [t for t in vocab if t in text]
        if picked:
            return picked[:max_tags]
    except Exception as exc:
        logger.debug("emotional_memory: label LLM failed (%s)", exc)
    # 回退: 规则
    return _classify_tags_from_pad(*new_pad)[:max_tags]


def maybe_label_emotion(
    old_pad: tuple[float, float, float],
    new_pad: tuple[float, float, float],
    user_text: str,
) -> Optional[list[str]]:
    """每轮对话结束后为对话打情感标签并落 v5.db (fire-and-forget 入口).

    设计: 不阻塞回复. 调用方用 threading.Thread(daemon=True) 包一层.
    返回打到的标签列表 (失败返 None).
    """
    tags = label_emotion(user_text, new_pad)
    if not tags:
        return None
    try:
        from v5 import store as store
        store.store(
            content=f"情感标签: {('、'.join(tags))}",
            type="emotion_label",
            weight=0.5,
            tags="v5,emotion," + ",".join(tags),
        )
        logger.info("emotional_memory: labeled %s", tags)
    except Exception as exc:
        logger.debug("emotional_memory: label store failed (%s)", exc)
    return tags


def search_by_emotion(emotion_tag: str, top_k: int = 5) -> list[dict]:
    """按情感标签检索历史上的情感记忆 (因果事件 / 标签记录).

    Args:
        emotion_tag: 标签词, 如 "开心"
        top_k: 返回条数

    Returns:
        list[dict]: [{id, content, type, weight, created}, ...]
    """
    out: list[dict] = []
    try:
        from v5 import store as store
        mems = store.list_all(type_filter="emotional_event", limit=200)
        mems += store.list_all(type_filter="emotion_label", limit=200)
        seen = set()
        for m in mems:
            if m.id in seen:
                continue
            hay = f"{m.content} {m.tags}"
            if emotion_tag in hay:
                seen.add(m.id)
                out.append({
                    "id": m.id, "content": m.content,
                    "type": m.type, "weight": m.weight,
                    "created": m.created,
                })
            if len(out) >= top_k:
                break
    except Exception as exc:
        logger.debug("emotional_memory: search_by_emotion failed (%s)", exc)
    return out


def build_emotion_diff_block() -> str:
    """情感对比注入: 当前情感与上次差异大时注一句 (spec 2.6 示例).

    返回 "" = 跳过. 5s 去重 + diff_threshold 门控.
    输出格式 (spec 4.2 统一): "\\n---\\n情感对比：\\n<内容>\\n"
    """
    if not _emotion_cfg("diff_inject", default=True):
        return ""
    try:
        import time as _t
        global _LAST_DIFF_INJECT
        _now = _t.time()
        if _now - _LAST_DIFF_INJECT < float(_emotion_cfg("debounce_seconds", default=5)):
            return ""
        _LAST_DIFF_INJECT = _now

        from v5.affect import AffectState
        st = AffectState.load().decay()
        p, a, d = st.pleasure, st.arousal, st.dominance
        cur_mood = _mood_from_pad(p, a, d)

        # 读上次状态
        last = {}
        try:
            if _EMOTION_STATE_PATH.is_file():
                last = _json.loads(_EMOTION_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            last = {}

        # 写回当前状态 (轨迹追踪, 每次都更)
        try:
            _EMOTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _EMOTION_STATE_PATH.write_text(
                _json.dumps({"p": p, "a": a, "d": d, "mood": cur_mood, "ts": _now},
                            ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass

        if not last:
            return ""
        lp, la, ld = float(last.get("p", 0.0)), float(last.get("a", 0.0)), float(last.get("d", 0.0))
        delta = abs(p - lp) + abs(a - la) + abs(d - ld)
        if delta < float(_emotion_cfg("diff_threshold", default=0.3)):
            return ""
        if cur_mood == last.get("mood"):
            return ""

        # 方向性句子
        if p - lp > 0.1:
            line = "今天比刚才开心了不少呢。"
        elif p - lp < -0.1:
            line = "刚才那股开心劲儿好像散了点。"
        elif a - la > 0.1:
            line = "现在感觉比刚才更精神了些。"
        else:
            line = f"心情从「{last.get('mood','平静')}」变成了「{cur_mood}」。"
        return f"\n---\n情感对比：\n{line}"
    except Exception as exc:
        logger.debug("emotional_memory: diff block failed (%s)", exc)
        return ""


def build_emotion_recall_block(user_text: str) -> str:
    """用户显式提到情绪时, 用 search_by_emotion 拉一条旧记忆注一句 (功能性用检索)."""
    if not user_text:
        return ""
    hit = next((w for w in _EMOTION_LEXICON if w in user_text), None)
    if not hit:
        return ""
    try:
        mems = search_by_emotion(hit, top_k=1)
        if not mems:
            return ""
        c = mems[0]["content"].replace("\n", " ")[:40]
        return f"\n---\n情感回忆：\n哥哥以前「{hit}」的时候, 曾 {c}"
    except Exception:
        return ""


if __name__ == "__main__":
    print("label_emotion(test):", label_emotion("哥哥夸了我, 好开心", (0.5, 0.3, 0.2)))
    print("search_by_emotion(开心):", len(search_by_emotion("开心")))
    print("build_emotion_diff_block():", repr(build_emotion_diff_block()))
