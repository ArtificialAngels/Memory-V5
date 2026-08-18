"""情境锚 (Context Anchor) — 记忆系统的时间/活动感知层 (Phase 1, 2026-08-14).

问题: 记忆系统此前"无感知"——不知道现在几点、周几、在做什么任务、在哪个会话,
写入/检索都没有时间锚点。

本模块提供统一的 now_context() 情境锚, 复用 cogno_5d 现成逻辑
(时间叙事 / 作息活动推断 / 前台窗口活动感知), 不重复造轮子。
供:
  - Phase 1: store.upsert() 写策略 (时间戳 / 相似合并决策)
  - Phase 2: 召回决策 (should_recall 按情境判断要不要翻记忆)
  - Phase 3: 时间锚定检索 (按 now 加权 / temporal 过滤)

用法:
    from v5.context_anchor import now_context
    ctx = now_context()   # {epoch, time_str, weekday, time_narrative, activity, window}
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("ikaros.v5.context_anchor")

# 惰性导入 cogno_5d (它依赖 ctypes 窗口 API, 沙箱/无桌面环境会失败, 全降级)
_cogno: Optional[object] = None


def _get_cogno():
    global _cogno
    if _cogno is None:
        try:
            from v5 import cogno_5d as _cogno
        except Exception as exc:  # pragma: no cover
            logger.debug("context_anchor: cogno_5d 不可用 (%s)", exc)
            _cogno = False
    return _cogno or None


def now_epoch() -> float:
    """当前 Unix 时间戳 (所有时间决策的统一锚)."""
    return time.time()


def time_str() -> str:
    """'2026/8/14 23:05' — 有 cogno_5d 用它的, 否则本地."""
    cog = _get_cogno()
    if cog and hasattr(cog, "get_time_str"):
        try:
            return cog.get_time_str()
        except Exception:
            pass
    return datetime.now().strftime("%Y/%m/%d %H:%M")


def weekday_str() -> str:
    """中文星期 (周一..周日)."""
    cog = _get_cogno()
    if cog and hasattr(cog, "get_weekday_str"):
        try:
            return cog.get_weekday_str(datetime.now().weekday())
        except Exception:
            pass
    return ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][datetime.now().weekday()]


def time_narrative() -> str:
    """压缩时间叙事 '周六（23:30）'."""
    cog = _get_cogno()
    if cog and hasattr(cog, "_get_time_narrative"):
        try:
            return cog._get_time_narrative()
        except Exception:
            pass
    now = datetime.now()
    return f"{weekday_str()}（{now.hour:02d}:{now.minute:02d}）"


def activity() -> str:
    """当前活动推断: 优先前台窗口真实感知, 降级作息推断.

    返回如 '在写代码' / '在浏览网页' / '哥哥的活动时间不太确定'。
    """
    cog = _get_cogno()
    if cog:
        # 1) 前台窗口标题 → 活动 (真实感知)
        try:
            title = cog._get_foreground_window_title()
            if title and hasattr(cog, "_match_activity"):
                act = cog._match_activity(title)
                if act:
                    return act
        except Exception:
            pass
        # 2) 作息推断
        try:
            return cog.infer_activity(datetime.now().hour, datetime.now().weekday())
        except Exception:
            pass
    hour = datetime.now().hour
    if 23 <= hour or hour < 6:
        return "深夜"
    if hour < 12:
        return "上午"
    if hour < 14:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


def foreground_window() -> str:
    """前台窗口标题 (best-effort, 失败返回空串)."""
    cog = _get_cogno()
    if cog and hasattr(cog, "_get_foreground_window_title"):
        try:
            return cog._get_foreground_window_title() or ""
        except Exception:
            pass
    return ""


def now_context() -> dict:
    """统一情境锚: 时间 + 活动 + 窗口. Phase 2/3 的召回/检索决策输入."""
    return {
        "epoch": now_epoch(),
        "time_str": time_str(),
        "weekday": weekday_str(),
        "time_narrative": time_narrative(),
        "activity": activity(),
        "window": foreground_window(),
        # "session_id" 由调用方 (对话层) 注入, 记忆层不假设
    }


# ────────────────────────────────────────────────────────────────────────────
# Phase 2: 召回决策 (should_recall) — "什么时候该调用记忆"
# 问题: 插件 on_pre_compress 每轮无条件翻记忆注入上下文, 寒暄/琐碎也翻,
#       浪费 token 且注入无关噪声。
# 方案: 线索词必召回; 寒暄/短琐碎跳过; 实质内容 (>= min_len) 召回。
# ────────────────────────────────────────────────────────────────────────────

# 明确需要翻记忆的线索 (提到过去/相关话题/回顾)
RECALL_CUES = (
    "记得", "上次", "之前", "回顾", "回忆", "我们聊过", "聊过", "关于", "最近",
    "查一下", "你记得", "之前说的", "上次说的", "提到", "说过",
    "remember", "recall", "earlier", "before", "previously", "what about",
    "you said", "last time", "do you remember",
)
# 寒暄/琐碎开头 (短消息跳过召回)
TRIVIAL_STARTS = (
    "你好", "hello", "hi", "嗨", "早上好", "下午好", "晚上好", "晚安",
    "谢谢", "感谢", "好的", "ok", "嗯", "在吗", "拜拜", "再见", "辛苦了",
)
# 情境里可信的活动 (未来可用于活动偏置检索; 现仅作上下文)
RECALL_MIN_LEN = 8  # 用户消息达到此长度才视为实质内容


def should_recall(user_text: str, *, min_len: int = RECALL_MIN_LEN) -> bool:
    """判断当前这轮对话是否值得翻记忆 (Phase 2).

    规则 (任一命中即召回):
      1. 含明确线索词 (记得/上次/回顾/关于/最近/remember...) → 必召回
      2. 寒暄/琐碎开头且消息短 (< 20 字) → 跳过 (你好/谢谢/晚安/ok...)
      3. 实质内容 (>= min_len 字) → 召回
    注: 线索词优先于寒暄判定 ("你好，你还记得上次聊的X吗" → 召回)。
    """
    t = (user_text or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(cue in low for cue in RECALL_CUES):
        return True
    if len(t) < 20 and any(t.startswith(s) for s in TRIVIAL_STARTS):
        return False
    return len(t) >= min_len
