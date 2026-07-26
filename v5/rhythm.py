# 详细说明见 docs/scripts/core/v5/v5/rhythm.md
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("ikaros.v5.rhythm")

# 本地时区偏移 (小时); 默认中国 UTC+8, 可由 env IKAROS_TZ_OFFSET 覆盖
_TZ_OFFSET = int(__import__("os").environ.get("IKAROS_TZ_OFFSET", "8"))


def _period_label(hour: int) -> str:
    """按小时返回时段标签 (spec 2.2 调性表的时段维度)."""
    if 0 <= hour < 5:
        return "深夜"
    if 5 <= hour < 8:
        return "清晨"
    if 8 <= hour < 11:
        return "上午"
    if 11 <= hour < 13:
        return "中午"
    if 13 <= hour < 18:
        return "下午"
    if 18 <= hour < 22:
        return "晚上"
    return "深夜"


def last_interaction_ts() -> float:
    """取 v5.db 最近一条记忆的 created (Unix epoch, UTC).

    近似 "上轮对话时间". 失败时返 0.0 (调用方按首轮处理).
    """
    try:
        from v5 import store
        mems = store.list_all(limit=1)
        if mems:
            return float(mems[0].created)
    except Exception as e:
        logger.debug("rhythm.last_interaction_ts failed: %s", e)
    return 0.0


def build_rhythm_block() -> str:
    """返回注入 system prompt 的节奏块. 空字符串 = 跳过 (由调用方兜底)."""
    now = time.time()
    last = last_interaction_ts()
    if last <= 0:
        last = now
    gap = max(0.0, now - last)

    # 用 gmtime 把 (epoch + 时区偏移) 解析为本地钟面时间;
    # 不能用 localtime(会再叠加系统本地时区, 导致 UTC+16 深夜误判)。
    lt = time.gmtime(now + _TZ_OFFSET * 3600)
    hour, minute = lt.tm_hour, lt.tm_min
    period = _period_label(hour)

    if gap < 60:
        gap_str = "刚刚"
    elif gap < 3600:
        gap_str = f"{int(gap // 60)}分钟"
    elif gap < 86400:
        h = int(gap // 3600)
        m = int((gap % 3600) // 60)
        gap_str = f"{h}小时{m}分" if m else f"{h}小时"
    else:
        d = int(gap // 86400)
        gap_str = f"{d}天"

    return (
        "\n---\n当前节奏：\n"
        f"距上轮: {gap_str} | 时段: {period}({hour:02d}:{minute:02d})"
    )


if __name__ == "__main__":
    print(build_rhythm_block())
