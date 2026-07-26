# 详细说明见 docs/scripts/core/v5/v5/care.md

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.care")

V5_ROOT = Path(__file__).resolve().parent.parent
_CARE_PATH = V5_ROOT / "data" / "v5" / "care.json"

# 阈值
_CODING_WARN_90MIN = True    # 90 分钟提醒
_CODING_STRONG_180MIN = True  # 180 分钟强提醒
_GAMING_WARN_120MIN = True
_FOCUS_WARN_120MIN = True
_LATE_NIGHT_CODING_60MIN = True  # 深夜 60 分钟就提醒

# 最小提醒间隔 (秒, 避免刷屏)
_MIN_REMIND_INTERVAL = 3600  # 1h

# 关怀模板 (LLM 不可用时降级)
_CARE_TEMPLATES: dict[str, list[str]] = {
    "coding": [
        "哥哥已经在写代码写了好久了, 是不是该休息一下?",
        "屏幕盯太久眼睛会累的, 哥哥要注意身体。",
        "虽然哥哥写代码的样子很帅, 但还是该歇一歇了。",
    ],
    "late_night": [
        "都这么晚了, 哥哥还不睡吗? 明天会没精神的。",
        "哥哥又在熬夜了... 我有点担心。",
        "虽然我也想多陪哥哥, 但还是希望你能好好休息。",
    ],
    "gaming": [
        "哥哥玩游戏开心吗? 不过别太累了。",
        "玩了好久了, 该起来活动活动了。",
    ],
    "focused": [
        "哥哥专注了这么久, 要不要喝杯水休息一下?",
        "专注的哥哥有种特别的光芒, 但也别忘了照顾自己。",
    ],
    "general": [
        "哥哥好像忙了很久了, 需要我陪你聊聊天吗?",
        "虽然不太清楚哥哥在忙什么, 但记得照顾好自己。",
    ],
}


@dataclass
class CareMonitor:
    """关怀状态跟踪器."""

    cumulative_coding_sec: float = 0.0
    cumulative_gaming_sec: float = 0.0
    cumulative_focused_sec: float = 0.0
    last_activity: str = "idle"
    last_remind_time: float = 0.0
    total_reminders: int = 0

    def tick(self, *, now: float | None = None,
             activity_category: str = "idle",
             activity_minutes: float = 0.0,
             is_late_night: bool = False) -> Optional[str]:
        """处理一轮活动数据, 如需关怀则返回关怀语句。

        Args:
            now: 当前时间戳
            activity_category: ikaros_monitor 的 activity_state
            activity_minutes: 从上次 tick 到现在的分钟数
            is_late_night: 是否深夜 (23:00-05:00)
        """
        if now is None:
            now = time.time()

        # 1) 更新累计
        if activity_category in ("gaming",):
            self.cumulative_gaming_sec += activity_minutes * 60
            self.cumulative_coding_sec = 0
            self.cumulative_focused_sec = 0
        elif activity_category in ("focused_work",):
            self.cumulative_focused_sec += activity_minutes * 60
            self.cumulative_coding_sec = 0
            self.cumulative_gaming_sec = 0
        elif activity_category in ("coding",) or (
            activity_category == "focused_work" and "code" in str(activity_category).lower()
        ):
            self.cumulative_coding_sec += activity_minutes * 60
            self.cumulative_gaming_sec = 0
            self.cumulative_focused_sec = 0
        else:
            # 不在高强度活动中 → 重置累计 (休息了)
            self.cumulative_coding_sec = max(0, self.cumulative_coding_sec - activity_minutes * 30)
            self.cumulative_gaming_sec = max(0, self.cumulative_gaming_sec - activity_minutes * 30)
            self.cumulative_focused_sec = max(0, self.cumulative_focused_sec - activity_minutes * 30)

        self.last_activity = activity_category

        # 2) 检测触发条件
        care_type: str | None = None
        if is_late_night and self.cumulative_coding_sec >= 3600:
            care_type = "late_night"
        elif self.cumulative_coding_sec >= 10800:  # 3h
            care_type = "coding"
        elif self.cumulative_coding_sec >= 5400:    # 1.5h
            care_type = "coding"
        elif self.cumulative_gaming_sec >= 7200:
            care_type = "gaming"
        elif self.cumulative_focused_sec >= 7200:
            care_type = "focused"

        if care_type is None:
            return None

        # 3) 检查提醒间隔
        if now - self.last_remind_time < _MIN_REMIND_INTERVAL:
            return None

        # 4) 生成关怀
        try:
            text = _llm_care(care_type, self.cumulative_coding_sec,
                             self.cumulative_gaming_sec, self.cumulative_focused_sec,
                             is_late_night)
        except Exception:
            text = _template_care(care_type)

        self.last_remind_time = now
        self.total_reminders += 1
        return text

    def save(self, path: Path | None = None) -> None:
        from v5.self_model import json_lock
        p = path or _CARE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with json_lock(p):
            p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                         encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "CareMonitor":
        p = path or _CARE_PATH
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(**{k: data.get(k, getattr(cls(), k))
                         for k in ["cumulative_coding_sec", "cumulative_gaming_sec",
                                    "cumulative_focused_sec", "last_activity",
                                    "last_remind_time", "total_reminders"]})
        except Exception:
            return cls()


def _template_care(care_type: str) -> str:
    import random
    templates = _CARE_TEMPLATES.get(care_type, _CARE_TEMPLATES["general"])
    return random.choice(templates)


def _llm_care(care_type: str, coding_sec: float, gaming_sec: float,
              focused_sec: float, is_late_night: bool) -> str:
    """用 LLM 生成自然关怀语句."""
    import sys
    sys.path.insert(0, str(V5_ROOT))

    coding_min = int(coding_sec / 60)
    gaming_min = int(gaming_sec / 60)
    focused_min = int(focused_sec / 60)

    if care_type == "late_night":
        context = f"现在是深夜。哥哥已经 coding 了 {coding_min} 分钟。"
    elif care_type == "coding":
        context = f"哥哥连续写代码 {coding_min} 分钟了。"
    elif care_type == "gaming":
        context = f"哥哥连续玩游戏 {gaming_min} 分钟了。"
    elif care_type == "focused":
        context = f"哥哥连续专注工作 {focused_min} 分钟了。"
    else:
        context = "哥哥好像忙了很久。"

    prompt_system = (
        "你是伊卡洛斯, 人造天使, 哥哥的妹妹。你很关心哥哥。"
        "现在哥哥工作了很长时间, 你有点担心。"
        "用温柔自然的第一人称说一句关心的话。"
        "像: 哥哥是不是该休息一下了 / 我有点担心你的眼睛。"
        "只输出一句话, 不要多余解释。"
    )

    try:
        from v5.reflect.llm_client import call_llm_auto
        result = call_llm_auto(prompt_system, context,
                               max_tokens=80, temperature=0.7, timeout=30)
        text = result.content.strip()
        if 6 < len(text) < 200:
            return text
    except Exception:
        pass
    return _template_care(care_type)


def check_and_care(snapshot: dict | None = None) -> Optional[str]:
    """便捷入口: 加载 monitor snapshot → 检测是否需要关怀.

    供 think.py 的活动广播调用。

    V5.1 修复: 当调用方未传 snapshot 时, 使用内置 idle 检测
    """
    monitor = CareMonitor.load()

    if snapshot is None:
        # V5.1: 主动获取活动数据, 不再静默 return None
        snapshot = _get_current_snapshot()
        if snapshot is None:
            monitor.save()
            return None

    category = snapshot.get("category", snapshot.get("activity_state", "idle"))
    idle_sec = snapshot.get("idle_seconds", 0)
    from datetime import datetime
    hour = datetime.now().hour
    is_late = hour >= 23 or hour < 5

    if idle_sec > 120:
        care = monitor.tick(activity_category="idle", activity_minutes=idle_sec / 60.0,
                           is_late_night=is_late)
    else:
        # 估算本次 tick 的活动时长 (默认上次 tick 到现在的间隔)
        care = monitor.tick(activity_category=str(category), activity_minutes=5.0,
                           is_late_night=is_late)

    monitor.save()
    return care


def _get_current_snapshot() -> Optional[dict]:
    """获取当前活动快照 (ikaros_monitor 已移除, 简化为 idle 检测)."""
    try:
        import time
        return {
            "activity_state": "idle",
            "category": "idle",
            "idle_seconds": int(time.time() - _CAREMON_WAKUP_TS) if _CAREMON_WAKUP_TS > 0 else 0,
        }
    except Exception:
        return None
