# 详细说明见 docs/scripts/core/v5/v5/proactive.md

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.proactive")

V5_ROOT = Path(__file__).resolve().parent
_STATE_PATH = V5_ROOT / "data" / "v5" / "proactive_state.json"

# 门控阈值
_GATE_IDLE_CHAT_SEC = 5 * 60      # 5min 无对话
_GATE_IDLE_AWAY_SEC = 5 * 60      # 5min 无活动 = away
_GATE_CURIOSITY = 0.4
_GATE_AROUSAL_MIN = -0.3          # 不能太困
_GATE_LONELINESS = 0.2            # 有一定倾诉欲


def _time_since_last_chat() -> float:
    """距上次对话的秒数."""
    try:
        st = json.loads(_STATE_PATH.read_text(encoding="utf-8")) if _STATE_PATH.is_file() else {}
        last = st.get("last_chat_ts", 0) or 0
        return time.time() - last if last else 99999.0
    except Exception:
        return 99999.0


def _time_since_last_speak() -> float:
    """距上次主动说话的秒数."""
    try:
        st = json.loads(_STATE_PATH.read_text(encoding="utf-8")) if _STATE_PATH.is_file() else {}
        last = st.get("last_speak_ts", 0) or 0
        return time.time() - last if last else 99999.0
    except Exception:
        return 99999.0


def _has_pending_task() -> bool:
    """检查是否有待执行任务."""
    task_path = V5_ROOT / "data" / "v5" / "task_pending.json"
    result_path = V5_ROOT / "data" / "v5" / "task_result.json"
    return task_path.is_file() or result_path.is_file()


def _get_activity_idle() -> float:
    """从 monitor 获取当前空闲秒数."""
    try:
        import sys
        sys.path.insert(0, str(V5_ROOT.parent))
        from services.monitor_adapter import get_current_idle
        return get_current_idle()
    except Exception:
        try:
            mp = V5_ROOT / "data" / "v5" / "monitor_snapshot.json"
            if mp.is_file():
                snap = json.loads(mp.read_text(encoding="utf-8"))
                return float(snap.get("idle_seconds", 0))
        except Exception:
            pass
    return 0.0


def mark_chat(now: float | None = None) -> None:
    """cloud_chat 每轮对话开始时调用."""
    if now is None:
        now = time.time()
    try:
        st = json.loads(_STATE_PATH.read_text(encoding="utf-8")) if _STATE_PATH.is_file() else {}
        st["last_chat_ts"] = now
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def should_speak() -> tuple[bool, str]:
    """决策: 伊卡洛斯现在是否应该主动说话?

    Returns:
        (should_speak, reason) — reason 为决策原因或拒绝原因
    """
    # 1. 空闲时间检查
    chat_idle = _time_since_last_chat()
    if chat_idle < _GATE_IDLE_CHAT_SEC:
        return False, f"刚聊过 ({chat_idle:.0f}s)"

    speak_idle = _time_since_last_speak()
    if speak_idle < 300:
        return False, f"刚说过话 ({speak_idle:.0f}s)"

    # 2. 无待执行任务
    if _has_pending_task():
        return False, "有待执行任务"

    # 3. 情感检查
    try:
        from v5.affect import AffectState
        state = AffectState.load().decay()
        if state.arousal < _GATE_AROUSAL_MIN:
            return False, f"太困 (A={state.arousal:+.2f})"
        if state.loneliness < _GATE_LONELINESS:
            return False, f"不孤独 (L={state.loneliness:+.2f})"
    except Exception:
        pass

    # 4. 好奇度检查
    try:
        from v5.self_model import SelfModel
        sm = SelfModel.load()
        if sm.get_curiosity() < _GATE_CURIOSITY:
            return False, f"好奇度不够 ({sm.get_curiosity():.2f})"
    except Exception:
        pass

    # 5. 哥哥是否离开
    activity_idle = _get_activity_idle()
    if activity_idle > _GATE_IDLE_AWAY_SEC:
        return False, f"哥哥离开了 ({activity_idle:.0f}s idle)"

    return True, "条件全部满足"


def is_user_away() -> bool:
    """检测哥哥是否离开了 (> 5min 无活动)."""
    return _get_activity_idle() > _GATE_IDLE_AWAY_SEC


def try_proactive() -> Optional[str]:
    """尝试主动搭话 (供 think 循环 metacog 节拍时调用).

    Returns:
        说话内容 string, 或 None (条件不满足).
    """
    ok, reason = should_speak()
    if not ok:
        logger.debug("proactive skip: %s", reason)
        return None

    try:
        from v5.metacog import surface_utterance
        u = surface_utterance()
        if not u:
            return None
        text = u.get("text", "")
        if not text or len(text) < 10:
            return None

        # 记录说话时间
        st = json.loads(_STATE_PATH.read_text(encoding="utf-8")) if _STATE_PATH.is_file() else {}
        st["last_speak_ts"] = time.time()
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")

        logger.info("proactive speak [self-determined]: %s", text[:60])
        return text
    except Exception as exc:
        logger.debug("proactive failed: %s", exc)
        return None


# ─── 记忆提醒调度器 (兼容 cloud_chat.py 旧接口) ───────────────────

import json as _json
from datetime import datetime as _dt

_SCHED_PATH = V5_ROOT / "data" / "v5" / "schedule.json"

class _TodoScheduler:
    """简单的待办/提醒调度器."""
    def __init__(self):
        self._items = []
        self._load()

    def _load(self):
        if _SCHED_PATH.is_file():
            try:
                self._items = _json.loads(_SCHED_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._items = []

    def _save(self):
        _SCHED_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SCHED_PATH.write_text(_json.dumps(self._items, ensure_ascii=False), encoding="utf-8")

    def remember_todo(self, text: str, due_ts: float = 0, kind: str = ""):
        self._load()
        self._items.append({"text": text, "due_ts": due_ts, "kind": kind, "created": time.time()})
        self._save()

_scheduler: _TodoScheduler | None = None

def get_scheduler() -> _TodoScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = _TodoScheduler()
    return _scheduler

def parse_remember_intent(text: str) -> dict | None:
    """解析 '记住/提醒我...' 意图."""
    import re as _re
    t = text.strip()
    # 匹配 "提醒我/记住/别忘了 ..."
    m = _re.match(r'(?:提醒我|记住|别忘了|帮我记一下)[：:\s]*(.+)', t)
    if not m:
        return None
    content = m.group(1).strip()
    if not content or len(content) < 2:
        return None
    due_ts = 0.0
    kind = "todo"
    # 解析时间: "明天3点" / "5分钟后" 等
    now = time.time()
    if "分钟后" in content:
        m2 = _re.search(r'(\d+)\s*分钟后', content)
        if m2:
            due_ts = now + int(m2.group(1)) * 60
            kind = "reminder"
    elif "小时后" in content:
        m2 = _re.search(r'(\d+)\s*小时后', content)
        if m2:
            due_ts = now + int(m2.group(1)) * 3600
            kind = "reminder"
    elif "明天" in content:
        due_ts = now + 86400
        kind = "reminder"
    elif "下周" in content:
        due_ts = now + 7 * 86400
        kind = "reminder"
    return {"text": content, "due_ts": due_ts, "kind": kind}

def fmt_due(due_ts: float) -> str:
    """格式化到期时间."""
    if due_ts <= 0:
        return "找机会"
    dt = _dt.fromtimestamp(due_ts)
    today = _dt.now().date()
    if dt.date() == today:
        return f"今天{dt.strftime('%H:%M')}"
    elif (dt.date() - today).days == 1:
        return f"明天{dt.strftime('%H:%M')}"
    return dt.strftime("%m月%d日 %H:%M")
