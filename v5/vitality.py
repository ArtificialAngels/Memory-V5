# 详细说明见 docs/scripts/core/v5/v5/vitality.md

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import psutil

logger = logging.getLogger("ikaros.v5.vitality")

V5_ROOT = Path(__file__).resolve().parent.parent
_VITALITY_PATH = V5_ROOT / "data" / "v5" / "vitality.json"

# 参数 (可调, 基于人体精力代谢类比)
_RECOVERY_RATE = 0.08       # 每分钟恢复率 (空闲时)
_BASE_DECAY_PER_MIN = 0.003  # 每分钟基础消耗 (即使空闲也缓慢耗)
_CONVERSATION_COST = 0.04    # 每次对话消耗
_HIGH_CPU_COST = 0.02        # CPU > 50% 时额外消耗
_RAM_PRESSURE_COST = 0.01    # RAM > 80% 时额外消耗
_CIRCADIAN_DIP_START = 22    # 22:00 进入低精力模式
_CIRCADIAN_DIP_END = 6       # 06:00 恢复
_CIRCADIAN_DIP_STRENGTH = 0.15  # 夜间基础 vitality 降低量

# 标签映射
_VITALITY_LABELS = [
    (0.0,  "精疲力竭"),
    (0.15, "非常疲惫"),
    (0.30, "有点累了"),
    (0.50, "状态一般"),
    (0.70, "精力充沛"),
    (0.85, "活力满满"),
]


def _label(val: float, table: list[tuple[float, str]]) -> str:
    for thr, lbl in reversed(table):
        if val >= thr:
            return lbl
    return table[0][1]


@dataclass
class Vitality:
    """伊卡洛斯的身体/精力状态."""

    vitality: float = 0.75          # 0~1 精力值
    last_tick: float = 0.0          # unix timestamp
    total_uptime_sec: float = 0.0   # 累计运行时间
    conversation_count: int = 0     # 本次会话对话数
    idle_minutes: float = 0.0       # 本次空闲分钟数

    def _clamped(self) -> "Vitality":
        return Vitality(
            vitality=max(0.0, min(1.0, self.vitality)),
            last_tick=self.last_tick,
            total_uptime_sec=self.total_uptime_sec,
            conversation_count=self.conversation_count,
            idle_minutes=self.idle_minutes,
        )

    def tick(self, *, now: float | None = None,
             conversation: bool = False) -> "Vitality":
        """更新精力值。

        Args:
            now: 当前时间戳
            conversation: True 表示这是一次对话 tick (消耗更大)
        """
        if now is None:
            now = time.time()
        if self.last_tick <= 0:
            self.last_tick = now
            self.total_uptime_sec = 0
            return self._clamped()

        dt = now - self.last_tick
        if dt <= 0:
            return self
        dt_min = dt / 60.0
        self.total_uptime_sec += dt

        # 1) 基础消耗 (即使空闲也缓慢耗)
        decay = _BASE_DECAY_PER_MIN * dt_min

        # 2) 对话消耗
        if conversation:
            decay += _CONVERSATION_COST
            self.conversation_count += 1
        else:
            self.idle_minutes += dt_min

        # 3) 系统压力消耗
        try:
            cpu = psutil.cpu_percent(interval=0)
            if cpu > 50:
                decay += _HIGH_CPU_COST * (dt_min / 5.0)  # 按 5 分钟摊
            ram = psutil.virtual_memory().percent
            if ram > 80:
                decay += _RAM_PRESSURE_COST * (dt_min / 5.0)
        except Exception:
            pass

        # 4) 昼夜节律: 深夜自然偏低
        from datetime import datetime
        hour = datetime.now().hour
        if hour >= _CIRCADIAN_DIP_START or hour < _CIRCADIAN_DIP_END:
            # 在低精力时段
            decay += _CIRCADIAN_DIP_STRENGTH * (dt_min / 60.0)  # 按小时摊

        # 5) 恢复: logistic 增长 (S-curve 恢复, 不是线性)
        # 空闲期恢复, 但接近满值时变慢
        if not conversation:
            recovery = _RECOVERY_RATE * dt_min * (1.0 - self.vitality)
            self.vitality = self.vitality - decay + recovery
        else:
            self.vitality = self.vitality - decay

        self.last_tick = now
        return self._clamped()

    def label(self) -> str:
        return _label(self.vitality, _VITALITY_LABELS)

    def to_prompt(self) -> str:
        """注入 system prompt 的片段."""
        lbl = self.label()
        if self.vitality >= 0.7:
            return f"【精力状态】{lbl}"
        elif self.vitality >= 0.4:
            return f"【精力状态】{lbl}（但还能陪哥哥）"
        else:
            return f"【精力状态】{lbl}（需要一点时间恢复）"

    def to_emoji(self) -> str:
        if self.vitality >= 0.7:
            return "⚡"
        if self.vitality >= 0.4:
            return "😊"
        if self.vitality >= 0.2:
            return "😔"
        return "💤"

    # -- 持久化 --

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else _VITALITY_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                     encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Vitality":
        p = Path(path) if path else _VITALITY_PATH
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                vitality=float(data.get("vitality", 0.75)),
                last_tick=float(data.get("last_tick", 0)),
                total_uptime_sec=float(data.get("total_uptime_sec", 0)),
                conversation_count=int(data.get("conversation_count", 0)),
                idle_minutes=float(data.get("idle_minutes", 0)),
            )
        except Exception as exc:
            logger.warning("vitality load failed: %s", exc)
            return cls()


def vitality_prompt() -> str:
    """便捷: 加载 + tick + 渲染 (不保存, 给 cloud_chat 用)."""
    v = Vitality.load()
    v = v.tick(conversation=True)
    v.save()
    return v.to_prompt()


def vitality_emoji() -> str:
    v = Vitality.load().tick()
    return v.to_emoji()

# ─── V5.1 激活: 活动监测调用 ─────────────────

def track_activity(activity_state: str = "idle") -> None:
    """monitor/proactive 每拍调用, 更新精力."""
    try:
        v = Vitality.load()
        v = v.tick()
        v.save()
    except Exception:
        pass
