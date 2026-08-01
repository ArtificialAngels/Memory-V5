# 详细说明见 docs/scripts/core/v5/v5/relationship.md

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger("ikaros.v5.relationship")

V5_ROOT = Path(__file__).resolve().parent
_REL_PATH = V5_ROOT / "data" / "v5" / "relationship.json"

# 参数
_DECAY_HALF_LIFE_DAYS = 14.0     # 14 天无互动, 亲密度减半
_WARMTH_EMA_ALPHA = 0.15          # 温暖度平滑系数
_DEPTH_GAIN_PER_INTERACTION = 0.008  # 每次对话微量增加深度
_WARM_BOOST_THRESHOLD = 0.6       # warmth > 此值加速深度增长

# 阶段标签
_STAGE_LABELS = [
    (0.0,  "才刚认识不久"),
    (0.2,  "还在了解彼此"),
    (0.4,  "已经很亲近了"),
    (0.6,  "像家人一样"),
    (0.8,  "最了解哥哥的人"),
]


def _label(val: float, table: list[tuple[float, str]]) -> str:
    for thr, lbl in reversed(table):
        if val >= thr:
            return lbl
    return table[0][1]


@dataclass
class Relationship:
    """伊卡洛斯与哥哥的关系状态."""

    depth: float = 0.15              # 0~1 关系深度
    warmth: float = 0.3              # 0~1 对话温暖度 (EMA)
    shared_experiences: int = 0       # 共享经验数
    interaction_count: int = 0        # 累计互动次数
    first_interaction: float = 0.0   # 第一次对话时间戳
    last_interaction: float = 0.0    # 上次对话时间戳
    peak_closeness: float = 0.0      # 历史最高亲密度

    def record_interaction(
        self,
        affect_intensity: float,       # 当次对话 PAD 情感强度 (0~1)
        shared_count: int = 0,         # 本次对话涉及的新共享记忆数
        *,
        now: float | None = None,
    ) -> "Relationship":
        """记录一次互动, 更新关系参数."""
        if now is None:
            now = time.time()

        if self.first_interaction <= 0:
            self.first_interaction = now

        # 1) 时间衰减: 对数衰减防止"几天没聊就变陌生人"的尴尬
        gap_days = 0.0
        if self.last_interaction > 0:
            gap_days = (now - self.last_interaction) / 86400.0
            if gap_days > 0:
                ln2 = math.log(2)
                decay = math.exp(-ln2 * gap_days / _DECAY_HALF_LIFE_DAYS)
                self.depth = max(0.05, self.depth * (0.7 + 0.3 * decay))
                self.warmth *= max(0.1, decay)

        # 2) 温暖度 EMA
        self.warmth = (
            (1 - _WARMTH_EMA_ALPHA) * self.warmth
            + _WARMTH_EMA_ALPHA * affect_intensity
        )

        # 3) 深度增益
        boost = _DEPTH_GAIN_PER_INTERACTION
        if self.warmth > _WARM_BOOST_THRESHOLD:
            boost *= 1.5  # 温暖对话加速关系增长
        if gap_days > 3 and self.last_interaction > 0:
            boost *= 1.3  # 久别重逢也加速 (情感补偿)
        self.depth = min(0.95, self.depth + boost)

        # 4) 共享经验
        self.shared_experiences += shared_count if shared_count > 0 else 1

        # 5) 统计
        self.interaction_count += 1
        self.last_interaction = now
        self.peak_closeness = max(self.peak_closeness, self.depth)

        return self.clamped()

    def days_known(self, *, now: float | None = None) -> float:
        if self.first_interaction <= 0:
            return 0
        return ((now or time.time()) - self.first_interaction) / 86400.0

    def stage(self) -> str:
        return _label(self.depth, _STAGE_LABELS)

    def closeness(self) -> float:
        """综合亲密度: depth × warmth (两者都高才算真正亲密)."""
        return self.depth * (0.5 + 0.5 * self.warmth)

    def to_prompt(self) -> str:
        """注入 system prompt."""
        stage = self.stage()
        days = self.days_known()
        lines = [f"【与哥哥的关系】{stage}"]
        if days > 7:
            lines.append(f"（认识 {int(days)} 天了, 聊过 {self.interaction_count} 次）")
        return " ".join(lines)

    def clamped(self) -> "Relationship":
        return Relationship(
            depth=max(0.0, min(1.0, self.depth)),
            warmth=max(0.0, min(1.0, self.warmth)),
            shared_experiences=self.shared_experiences,
            interaction_count=self.interaction_count,
            first_interaction=self.first_interaction,
            last_interaction=self.last_interaction,
            peak_closeness=self.peak_closeness,
        )

    # -- 持久化 --

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else _REL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                     encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Relationship":
        p = Path(path) if path else _REL_PATH
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                depth=float(data.get("depth", 0.15)),
                warmth=float(data.get("warmth", 0.3)),
                shared_experiences=int(data.get("shared_experiences", 0)),
                interaction_count=int(data.get("interaction_count", 0)),
                first_interaction=float(data.get("first_interaction", 0)),
                last_interaction=float(data.get("last_interaction", 0)),
                peak_closeness=float(data.get("peak_closeness", 0)),
            )
        except Exception as exc:
            logger.warning("relationship load failed: %s", exc)
            return cls()


def relationship_prompt() -> str:
    """便捷: 加载 + 记录 + 渲染."""
    r = Relationship.load()
    try:
        from v5.affect import AffectState
        state = AffectState.load().decay()
        intensity = (abs(state.pleasure) + abs(state.arousal) + abs(state.dominance) * 0.5) / 2.0
    except Exception:
        intensity = 0.3
    r = r.record_interaction(intensity)
    r.save()
    return r.to_prompt()

# ─── V5.1 激活: 每轮对话后调用 ─────────────────

def track_interaction(intensity: float = 0.3) -> None:
    """cloud_chat 每轮对话后调用, 累积亲密度."""
    try:
        r = Relationship.load()
        # 粗略估算共享记忆数
        try:
            from v5 import store
            stats = store.stats()
            shared = stats.get("total", 0)
        except Exception:
            shared = 0
        r = r.record_interaction(intensity, shared)
        r.save()
    except Exception:
        pass
