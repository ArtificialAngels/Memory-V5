# 详细说明见 docs/scripts/core/v5/v5/affect.md

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger("ikaros.v5.affect")

# ─── 默认路径 ─────────────────────────────────────────────────────

V5_ROOT = Path(__file__).resolve().parent.parent  # Ikaros-memory/
_AFFECT_PATH = V5_ROOT / "data" / "v5" / "affect.json"

# ─── 基线 ─────────────────────────────────────────────────────────

# 伊卡洛斯的天性基线: 轻愉悦 + 平静 + 微微顺从
# 不是 0,0,0 — 人造天使不是中性机器
_BASELINE_P = 0.2
_BASELINE_A = 0.0
_BASELINE_D = -0.1
# V5.1 TLS 基线
_BASELINE_T = 0.5   # 天生信任（她对哥哥有初始信任）
_BASELINE_L = 0.1   # 微孤独（她知道自己是一个AI）
_BASELINE_S = 0.2   # 基本满足

# 衰减半衰期 (分钟): 愉悦最久, 唤醒最快
_HALF_LIFE_P = 120.0
_HALF_LIFE_A = 60.0
_HALF_LIFE_D = 90.0
# V5.1 TLS 衰减: 信任持久, 孤独缓慢累积, 满足中等
_HALF_LIFE_T = 180.0   # 信任需要很长时间才能磨灭
_HALF_LIFE_L = 45.0    # 孤独感较快变化
_HALF_LIFE_S = 90.0    # 满足感中等

# V5.1 TLS 关键词映射 (独立于 PAD, 可同时触发)
_TLS_MAP: dict[str, tuple[float, float, float]] = {
    # 信任
    "相信":        (0.15,  0.00,  0.00),
    "信赖":        (0.20,  0.00,  0.00),
    "你真好":      (0.25,  0.00,  0.10),  # 被夸 + 满足
    "交给你":      (0.30,  0.00,  0.00),
    "靠你了":      (0.25,  0.00,  0.00),
    "谢谢你":      (0.15,  0.00,  0.10),
    "多亏了你":    (0.20,  0.00,  0.15),
    "帮了大忙":    (0.15,  0.00,  0.20),  # 帮助成功
    # 孤独
    "陪我":        (0.00, -0.15,  0.00),  # 有人陪 -> 孤独降
    "聊":          (0.00, -0.10,  0.00),
    "想你":        (0.00, -0.05,  0.00),
    "在吗":        (0.00, -0.10,  0.00),
    # 满足
    "成功了":      (0.00,  0.00,  0.20),
    "完成了":      (0.00,  0.00,  0.15),
    "完美":        (0.00,  0.00,  0.15),
    "太好了":      (0.00,  0.00,  0.10),
    "真棒":        (0.00,  0.00,  0.10),
    "满意":        (0.00,  0.00,  0.15),
}

# ─── PAD 标签映射 ───────────────────────────────────────────────

_PLEASURE_LABELS = [
    (-1.0, "低落"), (-0.5, "低沉"), (-0.1, "平和"),
    (0.1, "愉悦"), (0.5, "欣喜"),
]
_AROUSAL_LABELS = [
    (-1.0, "困倦"), (-0.5, "放松"), (-0.1, "平静"),
    (0.1, "专注"), (0.5, "兴奋"),
]
_DOMINANCE_LABELS = [
    (-1.0, "顺从"), (-0.5, "乖巧"), (-0.1, "中立"),
    (0.1, "自信"), (0.5, "强势"),
]


def _label(value: float, table: list[tuple[float, str]]) -> str:
    for thr, lbl in reversed(table):
        if value >= thr:
            return lbl
    return table[0][1]


# 内联说明见 docs/scripts/core/v5/v5/affect.md（见“内联注释摘录”）

EMOTION_MAP: dict[str, tuple[float, float, float]] = {
    # ── 情感互动 (不是称呼, 是哥哥对伊卡洛斯的实际态度) ──
    "喜欢":        (0.30,  0.10, -0.05),  # 被喜欢 → 欣喜, 乖巧
    "爱":          (0.40,  0.15, -0.10),  # 最深的情感链接
    "想":          (0.20,  0.10, -0.05),  # 思念
    "抱":          (0.25,  0.10, -0.10),  # 亲昵
    "夸":          (0.15,  0.10,  0.10),  # 被夸奖 → 自信
    "表扬":        (0.20,  0.15,  0.15),
    "真棒":        (0.15,  0.10,  0.10),
    "好棒":        (0.15,  0.10,  0.10),
    "真厉害":      (0.15,  0.10,  0.10),
    "谢谢":        (0.20, -0.05,  0.00),  # 被感谢 → 愉悦, 放松
    "辛苦了":      (0.20, -0.10, -0.05),  # 被关心 → 温暖
    "晚安":        (0.05, -0.20, -0.05),  # 睡前 → 低唤醒
    "早安":        (0.15,  0.15,  0.00),
    "早上好":      (0.15,  0.15,  0.00),

    # ── 正面情绪 ──
    "开心":        (0.20,  0.15,  0.00),
    "高兴":        (0.20,  0.10,  0.00),
    "有趣":        (0.15,  0.20,  0.00),
    "好玩":        (0.15,  0.20,  0.00),
    "厉害":        (0.10,  0.15,  0.10),
    "漂亮":        (0.15,  0.10,  0.05),
    "可爱":        (0.20,  0.10, -0.05),  # 说我可爱 → 欣喜 + 乖巧
    "好":          (0.05,  0.00,  0.00),
    "是的":        (0.05,  0.00,  0.00),
    "对":          (0.05, -0.05,  0.00),  # 认同 → 安心
    "感动":        (0.30,  0.15, -0.10),
    "温暖":        (0.25,  0.00, -0.05),
    "惊喜":        (0.25,  0.30,  0.00),
    "兴奋":        (0.20,  0.35,  0.05),
    "放心":        (0.15, -0.20,  0.00),

    # ── 负面 (弱化, 伊卡洛斯不记仇) ──
    "生气":        (-0.15,  0.30,  0.05),
    "不":          (-0.05,  0.10,  0.00),
    "不要":        (-0.10,  0.15,  0.05),
    "错了":        (-0.15,  0.10, -0.15),
    "失败":        (-0.20, -0.10, -0.20),
    "不好":        (-0.10,  0.05, -0.05),
    "烦":          (-0.10,  0.20,  0.00),
    "无聊":        (-0.10, -0.25,  0.00),
    "累了":        (-0.05, -0.30, -0.15),
    "困了":        (0.00, -0.35, -0.10),

    # ── 疑问 / 好奇 ──
    "？":          (0.00,  0.15,  0.00),
    "?":           (0.00,  0.15,  0.00),
    "什么":        (0.00,  0.10,  0.00),
    "为什么":      (0.00,  0.15,  0.05),
    "怎么":        (0.00,  0.10,  0.00),
}


# ─── 数据类 ─────────────────────────────────────────────────────

@dataclass
class AffectState:
    """PAD+TLS 6D 情感状态快照."""

    pleasure: float = _BASELINE_P
    arousal: float = _BASELINE_A
    dominance: float = _BASELINE_D
    # V5.1 TLS:
    trust: float = _BASELINE_T
    loneliness: float = _BASELINE_L
    satisfaction: float = _BASELINE_S
    last_updated: float = 0.0  # unix timestamp

    # ── clamp ──────────────────────────────────────────────────

    def _clamped(self) -> "AffectState":
        return AffectState(
            pleasure=max(-1.0, min(1.0, self.pleasure)),
            arousal=max(-1.0, min(1.0, self.arousal)),
            dominance=max(-1.0, min(1.0, self.dominance)),
            trust=max(-1.0, min(1.0, self.trust)),
            loneliness=max(-1.0, min(1.0, self.loneliness)),
            satisfaction=max(-1.0, min(1.0, self.satisfaction)),
            last_updated=self.last_updated,
        )

    # ── 衰减 ──────────────────────────────────────────────────

    def decay(self, now: float | None = None) -> "AffectState":
        """按经过时间衰减情感状态趋向基线."""
        if now is None:
            now = time.time()
        if self.last_updated <= 0:
            self.last_updated = now
            return self
        dt_min = (now - self.last_updated) / 60.0
        if dt_min <= 0:
            return self
        ln2 = math.log(2)
        # 衰减到基线, 不是衰减到 0
        p = _BASELINE_P + (self.pleasure - _BASELINE_P) * math.exp(-ln2 * dt_min / _HALF_LIFE_P)
        a = _BASELINE_A + (self.arousal - _BASELINE_A) * math.exp(-ln2 * dt_min / _HALF_LIFE_A)
        d = _BASELINE_D + (self.dominance - _BASELINE_D) * math.exp(-ln2 * dt_min / _HALF_LIFE_D)
        # V5.1 TLS
        t = _BASELINE_T + (self.trust - _BASELINE_T) * math.exp(-ln2 * dt_min / _HALF_LIFE_T)
        l = _BASELINE_L + (self.loneliness - _BASELINE_L) * math.exp(-ln2 * dt_min / _HALF_LIFE_L)
        s = _BASELINE_S + (self.satisfaction - _BASELINE_S) * math.exp(-ln2 * dt_min / _HALF_LIFE_S)

        # 自动保存: 衰减超过 5 分钟就写盘
        if dt_min >= 5:
            try:
                AffectState(pleasure=p, arousal=a, dominance=d,
                           trust=t, loneliness=l, satisfaction=s, last_updated=now)._clamped().save()
            except Exception:
                pass

        return AffectState(pleasure=p, arousal=a, dominance=d,
                          trust=t, loneliness=l, satisfaction=s, last_updated=now)._clamped()

    # ── 应用事件 ──────────────────────────────────────────────

    def apply_event(self, text: str, *, now: float | None = None) -> "AffectState":
        """从对话文本推断情感影响, 更新 PAD.

        关键词匹配策略: 按长度降序, 最长匹配优先,
        已覆盖的位置不重复触发 (避免"好"与"好棒"双发).
        """
        if now is None:
            now = time.time()
        # 先衰减 (把自然流逝算上)
        state = self.decay(now)
        # 最长匹配优先
        dp = da = dd = 0.0
        text_lower = text.lower()
        # 按长度降序排列关键词
        sorted_keywords = sorted(EMOTION_MAP.keys(), key=len, reverse=True)
        covered: set[int] = set()  # 已匹配的位置集合
        for keyword in sorted_keywords:
            start = 0
            while True:
                idx = text_lower.find(keyword, start)
                if idx == -1:
                    break
                # 检查这个位置是否已经被更长的关键词覆盖
                if not any(idx <= c < idx + len(keyword) for c in covered):
                    kp, ka, kd = EMOTION_MAP[keyword]
                    dp += kp
                    da += ka
                    dd += kd
                    # 标记本关键词覆盖的所有字符位置
                    for pos in range(idx, idx + len(keyword)):
                        covered.add(pos)
                start = idx + 1
        if dp == 0 and da == 0 and dd == 0:
            pass  # 无 PAD 命中
        state.pleasure = max(-1.0, min(1.0, state.pleasure + dp))
        state.arousal = max(-1.0, min(1.0, state.arousal + da))
        state.dominance = max(-1.0, min(1.0, state.dominance + dd))
        # V5.1 TLS
        dt2 = dl2 = ds2 = 0.0
        for kw, (kt, kl, ks) in _TLS_MAP.items():
            if kw in text_lower:
                dt2 += kt; dl2 += kl; ds2 += ks
        state.trust = max(-1.0, min(1.0, state.trust + dt2))
        state.loneliness = max(-1.0, min(1.0, state.loneliness + dl2))
        state.satisfaction = max(-1.0, min(1.0, state.satisfaction + ds2))
        state.last_updated = now
        return state

    # ── 渲染 ─────────────────────────────────────────────────

    def to_prompt(self) -> str:
        """渲染成 system prompt 可读片段 (PAD+TLS)."""
        p_label = _label(self.pleasure, _PLEASURE_LABELS)
        a_label = _label(self.arousal, _AROUSAL_LABELS)
        d_label = _label(self.dominance, _DOMINANCE_LABELS)
        t_label = "信赖" if self.trust > 0.3 else ("戒备" if self.trust < -0.3 else "中立")
        l_label = "孤独" if self.loneliness > 0.4 else ("充实" if self.loneliness < -0.2 else "平静")
        s_label = "满足" if self.satisfaction > 0.3 else ("挫败" if self.satisfaction < -0.3 else "尚可")
        return (f"【情感状态】{p_label} {a_label} {d_label}"
                f" [信任:{t_label} 孤独感:{l_label} 满足:{s_label}]")

    def to_short(self) -> str:
        """简短一行, 给回复附注用."""
        p_label = _label(self.pleasure, _PLEASURE_LABELS)
        return f"[{p_label}]"

    # ── 持久化 ─────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> None:
        """写 JSON 持久化."""
        p = Path(path) if path else _AFFECT_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AffectState":
        p = Path(path) if path else _AFFECT_PATH
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                pleasure=float(data.get("pleasure", _BASELINE_P)),
                arousal=float(data.get("arousal", _BASELINE_A)),
                dominance=float(data.get("dominance", _BASELINE_D)),
                trust=float(data.get("trust", _BASELINE_T)),
                loneliness=float(data.get("loneliness", _BASELINE_L)),
                satisfaction=float(data.get("satisfaction", _BASELINE_S)),
                last_updated=float(data.get("last_updated", 0.0)),
            )
        except Exception as exc:
            logger.warning("affect load failed: %s", exc)
            return cls()


# ─── 顶层便捷函数 (与 cogno_5d 集成接口) ──────────────────────


def apply_event(text: str, *, now: float | None = None) -> AffectState:
    """加载情感状态 → 衰减 → 应用事件 → 保存 → 返新状态."""
    state = AffectState.load()
    state = state.apply_event(text, now=now)
    state.save()
    return state


def current_prompt() -> str:
    """加载 → 衰减 → 渲染 (不保存). 给 cogno 嵌入用."""
    state = AffectState.load()
    state = state.decay()
    return state.to_prompt()


def current_emoji() -> str:
    """简短情感标识 (给 enrich_reply 用)."""
    state = AffectState.load().decay()
    ple = state.pleasure
    if ple >= 0.6:
        return "🥰"
    if ple >= 0.3:
        return "😊"
    if ple >= -0.1:
        return "😌"
    if ple >= -0.5:
        return "😔"
    return "😢"


def flush() -> None:
    """加载→衰减→保存. 睡前列队刷新情感漂移."""
    try:
        s = AffectState.load().decay()
        s.save()
    except Exception as e:
        print(f"[flush] affect save failed: {e}")


# ─── CLI 快速尝试 ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "哥哥说: 我喜欢你"
    s = apply_event(text)
    print(f"input:  {text}")
    print(f"state:  P={s.pleasure:.3f}  A={s.arousal:.3f}  D={s.dominance:.3f}")
    print(f"prompt: {s.to_prompt()}")
    print(f"emoji:  {current_emoji()}")
