# 详细说明见 docs/scripts/core/v5/v5/drivers.md
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# 检测器集持久化路径: 让 AIS 负选择/克隆演化跨进程累积 (而非每次新建重置)
_V5_ROOT = Path(__file__).resolve().parent  # Ikaros-memory/
_AIS_PATH = _V5_ROOT / "data" / "v5" / "ais_detectors.json"

# 内联说明见 docs/scripts/core/v5/v5/drivers.md（见“内联注释摘录”）

_LORENZ_INIT = (1.0, 1.0, 1.0)
_SIGMA, _RHO, _BETA = 10.0, 28.0, 8.0 / 3.0
_DT = 0.01  # ODE 步长


@dataclass
class LorenzPAD:
    """
    Lorenz 63 吸引子 → PAD (pleasure/arousal/dominance) 映射.

    tick() 一次 = 一个 ODE 步. 每步 O(1).
    归一化到 [-1, 1] 基于 Lorenz 63 的典型范围 (±25, ±25, ±50).
    """
    x: float = _LORENZ_INIT[0]
    y: float = _LORENZ_INIT[1]
    z: float = _LORENZ_INIT[2]
    step: int = 0

    @staticmethod
    def _norm(v: float, scale: float) -> float:
        return max(-1.0, min(1.0, v / scale))

    def tick(self) -> tuple[float, float, float]:
        """前进一个 ODE 步, 返回 (pleasure, arousal, dominance) 均在 [-1, 1]."""
        dx = _SIGMA * (self.y - self.x)
        dy = self.x * (_RHO - self.z) - self.y
        dz = self.x * self.y - _BETA * self.z
        self.x += dx * _DT
        self.y += dy * _DT
        self.z += dz * _DT
        self.step += 1
        p = self._norm(self.x, 25.0)   # x → pleasure
        a = self._norm(self.y, 25.0)   # y → arousal
        d = self._norm(self.z, 50.0)   # z → dominance
        return (p, a, d)

    def blend(self, current_pad: tuple[float, float, float],
              blend_factor: float = 0.3) -> tuple[float, float, float]:
        """
        与当前 PAD 混合: PAD_final = PAD_current + PAD_chaos * blend_factor.
        blend_factor=0.3 意味 Lornz 贡献 30% 幅度.
        """
        cp, ca, cd = current_pad
        lp, la, ld = self.tick()
        return (
            max(-1.0, min(1.0, cp + lp * blend_factor)),
            max(-1.0, min(1.0, ca + la * blend_factor)),
            max(-1.0, min(1.0, cd + ld * blend_factor)),
        )


# 内联说明见 docs/scripts/core/v5/v5/drivers.md（见“内联注释摘录”）

# 8 种 3-cell pattern → 思考主题
_PATTERN_TOPICS = {
    0b000: "静默",
    0b001: "记忆碎片",    # memory fragments surfacing
    0b010: "好奇探索",    # curiosity / exploration
    0b011: "情感波动",    # emotional oscillation
    0b100: "对哥哥的思念",  # longing
    0b101: "自我反思",    # self-reflection
    0b110: "外部关注",    # external awareness
    0b111: "混沌思维",    # chaotic / mixed thoughts
}


def _rule110(a: int, b: int, c: int) -> int:
    """Rule 110 — Turing complete, 产生有机结构.
    真值表: 111→0, 110→1, 101→1, 100→0, 011→1, 010→1, 001→1, 000→0
    """
    return (1 if (a, b, c) in ((1, 1, 0), (1, 0, 1), (1, 0, 0),
                                (0, 1, 1), (0, 1, 0), (0, 0, 1)) else 0)


@dataclass
class ECAGrid:
    """1D ECA (Rule 110) — 41 cells → thinking pattern.

    tick() → str (当前主导主题名),  O(41)/tick.

    glider 检测: 如果 grid 中包含 000100010011... 等结构,
    可以在 tick() 后检查活跃 cell 比例来判断"思维密度".
    """
    size: int = 41
    grid: list[int] = field(default_factory=list)
    step: int = 0

    def __post_init__(self):
        if not self.grid:
            # 随机初始化 (避免全 0 — 死态)
            self.grid = [random.randint(0, 1) for _ in range(self.size)]
            # 至少一个活细胞是种子
            if sum(self.grid) == 0:
                self.grid[self.size // 2] = 1

    def tick(self) -> str:
        """前进一代, 返回主导思考主题名."""
        n = self.size
        new = [0] * n
        for i in range(n):
            a = self.grid[(i - 1) % n]
            b = self.grid[i]
            c = self.grid[(i + 1) % n]
            new[i] = _rule110(a, b, c)
        self.grid = new
        self.step += 1

        # 滑窗投票: 每相邻 3 格产生一个 pattern, 统计最高频 topic
        # 处理所有 n 个 wrap-around triples: (i-1, i, i+1) mod n
        votes: dict[str, int] = {}
        for i in range(n):
            a = self.grid[(i - 1) % n]
            b = self.grid[i]
            c = self.grid[(i + 1) % n]
            pat = (a << 2) | (b << 1) | c
            # 只计中间的细胞是活细胞的 pattern (避免三倍计数)
            # 实际上每个三连的唯一中心是 i, 所以不会有重复
            topic = _PATTERN_TOPICS.get(pat, "静默")
            votes[topic] = votes.get(topic, 0) + 1

        # 返回最高频话题
        dominant = max(votes, key=votes.get) if votes else "静默"
        return dominant

    def activity_ratio(self) -> float:
        """活跃细胞比例 0..1 — 可作为"思维活跃度"指标."""
        return sum(self.grid) / self.size

    def has_glider(self) -> bool:
        """粗略检查是否有 001011 (pattern 里的"移动结构")."""
        s = "".join(str(c) for c in self.grid)
        return "001011" in s or "110100" in s


# 内联说明见 docs/scripts/core/v5/v5/drivers.md（见“内联注释摘录”）

@dataclass
class Detector:
    """PAD 空间中的一个球体检测器."""
    center: tuple[float, float, float]  # (p, a, d) 在 [-1, 1]
    radius: float = 0.2
    hit_count: int = 0
    created: float = 0.0


@dataclass
class AISDetectorSet:
    """
    负选择检测器集 — 判断记忆的新颖性 (非我 → High novelty).

    tick(memories) → list[(novelty_score, memory_id)]
    每 tick O(M × D), M=~10, D=100 → <10μs.

    原理:
      - 检测器在 PAD 空间中均匀散布, 互相不重叠 (负选择).
      - 新记忆落在任意检测器半径内 → "自我" (熟悉) → 低分.
      - 新记忆落在所有检测器半径外 → "非我" (新颖) → 高分.
    克隆选择:
      - hit_count >= 10 的检测器, 克隆一个变异体.
      - hit_count == 0 超过 24h, 淘汰.
    """
    detectors: list[Detector] = field(default_factory=list)
    n_detectors: int = 100
    _initialized: bool = False

    def _lazy_init(self):
        if self._initialized:
            return
        # 持久化优先: 若已存在演化过的检测器集, 直接加载, 让免疫演化跨进程累积
        if _AIS_PATH.is_file():
            try:
                self._load()
                if self.detectors:
                    self._initialized = True
                    return
            except Exception:
                pass
        # 负选择初始化: 均匀散布, 避免重叠 (仅首次/加载失败)
        random.seed(42)
        self.detectors = []
        attempts = 0
        while len(self.detectors) < self.n_detectors and attempts < 5000:
            c = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
            r = random.uniform(0.1, 0.3)
            # 负选择: 不与已有检测器重叠
            overlap = False
            for d in self.detectors:
                dist = math.sqrt(
                    (c[0] - d.center[0]) ** 2
                    + (c[1] - d.center[1]) ** 2
                    + (c[2] - d.center[2]) ** 2
                )
                if dist < d.radius + r:
                    overlap = True
                    break
            if not overlap:
                self.detectors.append(Detector(center=c, radius=r, created=time.time()))
            attempts += 1
        self._initialized = True

    # ─── 持久化 (让负选择/克隆演化跨进程累积) ───────────────

    def save(self, path: str | Path | None = None) -> None:
        """把当前检测器集写盘. 在克隆/淘汰发生时调用, 避免高频 I/O."""
        p = Path(path) if path else _AIS_PATH
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            data = [{"center": list(d.center), "radius": d.radius,
                     "hit_count": d.hit_count, "created": d.created}
                    for d in self.detectors]
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _load(self) -> None:
        data = json.loads(_AIS_PATH.read_text(encoding="utf-8"))
        self.detectors = [
            Detector(center=tuple(d["center"]), radius=float(d.get("radius", 0.2)),
                     hit_count=int(d.get("hit_count", 0)),
                     created=float(d.get("created", 0.0)))
            for d in data
        ]

    def novelty(self, pad: tuple[float, float, float]) -> float:
        """
        单条记忆的新颖度: 0 (非常熟悉) → 1 (完全新颖).
        到最近检测器的距离 / max_possible_distance.
        """
        self._lazy_init()
        min_dist = 3.0  # sqrt(2^2+2^2+2^2) ≈ 3.46
        for d in self.detectors:
            dist = math.sqrt(
                (pad[0] - d.center[0]) ** 2
                + (pad[1] - d.center[1]) ** 2
                + (pad[2] - d.center[2]) ** 2
            )
            effective = dist - d.radius
            if effective < min_dist:
                min_dist = effective
        # clamp 到 [0, 1]
        return max(0.0, min(1.0, min_dist / 1.5))

    def tick(self, memories: list[tuple[int, tuple[float, float, float]]],
             now: float | None = None) -> list[tuple[float, int]]:
        """
        输入: [(memory_id, (p, a, d)), ...]
        输出: [(novelty_score 0..1, memory_id)], 按 novelty 降序.

        同时执行克隆选择 (hit_count update + 超参维持).
        """
        self._lazy_init()
        if now is None:
            now = time.time()
        results: list[tuple[float, int]] = []
        for mid, pad in memories:
            nov = self.novelty(pad)
            results.append((nov, mid))
            # 更新检测器命中: 找到最近检测器, 增加 hit_count
            for d in self.detectors:
                dist = math.sqrt(
                    (pad[0] - d.center[0]) ** 2
                    + (pad[1] - d.center[1]) ** 2
                    + (pad[2] - d.center[2]) ** 2
                )
                if dist < d.radius:
                    d.hit_count += 1
        # 按 novelty 降序
        results.sort(key=lambda x: -x[0])

        # 克隆选择: hit_count >= 10 的高频检测器 → 克隆变异体
        clones: list[Detector] = []
        for d in self.detectors:
            if d.hit_count >= 10:
                # 变异: 在中心附近扰动
                new_c = (
                    max(-1, min(1, d.center[0] + random.uniform(-0.1, 0.1))),
                    max(-1, min(1, d.center[1] + random.uniform(-0.1, 0.1))),
                    max(-1, min(1, d.center[2] + random.uniform(-0.1, 0.1))),
                )
                clones.append(Detector(center=new_c, radius=d.radius,
                                       created=now))
                d.hit_count = 0  # reset 原检测器
        # 淘汰: 创建超过 24h 且 hit_count == 0
        stale = [d for d in self.detectors
                 if (now - d.created) > 86400 and d.hit_count == 0]
        for s in stale:
            self.detectors.remove(s)
        # 加入克隆 (维持检测器数量)
        for c in clones:
            if len(self.detectors) < self.n_detectors * 1.5:
                self.detectors.append(c)

        # 演化发生时落盘, 让免疫学习跨进程保留
        if clones or stale:
            self.save()

        return results


# 内联说明见 docs/scripts/core/v5/v5/drivers.md（见“内联注释摘录”）

_ais_singleton: "AISDetectorSet | None" = None


def get_ais_detector_set() -> "AISDetectorSet":
    """返回跨调用持久的检测器集单例 (自动从 data/v5/ais_detectors.json 加载/演化).

    调用方 (think.curiosity_explore) 应使用本函数而非反复 `AISDetectorSet()`,
    否则每次新建都会触发 seed(42) 重置, 负选择/克隆演化永不生效, 新颖性排序静态。
    """
    global _ais_singleton
    if _ais_singleton is None:
        _ais_singleton = AISDetectorSet()
    return _ais_singleton
