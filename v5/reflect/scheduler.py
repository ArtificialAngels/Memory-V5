# 详细说明见 docs/scripts/core/v5/v5/reflect/scheduler.md

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

logger = logging.getLogger("ikaros.memory.v5.scheduler")


# ─── V3 间隔对照 (V3 memory_reflect.py:67-70) ─────────────────────
# V3 常量名 → V4 常量名 (改名以示升级, 行为暂时对齐)
DEFAULT_CONSOLIDATE_INTERVAL = 3600      # 1h  — V3 _DEFAULT_CONSOLIDATE_INTERVAL
DEFAULT_DEDUP_INTERVAL = 21600          # 6h  — V3 _DEFAULT_DEDUP_INTERVAL
DEFAULT_PROMOTE_INTERVAL = 3600         # 1h — 原12h (V3), 2026-08-02 加速见下方注释
DEFAULT_DISTILL_INTERVAL = 86400        # 24h — V3 _DEFAULT_DISTILL_INTERVAL
# V3 bug 修复: cleanup 单独 interval, 不复用 dedup
DEFAULT_CLEANUP_INTERVAL = 21600        # 6h  — V3 复用 bug (line 649) 已修
# V4 新增: reflect (大模型灵魂层反思) — 3h, 哥哥 7-12 要求提升自省频率
DEFAULT_REFLECT_INTERVAL = 10800        # 3h
# V4 新增: vector_sync (向量回填/校正) — 5min (原24h→1h), 确保 v5.db 每条记忆都有 Chroma 向量
#   2026-08-02 根因修复: 写时同步失败率高(多进程 hnsw 并发写冲突/chromadb 缺失),
#   731/1433 记忆缺向量导致语义检索召回失败。op 改增量(只补缺失 id) + 5min 兜底,
#   缺失窗口从 24h → 5min, 新记忆几乎实时可召回。
DEFAULT_VECTOR_SYNC_INTERVAL = 300      # 5min
# V5 新增: narrative (自我叙事连续性) — 30d
DEFAULT_NARRATIVE_INTERVAL = 2592000   # 30d
# V5 新增: profile_sync (哥哥画像聚合) — 6h
DEFAULT_PROFILE_SYNC_INTERVAL = 21600  # 6h


@dataclass(frozen=True)
class ScheduleState:
    """反思调度状态 (持久化到 data/v4/reflect_state.json).

    V3 用普通 dict (memory_reflect.py:138-145), V4 用 frozen dataclass:
      - 不可变 (MappingProxyType 视图), 防止误改
      - 字段**不**限 5 个 last_*: 支持任意 last_run_key (测试可加 last_bad)
      - 自带 __eq__/__hash__, 适合做 diff

    设计选择: 用 _times: Mapping[str, float] 而不是固定 5 个字段.
    理由: 测试 / 第三方 op 可以注册任意 last_run_key, 不需要改 state schema.
    """
    _times: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({}),
        compare=True,
    )

    @classmethod
    def empty(cls) -> "ScheduleState":
        return cls(_times=MappingProxyType({}))

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self._times.get(key, default))

    def to_dict(self) -> dict:
        return dict(self._times)

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduleState":
        coerced = {k: float(v) for k, v in d.items()}
        return cls(_times=MappingProxyType(coerced))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduleState):
            return NotImplemented
        return dict(self._times) == dict(other._times)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._times.items())))

    def set(self, key: str, value: float) -> "ScheduleState":
        """返新 state (frozen 不可变). 旧 state 不动."""
        new_times = dict(self._times)
        new_times[key] = value
        return ScheduleState(_times=MappingProxyType(new_times))


@dataclass(frozen=True)
class ReflectOp:
    """单个反思操作的注册信息 (V4 显式 trigger).

    V3 隐式: 4 个 if 块 (memory_reflect.py:609-656), 加第 5 个操作要复制 4 行模板.
    V4 显式: 注册表模式, 新操作 = 加一行 ReflectOp + register().
    """
    name: str
    fn: Callable[[], int]
    interval_sec: int
    last_run_key: str  # 对应 ScheduleState 字段名


# ─── trigger 判断 (纯函数, 易测) ─────────────────────────────────

def should_run(state: ScheduleState, key: str, interval: int,
               *, now: float | None = None, force: bool = False) -> bool:
    """判断某个反思操作是否该跑.

    V3 隐式 (memory_reflect.py:156-158):
        def _should_run(state, key, interval):
            last = state.get(key, 0)
            return (time.time() - last) >= interval

    V4 改进:
      - 接受 state 字段名 (字符串), 不接受裸 dict, 防止 typo
      - 接受 now 参数 (单测可注入时间)
      - force=True 跳过判断
      - 显式返回 bool, 不返回 truthy 数值
    """
    if force:
        return True
    if now is None:
        now = time.time()
    last = state.get(key, 0.0)
    return (now - last) >= interval


def next_run_time(state: ScheduleState, key: str, interval: int,
                  *, now: float | None = None) -> float:
    """计算下次运行时间 (调试 / 日志用).

    单测覆盖: never_run → 立刻, just_run → interval 后, overdue → 已过期.
    """
    if now is None:
        now = time.time()
    last = state.get(key, 0.0)
    return last + interval


# ─── 状态持久化 (复用 V3 路径, 但走 V4 子目录) ───────────────────

# 内联说明见 docs/scripts/core/v5/v5/reflect/scheduler.md（见“内联注释摘录”）
_V5_DATA_DIR = Path(__file__).resolve().parent / "data" / "v5"
_STATE_FILE = _V5_DATA_DIR / "reflect_state.json"


def load_state(path: Path | None = None) -> ScheduleState:
    """加载反思状态. V5 子目录, 不污染 V3 state."""
    p = path or _STATE_FILE
    if p.exists():
        try:
            import json
            return ScheduleState.from_dict(json.loads(p.read_text("utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            # V4: 状态损坏时显式 log, 不静默返空
            logger.warning("reflect_state.json 损坏, 重新初始化: %s", e)
    return ScheduleState()


def save_state(state: ScheduleState, path: Path | None = None) -> None:
    """保存反思状态. 原子写: 写 tmp + rename, 防中途崩溃丢状态."""
    import json
    import os
    p = path or _STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, p)


# ─── 调度器 (注册表模式) ────────────────────────────────────────

def _run_profile_sync() -> int:
    """包装 run_sync 返回 int (scheduler 要求 int)."""
    from v5.profile import run_sync
    try:
        result = run_sync()
        return result.get("traits", 0)
    except Exception as e:
        logger.error("profile_sync failed: %s", e)
        return 0


class ReflectScheduler:
    """V4 反思调度器: trigger 显式, logic 委托, 错误显式.

    V3 reflect_cycle() (memory_reflect.py:594-678):
      - 5 个 if 块, 每个包 try/except, 返 -1
      - 加新操作要复制 4 行模板
      - 沉默失败: 任何一步挂, 主控不感知

    V4 ReflectScheduler:
      - 注册 ReflectOp 列表
      - 单次 run_all() 按 trigger 顺序执行
      - 错误上抛 (force=False) 或收集 (force=True + continue_on_error=True)
      - state 持久化由 caller 控制 (不藏在主控里)
    """

    def __init__(self, ops: list[ReflectOp] | None = None,
                 state: ScheduleState | None = None):
        self._ops: list[ReflectOp] = list(ops or [])
        self._state = state or load_state()
        # V5 新增: profile_sync 注册到默认 ops (仅当未显式传入 ops)
        if ops is None:
            self._ops.append(ReflectOp(
                name="profile_sync",
                fn=_run_profile_sync,
                interval_sec=DEFAULT_PROFILE_SYNC_INTERVAL,
                last_run_key="last_profile_sync",
            ))

    def register(self, op: ReflectOp) -> None:
        """注册反思操作. 重复注册同名 op 会覆盖 (设计选择)."""
        self._ops = [o for o in self._ops if o.name != op.name]
        self._ops.append(op)

    def get_op(self, name: str) -> "ReflectOp | None":
        """Look up a registered op by name. Returns None if not found.

        Public accessor for ``self._ops`` so callers (e.g. v5.tools.extra_tool)
        don't reach into private state and silently break on refactors.
        """
        for op in self._ops:
            if op.name == name:
                return op
        return None

    @property
    def state(self) -> ScheduleState:
        return self._state

    def ops_due(self, *, now: float | None = None,
                force: bool = False) -> list[ReflectOp]:
        """列出当前该跑的操作 (不改 state)."""
        if now is None:
            now = time.time()
        return [
            op for op in self._ops
            if should_run(self._state, op.last_run_key, op.interval_sec,
                          now=now, force=force)
        ]

    def run_one(self, op: ReflectOp, *, force: bool = False) -> int:
        """跑单个操作, 更新 state 对应字段.

        V4 行为:
          - should_run 为 False 且非 force → 返 0 不跑
          - 跑成功 → 更新 state 对应 last_run_key, 返 fn() 结果
          - 跑失败 → 异常上抛, state 不更新 (下次还会重试)
        """
        now = time.time()
        if not should_run(self._state, op.last_run_key, op.interval_sec,
                          now=now, force=force):
            logger.debug("op %s: 未到时间, 跳过", op.name)
            return 0
        logger.info("op %s: 开始 (interval=%ds)", op.name, op.interval_sec)
        n = op.fn()  # V4: 异常上抛, 不 try/except
        self._state = self._state.set(op.last_run_key, now)
        logger.info("op %s: 完成, 处理 %d 条", op.name, n)
        return n

    def run_all(self, *, force: bool = False,
                continue_on_error: bool = False) -> dict[str, int]:
        """跑所有到期的操作.

        Args:
            force: 忽略 trigger 判断, 全跑
            continue_on_error: True 时单步失败不中断, 收集到 results 返 -1
                             False 时失败上抛 (V3 默认行为相反)
        """
        results: dict[str, int] = {}
        for op in self.ops_due(force=force):
            try:
                results[op.name] = self.run_one(op, force=force)
            except Exception as e:
                logger.error("op %s: 失败, %s", op.name, e)
                if continue_on_error:
                    results[op.name] = -1
                else:
                    raise
        save_state(self._state)
        return results

    def dry_run(self) -> dict[str, dict]:
        """Dry-run: 列出每个 op 状态, 不真跑.

        给哥哥看的工具 — 'v4 反思周期现在该跑什么?' 一次回答.
        """
        now = time.time()
        out = {}
        for op in self._ops:
            due = should_run(self._state, op.last_run_key, op.interval_sec,
                             now=now)
            next_at = next_run_time(self._state, op.last_run_key,
                                    op.interval_sec, now=now)
            last = self._state.get(op.last_run_key, 0.0)
            out[op.name] = {
                "due": due,
                "last_run": last if last > 0 else None,
                "last_run_human": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last))
                    if last > 0 else "never"
                ),
                "next_run_in_sec": max(0, int(next_at - now)),
                "interval_sec": op.interval_sec,
            }
        return out
