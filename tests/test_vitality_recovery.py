"""vitality.tick() 空闲恢复语义回归测试 (2026-08-30 根治)。

## 修的是什么

旧版 `tick(conversation=True)` 里 `conversation` 一个标志同时管**两件正交的事**:

    (a) 收 _CONVERSATION_COST 一次性成本 + conversation_count 计数  <- 合理
    (b) 抑制**整段经过时间**的空闲恢复                              <- bug

(b) 的后果: 任何"每轮调一次 tick(conversation=True)"的路径都只减不增,
精力单调抽干到 0, persona 永远显示「精疲力竭」。

受害的不只是 Loop:
    - `vitality.py::vitality_prompt()` —— tick(conversation=True) 后 save,
      走 **cloud_chat 主链路**, 是最早也最持续的抽干源。
    - `vitality_tool.py::v5_vitality_tick(conversation=True)`。
    - `loop.py::_step_vitality` (2026-08-30 新增的 post 阶段)。

## 修法

恢复改按**真空闲分钟数**计算, 与 `conversation` 标志脱钩:

    idle_min = dt_min - conversation_minutes
    recovery = _RECOVERY_RATE * idle_min * (1 - vitality)

`conversation_minutes` 未传时按 0 计 —— 一次对话 tick 的自身耗时相对 tick
间隔可忽略, 所以默认整段经过时间都算空闲。要统一结算一段持续对话
(如长会话结束时补一次 tick) 就显式传分钟数, 那段不计恢复。

## 本文件守的不变量

1. 每轮对话 tick **不会**让精力单调下降到 0 (空闲恢复必须生效)。
2. 长时间空闲后能回满 —— 这是"休息恢复精力"的设计意图。
3. `conversation_minutes` 显式传值时, 那段确实不计恢复。
4. 一次性对话成本仍照收 (不能因为修恢复就把成本修没了)。
"""
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # core/ -> import memory_v5

from memory_v5.vitality import (  # noqa: E402
    Vitality, _CONVERSATION_COST, _RECOVERY_RATE,
)


def _fresh(vitality: float = 0.75, elapsed_min: float = 0.0) -> Vitality:
    """构造一个"距上次 tick 已过去 elapsed_min 分钟"的实例。"""
    now = time.time()
    v = Vitality(vitality=vitality, last_tick=now - elapsed_min * 60.0)
    return v


# ── S1: 核心回归 —— 每轮对话不再单调抽干 ──────────────────────────────

def test_conversation_tick_still_recovers_idle_time():
    """旧行为: conversation=True 会把整段经过时间的恢复全部跳过 -> 恒降。

    30 分钟空闲(期间没人说话) + 1 次对话:
      恢复 = 0.08 * 30 * (1-0.75) = 0.6  >>  对话成本 0.04
    净效应必须是**上升**并触顶, 而不是被 0.04 吃掉。
    """
    v = _fresh(vitality=0.75, elapsed_min=30.0)
    v = v.tick(conversation=True)
    assert v.vitality > 0.75, (
        f"对话 tick 把空闲恢复也跳过了 (期望 >0.75, 实际 {v.vitality})")


def test_repeated_conversation_ticks_do_not_drain_to_zero():
    """模拟真实节奏: 每轮间隔 1 分钟, 连聊 60 轮 —— 不能掉到 0。

    单轮账: 恢复 0.08*1*(1-v) vs 成本 0.04 + 基础衰减 0.003。
    v=0.5 时恢复 0.04 == 成本 0.04, 存在不动点 -> 系统稳定, 不会归零。
    旧行为(恢复恒为 0) 下 60 轮后必然是 0.0。
    """
    v = _fresh(vitality=0.75, elapsed_min=1.0)
    for _ in range(60):
        v = v.tick(conversation=True)
        v.last_tick -= 60.0          # 推进虚拟时钟: 下一轮间隔 1 分钟
    assert v.vitality > 0.05, (
        f"连续对话把精力抽干了 (60 轮后 {v.vitality}) —— "
        f"conversation 标志又在抑制空闲恢复")


def test_idle_only_recovers_toward_full():
    """纯空闲应当回满 (logistic: 接近 1 时变慢, 不越过 1)。"""
    v = _fresh(vitality=0.20, elapsed_min=120.0)
    v = v.tick()
    assert v.vitality == pytest.approx(1.0, abs=1e-6), (
        f"长时间空闲应回满, 实际 {v.vitality}")
    assert v.vitality <= 1.0


# ── S2: conversation_minutes 的转义口 ─────────────────────────────────

def test_conversation_minutes_suppresses_recovery_for_that_span():
    """显式声明"这 10 分钟都在对话中" -> 这 10 分钟不计恢复。

    用于长会话结束时统一结算的场景 (否则会把一整段活跃对话误算成空闲)。
    """
    base = _fresh(vitality=0.75, elapsed_min=10.0).tick()
    span = _fresh(vitality=0.75, elapsed_min=10.0).tick(
        conversation=True, conversation_minutes=10.0)
    # 声明整段在对话 -> 恢复被压到 0, 只剩衰减 + 一次性成本
    assert span.vitality < base.vitality, (
        "conversation_minutes 未生效: 该时段仍在计恢复")


def test_conversation_minutes_capped_at_elapsed():
    """conversation_minutes 超过实际经过时间时按经过时间截断 (不能算出负空闲)。"""
    v = _fresh(vitality=0.75, elapsed_min=2.0)
    v = v.tick(conversation=True, conversation_minutes=999.0)
    assert v.vitality >= 0.0
    assert v.idle_minutes >= 0.0, "空闲分钟数被算成负数"


# ── S3: 不能把原有语义修没了 ──────────────────────────────────────────

def test_conversation_still_costs_energy():
    """一次性对话成本仍要收 (修恢复 ≠ 免单)。

    构造 dt≈0 的对话 tick: 此时无恢复可言, 差额应当正好是 _CONVERSATION_COST
     (+ 极小的基础衰减 / 系统压力 / 昼夜项, 故用容差比较)。
    """
    a = _fresh(vitality=0.75, elapsed_min=0.0001).tick(conversation=False)
    b = _fresh(vitality=0.75, elapsed_min=0.0001).tick(conversation=True)
    assert a.vitality - b.vitality == pytest.approx(_CONVERSATION_COST, abs=1e-3)


def test_conversation_increments_counter():
    v = _fresh(elapsed_min=1.0)
    assert v.conversation_count == 0
    v = v.tick(conversation=True)
    assert v.conversation_count == 1
    v = v.tick()
    assert v.conversation_count == 1, "非对话 tick 不应增加对话计数"


def test_idle_minutes_accumulates_only_idle_span():
    """idle_minutes 只累计真空闲部分。"""
    v = _fresh(elapsed_min=10.0)
    v = v.tick(conversation=True, conversation_minutes=4.0)
    assert v.idle_minutes == pytest.approx(6.0, abs=1e-6)


def test_recovery_rate_sane():
    """守住参数量级: 恢复率必须显著大于单次对话成本, 否则修了也压不住抽干。"""
    assert _RECOVERY_RATE > _CONVERSATION_COST, (
        f"恢复率 {_RECOVERY_RATE} 应大于单次对话成本 {_CONVERSATION_COST}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
