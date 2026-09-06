"""memory_v5.tests.test_loop — 标准记忆循环引擎测试。

引擎机制 (注册 / 冷却 / 失败隔离 / 状态落盘 / 状态观测) 用 stub step 测,
**不碰真实 v5.db**, 可离线跑且毫秒级。

默认 step 表只测结构 (名字 / 阶段 / 间隔), 不执行 —— 执行会写真实库,
由冒烟脚本 (v5_call.py loop) 与线上 hook 覆盖。
详见 docs/v5-mcp-consolidation.md。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# tests/ -> memory_v5/ -> core/  (memory_v5 包在 core/ 下, 需把 core 入 path)
_CORE = Path(__file__).resolve().parents[2]
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import pytest

from memory_v5 import loop as loop_mod
from memory_v5.loop import (
    DEFAULT_REFLECT_INTERVAL,
    PHASE_MAINTENANCE, PHASE_POST, PHASE_PRE, PHASES,
    LoopContext, LoopEngine, LoopStep,
    make_default_engine, run_phase, status,
)


# ─── stub steps ──────────────────────────────────────────────────────

def _make_recorder(log: list):
    def _fn(ctx: LoopContext) -> dict:
        log.append(ctx)
        return {"seen_query": ctx.query, "seen_response": ctx.response}
    return _fn


def _boom(ctx: LoopContext) -> dict:
    raise RuntimeError("step exploded")


def _engine(tmp_path, steps, state=None) -> LoopEngine:
    return LoopEngine(steps=steps, state=state if state is not None else {})


# ─── S1: 默认 step 表结构 ────────────────────────────────────────────

def test_phases_are_documented_three():
    assert PHASES == ("pre", "post", "maintenance")


def test_default_step_table_shape():
    """默认表: pre 3 步 / post 3 步 / maintenance 1 步 (6h 冷却)。"""
    eng = make_default_engine(state={})
    assert [s.name for s in eng.steps(PHASE_PRE)] == ["identity", "recall", "project"]
    assert [s.name for s in eng.steps(PHASE_POST)] == [
        "vitality", "relationship", "anti_repeat"]
    m = eng.steps(PHASE_MAINTENANCE)
    assert [s.name for s in m] == ["reflect"]
    assert m[0].interval_sec == DEFAULT_REFLECT_INTERVAL == 21600


def test_pre_and_post_steps_have_no_cooldown():
    """pre / post 每轮都跑 (interval=0), 只有 maintenance 有冷却。"""
    eng = make_default_engine(state={})
    for phase in (PHASE_PRE, PHASE_POST):
        for s in eng.steps(phase):
            assert s.interval_sec == 0, f"{s.name} 不该有冷却"


def test_state_key_is_phase_qualified():
    """state key 带 phase 前缀 —— 同名 step 在不同 phase 不串味。"""
    s = LoopStep("x", lambda ctx: None, PHASE_POST)
    assert s.state_key == "post.x"


# ─── S2: 注册与校验 ──────────────────────────────────────────────────

def test_register_rejects_unknown_phase():
    eng = LoopEngine(steps=[], state={})
    with pytest.raises(ValueError):
        eng.register(LoopStep("bad", lambda ctx: None, "no_such_phase"))


def test_register_same_key_overwrites():
    """同名 step 覆盖, 不重复 (同 ReflectScheduler.register 的设计选择)。"""
    eng = LoopEngine(steps=[], state={})
    eng.register(LoopStep("a", lambda ctx: 1, PHASE_POST))
    eng.register(LoopStep("a", lambda ctx: 2, PHASE_POST))
    assert len(eng.steps(PHASE_POST)) == 1


# ─── S3: 执行语义 ────────────────────────────────────────────────────

def test_run_executes_all_due_steps_in_phase_only(tmp_path):
    """只跑该 phase 的 step, 不越界。"""
    log = []
    steps = [
        LoopStep("p1", _make_recorder(log), PHASE_PRE),
        LoopStep("q1", _make_recorder(log), PHASE_POST),
    ]
    eng = _engine(tmp_path, steps)
    r = eng.run(PHASE_PRE, LoopContext(query="hi"),
                state_path=tmp_path / "loop_state.json")
    assert r["ok"] is True
    assert r["ran"] == ["p1"]
    assert len(log) == 1


def test_context_reaches_step(tmp_path):
    """LoopContext 的 query / response / session_id 透传到 step。"""
    log = []
    eng = _engine(tmp_path, [LoopStep("s", _make_recorder(log), PHASE_POST)])
    eng.run(PHASE_POST, LoopContext(query="Q", response="R", session_id="sess-1"),
            state_path=tmp_path / "loop_state.json")
    assert log[0].query == "Q"
    assert log[0].response == "R"
    assert log[0].session_id == "sess-1"


def test_step_failure_is_collected_not_raised(tmp_path):
    """契约: 单个 step 炸了收集进 errors, 后续 step 照跑, 不上抛。"""
    log = []
    steps = [
        LoopStep("boom", _boom, PHASE_POST),
        LoopStep("after", _make_recorder(log), PHASE_POST),
    ]
    eng = _engine(tmp_path, steps)
    r = eng.run(PHASE_POST, LoopContext(),
                state_path=tmp_path / "loop_state.json")
    assert r["ok"] is False
    assert "boom" in r["errors"]
    assert "RuntimeError" in r["errors"]["boom"]
    assert r["ran"] == ["after"]          # 失败不阻断后续
    assert len(log) == 1


def test_unknown_phase_returns_error_not_raise(tmp_path):
    eng = _engine(tmp_path, [])
    r = eng.run("nope", LoopContext(), state_path=tmp_path / "loop_state.json")
    assert r["ok"] is False
    assert "unknown phase" in r["error"]


def test_disabled_step_is_reported_as_skipped(tmp_path):
    eng = _engine(tmp_path, [LoopStep("off", lambda ctx: 1, PHASE_POST, enabled=False)])
    r = eng.run(PHASE_POST, LoopContext(), state_path=tmp_path / "loop_state.json")
    assert r["skipped"]["off"] == "disabled"
    assert r["ran"] == []


# ─── S4: 冷却 ────────────────────────────────────────────────────────

def test_cooldown_skips_second_run_and_force_overrides(tmp_path):
    log = []
    steps = [LoopStep("slow", _make_recorder(log), PHASE_POST, interval_sec=3600)]
    eng = _engine(tmp_path, steps)
    sp = tmp_path / "loop_state.json"

    r1 = eng.run(PHASE_POST, LoopContext(), state_path=sp)
    assert r1["ran"] == ["slow"]

    r2 = eng.run(PHASE_POST, LoopContext(), state_path=sp)
    assert r2["ran"] == []
    assert r2["skipped"]["slow"] == "cooldown(3600s)"

    r3 = eng.run(PHASE_POST, LoopContext(), force=True, state_path=sp)
    assert r3["ran"] == ["slow"]
    assert len(log) == 2


def test_state_persists_across_engines(tmp_path):
    """状态落盘 —— 新建引擎 (模拟进程重启) 仍能看到上次运行时间。"""
    sp = tmp_path / "loop_state.json"
    steps = [LoopStep("slow", lambda ctx: 1, PHASE_POST, interval_sec=3600)]
    _engine(tmp_path, steps).run(PHASE_POST, LoopContext(), state_path=sp)

    eng2 = LoopEngine(steps=steps, state=loop_mod.load_state(sp))
    assert eng2.due_steps(PHASE_POST) == []
    assert eng2.state["post.slow"] > 0


def test_zero_interval_always_due(tmp_path):
    eng = _engine(tmp_path, [LoopStep("every", lambda ctx: 1, PHASE_PRE)])
    for _ in range(3):
        assert [s.name for s in eng.due_steps(PHASE_PRE)] == ["every"]
        eng.run(PHASE_PRE, LoopContext(), state_path=tmp_path / "loop_state.json")


# ─── S5: 状态观测 ────────────────────────────────────────────────────

def test_status_reports_all_three_phases(tmp_path):
    st = make_default_engine(state={}).status()
    assert set(st) == set(PHASES)
    for phase in PHASES:
        for name, info in st[phase].items():
            assert {"enabled", "due", "last_run", "last_run_human",
                    "next_run_in_sec", "interval_sec"} <= info.keys()


def test_status_reflect_due_when_never_run():
    """从未跑过 -> due=True, last_run=None (与 reflect scheduler 语义一致)。"""
    st = make_default_engine(state={}).status()
    info = st[PHASE_MAINTENANCE]["reflect"]
    assert info["due"] is True
    assert info["last_run"] is None
    assert info["last_run_human"] == "never"


def test_status_reflect_not_due_right_after_run():
    now = time.time()
    st = make_default_engine(state={"maintenance.reflect": now}).status()
    info = st[PHASE_MAINTENANCE]["reflect"]
    assert info["due"] is False
    assert 21500 < info["next_run_in_sec"] <= 21600


# ─── S6: 模块级入口 ──────────────────────────────────────────────────

def test_run_phase_accepts_kwargs():
    """run_phase(phase, **kwargs) 只吃 LoopContext 的字段, 多余键忽略。"""
    r = run_phase("__bad_phase__", query="x", bogus_key=1)
    assert r["ok"] is False
    assert "unknown phase" in r["error"]


def test_status_module_entry_is_read_only():
    st = status()
    assert isinstance(st, dict)
    assert set(st) == set(PHASES)


def test_load_state_tolerates_corrupt_file(tmp_path):
    """状态损坏 -> 显式 log + 返空, 不静默吞 (同 reflect scheduler 约定)。"""
    p = tmp_path / "loop_state.json"
    p.write_text("{ not json", encoding="utf-8")
    assert loop_mod.load_state(p) == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
