"""memory_v5.tests.test_tools_facade — 冷路径门面工具的契约测试。

核心断言: **门面对旧工具是纯委托** —— 同一个 action 的输出与被它吸收的旧工具
输出**逐字节一致**。这样 slim 模式切过去不会有任何行为漂移。

只跑只读 action, 不写真实 v5.db (写动作由线上 hook 与冒烟脚本覆盖)。
详见 docs/v5-mcp-consolidation.md。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# tests/ -> memory_v5/ -> core/
_CORE = Path(__file__).resolve().parents[2]
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import pytest

from memory_v5.tools import facade
from memory_v5.tools import care_tool, emotion_tool, extra_tool, self_tool
from memory_v5.tools import relationship_tool, vitality_tool


def _json_of(raw: str):
    """answer() 的输出是「自然语言一行 + JSON 一段」两段; 取 JSON 段解析。"""
    body = raw.split("\n", 1)[1] if "\n" in raw else raw
    return json.loads(body)


# ─── S1: 未知 action 的统一错误 ──────────────────────────────────────

@pytest.mark.parametrize("tool,kwargs", [
    ("v5_self", {}),
    ("v5_state", {}),
    ("v5_skill", {}),
    ("v5_reflection", {}),
    ("v5_directive", {}),
    ("v5_repeat", {}),
    ("v5_loop", {}),
])
def test_unknown_action_lists_valid_options(tool, kwargs):
    """未知 action -> ok=False + valid_actions 列表, 让模型一次改对, 不猜。"""
    fn = getattr(facade, tool)
    out = fn(action="__nope__", **kwargs)
    d = json.loads(out)
    assert d["ok"] is False
    assert d["tool"] == tool
    assert "unknown action" in d["error"]
    assert isinstance(d["valid_actions"], list) and d["valid_actions"]


# ─── S2: 纯委托 —— 门面输出 == 旧工具输出 ─────────────────────────────

@pytest.mark.parametrize("facade_action,legacy_fn,legacy_args", [
    ("model", self_tool.v5_self_model, ()),
    ("anchor", self_tool.v5_context_refresh, ()),
    # 2026-09-05: thought/curiosity/subconscious/discover 已从 slim action 移除
])
def test_v5_self_delegates_byte_identical(facade_action, legacy_fn, legacy_args):
    assert facade.v5_self(action=facade_action) == legacy_fn(*legacy_args)


@pytest.mark.parametrize("facade_action,legacy_fn", [
    ("emotion", emotion_tool.v5_emotion_status),
    ("care", care_tool.v5_care_status),
    ("relationship", relationship_tool.v5_relationship),
    # 2026-09-05: emotion_label/activity/compression 已从 slim action 移除
])
def test_v5_state_delegates_byte_identical(facade_action, legacy_fn):
    assert facade.v5_state(action=facade_action) == legacy_fn()


def test_v5_state_vitality_shape():
    """vitality 不在逐字节比对里 —— v5_vitality 调 tick() 有副作用
    (推进 last_tick / uptime), 连调两次结果本就不同。这里只验形状。"""
    d = json.loads(facade.v5_state(action="vitality"))
    assert {"vitality", "label", "emoji", "total_uptime_sec"} <= d.keys()


def test_v5_state_emotion_update_delegates():
    """emotion_update 有写动作 (改 affect.json), 只验委托链路与字段集。"""
    text = "今天天气不错"
    a = _json_of(facade.v5_state(action="emotion_update", text=text))
    b = _json_of(emotion_tool.v5_analyze_emotion(text))
    assert set(a) == set(b)
    assert {"pleasure", "arousal", "dominance", "mood_label", "delta",
            "intensity"} <= a.keys()


# 2026-09-05: test_v5_state_emotion_label_delegates 已删除 (emotion_label action 移除)
# 2026-09-05: test_v5_content_dissonance_shape 已删除 (v5_content 工具移除)


# ─── S3: 各门面返回值形状 ────────────────────────────────────────────

def test_v5_self_reflect_shape():
    d = json.loads(facade.v5_self(action="reflect", mode="reflect"))
    assert isinstance(d, dict)


def test_v5_skill_list_shape():
    d = _json_of(facade.v5_skill(action="list"))
    assert isinstance(d, list)
    for s in d:
        assert {"name", "description", "path"} <= s.keys()


def test_v5_reflection_stats_shape():
    raw = facade.v5_reflection(action="stats")
    assert isinstance(raw, str)
    assert isinstance(json.loads(raw), (dict, list))


def test_v5_directive_stats_shape():
    assert isinstance(json.loads(facade.v5_directive(action="stats")), dict)


def test_v5_repeat_stats_shape():
    assert isinstance(json.loads(facade.v5_repeat(action="stats")), dict)


def test_v5_repeat_check_shape():
    d = json.loads(facade.v5_repeat(action="check", character="ikaros",
                                    candidate_text="这是一段测试文本"))
    assert isinstance(d, dict)


# ─── S4: v5_loop 门面 ───────────────────────────────────────────────

def test_v5_loop_status_returns_all_phases():
    d = json.loads(facade.v5_loop(action="status"))
    assert d["ok"] is True
    assert set(d["phases"]) == {"pre", "post", "maintenance"}


def test_v5_loop_run_rejects_unknown_phase():
    d = json.loads(facade.v5_loop(action="run", phase="nope"))
    assert d["ok"] is False
    assert d["valid_phases"] == ["pre", "post", "maintenance"]


def test_v5_loop_run_post_executes(monkeypatch, tmp_path):
    """post 阶段用 stub step 跑, 不碰真实库。"""
    from memory_v5 import loop as loop_mod

    calls = []

    def _stub(ctx):
        calls.append(ctx.response)
        return {"ok": True}

    monkeypatch.setattr(loop_mod, "_default_steps",
                        lambda: [loop_mod.LoopStep("stub", _stub, "post")])
    monkeypatch.setattr(loop_mod, "_STATE_FILE", tmp_path / "loop_state.json")
    raw = facade.v5_loop(action="run", phase="post",
                         response="测试回复", character="ikaros")
    assert "跑了 1 步" in raw.split("\n", 1)[0]   # answer() 的自然语言首行
    d = _json_of(raw)
    assert d["ran"] == ["stub"]
    assert d["results"]["stub"] == {"ok": True}
    assert calls == ["测试回复"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
