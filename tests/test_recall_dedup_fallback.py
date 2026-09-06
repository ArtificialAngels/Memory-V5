"""v5_recall 去重兜底 (2026-08-30): 候选被冷却一空时放宽去重, 不交空纸条。

背景 (实测证据):
    recall_ledger **落盘持久化** (data/v5/recall_log_<sid>.json), turn 单调递增
    且永不清零; 插件侧 session_id 硬编码 'dsh' (所有会话共用一本账)。
    于是「同一话题连着问几轮」时 top-k 候选会整批处于冷却中 → 装配 0 条 →
    返回 "(无相关记忆)"。

设计裁定:
    去重是**优化** (省 token / 少点"又讲一遍"的噪声); 返回空上下文是**功能性失败**。
    宁可重复, 不可失忆 —— 候选被冷却一空时重新用全量候选。

实测影响面: 不同 query 的 cooled 仅 0~2 条, 兜底极少触发, 不抵消去重收益。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CORE = Path(__file__).resolve().parents[2]  # tests/ -> memory_v5/ -> core/
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from memory_v5 import recall_ledger as rl
from memory_v5 import memory_retrieval as mr
from memory_v5.tools import recall_tool as rt


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    """ledger 落盘到 tmp, 不污染真实 data/v5/recall_log_*.json。"""
    monkeypatch.setattr(rl, "_LEDGER_ROOT", Path(tmp_path), raising=False)


def _rows(n: int, prefix: str = "m"):
    """合成检索结果。content 给足长度, 免得被预算判成空条目。"""
    return [
        {
            "id": f"{prefix}{i}",
            "content": f"第 {i} 条记忆正文，内容足够长以便通过预算装配的最小长度检查。",
            "type": "fact",
            "score": 0.9 - i * 0.01,
            "tags": "",
        }
        for i in range(n)
    ]


@pytest.fixture
def _stub_retrieve(monkeypatch):
    """把 unified_retrieve 换成可控桩, 不依赖 embedding / v5.db。

    ⚠️ 必须 patch **源头模块** `memory_v5.memory_retrieval.unified_retrieve`,
    不能 patch `rt.unified_retrieve`:
        recall_tool.v5_recall 是**函数体内惰性导入**
        (`from memory_v5.memory_retrieval import unified_retrieve`),
        每次调用都重新取一次, 根本不读模块属性。
        patch 模块属性会静默无效 → 真的去查 v5.db + embedding,
        表现为"断言数字对不上"且测试变慢, 极易误判成兜底逻辑写错。
    """
    calls = {"n": 0}

    def fake(query, top_k=20, include_dsh_only=True, **kw):
        calls["n"] += 1
        return _rows(3)

    monkeypatch.setattr(mr, "unified_retrieve", fake, raising=False)
    return calls


def _stats_of(out: str) -> dict:
    """v5_recall 经 @safe_tool 包装, 输出是「自然语言\nJSON」两段。"""
    body = out.split("\n", 1)[1] if "\n" in out else out
    return json.loads(body)["stats"]


def _context_of(out: str) -> str:
    body = out.split("\n", 1)[1] if "\n" in out else out
    return json.loads(body)["context"]


# ── 1) 首次召回: 无冷却, 正常装配 ──
def test_first_call_no_dedup(_stub_retrieve):
    st = _stats_of(rt.v5_recall("某查询", session_id="s1"))
    assert st["cooled"] == 0
    assert st["dedup_relaxed"] is False
    assert st["placed"] == 3


# ── 2) 同 query 第二轮: 全被冷却 → 放宽去重, 仍给出内容 ──
def test_second_call_all_cooled_relaxes(_stub_retrieve):
    st1 = _stats_of(rt.v5_recall("同一话题", session_id="s2"))
    assert st1["placed"] == 3

    st2 = _stats_of(rt.v5_recall("同一话题", session_id="s2"))
    assert st2["cooled"] == 3, "第二轮应整批处于冷却中（这是兜底要救的场景）"
    assert st2["dedup_relaxed"] is True
    assert st2["placed"] == 3, "兜底后必须仍然装配出内容, 不能是空上下文"

    ctx = _context_of(rt.v5_recall("同一话题", session_id="s2"))
    assert ctx and "无相关记忆" not in ctx


# ── 3) 部分冷却时**不**放宽 —— 兜底不能抵消去重收益 ──
def test_partial_cooling_does_not_relax(monkeypatch):
    seq = [
        _rows(3, "a"),                       # turn 1: 3 条
        _rows(2, "a") + _rows(2, "b"),       # turn 2: 2 旧 + 2 新
    ]
    it = iter(seq)
    monkeypatch.setattr(mr, "unified_retrieve", lambda *a, **kw: next(it), raising=False)

    rt.v5_recall("话题", session_id="s3")
    st2 = _stats_of(rt.v5_recall("话题", session_id="s3"))
    # a0/a1 冷却, b0/b1 是新的 → fresh 非空, 不该放宽
    assert st2["cooled"] == 2
    assert st2["dedup_relaxed"] is False


# ── 4) 检索本来就空 → 不放宽, 且真的是空上下文 ──
def test_empty_retrieval_is_not_relaxed(monkeypatch):
    monkeypatch.setattr(mr, "unified_retrieve", lambda *a, **kw: [], raising=False)
    st = _stats_of(rt.v5_recall("无此记忆", session_id="s4"))
    assert st["retrieved"] == 0
    assert st["dedup_relaxed"] is False, "检索本身为空不是去重的锅, 不该放宽"
    assert _context_of(rt.v5_recall("无此记忆", session_id="s4")) == "(无相关记忆)"


# ── 5) 空 query 短路, 不碰检索 ──
def test_empty_query_short_circuits(_stub_retrieve):
    rt.v5_recall("   ", session_id="s5")
    assert _stub_retrieve["n"] == 0, "空 query 应直接跳过, 不空转检索"


# ── 6) 兜底后仍记录 served（下轮若仍全冷却会再次放宽, 不会卡死）──
def test_relaxed_path_still_records_served(_stub_retrieve):
    rt.v5_recall("重复话题", session_id="s6")
    rt.v5_recall("重复话题", session_id="s6")  # 放宽
    st3 = _stats_of(rt.v5_recall("重复话题", session_id="s6"))
    assert st3["cooled"] == 3
    assert st3["dedup_relaxed"] is True
    assert st3["placed"] == 3, "连续重复提问也必须每轮都有内容, 不能逐轮劣化到空"
