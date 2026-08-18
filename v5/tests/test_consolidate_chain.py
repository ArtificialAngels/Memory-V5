"""reflect/consolidate 三级提取兜底链测试 (R9 测试缺口 #3 / L2).

extract_with_fallback: LLM (mock) → 规则 (真实 rule_based_extract) → 安全兜底。
验证每级触发条件与返回字段 (type/content/weight/source)。

参考: consolidate.py FACT_PATTERNS / rule_based_extract / safe_extract /
      extract_with_fallback (Ekko ModelMemoryExtractor → RuleBased → SafeFallback)。
"""
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # core

from v5.reflect import consolidate  # noqa: E402
from v5.reflect import llm_client  # noqa: E402


class _FakeResp:
    """模拟 llm_client.call_llm 的返回值 (只读 .content)."""

    def __init__(self, content: str):
        self.content = content


def test_llm_path_used_when_available(monkeypatch):
    """LLM 返回有效行 → 直接走 llm_extract, 不再触规则/兜底."""
    def fake_call_llm(prompt, *args, **kwargs):
        return _FakeResp("preference|我喜欢深夜写代码\nfact|哥哥是后端工程师")
    monkeypatch.setattr(llm_client, "call_llm", fake_call_llm)
    facts = consolidate.extract_with_fallback("任何内容都行", use_llm=True)
    assert len(facts) == 2
    assert {f["source"] for f in facts} == {"llm_extract"}
    assert {f["type"] for f in facts} == {"preference", "fact"}
    assert all(f["weight"] == 0.6 for f in facts)


def test_llm_failure_falls_back_to_rules(monkeypatch):
    """LLM 抛异常 → 规则层 (真实正则) 兜住, 返回 rule_based."""
    def boom(*args, **kwargs):
        raise RuntimeError("llm down")
    monkeypatch.setattr(llm_client, "call_llm", boom)
    facts = consolidate.extract_with_fallback(
        "我叫小明，我喜欢在安静的图书馆看书", use_llm=True)
    assert facts
    assert all(f["source"] == "rule_based" for f in facts)
    assert all(f["weight"] == 0.5 for f in facts)
    types = {f["type"] for f in facts}
    assert "identity" in types and "preference" in types


def test_llm_none_falls_back_to_rules(monkeypatch):
    """LLM 返回 None → 视为无结果, 落到规则层."""
    monkeypatch.setattr(llm_client, "call_llm", lambda *a, **k: None)
    facts = consolidate.extract_with_fallback("我叫小明", use_llm=True)
    assert facts and facts[0]["source"] == "rule_based"
    assert facts[0]["type"] == "identity"
    assert facts[0]["content"] == "我叫"


def test_rules_real_no_llm():
    """规则层零 LLM 独立可跑: relation / avoidance 正则命中."""
    facts = consolidate.extract_with_fallback("哥哥是后端工程师，我不吃香菜",
                                              use_llm=False)
    types = {f["type"] for f in facts}
    assert "relation" in types and "avoidance" in types
    assert all(f["source"] == "rule_based" and f["weight"] == 0.5 for f in facts)


def test_safe_fallback_when_llm_empty_and_no_rules(monkeypatch):
    """LLM 返空 + 无规则命中 → 安全兜底 (conversation 截取)."""
    monkeypatch.setattr(llm_client, "call_llm", lambda *a, **k: _FakeResp(""))
    content = "这台旧机器在深夜发出低沉的嗡嗡声，我决定先观察一下再决定"
    facts = consolidate.extract_with_fallback(content, use_llm=True)
    assert len(facts) == 1
    f = facts[0]
    assert f["type"] == "conversation"
    assert f["source"] == "safe_fallback"
    assert f["weight"] == 0.3
    assert f["content"] == content[:200]


def test_safe_extract_short_content_returns_empty():
    """兜底层对过短内容不产出 (防噪音)."""
    assert consolidate.safe_extract("短") == []
    assert consolidate.safe_extract("") == []


def test_three_level_chain_deterministic_order(monkeypatch):
    """三级链总顺序: LLM 有货 → LLM; LLM 无货 → 规则; 规则无货 → 兜底.

    用空规则内容验证最终落到兜底 (整链端到端, 无 mock 规则层)。
    """
    monkeypatch.setattr(llm_client, "call_llm",
                        lambda *a, **k: _FakeResp("哦哦"))
    content = "下午三点的时候，窗外的光线刚好落在桌角的那盆绿萝上"
    assert len(content) >= 10
    facts = consolidate.extract_with_fallback(content, use_llm=True)
    assert len(facts) == 1
    assert facts[0]["source"] == "safe_fallback"
