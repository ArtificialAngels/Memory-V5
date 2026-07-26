"""v5 历史摘要单元测试 (R4). mock 本地 LLM, 验证触发/复用/丢弃/隔离."""
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

import v5.summary as sm  # noqa: E402


def _history(n: int) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i}"} for i in range(n)]


class _FakeResp:
    def __init__(self, content):
        self.content = content


def test_short_history_no_inject(monkeypatch):
    monkeypatch.setattr(sm, "_load_cache", lambda: {"last_summary": "", "last_round": -1})
    monkeypatch.setattr(sm, "_save_cache", lambda d: None)
    block = sm.build_summary_block(_history(5), round_index=2)
    assert block == ""  # 不足阈值且无缓存 → 不注入


def test_triggers_and_caches(monkeypatch):
    calls = {"n": 0}

    def fake_compress(old_turns, max_sentences, timeout):
        calls["n"] += 1
        return "哥哥修了语音识别的 CUDA 问题。"

    monkeypatch.setattr(sm, "_compress", fake_compress)
    monkeypatch.setattr(sm, "_load_cache", lambda: {"last_summary": "", "last_round": -1})
    saved = {}
    monkeypatch.setattr(sm, "_save_cache", lambda d: saved.update(d))

    block = sm.build_summary_block(_history(25), round_index=12)
    assert "近期对话摘要" in block
    assert "哥哥修了语音识别" in block
    assert calls["n"] == 1
    assert saved.get("last_summary") == "哥哥修了语音识别的 CUDA 问题。"


def test_reuse_within_window(monkeypatch):
    calls = {"n": 0}

    def fake_compress(old_turns, max_sentences, timeout):
        calls["n"] += 1
        return "摘要A"

    # 已有缓存 (round=12), 当前 round=15 (gap=3 < reuse 10) → 复用, 不重算
    monkeypatch.setattr(sm, "_compress", fake_compress)
    monkeypatch.setattr(sm, "_load_cache",
                        lambda: {"last_summary": "摘要A", "last_round": 12})
    monkeypatch.setattr(sm, "_save_cache", lambda d: None)

    block = sm.build_summary_block(_history(25), round_index=15)
    assert "摘要A" in block
    assert calls["n"] == 0  # 复用, 未调用 LLM


def test_discard_when_too_old(monkeypatch):
    def fake_compress(old_turns, max_sentences, timeout):
        return "过期摘要"

    # 缓存 round=12, 当前 round=50 (gap=38 > max_age 30) 且 history 短(不触发生成)
    # → 超龄旧摘要丢弃, 不注入
    monkeypatch.setattr(sm, "_compress", fake_compress)
    monkeypatch.setattr(sm, "_load_cache",
                        lambda: {"last_summary": "旧摘要", "last_round": 12})
    saved = {}
    monkeypatch.setattr(sm, "_save_cache", lambda d: saved.update(d))

    block = sm.build_summary_block(_history(5), round_index=50)
    assert block == ""  # 超龄丢弃且不重新生成 (history 不足阈值)


def test_failure_isolated(monkeypatch):
    def fake_compress(old_turns, max_sentences, timeout):
        raise RuntimeError("llm down")

    monkeypatch.setattr(sm, "_compress", fake_compress)
    monkeypatch.setattr(sm, "_load_cache", lambda: {"last_summary": "", "last_round": -1})
    monkeypatch.setattr(sm, "_save_cache", lambda d: None)

    # 失败不应抛, 应静默返空
    block = sm.build_summary_block(_history(25), round_index=12)
    assert block == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
