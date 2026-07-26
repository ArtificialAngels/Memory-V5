"""v5 情感引擎增强单元测试 (R6). 隔离本地 LLM / v5.db / 文件 IO."""
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # Ikaros-memory/

import v5.emotional_memory as em  # noqa: E402
import v5.affect as affect_mod  # noqa: E402


class _FakeResp:
    def __init__(self, content):
        self.content = content


def _patch_llm(monkeypatch, content):
    import v5.reflect.llm_client as lc

    def _fake(system, user, *, provider="local", max_tokens=16,
              temperature=0.1, timeout=5, **kwargs):
        return _FakeResp(content)
    monkeypatch.setattr(lc, "call_llm", _fake)


def _make_affect(p, a, d):
    class _S:
        pleasure, arousal, dominance = p, a, d

        def decay(self):
            return self

        @classmethod
        def load(cls):
            return cls()
    return _S


def test_label_emotion_uses_vocab(monkeypatch):
    _patch_llm(monkeypatch, "开心、期待")
    tags = em.label_emotion("哥哥夸了我, 好开心", (0.6, 0.3, 0.2))
    assert "开心" in tags
    assert "期待" in tags
    assert len(tags) <= 2


def test_label_emotion_rule_fallback(monkeypatch):
    import v5.reflect.llm_client as lc

    def _boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr(lc, "call_llm", _boom)
    tags = em.label_emotion("x", (0.5, 0.3, 0.2))
    assert tags and tags[0] in ("开心", "愉悦")


class _Mem:
    def __init__(self, id, content, type="emotion_label", tags="", weight=0.5, created=0.0):
        self.id = id
        self.content = content
        self.type = type
        self.tags = tags
        self.weight = weight
        self.created = created


def test_search_by_emotion_filters_by_tag(monkeypatch):
    mems = [
        _Mem(1, "情感标签: 开心", tags="v5,emotion,开心"),
        _Mem(2, "情感标签: 难过", tags="v5,emotion,难过"),
        _Mem(3, "情感标签: 开心", tags="v5,emotion,开心"),
    ]
    import v5.store as sm

    def _list(type_filter=None, limit=50):
        return [m for m in mems if m.type == type_filter]
    monkeypatch.setattr(sm, "list_all", _list)
    res = em.search_by_emotion("开心", top_k=5)
    assert len(res) == 2
    assert all("开心" in r["content"] for r in res)


def test_build_emotion_diff_block_injects(monkeypatch, tmp_path):
    monkeypatch.setattr(affect_mod, "AffectState", _make_affect(0.6, 0.3, 0.2))
    state_file = tmp_path / "emotion_state.json"
    state_file.write_text('{"p": -0.4, "a": -0.1, "d": 0.0, "mood": "低落", "ts": 0}',
                           encoding="utf-8")
    monkeypatch.setattr(em, "_EMOTION_STATE_PATH", state_file)
    em._LAST_DIFF_INJECT = 0.0
    block = em.build_emotion_diff_block()
    assert block.startswith("\n---\n情感对比：")
    assert "开心" in block


def test_build_emotion_diff_block_debounce(monkeypatch, tmp_path):
    monkeypatch.setattr(affect_mod, "AffectState", _make_affect(0.6, 0.3, 0.2))
    state_file = tmp_path / "emotion_state.json"
    state_file.write_text('{"p": -0.4, "a": -0.1, "d": 0.0, "mood": "低落", "ts": 0}',
                           encoding="utf-8")
    monkeypatch.setattr(em, "_EMOTION_STATE_PATH", state_file)
    em._LAST_DIFF_INJECT = time.time()  # 刚注过 → 5s 内跳过
    assert em.build_emotion_diff_block() == ""


def test_build_emotion_recall_block_lexicon(monkeypatch):
    monkeypatch.setattr(em, "search_by_emotion",
                        lambda tag, top_k=1: [{"content": "和哥哥看了电影", "type": "emotion_label"}])
    block = em.build_emotion_recall_block("今天好开心呀")
    assert block.startswith("\n---\n情感回忆：")
    assert "和哥哥看了电影" in block


def test_build_emotion_recall_block_no_lexicon(monkeypatch):
    monkeypatch.setattr(em, "search_by_emotion", lambda tag, top_k=1: [])
    assert em.build_emotion_recall_block("今天天气不错") == ""


def test_maybe_label_emotion_stores(monkeypatch):
    _patch_llm(monkeypatch, "开心")
    stored = {}
    import v5.store as sm

    def _store(content, type="fact", weight=0.6, tags="", **kw):
        stored["content"] = content
        stored["type"] = type
        stored["tags"] = tags
        return 1
    monkeypatch.setattr(sm, "store", _store)
    tags = em.maybe_label_emotion((0.0, 0, 0), (0.5, 0.3, 0.2), "哥哥夸我")
    assert tags == ["开心"]
    assert stored["type"] == "emotion_label"
    assert "开心" in stored["tags"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
