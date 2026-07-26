"""v5 用户画像单元测试 (R5). mock v4.store, 验证负面优先/置信度门控/隔离."""
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

import v5.profile as pf  # noqa: E402


class _M:
    def __init__(self, content, weight):
        self.content = content
        self.weight = weight


def test_empty_when_no_data(monkeypatch):
    monkeypatch.setattr(pf, "_read", lambda kind, limit=50: [])
    assert pf.build_profile_block() == ""


def test_dislikes_prioritized_and_gated(monkeypatch):
    def fake_read(kind, limit=50):
        if kind == "dislike":
            return [("啰嗦", 0.9), ("验证脚本输出", 0.5)]  # 0.5 低于门控
        return []
    monkeypatch.setattr(pf, "_read", fake_read)
    block = pf.build_profile_block()
    assert "不喜欢啰嗦" in block
    assert "验证脚本输出" not in block  # 置信度不足被门控


def test_preferences_used_when_no_dislikes(monkeypatch):
    def fake_read(kind, limit=50):
        if kind == "preference":
            return [("简洁直接", 0.95)]
        return []
    monkeypatch.setattr(pf, "_read", fake_read)
    block = pf.build_profile_block()
    assert "偏好简洁直接" in block


def test_record_writes_to_store(monkeypatch):
    saved = {}
    def fake_store(content, type, weight, tags):
        saved["content"] = content
        saved["type"] = type
        return 1
    monkeypatch.setattr(pf, "_read", lambda kind, limit=50: [])
    # 直接测 record 的 store 调用
    import v5.store as sm
    monkeypatch.setattr(sm, "store", fake_store)
    mid = pf.record("dislike", "客套话", weight=0.85)
    assert mid == 1
    assert saved["type"] == "dislike"
    assert saved["content"] == "客套话"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
