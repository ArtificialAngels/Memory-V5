"""v5 结构化管道格式守卫回归测试 (Task #15).

Guard 必须拦截: LLM 旁白前缀 / 泄漏 JSON / markdown fence / 超长原始 dump;
必须放行: 正常的结构化陈述 (事实 / 偏好 / 教训 / user_trait).
"""
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent  # core/v5
sys.path.insert(0, str(V5_ROOT.parent))

from v5.validation import is_clean_structured_content, guard_structured_content  # noqa: E402


def test_blocks_english_narration():
    bad = "Okay, the user wants me to remember that he likes concise replies."
    assert not is_clean_structured_content(bad)
    assert any(e.code.value == "V5-0109" for e in guard_structured_content(bad))


def test_blocks_chinese_narration():
    bad = "好的，用户说哥哥喜欢短句直接的回答"
    assert not is_clean_structured_content(bad)


def test_blocks_markdown_fence():
    bad = "```json\n[{\"content\": \"哥哥喜欢短句\", \"type\": \"preference\"}]\n```"
    assert not is_clean_structured_content(bad)


def test_blocks_leaked_json():
    bad = '{"content": "哥哥喜欢短句", "type": "preference"}'
    assert not is_clean_structured_content(bad)


def test_blocks_runaway_dump():
    bad = "哥哥喜欢" * 300  # 1200 字, 远超 800 字上限
    assert not is_clean_structured_content(bad)


def test_allows_medium_statement():
    # 接近上限但合理的陈述仍应放行 (守卫只拦明显 dump)
    ok = "哥哥的沟通偏好很明确: 喜欢短句直接、带具体例子的回答, 不喜欢空泛的修辞和绕弯子"
    assert len(ok) < 800
    assert is_clean_structured_content(ok)


def test_allows_clean_fact():
    ok = "哥哥喜欢短句直接的回答, 带例子比纯结论更好"
    assert is_clean_structured_content(ok)
    assert guard_structured_content(ok) == []


def test_allows_clean_preference_with_example():
    ok = "哥哥偏好直接、命令式的语气, 说人话比修辞更有效"
    assert is_clean_structured_content(ok)


def test_allows_clean_lesson():
    ok = "反复确认的规律: 当哥哥沉默超过 30 秒, 通常代表他在思考而非不满"
    assert is_clean_structured_content(ok)


def test_empty_is_clean_here():
    # 空串由 NotEmptyRule 在 store() 层处理, 守卫不重复报
    assert is_clean_structured_content("")
    assert is_clean_structured_content("   ")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
