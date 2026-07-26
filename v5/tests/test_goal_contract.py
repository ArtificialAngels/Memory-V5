"""
v5.tests.test_goal_contract — GoalContract dataclass + draft_contract() 单测

覆盖 (21 case):
  - GoalContract dataclass 行为 (4)
  - from_dict 容错 (3)
  - _extract_json_object 边界 (5)
  - API key 解析 (3)
  - draft_contract 失败路径 (2)
  - draft_contract 真实调用 (4, mock 不打 cloud)

设计: 所有真调 LLM 的 case 用 mock, 不发真请求; 真调路径在 ad-hoc 验证过
(详见会话记录 2026-07-07, 21 passed / 0 failed).

依赖:
  - sys.path.insert(0, Ikaros-memory 根), 让 `import goal_contract` 找得到。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT.parent))

import goal_contract as gc  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# 1) GoalContract dataclass 行为 (4)
# ──────────────────────────────────────────────────────────────────────

class TestGoalContractDataclass:
    def test_outcome_writes_correctly(self):
        c = gc.GoalContract(outcome="X 工作")
        assert c.outcome == "X 工作"

    def test_render_block_skips_empty_fields(self):
        c = gc.GoalContract(outcome="X", verification="pytest 全绿")
        block = c.render_block()
        assert "Outcome" in block
        assert "Verification" in block
        # 空字段不出现在 block 里
        assert "Boundaries" not in block
        assert "Constraints" not in block
        assert "Stop when" not in block

    def test_is_empty_default_returns_true(self):
        assert gc.GoalContract().is_empty() is True

    def test_is_empty_with_content_returns_false(self):
        c = gc.GoalContract(outcome="X")
        assert c.is_empty() is False


# ──────────────────────────────────────────────────────────────────────
# 2) from_dict 容错 (3)
# ──────────────────────────────────────────────────────────────────────

class TestFromDictTolerance:
    def test_numeric_value_coerced_to_str(self):
        d = gc.GoalContract.from_dict({"outcome": 123})
        assert d.outcome == "123"
        assert isinstance(d.outcome, str)

    def test_unknown_fields_ignored(self):
        d = gc.GoalContract.from_dict({"outcome": "x", "unknown_field": "y"})
        # 未知字段不是 GoalContract 属性, 应被丢弃
        assert not hasattr(d, "unknown_field")
        assert d.outcome == "x"

    def test_none_treated_as_empty_string(self):
        d = gc.GoalContract.from_dict({"verification": None, "constraints": None})
        assert d.verification == ""
        assert d.constraints == ""


# ──────────────────────────────────────────────────────────────────────
# 3) _extract_json_object 边界 (5)
# ──────────────────────────────────────────────────────────────────────

class TestJsonExtraction:
    def test_pure_json_parses_directly(self):
        assert gc._extract_json_object('{"a": 1}') == {"a": 1}

    def test_json_with_prefix_chatter(self):
        raw = 'blah blah {"x": 2, "y": "z"} trailing text'
        assert gc._extract_json_object(raw) == {"x": 2, "y": "z"}

    def test_no_json_returns_none(self):
        assert gc._extract_json_object("just text no braces") is None

    def test_empty_string_returns_none(self):
        assert gc._extract_json_object("") is None

    def test_nested_object_extracted(self):
        assert gc._extract_json_object('{"a": {"b": 1}}') == {"a": {"b": 1}}


# ──────────────────────────────────────────────────────────────────────
# 4) API key 解析 (3)
# ──────────────────────────────────────────────────────────────────────

class TestApiKeyResolution:
    def test_base_url_is_http_or_https(self):
        _, base, _ = gc._get_api_key_and_base()
        assert base.startswith("http"), f"unexpected base: {base}"

    def test_model_non_empty(self):
        _, _, model = gc._get_api_key_and_base()
        assert model, "model must not be empty"

    def test_either_key_or_local_fallback(self):
        ak, base, _ = gc._get_api_key_and_base()
        # 要么有 API key (走 cloud), 要么 base 指向本地 LLM (走 :8080)
        assert ak or "127.0.0.1" in base, (
            f"no API key and base is not local: ak={bool(ak)}, base={base}"
        )


# ──────────────────────────────────────────────────────────────────────
# 5) draft_contract 失败路径 (2)
# ──────────────────────────────────────────────────────────────────────

class TestDraftContractFailurePath:
    def test_empty_objective_returns_none(self):
        assert gc.draft_contract("") is None

    def test_whitespace_only_objective_returns_none(self):
        assert gc.draft_contract("   \n  \t  ") is None


# ──────────────────────────────────────────────────────────────────────
# 6) draft_contract 真实调用 — mock 不打 cloud (4)
# ──────────────────────────────────────────────────────────────────────

class TestDraftContractLiveCall:
    """所有真调 LLM 的 case 用 mock, 不发真请求。

    真调路径在会话 2026-07-07 ad-hoc 验证过 (21 passed / 0 failed).
    这里只验: 走对路径 / 解析对响应 / 退化到 None 当响应为空.
    """

    def _mock_call(self, raw: str):
        """构造一个 _call_llm_sync 的 mock, 返回 raw."""
        return patch.object(gc, "_call_llm_sync", return_value=raw)

    def test_draft_returns_non_none(self):
        with self._mock_call('{"outcome": "x", "verification": "y"}'):
            out = gc.draft_contract("some objective")
        assert out is not None

    def test_draft_returns_goal_contract_instance(self):
        with self._mock_call('{"outcome": "x", "verification": "y"}'):
            out = gc.draft_contract("some objective")
        assert isinstance(out, gc.GoalContract)

    def test_draft_not_empty(self):
        with self._mock_call('{"outcome": "x", "verification": "y"}'):
            out = gc.draft_contract("some objective")
        assert not out.is_empty()

    def test_render_block_contains_outcome(self):
        with self._mock_call('{"outcome": "完成排序", "verification": "find . -name *.py | wc -l"}'):
            out = gc.draft_contract("some objective")
        assert "Outcome" in out.render_block()

    def test_draft_returns_none_when_llm_returns_empty(self):
        with self._mock_call(""):
            assert gc.draft_contract("some objective") is None

    def test_draft_returns_none_when_llm_returns_non_json(self):
        with self._mock_call("just chatter no JSON at all"):
            assert gc.draft_contract("some objective") is None

    def test_draft_returns_none_when_json_is_empty_contract(self):
        # LLM 回了 JSON 但五字段都是空串 → from_dict → is_empty → 返回 None
        with self._mock_call('{"outcome": "", "verification": "", "constraints": "", "boundaries": "", "stop_when": ""}'):
            assert gc.draft_contract("some objective") is None