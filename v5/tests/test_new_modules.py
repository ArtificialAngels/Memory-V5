"""
v5.tests.test_new_modules — V5 9-module smoke test

不依赖 LLM / v5.db / ChromaDB 的纯单元测试。
验证核心数据结构、状态机逻辑、持久化 roundtrip。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))


class TestVitality:
    """#8 隐喻身体状态."""

    def test_initial_value(self):
        from v5.vitality import Vitality
        v = Vitality()
        assert 0.7 <= v.vitality <= 0.8

    def test_tick_conversation_decreases(self):
        from v5.vitality import Vitality
        import time
        v = Vitality()
        v.last_tick = time.time() - 120  # simulate 2 min elapsed
        old = v.vitality
        v = v.tick(conversation=True)
        assert v.vitality < old, f"conversation should decrease vitality (old={old} new={v.vitality})"

    def test_label_mapping(self):
        from v5.vitality import Vitality
        v = Vitality(vitality=0.9)
        assert v.label() == "活力满满"
        v.vitality = 0.2
        assert v.label() == "非常疲惫"

    def test_save_load_roundtrip(self):
        from v5.vitality import Vitality
        v = Vitality(vitality=0.5, conversation_count=3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            v.save(path)
            v2 = Vitality.load(path)
            assert v2.vitality == 0.5
            assert v2.conversation_count == 3
        finally:
            Path(path).unlink(missing_ok=True)


class TestEmotionalMemory:
    """#1 情感因果记忆."""

    def test_no_record_on_small_delta(self):
        from v5.emotional_memory import maybe_record_emotion
        result = maybe_record_emotion(
            (0.1, 0.0, -0.1),
            (0.11, 0.01, -0.09),
            "hello",
        )
        assert result is None

    def test_delta_above_threshold_smoke(self):
        """验证规则降级路径不崩溃 (不依赖 v5.db / LLM)."""
        from v5.emotional_memory import _rule_based_causal
        result = _rule_based_causal("哥哥真棒", (0.1, 0, 0), (0.4, 0, 0))
        assert result is not None
        assert len(result) > 5


class TestRelationship:
    """#2 关系亲密度."""

    def test_initial_stage(self):
        from v5.relationship import Relationship
        r = Relationship()
        assert r.stage() == "才刚认识不久"

    def test_interaction_increases_depth(self):
        from v5.relationship import Relationship
        r = Relationship()
        old_depth = r.depth
        r = r.record_interaction(0.5)
        assert r.depth > old_depth

    def test_warmth_ema(self):
        from v5.relationship import Relationship
        r = Relationship(warmth=0.5)
        r = r.record_interaction(0.8)
        assert r.warmth > 0.5  # should move toward 0.8

    def test_save_load_roundtrip(self):
        from v5.relationship import Relationship
        r = Relationship(depth=0.4, warmth=0.6, interaction_count=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            r.save(path)
            r2 = Relationship.load(path)
            assert r2.depth == 0.4
            assert r2.warmth == 0.6
            assert r2.interaction_count == 10
        finally:
            Path(path).unlink(missing_ok=True)


class TestCare:
    """#4 主动关怀."""

    def test_no_care_on_fresh_state(self):
        from v5.care import CareMonitor
        c = CareMonitor()
        care = c.tick(activity_category="coding", activity_minutes=5)
        assert care is None

    def test_care_triggered_after_long_coding(self):
        from v5.care import CareMonitor
        c = CareMonitor()
        care = c.tick(activity_category="coding", activity_minutes=95,
                      is_late_night=False)
        # After 95min coding → should trigger
        assert care is not None

    def test_save_load_roundtrip(self):
        from v5.care import CareMonitor
        c = CareMonitor(total_reminders=5, cumulative_coding_sec=5400)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            c.save(Path(path))
            c2 = CareMonitor.load(Path(path))
            assert c2.total_reminders == 5
            assert c2.cumulative_coding_sec == 5400
        finally:
            Path(path).unlink(missing_ok=True)


class TestDissonance:
    """#5 认知失调."""

    def test_non_check_type_skips(self):
        from v5.dissonance import detect_dissonance
        result = detect_dissonance("test", mem_type="conversation")
        assert result["conflicts"] == []
        assert result["checked"] == 0

    def test_check_types_defined(self):
        """验证检查类型白名单导入不崩溃."""
        from v5.dissonance import _CHECK_TYPES
        assert "fact" in _CHECK_TYPES
        assert "preference" in _CHECK_TYPES


class TestNarrative:
    """#7 自我叙事."""

    def test_simple_narrative_no_crash(self):
        from v5.narrative import _simple_narrative
        rows = [{"type": "emotional_event", "content": "开心"},
                {"type": "lesson", "content": "学到了耐心"}]
        result = _simple_narrative(rows)
        assert isinstance(result, str)
        assert len(result) > 10
