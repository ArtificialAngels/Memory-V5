"""v5 节奏感知单元测试 (R2). 纯规则, 不依赖外部服务."""
import os
import sys
import time
from pathlib import Path

import pytest

# 把 Ikaros-memory 加入 path 以便 import v5
_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent.parent  # Ikaros-memory/
if str(_V5) not in sys.path:
    sys.path.insert(0, str(_V5))

from v5 import rhythm  # noqa: E402


def test_period_label():
    assert rhythm._period_label(2) == "深夜"
    assert rhythm._period_label(9) == "上午"
    assert rhythm._period_label(12) == "中午"
    assert rhythm._period_label(15) == "下午"
    assert rhythm._period_label(20) == "晚上"
    assert rhythm._period_label(23) == "深夜"


def test_build_rhythm_block_format():
    block = rhythm.build_rhythm_block()
    assert block.startswith("\n---\n当前节奏：")
    assert "距上轮:" in block
    assert "时段:" in block
    # 形如 时段: 深夜(02:15)
    assert "(" in block and ")" in block


def test_gap_recent_is_just_now():
    # monkeypatch last_interaction_ts 返回 now → 距上轮应为"刚刚"
    real = rhythm.last_interaction_ts
    rhythm.last_interaction_ts = lambda: time.time()
    try:
        block = rhythm.build_rhythm_block()
        assert "刚刚" in block
    finally:
        rhythm.last_interaction_ts = real


def test_gap_hours_format():
    real = rhythm.last_interaction_ts
    rhythm.last_interaction_ts = lambda: time.time() - 3 * 3600 - 42 * 60
    try:
        block = rhythm.build_rhythm_block()
        assert "3小时42分" in block
    finally:
        rhythm.last_interaction_ts = real


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
