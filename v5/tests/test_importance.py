# -*- coding: utf-8 -*-
"""统一重要性模块 (P2 融汇, 2026-08-14) 测试."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # core/

from v5.importance import effective_importance, memory_importance


def test_ei_access_boost():
    now = 1_800_000_000.0
    cold = effective_importance(0.6, 0, 0.0, now)
    hot = effective_importance(0.6, 7, now, now)
    assert hot > cold


def test_ei_decay():
    now = 1_800_000_000.0
    recent = effective_importance(0.6, 0, now, now)
    stale = effective_importance(0.6, 0, now - 60 * 86400, now)
    assert recent > stale


def test_ei_reinforcement():
    now = 1_800_000_000.0
    base = effective_importance(0.6, 0, 0.0, now)
    rein = effective_importance(0.6, 0, 0.0, now, reinforcement=1.0)
    assert rein > base


def test_memory_importance_from_dict():
    now = 1_800_000_000.0
    d = {"weight": 0.8, "access_count": 3, "last_accessed": now,
         "reinforcement": 0.5}
    assert memory_importance(d, now) == effective_importance(0.8, 3, now, now, 0.5)


def test_memory_importance_from_row_like():
    now = 1_800_000_000.0

    class _R:
        weight = 0.7
        access_count = 1
        last_accessed = now
        reinforcement = 0.0

    assert memory_importance(_R(), now) == effective_importance(0.7, 1, now, now, 0.0)


def test_ei_bounded():
    now = 1_800_000_000.0
    # 极端值: weight 超界被 clamp 到 1.0; access_factor 按 log2 增长 (10^6 次访问→20);
    # reinforcement 封顶 +0.5。EI = 1.0 × 20 × 1.5 = 30 (数值有限, 不爆炸)
    v = effective_importance(5.0, 10**6, now, now, 10.0)
    assert v == pytest.approx(30.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
