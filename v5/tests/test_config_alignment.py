# -*- coding: utf-8 -*-
"""配置双源防漂移测试 (P7, 2026-08-14).

preprocess_config.yaml 是权威源, preprocess_config.py._DEFAULTS 是 yaml 缺失时的
兜底。二者一旦漂移, 回退会落到错误值 (曾发生: _DEFAULTS.min_fused_score=0.6 而
yaml 标定 0.3, yaml 缺失时语义召回全被打空)。本测试强制二者键结构对齐。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # core/

import v5.preprocess_config as pc


def _leaf_keys(d: dict, path: str = "") -> set:
    out = set()
    for k, v in d.items():
        p = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            out |= _leaf_keys(v, p)
        else:
            out.add(p)
    return out


def _load_yaml() -> dict:
    import yaml
    return yaml.safe_load(pc._CONFIG_PATH.read_text(encoding="utf-8")) or {}


def test_defaults_cover_yaml_keys():
    yaml_data = _load_yaml()
    yaml_keys = _leaf_keys({k: v for k, v in yaml_data.items() if k != "validation"})
    def_keys = _leaf_keys(pc._DEFAULTS)
    missing = yaml_keys - def_keys
    assert not missing, f"_DEFAULTS 缺 yaml 键 (yaml 缺失时回退不完整): {sorted(missing)}"


def test_defaults_no_stale_keys():
    yaml_data = _load_yaml()
    yaml_keys = _leaf_keys({k: v for k, v in yaml_data.items() if k != "validation"})
    def_keys = _leaf_keys(pc._DEFAULTS)
    stale = def_keys - yaml_keys
    assert not stale, f"_DEFAULTS 有 yaml 已删的陈旧键: {sorted(stale)}"


def test_critical_fallback_values():
    """关键兜底值必须与 yaml 标定一致 (防回退打到错误值)."""
    mr = pc._DEFAULTS["memory_retrieval"]
    assert mr["min_fused_score"] == 0.3, "min_fused_score 0.6 会把语义召回打空"
    assert "model" not in pc._DEFAULTS["summary"], "summary.model 已废弃, 不应残留"
    for k in ("type_decay", "situational", "intent", "base_weight_factor",
              "merge_reinforce_increment", "auto_route", "graph_min_score"):
        assert k in mr, f"_DEFAULTS.memory_retrieval 缺 {k} (Phase 4/路由键)"


def test_merged_cfg_has_phase4():
    """cfg() 合并后 (yaml 覆盖) 必须含 Phase 4 全套加权键."""
    cfg = pc.cfg()
    mr = cfg.get("memory_retrieval", {})
    assert mr.get("min_fused_score") == 0.3
    assert "type_decay" in mr and "intent" in mr and "situational" in mr
    assert "model" not in cfg.get("summary", {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
