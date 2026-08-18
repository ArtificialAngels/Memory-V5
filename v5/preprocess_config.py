# 详细说明见 docs/scripts/core/v5/v5/preprocess_config.md
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("ikaros.v5.preprocess_config")

_CONFIG_PATH = Path(__file__).resolve().parent / "preprocess_config.yaml"
_CACHE: dict | None = None

_DEFAULTS: dict = {
    # ⚠️ P7 (2026-08-14): _DEFAULTS 与 preprocess_config.yaml 同步 (防漂移).
    # yaml 是权威源; 本默认值仅在 yaml 缺失/损坏时兜底, 必须与 yaml 一致,
    # 否则回退会落到错误值 (如 min_fused_score 0.6 会把语义召回打空)。
    # 防漂移测试: tests/test_config_alignment.py。
    # R2 节奏感知
    "rhythm": {
        "tz_offset": int(os.environ.get("IKAROS_TZ_OFFSET", "8")),  # 中国 UTC+8
        "inject_when_recent": True,
    },
    # R3 记忆多路检索
    "memory_retrieval": {
        "vector_weight": 0.7,
        "fts_weight": 0.3,
        "time_decay_per_day": 0.05,
        "min_fused_score": 0.3,   # 2026-07-26 标定: 0.6 会把有效召回全过滤掉
        "top_k": 5,
        "auto_route": True,
        "graph_min_score": 0.2,
        "frequency_weight": 0.05,
        "reinforcement_weight": 0.10,
        "freshness_weight": 0.08,
        "long_term_boost": 0.05,
        # Phase 4 全套加权
        "base_weight_factor": 0.5,
        "merge_reinforce_increment": 0.05,
        "type_decay": {
            "conversation": {"per_day": 0.05, "floor": 0.2},
            "fact": {"per_day": 0.03, "floor": 0.4},
            "emotion_label": {"per_day": 0.05, "floor": 0.2},
            "emotional_event": {"per_day": 0.04, "floor": 0.3},
            "preference": {"per_day": 0.02, "floor": 0.5},
            "user_trait": {"per_day": 0.01, "floor": 0.6},
            "identity": {"per_day": 0.005, "floor": 0.7},
            "decision": {"per_day": 0.005, "floor": 0.7},
            "lesson": {"per_day": 0.01, "floor": 0.6},
            "convention": {"per_day": 0.005, "floor": 0.7},
            "pitfall": {"per_day": 0.005, "floor": 0.7},
            "default": {"per_day": 0.05, "floor": 0.2},
        },
        "situational": {
            "enabled": True,
            "project_activity_boost": 0.10,
            "hour_match_boost": 0.05,
        },
        "type_boost": {
            "emotion": 1.2, "fact": 1.1, "user_trait": 1.15,
            "preference": 1.05, "conversation": 1.0, "default": 1.0,
        },
        "intent": {
            "enabled": True,
            "why": {"decision": 1.3, "lesson": 1.1, "conversation": 1.1},
            "when": {"conversation": 1.15, "fact": 1.1},
            "entity": {"fact": 1.2, "preference": 1.1, "identity": 1.1},
            "general": {},
        },
    },
    # R4 历史摘要 (model 字段已废弃移除, 摘要固定走云端 deepseek)
    "summary": {
        "trigger_rounds": 20,
        "reuse_rounds": 10,
        "max_age_rounds": 30,
        "max_sentences": 3,
        "timeout_s": 5,
    },
    # R5 关系引擎
    "relationship": {
        "confidence_gate": 0.7,
        "inject_on_signal": True,
    },
    # R6 情感引擎增强
    "emotion": {
        "label_model": "local-llm",
        "max_tags": 2,
        "diff_inject": True,
        "diff_threshold": 0.3,
        "debounce_seconds": 5,
        "timeout_s": 5,
        "tag_vocab": [
            "开心", "愉悦", "平静", "低落", "难过", "焦虑", "专注",
            "期待", "温柔", "害羞", "满足", "生气", "困惑", "怀旧", "无聊",
        ],
    },
    # 4.3 Token 预算
    "token_budget": {
        "min": 800, "max": 1200, "char_x": 1.0,
    },
    # 运行时内存缓存
    "cache": {
        "embedding_enabled": True,
        "embedding_max": 512,
        "vector_index_singleton": True,
        "vector_refresh_seconds": 30,
        "retrieve_ttl_seconds": 20,
        "retrieve_ttl_enabled": True,
        "ontology_align_enabled": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(dict(base[k]), v)
        else:
            base[k] = v
    return base


def cfg() -> dict:
    """返回配置 dict (进程内缓存). 失败返回内置默认."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data = _deep_merge(dict(_DEFAULTS), {})
    try:
        try:
            import yaml  # noqa: F401
        except ImportError:
            logger.debug("pyyaml not available, preprocess_config uses defaults")
            _CACHE = data
            return data
        if _CONFIG_PATH.is_file():
            loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = _deep_merge(data, loaded)
    except Exception as e:
        logger.warning("preprocess_config load failed, using defaults: %s", e)
    _CACHE = data
    return data


def get(*keys: str, default: Any = None) -> Any:
    """链式取值: get('rhythm','tz_offset', default=8)."""
    d: Any = cfg()
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


if __name__ == "__main__":
    import json
    print(json.dumps(cfg(), ensure_ascii=False, indent=2))
