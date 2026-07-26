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
    # R2 节奏感知
    "rhythm": {
        "tz_offset": int(os.environ.get("IKAROS_TZ_OFFSET", "8")),  # 中国 UTC+8
        "inject_when_recent": True,   # 即使刚聊过也注入精确数据
    },
    # R3 记忆多路检索
    "memory_retrieval": {
        "vector_weight": 0.7,
        "fts_weight": 0.3,
        "time_decay_per_day": 0.05,   # fused × (1 - 0.05 × days)
        "min_fused_score": 0.6,
        "top_k": 5,
        "type_boost": {
            "emotion": 1.2, "fact": 1.1, "conversation": 0.8, "default": 1.0,
        },
    },
    # R4 历史摘要
    "summary": {
        "trigger_rounds": 20,    # history 长度超过才压缩
        "reuse_rounds": 10,      # 10 轮内复用上次摘要
        "max_age_rounds": 30,    # 超过 30 轮旧摘要丢弃
        "model": "local-llm",
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
        "diff_threshold": 0.3,   # |ΔP|+|ΔA|+|ΔD| 超过才注差异句
        "debounce_seconds": 5,    # spec 4.1: 5s 内不重复注
        "timeout_s": 5,
        "tag_vocab": [
            "开心", "愉悦", "平静", "低落", "难过", "焦虑", "专注",
            "期待", "温柔", "害羞", "满足", "生气", "困惑", "怀旧", "无聊",
        ],
    },
    # 4.3 Token 预算
    "token_budget": {
        "min": 800, "max": 1200, "char_x": 1.0,  # char_x 是安全系数: 中文~1token/字, 其他~0.5
    },
    # 运行时内存缓存 (性能优化: 削 embedding 忙时尖峰 + 进程冷启动 chroma 重开 850ms)
    # spec 之外的哥哥优化项: 同 session query embedding 缓存 + VectorIndex 单例(每轮覆写)
    "cache": {
        "embedding_enabled": True,
        "embedding_max": 512,           # LRU 容量 (进程级)
        "vector_index_singleton": True, # 复用 chroma 客户端, 不再每轮重开
        "vector_refresh_seconds": 30,   # 最多 30s 重开一次, 拾取外部(反思循环)新增记忆
        "retrieve_ttl_seconds": 20,    # 检索结果短 TTL: 同 query 20s 内直接返回, 跳过 embedding+chroma
        "retrieve_ttl_enabled": True,
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
