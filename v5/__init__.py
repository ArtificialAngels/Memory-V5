# 详细说明见 docs/scripts/core/v5/v5/__init__.md

from __future__ import annotations

__version__ = "5.1.0"
SCHEMA_VERSION = __version__  # data/v5/*.json schema version


# ─── Ekko 启发: 受控种类 (Kind) 系统 ──────────────────────
# 限制模型可以写入的键空间, 防止 state 污染.
# 每个受控 kind 定义: domain / state_key (data/v5/*.json) / type / 描述
# 写入走 validate_state_key() 验证, 拒绝未注册的键.

CONTROLLED_KINDS: dict[str, dict[str, str]] = {
    # 自我/身份
    "identity.name":      {"domain": "profile", "state_file": "self_model.json", "type": "fact",
                           "desc": "用户姓名"},
    "identity.nature":    {"domain": "profile", "state_file": "self_model.json", "type": "fact",
                           "desc": "伊卡洛斯的本质"},
    "identity.creator":   {"domain": "profile", "state_file": "self_model.json", "type": "fact",
                           "desc": "创造者"},
    "identity.created":   {"domain": "profile", "state_file": "self_model.json", "type": "fact",
                           "desc": "创建日期"},
    "identity.vibe":      {"domain": "profile", "state_file": "self_model.json", "type": "preference",
                           "desc": "画风/气质"},
    # 情感
    "affect":             {"domain": "affect",  "state_file": "affect.json",     "type": "state",
                           "desc": "六维情感状态 (PAD + TLS)"},
    # 关系
    "relationship":       {"domain": "relation","state_file": "relationship.json","type": "state",
                           "desc": "与哥哥的关系模型"},
    # 活力
    "vitality":           {"domain": "vitality","state_file": "vitality.json",   "type": "state",
                           "desc": "精力状态"},
    # 自我叙事
    "self_narrative":     {"domain": "narrative","state_file": "self_model.json", "type": "narrative",
                           "desc": "月度自我叙事"},
    # 元认知
    "metacog":            {"domain": "metacog", "state_file": "self_model.json", "type": "state",
                           "desc": "内省统计"},
    # 关怀
    "care":               {"domain": "care",    "state_file": "care.json",       "type": "state",
                           "desc": "关怀策略状态"},
    # 探索欲
    "curiosity":          {"domain": "curiosity","state_file": "self_model.json", "type": "state",
                           "desc": "好奇心/探索欲"},
}


def validate_state_key(key: str) -> dict | None:
    """校验 state key 是否在受控种类中. 返回 kind 信息, 未注册返回 None."""
    return CONTROLLED_KINDS.get(key)


def registered_state_keys() -> list[str]:
    """返回所有已注册的受控 key."""
    return list(CONTROLLED_KINDS.keys())


# ─── 导出 ────────────────────────────────────────────
__all__ = [
    "AffectState", "EMOTION_MAP", "load_state", "save_state",
    "SCHEMA_VERSION", "CONTROLLED_KINDS", "validate_state_key",
    "registered_state_keys",
]
