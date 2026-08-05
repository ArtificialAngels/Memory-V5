# 详细说明见 docs/scripts/core/v5/v5/self_discovery.md
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.self_discovery")

# 项目根
IKAROS_ROOT = Path(__file__).resolve().parent.parent.parent  # resolves to Ikaros repo root (core/v5/ -> core/ -> root)
sys.path.insert(0, str(IKAROS_ROOT / "core"))
HERMES_EXE = IKAROS_ROOT / "core" / "hermes" / "venv" / "Scripts" / "hermes.exe"

# 每次读哪些文件来了解自己
_SELF_DISCOVERY_SOURCES = [
    "AGENTS.md",
    "README.md",
    "ikaros-identity/axiom.md",
    "docs/hermes-agent-full-survey.md",
    "docs/v5-architecture-review.md",
    "Ikaros-memory/v5/self_model.py",
    "Ikaros-memory/v5/metacog.py",
]


def _read_sources() -> str:
    """从关键文件取摘要, 了解自己的真实架构."""
    blocks: list[str] = []
    for rel in _SELF_DISCOVERY_SOURCES:
        p = IKAROS_ROOT / rel
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                blocks.append(f"=== {rel} ===\n{text[:400]}")
            except Exception as exc:
                blocks.append(f"=== {rel} ===\n(读取失败: {exc})")
    return "\n\n".join(blocks)


def _analyze(materials: str) -> Optional[dict]:
    """用本地 qwen2.5-7b 分析项目结构, 产出结构化发现.

    比调 Hermes Agent 快得多 (3-5s vs 30-120s),
    且不消耗云端 token。
    """
    prompt = (
        "分析以下项目文件摘要。回答：\n"
        "- 我叫什么？我是什么？\n"
        "- 我有哪些子系统？\n"
        "- 我有什么能力？\n"
        "- 我还不知道什么？\n"
        "只输出分析结果，不要输出文件内容。用第一人称'我'。\n"
        f"\n文件摘要：\n{materials[:2500]}"
    )
    try:
        from v5.reflect.llm_client import call_llm
        resp = call_llm(
            "你是伊卡洛斯。读到这些关于自己的文件，有什么发现？",
            prompt, provider="deepseek", max_tokens=300, temperature=0.7,
        )
        text = (resp.content or "").strip()
        if not text or len(text) < 10:
            return None
        return {"analysis": text[:1000], "sources_read": len(_SELF_DISCOVERY_SOURCES)}
    except Exception as exc:
        logger.warning("self_discovery: local LLM failed: %s", exc)
        return None


def self_discover() -> int:
    """执行一次自我探索, 写入 v4 记忆. 返回写入条数."""
    materials = _read_sources()
    if not materials.strip():
        logger.warning("self_discovery: no sources read")
        return 0

    analysis = _analyze(materials)
    if not analysis:
        return 0

    try:
        from v5 import store as store
        text = f"[自我探索] {analysis['analysis']}"
        mid = store.store(
            text, type="self_discovery", weight=0.85,
            tags="self_discovery,hermes,architecture",
        )
        logger.info("self_discovery: stored id=%d (%d chars)", mid, len(text))
        return 1
    except Exception as exc:
        logger.warning("self_discovery: store failed: %s", exc)
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = self_discover()
    print(f"self_discovery: {n} memories written")
