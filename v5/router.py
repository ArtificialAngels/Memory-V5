# 详细说明见 docs/scripts/core/v5/v5/router.md

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.router")

# ─── 任务关键词 (命中任一即为 task) ───────────────────────────

_TASK_KEYWORDS: list[str] = [
    "帮我", "写一个", "写一段", "创建一个", "生成", "创建",
    "修改", "重构", "优化", "分析", "比较", "总结", "解决",
    "修复", "调试", "解释", "实现", "部署", "配置", "安装",
    "测试", "搜索", "查一下", "看看", "评估", "设计",
    "怎么做", "怎么实现", "怎么修改", "如何",
    "refactor", "debug", "fix", "implement", "write", "create",
    "analyze", "compare", "summarize", "explain",
]

# ─── 对话关键词 (命中任一即为 conversation) ──────────────────

_CONV_KEYWORDS: list[str] = [
    "你开心吗", "你好吗", "你在干嘛", "你在做什么",
    "晚安", "早安", "下午好", "早上好",
    "我想你", "喜欢你", "爱你", "抱抱",
    "开心", "好不好", "好呀", "嗯",
    "哈哈", "呵呵", "嘿嘿",
    "谢谢", "辛苦了", "感谢",
    "没事", "没关系", "好吧",
]

# ─── 正则模式 (文件路径 / 代码块 / 符号等) ─────────────────

_TASK_PATTERNS: list[str] = [
    r"[A-Za-z]:\\|\\\\",         # Windows 路径
    r"\.py\b|\.ts\b|\.js\b|\.rs\b",  # 文件扩展名
    r"```",                       # 代码块
    r"\/\/|\/\*",                 # 注释符号
    r"def |class |import |from ", # 代码关键词
    r"git\s+\w+",                 # git 命令
]


def classify(text: str) -> str:
    """分类: 'task' | 'conversation'.

    使用关键词 + 正则匹配, O(n) 快速路径.
    不做 LLM 调用分类 (太慢 + 浪费 token).
    """
    text_lower = text.strip().lower()
    if not text_lower:
        return "conversation"

    # 优先级1: 强任务信号 (代码/路径/命令)
    for pat in _TASK_PATTERNS:
        if re.search(pat, text):
            return "task"

    # 优先级2: 任务关键词
    for kw in _TASK_KEYWORDS:
        if kw in text_lower:
            # 但如果是 "帮我看看" 这种偏对话的, 走 conversation
            if text_lower in ("帮我看看", "帮我看一下"):
                return "conversation"
            return "task"

    # 优先级3: 对话关键词
    for kw in _CONV_KEYWORDS:
        if kw in text_lower:
            return "conversation"

    # 优先级4: 短文本 ≤ 5 字 → 对话 (自然口语)
    if len(text_lower) <= 5:
        return "conversation"

    # 默认: conversation (宁可为对话错误使用云 LLM, 也不把对话当任务处理)
    return "conversation"


def optimize_task(text: str) -> Optional[str]:
    """用云端 LLM (DeepSeek) 把模糊任务指令优化为结构化指令.

    云端 LLM 强制思考模式, 输出是内部推理而非 JSON.
    策略: 把推理内容里最后 3 句行动指南提取为优化指令.
    """
    memory_context = _search_relevant(text)
    raw = _call_task_refiner(text, memory_context)
    if not raw or len(raw) < 20:
        return None

    # qwen3 思考输出的最后 3-5 句通常是行动指令
    sentences = re.split(r"[。！\n]", raw)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return None
    # 取最后 3 句, 但不超过 300 chars
    tail = sentences[-3:]
    result = "。\n".join(tail)
    if len(result) > 300:
        # 如果太长, 取最后一句
        result = sentences[-1]
    if len(result) > 300:
        result = result[:297] + "..."
    return result.strip()


def _search_relevant(text: str) -> str:
    """搜 V4 找与当前任务相关的记忆上下文."""
    try:
        from v5 import store as store
        hits = store.search(text, top_k=3, min_weight=0.4)
        if not hits:
            return ""
        lines = []
        for h in hits:
            label = f"[{h.type}][w={h.weight}]"
            lines.append(f"{label} {h.content[:200]}")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("router: memory search failed (%s)", exc)
        return ""


def _call_task_refiner(text: str, memory_context: str) -> Optional[str]:
    """调云端 LLM (DeepSeek) 优化任务指令. 强制输出 JSON."""
    system = (
        "你是一个任务指令优化器。输出严格 JSON, 不要多余话, 不要推理过程:\n"
        "{\n"
        '  "goal": "一句话目标",\n'
        '  "context": "背景",\n'
        '  "output": "期望产出物",\n'
        '  "constraints": "限制条件",\n'
        '  "skills": ["相关技能名"]\n'
        "}\n"
        "不要解释, 不要聊天, 只输出 JSON。"
    )
    # Inject contextually relevant operation rules (semantic retrieval via :8587)
    try:
        from v5.rules_retriever import retrieve_relevant_rules
        rules_block = retrieve_relevant_rules(text)
        if rules_block:
            system += "\n\n" + rules_block
    except Exception:
        pass
    user = f"原始指令: {text}"
    if memory_context:
        user += f"\n\n相关记忆:\n{memory_context}"

    try:
        from v5.reflect.llm_client import call_llm_auto
        result = call_llm_auto(
            system, user,
            max_tokens=512, timeout=60,
        )
        out = result.content.strip()
        if len(out) < 10:
            return None
        return out
    except Exception as exc:
        logger.warning("router: task refiner failed (%s), using raw text", exc)
        return None


# ─── 便捷入口 ───────────────────────────────────────────────

def route(text: str) -> dict:
    """完整路由: 分类 + (如果是任务则优化).

    Returns:
        {
            "type": "conversation" | "task",
            "optimized_text": str | None,  # 仅任务时有
            "elapsed_ms": float,
        }
    """
    t0 = time.time()
    result: dict = {"type": "conversation", "optimized_text": None}

    cls = classify(text)
    result["type"] = cls

    if cls == "task":
        optimized = optimize_task(text)
        if optimized:
            result["optimized_text"] = optimized

    result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return result


# ─── CLI 测试 ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "哥哥好",
        "帮我写个 Python 脚本, 把 Downloads 文件夹按后缀名分类",
        "你开心吗",
        "晚安",
        "分析一下这个函数的时间复杂度 def foo(n): for i in range(n): print(i)",
        "我想你了",
        "修改 AGENTS.md 第 3 节, 把 bridge-rs 引用去掉",
    ]
    for t in tests:
        r = route(t)
        cls = r["type"]
        opt = r.get("optimized_text", "")
        print(f"[{cls:14s}] {t[:40]}")
        if opt:
            print(f"  {'优化':>14s} {opt[:100]}...")
        print()
