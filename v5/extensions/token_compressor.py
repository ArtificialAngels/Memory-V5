"""
token_compressor.py — V5 token 级压缩预通道 (骨架 / EXPERIMENTAL)
=================================================================

问题背景
--------
V5 当前所有"上下文缩减"手段都是 **LLM 语义摘要**(summary.trigger_rounds=20 /
Hermes ContextCompressor) 或 **硬性 top_k 截断**。配置里的 `token_budget`
(min 800 / max 1200 / char_x) 在 `preprocess_config.yaml:58` 定义、在
`preprocess_config.py:59` 加载, 但 **主检索代码无任何消费点**
(`memory_retrieval.py` 不引用它) —— 等于一个空配置。

这正对应上一轮结论: V5 相对 LLMLingua 的硬差距是"完全没有 token 级压缩",
所有缩减都靠语义摘要(有损 + 依赖 LLM)或硬截断(非压缩)。

本骨架目标
----------
  (a) 压缩旧对话轮 —— 中段旧轮冗余 token 删掉, 近期 tail 原样保留;
  (b) 压缩检索块 —— 把 retrieve() 返回的旧记忆 content 裁到预算内, 而非整条丢弃;
  (c) 预算强制执行 —— 给定一组文本块 + max token 预算, 保高相关、压/截低相关。

默认**离线规则压缩**(零 LLM 成本, 适配本地 1.7B 小模型); 同时**委派微软
`llmlingua` 现成库**做真实 token 级压缩——导入守护, 环境装了 `llmlingua` 就自动
启用, 未装/离线自动回退规则(不破坏 U 盘便携性)。`llm_compress(quality=...)`
还支持显式调本地 :8080 做更高质量压缩。

接入点(详见同目录 EXTENSIONS.md)
---------------------------------
  - hermes 插件 `on_pre_compress` 组装 memory-context 前, 对检索结果跑
    `compress_retrieval_block()`;
  - :8080 本地小模型走 V5 构建 system/记忆前缀时, 对整段跑 `enforce_budget()`。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("ikaros.v5.ext.token_compressor")


# ─── token 估算 (复用 token_budget.char_x 安全系数) ───────────────

def _char_x() -> float:
    try:
        from v5 import preprocess_config as pc
        return float(pc.cfg().get("token_budget", {}).get("char_x", 1.0))
    except Exception:
        return 1.0


def est_tokens(text: str) -> int:
    """粗略 token 估算: 中文~1token/字, 其他~0.5token/字符(char_x 系数)."""
    if not text:
        return 0
    return max(1, int(len(text) * _char_x()))


# ─── 现成库委派: 微软 llmlingua (真实 token 级压缩) ─────────────
# 导入守护: Ikaros 是 U 盘离线便携应用, 不强装 llmlingua (它首次运行需从
# HuggingFace 下载模型权重, 会破坏离线便携性)。装了就用, 没装/离线自动回退规则。
try:
    from llmlingua import PromptCompressor  # pip install llmlingua
    _LLMLINGUA_AVAILABLE = True
except Exception:  # pragma: no cover - 离线便携环境常态
    PromptCompressor = None
    _LLMLINGUA_AVAILABLE = False

_LLMLINGUA_INSTANCE = None


def _get_llmlingua():
    """懒加载 llmlingua 单例; 任何失败(离线/无模型)返回 None 触发回退。"""
    global _LLMLINGUA_INSTANCE
    if not _LLMLINGUA_AVAILABLE:
        return None
    if _LLMLINGUA_INSTANCE is None:
        try:
            _LLMLINGUA_INSTANCE = PromptCompressor()
        except Exception as exc:  # 首次实例化会下载 HF 模型, 离线即失败
            logger.warning("token_compressor: llmlingua 初始化失败(可能离线/无模型), "
                           "回退规则压缩 (%s)", exc)
            return None
    return _LLMLINGUA_INSTANCE


def llmlingua_compress(text: str, *, target_token: Optional[int] = None,
                       rate: float = 0.5) -> Optional[str]:
    """调现成库 llmlingua 做 token 级压缩 (LLMLingua / LongLLMLingua)。

    返回压缩文本; 不可用(未安装/离线/压缩失败)返回 None, 调用方须回退规则。
    API: PromptCompressor().compress_prompt(text, instruction="", question="",
        target_token=...) -> {"compressed_prompt":..., "ratio":...}
    """
    comp = _get_llmlingua()
    if comp is None:
        return None
    try:
        kw = {"target_token": target_token} if target_token else {"rate": rate}
        res = comp.compress_prompt(text, instruction="", question="", **kw)
        compressed = (res or {}).get("compressed_prompt")
        return compressed or None
    except Exception as exc:
        logger.warning("token_compressor: llmlingua 压缩失败, 回退规则 (%s)", exc)
        return None


def compress_text(text: str, *, quality: str = "auto",
                  target_token: Optional[int] = None, ratio: float = 0.5) -> str:
    """统一压缩入口。

    quality:
      "auto" (默认) llmlingua(装了就用) → 规则回退, 不调 LLM, 离线零成本;
      "llm"          本地 :8080 高质量压缩 → 规则回退;
      "rule"         仅规则压缩(最稳, 离线安全)。
    """
    if quality == "llm":
        return llm_compress(text, max_tokens=target_token or 200, quality="llm")
    got = llmlingua_compress(text, target_token=target_token, rate=ratio)
    if got is not None:
        return got
    return rule_compress(text, ratio=ratio)


# ─── 规则压缩 (不调 LLM, 适配本地 1.7B 小模型零成本) ─────────────

_FILLER_RE = re.compile(r"(?:\s*[，。、；：,.!?；：]+\s*){2,}")  # 连续重复标点
_WS_RE = re.compile(r"\s+")

_FILLERS = {
    "好的。", "好的", "嗯。", "嗯", "哦。", "哦", "啊。", "好的好的",
    "明白了。", "明白了", "收到。", "收到", "ok。", "ok", "okay",
}

# 低于此 token 数的文本不做中段截断(否则会把关键信息截残)
_MIN_COMPRESS_TOKENS = 40


def rule_compress(text: str, ratio: float = 0.5) -> str:
    """确定性、零 LLM 的文本瘦身 (LLMLingua 思路的极简规则版, 非语义).

    ratio: 目标压缩比(0~1), 软指引, 不保证精确。
    策略:
      1. 先按行删纯语气 filler 短句 + 相邻重复行 (保留换行语义)
      2. 折叠行内空白 / 连续重复标点
      3. 仅当原文 token >= _MIN_COMPRESS_TOKENS 且仍超 ratio 目标,
         从中间等长截断(保头尾, 缓解 lost-in-the-middle); 短文本直接保留
    """
    if not text:
        return ""
    # 1) 行级清洗必须在折叠空白之前做, 否则 \\n 被压成空格会让 filler 检测失效
    kept_lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if len(s) >= 2 and s not in _FILLERS:
            kept_lines.append(s)
    deduped = []
    for ln in kept_lines:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)
    t = "\n".join(deduped)
    # 2) 行内空白折叠 + 连续标点折叠
    t = _WS_RE.sub(" ", t).strip()
    t = _FILLER_RE.sub(lambda m: m.group(0)[:1], t)

    # 3) 仅对较长文本做中段截断, 短文本直接保留(避免把关键信息截没)
    target = max(1, int(est_tokens(text) * ratio))
    if est_tokens(text) >= _MIN_COMPRESS_TOKENS and est_tokens(t) > target:
        toks = max(1, est_tokens(t))
        keep = max(1, target // 2)
        head_len = max(1, int(len(t) * (keep / toks)))
        tail_len = max(1, int(len(t) * (keep / toks)))
        head = t[:head_len]
        tail = t[len(t) - tail_len:]
        t = (head + " … " + tail).strip()
    return t


def llm_compress(text: str, *, max_tokens: int = 200,
                 provider: str = "local", quality: str = "auto") -> str:
    """可选高质量压缩: 优先级 llmlingua(现成库) → 本地 :8080 → 规则回退。

    quality:
      "auto" (默认) 装了 llmlingua 就用它, 否则本地 LLM, 再否则规则;
      "llm"          强制本地 :8080;
      "rule"         强制规则(零成本, 离线安全)。
    """
    if quality == "rule":
        return rule_compress(text)
    if quality == "auto":
        got = llmlingua_compress(text, target_token=max_tokens)
        if got is not None:
            return got
    # llm path (auto 且 llmlingua 不可用时, 或强制 llm)
    try:
        from v5.reflect import llm_client
        prompt = (
            "压缩下面这段文本, 删除冗余/重复/口语 filler, 保留全部事实与关键指代, "
            f"控制在 {max_tokens} token 内。只输出压缩结果, 不要解释:\n\n" + text
        )
        out = llm_client.call_llm(prompt, "", provider=provider,
                                  max_tokens=max_tokens, temperature=0.0)
        return out.content.strip() or text
    except Exception as exc:
        logger.debug("token_compressor: 本地 LLM 压缩失败, 回退规则 (%s)", exc)
    return rule_compress(text)


# ─── 旧轮压缩 ───────────────────────────────────────────────────

def compress_old_rounds(
    rounds: list[dict],
    *,
    tail_keep: int = 6,
    budget_tokens: Optional[int] = None,
    ratio: float = 0.5,
) -> list[dict]:
    """压缩对话旧轮, 保护最近 tail_keep 轮原样 (与 Hermes 压缩器头尾保护互补)。

    rounds: [{"role":..., "content":..., "score":...}, ...]  按时间升序
    返回同结构; 旧轮 rule_compress 瘦身, tail 原样; 超 budget 的旧轮进一步裁切。
    """
    if not rounds:
        return []
    n = len(rounds)
    cut = max(0, n - tail_keep)
    out: list[dict] = []
    for i, r in enumerate(rounds):
        if i >= cut:
            out.append(dict(r))  # tail 原样
            continue
        content = r.get("content", "") or ""
        compressed = compress_text(content, ratio=ratio) if len(content) > 40 else content
        new_r = dict(r)
        new_r["content"] = compressed
        new_r["_compressed"] = True
        out.append(new_r)
    if budget_tokens:
        out = _enforce_budget(out, budget_tokens, key="content")
    return out


# ─── 检索块压缩 ─────────────────────────────────────────────────

def compress_retrieval_block(
    results: list[dict],
    *,
    budget_tokens: Optional[int] = None,
    max_chars_per_item: int = 150,
) -> list[dict]:
    """压缩 retrieve() 返回的旧记忆块。

    V5 现状: hermes 插件 `on_pre_compress` 直接 `text[:150]` 硬截断(__init__.py:361)。
    本骨架改为: 高相关(score>=0.6)原样, 低相关先 rule_compress 再裁到
    max_chars_per_item —— 避免"要么全要要么全弃"的信息损失。
    """
    if not results:
        return []
    out: list[dict] = []
    for r in results:
        score = float(r.get("score", 0) or 0)
        content = r.get("content", "") or ""
        if score >= 0.6 and len(content) <= max_chars_per_item:
            out.append(dict(r))
            continue
        compressed = compress_text(content, ratio=0.7)
        if len(compressed) > max_chars_per_item:
            # 留 1 字符给省略号, 保证最终长度 <= max_chars_per_item
            compressed = compressed[: max_chars_per_item - 1].rstrip() + "…"
        new_r = dict(r)
        new_r["content"] = compressed
        new_r["_compressed"] = True
        out.append(new_r)
    if budget_tokens:
        out = _enforce_budget(out, budget_tokens, key="content")
    return out


# ─── 预算强制执行 ───────────────────────────────────────────────

def _enforce_budget(blocks: list[dict], budget_tokens: int, *,
                    key: str = "content") -> list[dict]:
    """按 score 降序保留, 直到逼近 budget_tokens; 超预算的块丢弃(已压缩过)。"""
    ranked = sorted(blocks, key=lambda b: -float(b.get("score", 0) or 0))
    total, kept = 0, []
    for b in ranked:
        cost = est_tokens(b.get(key, "") or "")
        if total + cost > budget_tokens and kept:  # 至少留一条
            break
        kept.append(b)
        total += cost
    return kept


def enforce_budget(texts: list[str], budget_tokens: int) -> list[str]:
    """纯文本版: 给定若干文本块, 按输入顺序截到预算内(先到先得)。"""
    out, total = [], 0
    for t in texts:
        cost = est_tokens(t)
        if total + cost > budget_tokens and out:
            break
        out.append(t)
        total += cost
    return out
