# 详细说明见 docs/scripts/core/v5/v5/reflect/llm_client.md

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

# 内联说明见 docs/scripts/core/v5/v5/reflect/llm_client.md（见“内联注释摘录”）

logger = logging.getLogger("ikaros.memory.v5.llm")

# ─── 路径配置 (与 v4 其他模块一致) ──────────────────────────────

V4_ROOT = Path(__file__).resolve().parent.parent
V5_DATA_DIR = V4_ROOT / "data" / "v5"

# dotenv：加载 Hermes Agent 的 .env (若设置了 HERMES_HOME) 与仓库本地 .env
try:
    from dotenv import load_dotenv
    _hermes_home = os.environ.get("HERMES_HOME")
    if _hermes_home:
        _hh_env = Path(_hermes_home) / ".env"
        if _hh_env.exists():
            load_dotenv(_hh_env, override=False)
    # 仓库本地 .env 允许覆盖 (override=True, 优先级最高)
    _v4_env = V4_ROOT / ".env"
    if _v4_env.exists():
        load_dotenv(_v4_env, override=True)
except ImportError:
    pass  # dotenv 不可用, 走 os.environ 裸读

# ─── 本地 LLM (:8080) — 懒加载 / 按需服务 (2026-07-26 后重构) ───────
# V5 认知任务(consolidate/distill/reflect/emotion 标注)统一走云端 DeepSeek,
# 不消费本地 LLM (见 call_llm_auto)。但 :8080 作为「常规 llama 服务」保留,
# 由 agent / 本地 chat 等按需调用: provider="local" 时 _call_local 会先触发
# ensure_local_llm() 热载入模型(看门狗只做端口巡检, 不在启动/巡检时拉起)。
# 配置逻辑统一来自 core/v5/models/model_config.py (经看门狗 _load_model_cfg 读取)。

# ─── 大模型 (DeepSeek) ────────────────────────────────

# 哥哥 (2026-07-05) 选定 V4 flash, 验证于 Context7 /websites/api-docs_deepseek
# 端点: https://api.deepseek.com (OpenAI-compatible, 跟 V3 LOCAL_LLM_URL 同形)
DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_TIMEOUT = int(os.environ.get("DEEPSEEK_TIMEOUT", "120"))
# 哥哥 key 已设到 Hermes Agent .env (DEEPSEEK_API_KEY=sk-...)
# V4 不直接读 .env, 只从 os.environ 拿 (避免代码进 git)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ─── 本地 LLM (:8080) ───────────────────────────────
# 懒加载服务: 首次调用时才由 ensure_local_llm() 热载入模型。
LOCAL_LLM_URL = os.environ.get("IKAROS_LOCAL_LLM_URL", "http://127.0.0.1:8080")
LOCAL_LLM_TIMEOUT = int(os.environ.get("IKAROS_LOCAL_LLM_TIMEOUT", "180"))
LOCAL_LLM_ALIAS = os.environ.get("IKAROS_LOCAL_LLM_ALIAS", "local-llm")


# ─── 统一接口 ──────────────────────────────────────────────────

ProviderName = Literal["local", "deepseek"]


@dataclass(frozen=True)
class LLMResponse:
    """统一 LLM 响应 (小模型/大模型都返这个)."""
    content: str
    provider: ProviderName
    model: str
    elapsed_sec: float
    raw: dict | None = None  # 原始响应 (调试用)


# ─── 重试 (防御 LLM 偶发 404 / 5xx / 网络抖动) ───────────────
# 哥哥 (2026-07-07) 修 "distill 偶发 404": 大模型/本地偶发 404 不该废掉整轮反思
MAX_RETRIES = 2
RETRY_BACKOFF_SEC = 1.5


def call_llm(
    system: str,
    user: str,
    *,
    provider: ProviderName = "deepseek",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int | None = None,
    extra: dict | None = None,
) -> LLMResponse:
    """统一 LLM 调用入口 (带重试 + 退避, 缓解偶发 404/5xx/抖动).

    Args:
        system: system prompt
        user: user prompt
        provider: "local" (本地 :8080 llama 服务, 懒加载热载入) 或 "deepseek" (V4 flash)
        max_tokens: 最大输出 token
        temperature: 0 = 确定性, 1 = 创造性
        timeout: 超时秒数, None 用 provider 默认
        extra: 合并进请求体的额外字段 (如 {"enable_thinking": False} 关闭思考链)

    Raises:
        RuntimeError: 重试耗尽后仍失败 / API key 缺失 / 解析失败 (显式, 不静默)
    """
    t0 = time.time()
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "local":
                # 本地 :8080 llama 服务 — 懒加载(首次调用热载入模型)
                resp = _call_local(system, user, max_tokens, temperature,
                                   timeout or LOCAL_LLM_TIMEOUT, extra)
            elif provider == "deepseek":
                resp = _call_deepseek(system, user, max_tokens, temperature, timeout or DEEPSEEK_TIMEOUT, extra)
            else:
                raise ValueError(f"unknown provider: {provider!r}")
            elapsed = time.time() - t0
            logger.info("LLM call: provider=%s model=%s attempt=%d/%d elapsed=%.2fs len=%d",
                        provider, resp.model, attempt, MAX_RETRIES, elapsed, len(resp.content))
            return resp
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                logger.warning("LLM call attempt %d/%d failed (%s); retry in %.1fs",
                               attempt, MAX_RETRIES, e, RETRY_BACKOFF_SEC)
                time.sleep(RETRY_BACKOFF_SEC)
            else:
                logger.error("LLM call failed after %d attempts: %s", MAX_RETRIES, e)
    elapsed = time.time() - t0
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts ({elapsed:.1f}s): {last_err}") from last_err


def call_llm_auto(
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int | None = None,
) -> LLMResponse:
    """统一走云端 DeepSeek (带重试 + 退避).

    小模型方案已从 V5 剔除 (2026-07-26), 原 "本地优先、云端兜底" 退化为
    纯云端。无 DEEPSEEK_API_KEY 时直接抛 RuntimeError (显式, 不静默)。
    失败语义同 call_llm: 重试耗尽后抛 RuntimeError。
    """
    if not has_api_key():
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. V5 已移除本地小模型, 所有认知任务必须走云端。\n"
            "Set via:\n"
            "  setx DEEPSEEK_API_KEY \"sk-...\"   (Windows)\n"
            "  export DEEPSEEK_API_KEY=\"sk-...\" (Linux/macOS/git-bash)\n"
            "Or set it in this repo's .env (DEEPSEEK_API_KEY=sk-...) or HERMES_HOME/.env"
        )
    return call_llm(system, user, provider="deepseek", max_tokens=max_tokens,
                    temperature=temperature, timeout=timeout)


# ─── DeepSeek ───────────────────────────────────────

def _call_deepseek(system: str, user: str, max_tokens: int,
                   temperature: float, timeout: int,
                   extra: dict | None = None) -> LLMResponse:
    """调 DeepSeek API.

    哥哥 (2026-07-05) 验证:
      endpoint = https://api.deepseek.com/v1/chat/completions
      model = deepseek-v4-flash
      auth = Bearer ${DEEPSEEK_API_KEY}
      thinking = {"type": "enabled"} (可选, 默认 off)
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Set via:\n"
            "  setx DEEPSEEK_API_KEY \"sk-...\"   (Windows)\n"
            "  export DEEPSEEK_API_KEY=\"sk-...\" (Linux/macOS/git-bash)\n"
            "Or set it in this repo's .env (DEEPSEEK_API_KEY=sk-...) or HERMES_HOME/.env"
        )

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra:
        body.update(extra)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        # V4: 显式错误, 不吞
        body_text = e.response.text[:500] if e.response else ""
        raise RuntimeError(
            f"DeepSeek API HTTP {e.response.status_code}: {body_text}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"DeepSeek API call failed: {e}") from e

    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or ""
# 内联说明见 docs/scripts/core/v5/v5/reflect/llm_client.md（见“内联注释摘录”）
        if not content.strip() and msg.get("reasoning_content"):
            content = msg.get("reasoning_content", "") or ""
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"DeepSeek response shape unexpected: {data}") from e

    if not content.strip():
        raise RuntimeError("DeepSeek returned empty content (both content and reasoning_content)")
    return LLMResponse(
        content=content.strip(),
        provider="deepseek",
        model=DEEPSEEK_MODEL,
        elapsed_sec=0.0,
        raw=data,
    )


# ─── 本地 LLM (:8080) ───────────────────────────────

def _ensure_local_llm_loaded() -> None:
    """确保本地 llama-server (:8080) 已就绪 — 否则热载入。

    复用看门狗模块的 ensure_local_llm() (detached spawn + 等 /health)。
    看门狗只做端口巡检, 不调用本函数; 热载入由 agent 调用本地 LLM 触发。
    """
    # standalone：用仓库内置的 llama 启动器 (不依赖 Ikaros 看门狗 wd_import)
    from v5.llama_launcher import ensure_local_llm
    if not ensure_local_llm(timeout=LOCAL_LLM_TIMEOUT):
        raise RuntimeError(
            "本地 LLM (:8080) 热载入失败。检查模型文件 (models/*.gguf) 或 "
            "IKAROS_LLAMA_SERVER 是否可用；可先 `python models/download_models.py` 下载模型。"
        )


def _call_local(system: str, user: str, max_tokens: int,
                temperature: float, timeout: int,
                extra: dict | None = None) -> LLMResponse:
    """调本地 llama-server (:8080) — OpenAI-compatible /v1/chat/completions。

    懒加载: 若 :8080 未就绪, 先由 _ensure_local_llm_loaded() 拉起模型(热载入),
    之后复用常驻服务。失败抛 RuntimeError (显式, 不静默)。
    """
    _ensure_local_llm_loaded()

    url = f"{LOCAL_LLM_URL.rstrip('/')}/v1/chat/completions"
    body = {
        "model": LOCAL_LLM_ALIAS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra:
        body.update(extra)

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        body_text = e.response.text[:500] if e.response else ""
        raise RuntimeError(f"Local LLM HTTP {e.response.status_code}: {body_text}") from e
    except Exception as e:
        raise RuntimeError(f"Local LLM call failed: {e}") from e

    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or ""
        # 本地模型可能返回 reasoning_content (思考链): 内容空时回退
        if not content.strip() and msg.get("reasoning_content"):
            content = msg.get("reasoning_content", "") or ""
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Local LLM response shape unexpected: {data}") from e

    if not content.strip():
        raise RuntimeError("Local LLM returned empty content (both content and reasoning_content)")
    return LLMResponse(
        content=content.strip(),
        provider="local",
        model=LOCAL_LLM_ALIAS,
        elapsed_sec=0.0,
        raw=data,
    )


# ─── helpers ──────────────────────────────────────────────────

def has_api_key() -> bool:
    """检查 DEEPSEEK_API_KEY 是否可用 (不返值, 只 bool)."""
    return bool(DEEPSEEK_API_KEY)


def stats() -> dict:
    """返回当前 LLM client 配置 (不泄露 key)."""
    return {
        "local": {
            "url": LOCAL_LLM_URL,
            "alias": LOCAL_LLM_ALIAS,
            "timeout": LOCAL_LLM_TIMEOUT,
            "mode": "lazy-on-demand (hot-loaded on first call via ensure_local_llm)",
        },
        "deepseek": {
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
            "timeout": DEEPSEEK_TIMEOUT,
            "api_key_set": has_api_key(),
        },
    }
