"""Standalone llama-server launcher for the V5 local LLM.

Replaces the Ikaros watchdog module (``wd_import``). Provides ``ensure_local_llm()``
which, on first use, spawns ``llama-server`` with the chat model declared in
``../models/model_config.py`` and polls ``/health`` until the server is ready.

The launcher is idempotent: if the server is already answering on its health
endpoint it returns ``True`` immediately without spawning a new process.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = __import__("logging").getLogger("ikaros.memory.v5.llama_launcher")

# <repo>/models  (this file lives in <repo>/v5/)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))


def _locate_binary() -> str:
    """Return a llama-server binary path via env, PATH, or repo-local fallback."""
    env = os.environ.get("IKAROS_LLAMA_SERVER")
    if env:
        return env
    for cand in ("llama-server.exe", "llama-server", "llama.cpp"):
        p = shutil.which(cand)
        if p:
            return p
    fallback = MODELS_DIR.parent / "runtime" / "llama" / "b10000-cuda" / "llama-server.exe"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(
        "找不到 llama-server。请设置环境变量 IKAROS_LLAMA_SERVER 指向二进制，"
        "或将其加入 PATH，或放到 <repo>/runtime/llama/b10000-cuda/。"
    )


def _health_ok(base_url: str, timeout: float = 2.0) -> bool:
    try:
        import httpx
    except ImportError:
        # 没有 httpx 时退化为 TCP 探测（llama-server /health 也是 http，但至少能判断是否端口通）
        import socket
        try:
            host, port = base_url.rsplit(":", 1)[0].split("://", 1)[-1], int(base_url.rsplit(":", 1)[1])
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False
    try:
        r = httpx.get(base_url.rstrip("/") + "/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def ensure_local_llm(*, timeout: int = 180, poll: float = 1.0) -> bool:
    """确保本地 llama-server 已就绪，否则热载入。返回是否可用。

    Args:
        timeout: 等待服务就绪的最大秒数。
        poll: 健康检查轮询间隔（秒）。
    """
    try:
        from model_config import resolve_model_config, server_args
    except Exception as e:  # noqa: BLE001
        logger.error("llama_launcher: cannot load models/model_config.py: %s", e)
        return False

    cfg = resolve_model_config()
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8080))
    base = f"http://{host}:{port}"

    if _health_ok(base, timeout=2.0):
        return True

    try:
        binary = _locate_binary()
    except FileNotFoundError as e:
        logger.error("llama_launcher: %s", e)
        return False

    args = [binary] + server_args()
    logger.info("llama_launcher: spawning %s (timeout=%ds)", binary, timeout)
    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("llama_launcher: spawn failed: %s", e)
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health_ok(base, timeout=2.0):
            logger.info("llama_launcher: local LLM ready at %s", base)
            return True
        time.sleep(poll)
    return _health_ok(base, timeout=2.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = ensure_local_llm()
    print("local LLM ready:" if ok else "local LLM FAILED", ok)
