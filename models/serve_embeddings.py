#!/usr/bin/env python3
"""Serve the nomic-embed-text embedding model via llama-server on :8587.

V5's semantic search (Chroma vectors) needs an ``/embedding`` endpoint. You can
either point ``IKAROS_EMBED_URL`` at an existing embedding service (OpenAI, Ollama,
a remote llama-server, ...) OR self-host the nomic model with this helper.

Prereqs:
  - the embedding GGUF present (see ``download_models.py --only embed``)
  - a llama-server binary (env IKAROS_LLAMA_SERVER, or on PATH)

Env overrides:
  EMBED_PORT        (default 8587)
  EMBED_HOST        (default 127.0.0.1)
  EMBED_MODEL_FILE  (default nomic-embed-text-v2-moe.f16.gguf)
  IKAROS_LLAMA_SERVER  (llama-server binary)

Usage:
  python models/serve_embeddings.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent


def _locate_binary() -> str:
    env = os.environ.get("IKAROS_LLAMA_SERVER")
    if env:
        return env
    for cand in ("llama-server.exe", "llama-server", "llama.cpp"):
        p = shutil.which(cand)
        if p:
            return p
    raise FileNotFoundError("找不到 llama-server；设置 IKAROS_LLAMA_SERVER 或加入 PATH。")


def main() -> None:
    host = os.environ.get("EMBED_HOST", "127.0.0.1")
    port = os.environ.get("EMBED_PORT", "8587")
    model_file = os.environ.get("EMBED_MODEL_FILE", "nomic-embed-text-v2-moe.f16.gguf")
    model_path = MODELS_DIR / model_file
    if not model_path.is_file():
        sys.exit(
            f"embedding 模型不存在: {model_path}\n"
            f"先运行: python models/download_models.py --only embed"
        )
    binary = _locate_binary()
    cmd = [
        binary,
        "-m", str(model_path),
        "--host", host,
        "--port", str(port),
        "--embed",                 # 启用 /embedding 端点
        "-ngl", os.environ.get("EMBED_GPU_LAYERS", "auto"),
    ]
    print("launching embedding server:", " ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
