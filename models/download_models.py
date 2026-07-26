#!/usr/bin/env python3
"""Download the GGUF models V5 needs (chat LLM + embedding model).

By default downloads BOTH into this ``models/`` directory:
  - Qwen3-1.7B (Q4_K_M)  -> chat / local LLM (:8080)
  - nomic-embed-text-v2-moe (f16) -> embedding service (:8587)

After download, ``model_config.py`` auto-detects the chat model on next run,
so no manual config is required. Embedding models are excluded from the chat
model scan automatically.

Usage:
  python models/download_models.py                 # both
  python models/download_models.py --only llm      # chat model only
  python models/download_models.py --only embed    # embedding model only
  python models/download_models.py --hf-mirror     # use hf-mirror.com (China)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent

# local filename -> HuggingFace resolve URL
DEFAULTS = {
    "Qwen_Qwen3-1.7B-Q4_K_M.gguf": (
        "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/"
        "Qwen3-1.7B-Q4_K_M.gguf"
    ),
    "nomic-embed-text-v2-moe.f16.gguf": (
        "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/"
        "nomic-embed-text-v2-moe.f16.gguf"
    ),
}


def _url(mirror: bool, url: str) -> str:
    return url.replace("https://huggingface.co", "https://hf-mirror.com") if mirror else url


def _download(url: str, dest: Path) -> None:
    import httpx

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  GET {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        with open(tmp, "wb") as f:
            for chunk in r.iter_raw(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    sys.stdout.write(f"\r    {pct:3d}%  {downloaded//1_048_576}MB/{total//1_048_576}MB")
                    sys.stdout.flush()
        print()
    tmp.replace(dest)
    print(f"  saved: {dest.name} ({dest.stat().st_size//1_048_576}MB, {time.time()-t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download V5 GGUF models")
    ap.add_argument("--only", choices=("llm", "embed", "both"), default="both")
    ap.add_argument("--hf-mirror", action="store_true", help="use hf-mirror.com")
    args = ap.parse_args()

    wanted = (
        list(DEFAULTS) if args.only == "both"
        else (["Qwen_Qwen3-1.7B-Q4_K_M.gguf"] if args.only == "llm"
              else ["nomic-embed-text-v2-moe.f16.gguf"])
    )
    for name, url in DEFAULTS.items():
        if name not in wanted:
            continue
        dest = MODELS_DIR / name
        print(f"[model] {name}")
        try:
            _download(_url(args.hf_mirror, url), dest)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            print("  手动下载后放到 models/ 目录即可。")
    print("done.")


if __name__ == "__main__":
    main()
