#!/usr/bin/env python3
"""Install the V5 memory plugin into a fresh Hermes Agent (or standalone venv).

What it does (idempotent):
  1. Create a virtualenv at --venv (default: <repo>/.venv).
  2. pip install -r requirements.txt (chromadb, httpx, mcp, python-dotenv).
  3. Optionally download GGUF models (--download-models / --download-embed).
  4. Emit hermes-mcp-v5.json — an MCP server config snippet to paste into
     Hermes Agent's config mcp_servers section.
  5. Optionally deploy the Hermes provider plugin into a Hermes Agent install
     (--hermes-agent PATH), self-contained (bundles the v5/ package).

Usage:
  python install.py                         # venv + deps + mcp config
  python install.py --download-models       # also pull chat + embed GGUF
  python install.py --hermes-agent "C:/.../hermes-agent"   # deploy provider plugin
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _quote(p: Path) -> str:
    # Hermes config accepts forward slashes on every platform.
    return str(p).replace("\\", "/")


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run(cmd: list[str], **kw) -> int:
    print("+", " ".join(_quote(Path(c)) if "/" in c or "\\" in c else c for c in cmd))
    return subprocess.run(cmd, **kw).returncode


def main() -> int:
    repo = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Install V5 memory plugin")
    ap.add_argument("--venv", default=str(repo / ".venv"), help="venv path (default <repo>/.venv)")
    ap.add_argument("--python", default=sys.executable, help="python interpreter to create the venv")
    ap.add_argument("--download-models", action="store_true", help="download chat + embedding GGUF")
    ap.add_argument("--download-embed", action="store_true", help="download embedding GGUF only")
    ap.add_argument("--hermes-agent", default=None, help="path to Hermes Agent install (deploy provider)")
    ap.add_argument("--no-venv", action="store_true", help="install into current interpreter (not recommended)")
    args = ap.parse_args()

    venv = Path(args.venv)
    if args.no_venv:
        py = Path(sys.executable)
        print(f"[install] installing into current interpreter: {py}")
    else:
        if not venv.exists():
            print(f"[install] creating venv at {venv}")
            if _run([args.python, "-m", "venv", str(venv)]) != 0:
                return 1
        py = _venv_python(venv)
        if not py.exists():
            print(f"[install] venv python missing: {py}", file=sys.stderr)
            return 1

    # 2) deps
    req = repo / "requirements.txt"
    if _run([str(py), "-m", "pip", "install", "-U", "pip"]) != 0:
        print("[install] pip upgrade failed (non-fatal)", file=sys.stderr)
    if _run([str(py), "-m", "pip", "install", "-r", str(req)]) != 0:
        print("[install] dependency install failed", file=sys.stderr)
        return 1

    # 3) optional model download
    if args.download_models or args.download_embed:
        only = "embed" if args.download_embed and not args.download_models else (
            "both" if args.download_models else "embed")
        dl = repo / "models" / "download_models.py"
        if _run([str(py), str(dl), "--only", only]) != 0:
            print("[install] model download failed (non-fatal); run manually later", file=sys.stderr)

    # 4) MCP config snippet
    mcp_cfg = {
        "mcp_servers": {
            "ikaros-v5": {
                "command": _quote(py),
                "args": [_quote(repo / "v5" / "mcp_server.py")],
                "env": {
                    "V5_MEMORY_ROOT": _quote(repo),
                    "PYTHONPATH": _quote(repo),
                    "IKAROS_EMBED_URL": "http://127.0.0.1:8587/embedding",
                },
            }
        }
    }
    out = repo / "hermes-mcp-v5.json"
    out.write_text(json.dumps(mcp_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[install] wrote MCP config snippet: {out}")

    # .env.example
    env_example = (
        "# V5 memory plugin environment (copy to .env and fill in)\n"
        "# Embedding endpoint (semantic search). Self-host via models/serve_embeddings.py,\n"
        "# or point at OpenAI / Ollama / any /embedding-compatible service.\n"
        "IKAROS_EMBED_URL=http://127.0.0.1:8587/embedding\n"
        "# Optional: local LLM (chat/cognition) via llama-server. If empty, V5 tries PATH / repo runtime.\n"
        "# IKAROS_LLAMA_SERVER=/path/to/llama-server\n"
        "# Cloud LLM for background reflection (optional; without it reflection degrades gracefully):\n"
        "# DEEPSEEK_API_KEY=sk-...\n"
        "# HERMES_HOME=/path/to/hermes-agent   (to load Hermes .env for keys)\n"
    )
    ep = repo / ".env.example"
    ep.write_text(env_example, encoding="utf-8")

    # 5) deploy provider plugin
    if args.hermes_agent:
        ha = Path(args.hermes_agent)
        plug_dir = ha / "plugins" / "memory" / "ikaros_v5"
        plug_dir.mkdir(parents=True, exist_ok=True)
        # copy provider
        shutil.copy2(repo / "hermes-plugin" / "ikaros_v5" / "__init__.py", plug_dir / "__init__.py")
        # bundle v5/ package for self-contained operation
        dest_v5 = plug_dir / "v5"
        if dest_v5.exists():
            shutil.rmtree(dest_v5)
        shutil.copytree(repo / "v5", dest_v5, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        print(f"[install] deployed self-contained provider -> {plug_dir}")
        print(f"[install] hermes will auto-discover it as memory provider 'ikaros-v5'.")

    print("[install] done.")
    print("Next: add the contents of hermes-mcp-v5.json under mcp_servers in your Hermes config,")
    print("       then (re)start Hermes Agent. For embeddings, run models/serve_embeddings.py or")
    print("       set IKAROS_EMBED_URL to your endpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
