#!/usr/bin/env python3
# 详细说明见 docs/scripts/core/v5/models/model_config.md
"""记忆架构本地 LLM 的模型加载配置（动态、不写死模型名）。

设计目标（来自项目要求）：
- 模型不写死在代码/注释里；加载信息统一放 ``Ikaros-memory/models``。
- 首次运行时若该目录没有配置文件，则扫描模型目录、自动选定初始加载模型并落盘。
- 后续运行读取已保存的选择，支持手动改 ``model_config.json`` 切换模型。

参数全部对应官方 llama.cpp ``llama-server --help``（b10000-cuda 实测）：

    -m,  --model FNAME       模型路径
    -a,  --alias STRING      给模型起 API 别名（逗号分隔，供路由/请求用）
    --host HOST              监听地址
    --port PORT              监听端口（默认 8080）
    -c,  --ctx-size N        上下文长度
    -ngl, --n-gpu-layers N   放入显存的层数（auto = 由 llama 自动决定）
    -fa, --flash-attn auto   Flash Attention（默认 auto）
    -cb, --cont-batching     连续批处理（默认开）
    --jinja                  用 jinja 模板做 chat（默认开）

llama.cpp 还提供 **router server** 模式（--models-dir 扫描目录 +
--models-preset INI 预设 + --models-autoload 自启），可实现多模型路由。
本模块采用「单一初始模型 + 配置文件」的轻量方案（最稳、最契合单 LLM 记忆后端）；
需要多模型路由时，可据此配置切换为 router 模式。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODELS_DIR / "model_config.json"

# embedding / 检索类模型绝不能当聊天 LLM 加载，扫描时排除。
_EMBEDDING_HINTS = ("embed", "nomic", "e5", "bge", "bce", "gte", "voyage", "rerank")

# 首跑自动选定的优先级提示（仅用于「无配置时的默认」，非写死）：
# 含 "1.7b" 的模型（当前项目实际使用的本地 LLM）优先；否则取体积最小者（最稳）。
_PREFERRED_HINT = "1.7b"


def _is_embedding(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _EMBEDDING_HINTS)


def _scan_chat_models() -> list[Path]:
    """扫描 models 目录下所有 .gguf，排除 embedding 类，返回聊天 LLM 候选。"""
    candidates = [
        p for p in sorted(MODELS_DIR.glob("*.gguf"))
        if p.is_file() and not _is_embedding(p.name)
    ]
    return candidates


def _select_default(candidates: list[Path]) -> Path:
    """从无配置时的候选里挑一个初始模型。"""
    if not candidates:
        raise FileNotFoundError(
            f"No chat LLM .gguf found in {MODELS_DIR} "
            f"(embedding models are excluded automatically). "
            f"Drop a chat GGUF here or set IKAROS_MODEL_LLM."
        )
    # 1) 优先含 preferred hint 的（当前项目实际使用的本地 LLM）
    for p in candidates:
        if _PREFERRED_HINT in p.name.lower():
            return p
    # 2) 只有一个候选 → 直接用
    if len(candidates) == 1:
        return candidates[0]
    # 3) 多个 → 取体积最小（加载最快、对显存最友好）
    return min(candidates, key=lambda p: p.stat().st_size)


def default_config(model_path: Path) -> dict:
    """构造默认配置（initial model 指向扫描选出的模型）。"""
    return {
        "initial_model": model_path.name,
        "alias": "local-llm",
        "host": "127.0.0.1",
        "port": 8080,
        "ctx_size": 8192,
        "gpu_layers": "auto",
        "flash_attn": "auto",
        "cont_batching": True,
        "jinja": True,
        # 说明：alias 为 API 请求时的 model 字段；与请求端保持一致即可，不绑定具体模型名。
    }


def resolve_model_config(force_rescan: bool = False) -> dict:
    """读取（或首跑创建）模型加载配置。

    - 配置存在且 ``initial_model`` 指向的文件仍在 → 直接用。
    - 否则扫描目录、选定默认模型、写回 ``model_config.json``。
    """
    if not force_rescan and CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            initial = cfg.get("initial_model")
            if initial and (MODELS_DIR / initial).is_file():
                return cfg
        except (json.JSONDecodeError, OSError):
            pass  # 损坏则重建

    candidates = _scan_chat_models()
    chosen = _select_default(candidates)
    cfg = default_config(chosen)
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cfg


def model_path() -> Path:
    """返回当前配置的模型绝对路径。"""
    cfg = resolve_model_config()
    return MODELS_DIR / cfg["initial_model"]


def server_args() -> list[str]:
    """返回 llama-server 的「模型 + 服务」参数列表（不含二进制本身）。

    调用方自行在前面拼上 llama-server 二进制路径。
    """
    cfg = resolve_model_config()
    args: list[str] = [
        "-m", str(MODELS_DIR / cfg["initial_model"]),
        "--host", cfg.get("host", "127.0.0.1"),
        "--port", str(cfg.get("port", 8080)),
        "-c", str(cfg.get("ctx_size", 8192)),
        "-ngl", str(cfg.get("gpu_layers", "auto")),
        "--flash-attn", cfg.get("flash_attn", "auto"),
        "--alias", cfg.get("alias", "local-llm"),
    ]
    if cfg.get("cont_batching", True):
        args.append("--cont-batching")
    if cfg.get("jinja", True):
        args.append("--jinja")
    return args


def _llama_bin() -> str:
    """llama-server 二进制路径：优先环境变量，否则按 PATH 解析，最后回退到仓库内 runtime 目录。"""
    env = os.environ.get("IKAROS_LLAMA_SERVER")
    if env:
        return env
    import shutil
    for cand in ("llama-server.exe", "llama-server", "llama.cpp"):
        p = shutil.which(cand)
        if p:
            return p
    # 回退：仓库内 runtime/llama/b10000-cuda/（Ikaros 约定；也可自行放置二进制）
    fallback = MODELS_DIR.parent / "runtime" / "llama" / "b10000-cuda" / "llama-server.exe"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(
        "找不到 llama-server。请设置环境变量 IKAROS_LLAMA_SERVER 指向二进制，"
        "或将其加入 PATH，或放到 <repo>/runtime/llama/b10000-cuda/。"
    )


def emit_bat() -> str:
    """生成一行可直接 ``call`` 的 bat 启动命令（含二进制与全部参数）。"""
    cfg = resolve_model_config()
    mp = MODELS_DIR / cfg["initial_model"]
    cmd = (
        f'"{_llama_bin()}" '
        f'-m "{mp}" '
        f'--host {cfg.get("host", "127.0.0.1")} '
        f'--port {cfg.get("port", 8080)} '
        f'-c {cfg.get("ctx_size", 8192)} '
        f'-ngl {cfg.get("gpu_layers", "auto")} '
        f'--flash-attn {cfg.get("flash_attn", "auto")} '
        f'--alias {cfg.get("alias", "local-llm")}'
    )
    if cfg.get("cont_batching", True):
        cmd += " --cont-batching"
    if cfg.get("jinja", True):
        cmd += " --jinja"
    return cmd


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Ikaros 本地 LLM 模型加载配置")
    ap.add_argument("--model-path", action="store_true", help="打印当前模型绝对路径")
    ap.add_argument("--alias", action="store_true", help="打印 API alias")
    ap.add_argument("--port", action="store_true", help="打印监听端口")
    ap.add_argument("--args", action="store_true", help="打印 llama-server 参数列表(空格分隔)")
    ap.add_argument("--emit-bat", action="store_true", help="打印一行可直接 call 的 bat 启动命令")
    ap.add_argument("--print-json", action="store_true", help="打印完整配置 JSON")
    ap.add_argument("--force-rescan", action="store_true", help="忽略现有配置，重新扫描目录")
    args = ap.parse_args()

    if args.force_rescan:
        resolve_model_config(force_rescan=True)

    if args.model_path:
        print(model_path())
    elif args.alias:
        print(resolve_model_config().get("alias", "local-llm"))
    elif args.port:
        print(resolve_model_config().get("port", 8080))
    elif args.args:
        print(" ".join(server_args()))
    elif args.emit_bat:
        print(emit_bat())
    elif args.print_json:
        print(json.dumps(resolve_model_config(), ensure_ascii=False, indent=2))
    else:
        # 默认：打印摘要
        cfg = resolve_model_config()
        print(f"model : {cfg['initial_model']}")
        print(f"alias : {cfg.get('alias', 'local-llm')}")
        print(f"port  : {cfg.get('port', 8080)}")
        print(f"config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
