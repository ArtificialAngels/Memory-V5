# 详细说明见 docs/scripts/core/memory_v5/action_log.md

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.action_log")

# ─── 路径 ───

_IKAROS_ROOT = Path(os.environ.get("IKAROS_ROOT") or r"E:\Ikaros")
_LOG_DIR = _IKAROS_ROOT / "data" / "ikaros-coordination" / "action_log"

# ─── 统计 (进程生命周期内) ───

_stats = {
    "subprocess": {"total": 0, "ok": 0, "fail": 0},
    "file_write": {"total": 0, "ok": 0, "fail": 0},
    "terminal": {"total": 0, "ok": 0, "fail": 0},
}


def _write_entry(entry: dict) -> None:
    """写一条 JSONL 审计记录."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = _LOG_DIR / f"{day}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.debug("action_log write failed: %s", e)


def _entry(action: str, label: str, **kwargs) -> dict:
    """构造一条审计记录."""
    return {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "action": action,
        "label": label,
        **kwargs,
    }


# ─── subprocess 包装 ───

def log_subprocess(
    cmd: list[str],
    *,
    label: str = "",
    cwd: str | Path | None = None,
    env: dict | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
    creationflags=0,
    timeout: float | None = None,
    **kwargs,
) -> subprocess.Popen:
    """替代裸 subprocess.Popen, 带审计日志.

    Returns: Popen 实例 (与原 API 一致)
    """
    t0 = time.time()
    cat = "subprocess"
    _stats[cat]["total"] += 1
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, env=env,
            stdin=stdin, stdout=stdout, stderr=stderr,
            creationflags=creationflags, **kwargs,
        )
        dur = round((time.time() - t0) * 1000, 1)
        _stats[cat]["ok"] += 1
        _write_entry(_entry(
            "subprocess.Popen", label,
            cmd=cmd, pid=proc.pid, cwd=str(cwd) if cwd else None,
            status="ok", duration_ms=dur,
        ))
        return proc
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 1)
        _stats[cat]["fail"] += 1
        _write_entry(_entry(
            "subprocess.Popen", label,
            cmd=cmd, cwd=str(cwd) if cwd else None,
            status="fail", duration_ms=dur, error=str(e),
        ))
        raise


def log_subprocess_run(
    cmd: list[str],
    *,
    label: str = "",
    cwd: str | Path | None = None,
    env: dict | None = None,
    capture_output: bool = False,
    text: bool = False,
    timeout: float | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """替代裸 subprocess.run, 带审计日志.

    Returns: CompletedProcess 实例 (与原 API 一致)
    """
    t0 = time.time()
    cat = "subprocess"
    _stats[cat]["total"] += 1
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, env=env,
            capture_output=capture_output, text=text,
            timeout=timeout, **kwargs,
        )
        dur = round((time.time() - t0) * 1000, 1)
        ok = result.returncode == 0
        _stats[cat]["ok" if ok else "fail"] += 1
        _write_entry(_entry(
            "subprocess.run", label,
            cmd=cmd, cwd=str(cwd) if cwd else None,
            returncode=result.returncode,
            status="ok" if ok else "fail",
            duration_ms=dur,
        ))
        return result
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 1)
        _stats[cat]["fail"] += 1
        _write_entry(_entry(
            "subprocess.run", label,
            cmd=cmd, cwd=str(cwd) if cwd else None,
            status="fail", duration_ms=dur, error=str(e),
        ))
        raise


# ─── file.write 包装 ───

def log_file_write(
    path: str | Path,
    content: bytes | str,
    *,
    label: str = "",
    mode: str = "wb",
) -> int:
    """替代裸 open().write(), 带审计日志.

    Args:
        path: 文件路径
        content: 写入内容 (str 用 'w' mode, bytes 用 'wb' mode)
        label: 审计标签 (如 "screenshot", "tts_cache")
        mode: 文件打开模式 (默认自动推断: bytes→wb, str→w)

    Returns: 写入字节数
    """
    t0 = time.time()
    cat = "file_write"
    _stats[cat]["total"] += 1
    path = Path(path)
    # 自动推断 mode
    if isinstance(content, bytes) and "b" not in mode:
        mode = "wb"
    elif isinstance(content, str) and "b" in mode:
        mode = "w"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding=None if "b" in mode else "utf-8") as f:
            f.write(content)
        nbytes = len(content) if isinstance(content, bytes) else len(content.encode("utf-8"))
        dur = round((time.time() - t0) * 1000, 1)
        _stats[cat]["ok"] += 1
        _write_entry(_entry(
            "file.write", label,
            path=str(path), nbytes=nbytes, mode=mode,
            status="ok", duration_ms=dur,
        ))
        return nbytes
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 1)
        _stats[cat]["fail"] += 1
        _write_entry(_entry(
            "file.write", label,
            path=str(path), mode=mode,
            status="fail", duration_ms=dur, error=str(e),
        ))
        raise


# ─── terminal 包装 ───

def log_terminal(
    cmd: str,
    *,
    label: str = "",
    timeout: float = 30,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """替代裸 os.system / shell subprocess, 带审计日志.

    Args:
        cmd: shell 命令字符串
        label: 审计标签
        timeout: 超时秒数
        cwd: 工作目录

    Returns: CompletedProcess 实例
    """
    t0 = time.time()
    cat = "terminal"
    _stats[cat]["total"] += 1
    try:
        result = subprocess.run(
            cmd, shell=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True,
            timeout=timeout,
        )
        dur = round((time.time() - t0) * 1000, 1)
        ok = result.returncode == 0
        _stats[cat]["ok" if ok else "fail"] += 1
        _write_entry(_entry(
            "terminal", label,
            cmd=cmd, cwd=str(cwd) if cwd else None,
            returncode=result.returncode,
            status="ok" if ok else "fail",
            duration_ms=dur,
        ))
        return result
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 1)
        _stats[cat]["fail"] += 1
        _write_entry(_entry(
            "terminal", label,
            cmd=cmd, cwd=str(cwd) if cwd else None,
            status="fail", duration_ms=dur, error=str(e),
        ))
        raise


# ─── 统计查询 ───

def stats() -> dict:
    """返回当前进程生命周期的审计统计."""
    return {
        "totals": {k: dict(v) for k, v in _stats.items()},
        "log_dir": str(_LOG_DIR),
    }


# ─── CLI 测试 ───

if __name__ == "__main__":
    print("=== action_log test ===")

    # 1. subprocess.run
    r = log_subprocess_run(["cmd", "/c", "echo", "hello"], label="test_echo", capture_output=True, text=True)
    print(f"  subprocess.run: {r.stdout.strip()}")

    # 2. file.write
    test_path = Path(os.environ.get("IKAROS_ROOT", r"E:\Ikaros")) / ".tmp" / "action_log_test.txt"
    n = log_file_write(test_path, "test content", label="test_write")
    print(f"  file.write: {n} bytes → {test_path}")

    # 3. terminal
    r2 = log_terminal("echo terminal_test", label="test_terminal")
    print(f"  terminal: {r2.stdout.strip()}")

    # 4. stats
    s = stats()
    print(f"\n  stats: {s['totals']}")
    print(f"  log_dir: {s['log_dir']}")

    # 5. 查看审计日志
    day = datetime.now().strftime("%Y-%m-%d")
    log_file = _LOG_DIR / f"{day}.jsonl"
    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        print(f"\n  审计日志 ({len(lines)} 条, 最后 3 条):")
        for line in lines[-3:]:
            entry = json.loads(line)
            print(f"    [{entry['action']}] {entry['label']} → {entry['status']} ({entry['duration_ms']}ms)")

    print("\n=== ALL TESTS PASSED ===")
