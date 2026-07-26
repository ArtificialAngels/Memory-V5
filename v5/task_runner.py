# 详细说明见 docs/scripts/core/v5/v5/task_runner.md

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ikaros.v5.task_runner")

V5_ROOT = Path(__file__).resolve().parent.parent  # Ikaros-memory/
_TASK_DIR = V5_ROOT / "data" / "v5"
_RESULT_PATH = _TASK_DIR / "task_result.json"
_PENDING_PATH = _TASK_DIR / "task_pending.json"


def call_async(text: str, optimized: Optional[str] = None) -> dict:
    """后台执行任务 (子线程调 cloud LLM), 立即返回.

    Args:
        text: 原始用户输入
        optimized: 已优化的指令 (如有)

    Returns:
        {"status": "running", "task_id": "xxx"}
    """
    task_id = uuid.uuid4().hex[:12]
    _TASK_DIR.mkdir(parents=True, exist_ok=True)

    # 写运行中标记 (防止重复触发)
    _write_json(_RESULT_PATH, {
        "task_id": task_id,
        "status": "running",
        "text": text,
        "optimized": optimized,
        "started_at": time.time(),
    })

    # 后台线程执行
    t = threading.Thread(
        target=_execute_async,
        args=(task_id, text, optimized),
        daemon=True,
        name=f"task-{task_id}",
    )
    t.start()
    _register_session(task_id, "", "running", text)
    return {"status": "running", "task_id": task_id}


def _resolve_hermes() -> Optional[str]:
    """稳健定位 hermes 可执行文件：优先 PATH (shutil.which)。

    standalone 下运行于 hermes-agent 内部, hermes 已在 PATH 上；
    不再硬编码任何项目专属的 venv 路径。
    """
    import shutil as _shutil
    _cand = _shutil.which("hermes.exe") or _shutil.which("hermes")
    if _cand and Path(_cand).is_file():
        return _cand
    # standalone 不硬编码 hermes-agent 路径 (运行于 hermes-agent 内, 走 PATH)；找不到返回 None
    return None


def _execute_async(task_id: str, text: str, optimized: Optional[str]) -> None:
    """后台: 委托 Hermes Agent 执行任务 (hermes chat -q), 写结果文件.

    Hermes Agent 有全套工具/技能/记忆/子代理, 比裸调 API 强得多.
    """
    try:
        import subprocess as _sp
        import sys as _sys

        # 定位 hermes 可执行文件 (稳健解析, 拒绝坏符号链接)
        _hermes = _resolve_hermes()
        if not _hermes:
            raise FileNotFoundError("hermes not found on PATH or in project venv")

        user_content = optimized if optimized else text
        # 构建指令: 简短任务标签让 Hermes Agent 的 router 自己决定用什么工具
        goal = f"执行这个任务: {user_content}"

        _result = _sp.run(
            [_hermes, "chat", "-q", goal, "--max-turns", "3", "--pass-session-id"],
            capture_output=True, text=True, timeout=300,
            cwd=str(V5_ROOT),
        )

        # 捕获 Hermes 子代理 session_id
        _sub_session_id = ""
        _reply_lines: list[str] = []
        for _ln in _result.stdout.split("\n") + (_result.stderr or "").split("\n"):
            if _ln.strip().startswith("session_id:"):
                _sub_session_id = _ln.strip().split("session_id:")[-1].strip()
            else:
                _reply_lines.append(_ln)
        reply = "\n".join(_reply_lines).strip()
        if not reply or _result.returncode != 0:
            reply = _result.stderr.strip() or "（任务执行失败）"

        _write_json(_RESULT_PATH, {
            "task_id": task_id, "status": "done",
            "text": text, "optimized": optimized,
            "result": reply, "completed_at": time.time(),
            "session_id": _sub_session_id,
        })
        logger.info("task %s: done (%d chars)", task_id, len(reply))

    except Exception as e:
        logger.error("task %s: failed (%s)", task_id, e)
        _write_json(_RESULT_PATH, {
            "task_id": task_id,
            "status": "failed",
            "text": text,
            "error": str(e),
            "completed_at": time.time(),
        })


def check_result() -> Optional[dict]:
    """检查是否有已完成的任务结果. 有则返回结果 dict, 不删文件."""
    if not _RESULT_PATH.is_file():
        return None
    try:
        data = json.loads(_RESULT_PATH.read_text(encoding="utf-8"))
        if data.get("status") in ("done", "failed"):
            return data
        return None
    except Exception:
        return None


def check_pending_reminder() -> Optional[dict]:
    """检查是否有挂起的提醒 (用户说没空时设的)."""
    if not _PENDING_PATH.is_file():
        return None
    try:
        data = json.loads(_PENDING_PATH.read_text(encoding="utf-8"))
        return data
    except Exception:
        return None


def set_reminder(data: dict) -> None:
    """设一个提醒 (用户没空时调用)."""
    _TASK_DIR.mkdir(parents=True, exist_ok=True)
    data["remind_at"] = time.time()  # 记下设置时间
    _PENDING_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def consume_result() -> Optional[dict]:
    """消费结果 (用户说有空时调用). 读取后删除文件."""
    data = check_result()
    if data:
        try:
            _RESULT_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    return data


def consume_reminder() -> Optional[dict]:
    """消费提醒. 读取后删除."""
    data = check_pending_reminder()
    if data:
        try:
            _PENDING_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    return data


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ─── 子代理会话追踪 ──────────────────────────────

def check_running_tasks() -> list[dict]:
    """检查所有正在运行/已完成待交付的任务."""
    tasks: list[dict] = []
    for p in [_RESULT_PATH, _PENDING_PATH]:
        if p.is_file():
            try:
                d = _read_json(p)
                if d:
                    tasks.append(d)
            except Exception:
                pass
    return tasks


def resume_sub_session(session_id: str, extra_prompt: str = "") -> str | None:
    """主 chat 推动子任务: 用 session_id resume 并追加提示.

    Args:
        session_id: Hermes 子代理 session ID
        extra_prompt: 额外指令 (如 "继续 / 检查状态 / 汇报进度")

    Returns:
        子代理的文字回复, 或 None (session 不存在/失败)
    """
    import subprocess as _sp
    _hermes = _resolve_hermes()
    if not _hermes:
        return None
    try:
        _args = [_hermes, "chat", "-Q", "--resume", session_id,
                 "--max-turns", "2"]
        if extra_prompt:
            _args += ["-q", extra_prompt]
        _result = _sp.run(_args, capture_output=True, text=True, timeout=60,
                          cwd=str(V5_ROOT))
        if _result.returncode == 0:
            return _result.stdout.strip() or None
    except Exception:
        pass
    return None


def task_status_summary() -> str:
    """生成可注入 main chat 的任务摘要."""
    tasks = list_task_sessions()
    if not tasks:
        return ""
    lines = ["\n### 子代理任务"]
    for t in tasks:
        sid = t.get("session_id", "")
        status = t.get("status", "?")
        goal = (t.get("text") or t.get("optimized") or "未知任务")[:50]
        if status == "running":
            lines.append(f"- [{status}] {goal} (session: {sid[:12]}...)")
        elif status == "done":
            summary = (t.get("result") or "")[:80]
            lines.append(f"- [done] {goal} → {summary}")
        else:
            lines.append(f"- [{status}] {goal}")
    return "\n".join(lines)


# ─── 子代理 Session 注册表 (监控面板可见) ─────────────────────

_TASK_SESSIONS_PATH = _TASK_DIR / "task_sessions.json"

def _register_session(task_id: str, session_id: str, status: str, goal: str):
    """写入子代理 session 到注册表(可被监控面板/history-read)."""
    try:
        entries = _read_json(_TASK_SESSIONS_PATH) or []
        # 去重: 同 session_id 覆盖
        entries = [e for e in entries if e.get("session_id") != session_id]
        entries.append({
            "task_id": task_id,
            "session_id": session_id,
            "status": status,
            "goal": goal[:80],
            "updated": time.time(),
        })
        _TASK_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write_json(_TASK_SESSIONS_PATH, entries[-50:])  # keep last 50
    except Exception:
        pass

def list_task_sessions() -> list[dict]:
    """列出最近的子代理 session."""
    return _read_json(_TASK_SESSIONS_PATH) or []

def get_task_session(session_id: str) -> dict | None:
    """按 session_id 查找子代理 session."""
    entries = _read_json(_TASK_SESSIONS_PATH) or []
    for e in entries:
        if e.get("session_id") == session_id:
            return e
    return None
