#!/usr/bin/env python3
# 详细说明见 docs/scripts/bin/ikaros-tools.md
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# 5 维身份
WHO = "Ikaros (self-initiated)"

# 路径
_HERE = Path(__file__).resolve().parent
_LOG_SCRIPT = _HERE / "ikaros-action-log.py"

# 延迟 import action_log (避免循环)
_action_log = None


def _get_action_log():
    """获取 action_log 模块 (单例, 延迟 import)."""
    global _action_log
    if _action_log is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_action_log", str(_LOG_SCRIPT))
        m = importlib.util.module_from_spec(spec)
        sys.modules["_action_log"] = m
        spec.loader.exec_module(m)
        _action_log = m
    return _action_log


# ─── 进程 ────────────────────────────────────────

def ik_kill(pid: int, why: str = "manual",
            grace_ms: int = 2000) -> bool:
    """杀进程 (Windows: taskkill /F, Unix: SIGTERM→SIGKILL).

    自动 action log: start + end. 失败返 False.
    """
    al = _get_action_log()
    with al.action(
            f"kill pid {pid}",
            action="process.kill",
            target=f"pid:{pid}",
            why=why,
            expected_duration_ms=grace_ms + 500) as a:
        ok = False
        try:
            if sys.platform == "win32":
                # /T = 子进程一起杀, /F = 强制. taskkill 输出是 GBK, 用 bytes
                r = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=10)
                # rc=0: 杀成功, rc=128: 进程不存在
                ok = (r.returncode == 0)
            else:
                import signal
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(grace_ms / 1000.0)
                    os.kill(pid, signal.SIGKILL)
                    ok = True
                except ProcessLookupError:
                    ok = True  # 已死
        except Exception as e:
            a.notes = f"err: {e}"
            a.done(result="fail", duration_ms=0, complete=False, completion_pct=0)
            return False
        a.done(result="ok" if ok else "fail",
               duration_ms=grace_ms,
               complete=ok,
               completion_pct=100 if ok else 50)
        return ok


def ik_start(bin_path: str, args: Optional[Sequence[str]] = None,
             why: str = "manual", cwd: Optional[str] = None,
             detached: bool = True) -> Optional[int]:
    """启动进程. 返 PID (detached 模式) 或 None.

    自动 action log: start (注意: 不写 end, 因为子进程要长期跑; 用孤儿
    检测来看是否真死).
    """
    al = _get_action_log()
    args = list(args or [])
    cmd_str = " ".join([bin_path] + args)
    with al.action(
            f"start {Path(bin_path).name}",
            action="process.start",
            target=cmd_str,
            why=why,
            expected_duration_ms=2000) as a:
        try:
            kwargs = {}
            if detached and sys.platform == "win32":
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS
            if cwd:
                kwargs["cwd"] = cwd
            proc = subprocess.Popen(
                [bin_path] + args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs)
            a.notes = f"pid={proc.pid}"
            a.done(result="started", duration_ms=50,
                   complete=True, completion_pct=100)
            return proc.pid
        except Exception as e:
            a.notes = f"err: {e}"
            a.done(result="fail", duration_ms=0,
                   complete=False, completion_pct=0)
            return None


# ─── 命令 ────────────────────────────────────────

def ik_run(cmd: Sequence[str], why: str = "manual",
           timeout_s: int = 60, cwd: Optional[str] = None) -> dict:
    """跑 shell 命令. 返 {rc, stdout, stderr, duration_ms}.

    自动 action log: start + end.
    """
    al = _get_action_log()
    cmd_str = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    with al.action(
            f"run {Path(cmd_str.split()[0]).name if cmd_str else 'cmd'}",
            action="terminal.run",
            target=cmd_str,
            why=why,
            expected_duration_ms=timeout_s * 1000) as a:
        t0 = time.time()
        try:
            r = subprocess.run(
                list(cmd) if not isinstance(cmd, str) else cmd,
                capture_output=True, text=True,
                timeout=timeout_s, cwd=cwd)
            dt = int((time.time() - t0) * 1000)
            ok = (r.returncode == 0)
            a.notes = f"rc={r.returncode}"
            a.done(result="ok" if ok else "fail",
                   duration_ms=dt,
                   complete=ok,
                   completion_pct=100 if ok else 80)
            return {"rc": r.returncode, "stdout": r.stdout,
                    "stderr": r.stderr, "duration_ms": dt}
        except subprocess.TimeoutExpired:
            dt = int((time.time() - t0) * 1000)
            a.notes = "timeout"
            a.done(result="timeout", duration_ms=dt,
                   complete=False, completion_pct=50)
            return {"rc": -1, "stdout": "", "stderr": "timeout",
                    "duration_ms": dt}
        except Exception as e:
            dt = int((time.time() - t0) * 1000)
            a.notes = f"err: {e}"
            a.done(result="fail", duration_ms=dt,
                   complete=False, completion_pct=0)
            return {"rc": -1, "stdout": "", "stderr": str(e),
                    "duration_ms": dt}


# ─── 文件 ────────────────────────────────────────

def ik_write(path: str, content: str, why: str = "manual",
              append: bool = False, encoding: str = "utf-8") -> bool:
    """写文件. 自动 action log."""
    al = _get_action_log()
    with al.action(
            f"write {Path(path).name}",
            action="file.write" if not append else "file.append",
            target=path,
            why=why,
            expected_duration_ms=200) as a:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(p, mode, encoding=encoding) as f:
                f.write(content)
            a.done(result="ok", duration_ms=20,
                   complete=True, completion_pct=100)
            return True
        except Exception as e:
            a.notes = f"err: {e}"
            a.done(result="fail", duration_ms=0,
                   complete=False, completion_pct=0)
            return False


def ik_delete(path: str, why: str = "manual") -> bool:
    """删文件. 自动 action log."""
    al = _get_action_log()
    with al.action(
            f"delete {Path(path).name}",
            action="file.delete",
            target=path,
            why=why,
            expected_duration_ms=200) as a:
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
            a.done(result="ok", duration_ms=10,
                   complete=True, completion_pct=100)
            return True
        except Exception as e:
            a.notes = f"err: {e}"
            a.done(result="fail", duration_ms=0,
                   complete=False, completion_pct=0)
            return False


# ─── CLI ────────────────────────────────────────

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="ikaros-tools (带动作日志)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_kill = sub.add_parser("kill", help="杀进程")
    p_kill.add_argument("--pid", type=int, required=True)
    p_kill.add_argument("--why", default="cli")

    p_start = sub.add_parser("start", help="启动进程")
    p_start.add_argument("--bin", required=True, help="binary 路径")
    p_start.add_argument("--args", default="", help="空格分隔参数")
    p_start.add_argument("--why", default="cli")
    p_start.add_argument("--cwd", default=None)

    p_run = sub.add_parser("run", help="跑命令")
    p_run.add_argument("--cmd", required=True, help="完整命令字符串")
    p_run.add_argument("--why", default="cli")
    p_run.add_argument("--timeout", type=int, default=60)
    p_run.add_argument("--cwd", default=None)

    p_write = sub.add_parser("write", help="写文件")
    p_write.add_argument("--path", required=True)
    p_write.add_argument("--content", required=True)
    p_write.add_argument("--why", default="cli")
    p_write.add_argument("--append", action="store_true")

    p_del = sub.add_parser("delete", help="删文件")
    p_del.add_argument("--path", required=True)
    p_del.add_argument("--why", default="cli")

    args = p.parse_args()

    if args.cmd == "kill":
        ok = ik_kill(args.pid, why=args.why)
        sys.exit(0 if ok else 1)
    elif args.cmd == "start":
        arg_list = args.args.split() if args.args else []
        pid = ik_start(args.bin, args=arg_list, why=args.why, cwd=args.cwd)
        print(f"pid={pid}")
        sys.exit(0 if pid else 1)
    elif args.cmd == "run":
        import shlex
        cmd_list = shlex.split(args.cmd)
        r = ik_run(cmd_list, why=args.why, timeout_s=args.timeout, cwd=args.cwd)
        print(f"rc={r['rc']}\n--- stdout ---\n{r['stdout']}\n--- stderr ---\n{r['stderr']}")
        sys.exit(r['rc'])
    elif args.cmd == "write":
        ok = ik_write(args.path, args.content, why=args.why, append=args.append)
        sys.exit(0 if ok else 1)
    elif args.cmd == "delete":
        ok = ik_delete(args.path, why=args.why)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
