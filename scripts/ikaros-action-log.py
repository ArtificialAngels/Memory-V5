#!/usr/bin/env python3
# 详细说明见 docs/scripts/bin/ikaros-action-log.md
import argparse
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

def _ikaros_root() -> Path:
    env = os.environ.get("IKAROS_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for d in (here, *here.parents):
        if (d / "core" / "memory_v5").is_dir():
            return d
    return here.parents[2] if len(here.parents) > 2 else here

ROOT = _ikaros_root()
LOG_FILE = ROOT / "data" / "logs" / "ikaros-actions.jsonl"

# 5 维信息
WHO_DEFAULT = "Ikaros (self-initiated)"


def _now_iso() -> str:
    """ISO 8601 + 北京时区."""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).isoformat(timespec="milliseconds")


def _ensure_log_dir():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _append(entry: dict):
    _ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


# ---- 核心 API ----
def action_start(
    intent: str,
    action: str = "unknown",
    target: str = "",
    why: str = "",
    who: str = WHO_DEFAULT,
    expected_duration_ms: int = 5000,
    context: dict | None = None,
) -> str:
    """记 1 条 START. 返 action_id (uuid hex)."""
    aid = uuid.uuid4().hex[:12]
    entry = {
        "ts": _now_iso(),
        "type": "action.start",
        "id": aid,
        "intent": intent,
        "action": action,
        "target": target,
        "who": who,
        "why": why,
        "context": context or {},
        "expected_duration_ms": expected_duration_ms,
    }
    _append(entry)
    return aid


def action_end(
    action_id: str,
    result: str = "ok",
    duration_ms: int | None = None,
    exit_code: int | None = None,
    completion_pct: int = 100,
    notes: str = "",
) -> None:
    """记 1 条 END. 配对上面的 action_start."""
    entry = {
        "ts": _now_iso(),
        "type": "action.end",
        "id": action_id,
        "result": result,  # ok / fail / timeout / stuck
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "complete": completion_pct >= 100,
        "completion_pct": completion_pct,
        "notes": notes,
    }
    _append(entry)


def action_done(
    action_id: str,
    result: str = "ok",
    duration_ms: int | None = None,
    exit_code: int | None = None,
    notes: str = "",
    complete: bool = True,
    completion_pct: int = 100,
) -> None:
    """action_end 的便捷别名. complete/completion_pct 由 caller 决定 (默认 true/100)."""
    action_end(action_id, result, duration_ms, exit_code, completion_pct, notes)


# ---- Python decorator / context manager ----
class _Action:
    """action context manager. 用法:
        with action("kill llama", action="process.kill", target=pid) as a:
            process.kill(pid)
            a.done(result="ok")
    """
    def __init__(self, intent: str, **kw):
        self.intent = intent
        self.kw = kw
        self.id: str | None = None
        self.start_ts: float = 0.0
        self._ended = False  # 幂等: 第二次 done 不写

    def __enter__(self):
        self.id = action_start(self.intent, **self.kw)
        self.start_ts = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.done(result="ok")
        else:
            self.done(result="fail",
                     notes=f"{exc_type.__name__}: {exc_val}")
        return False  # 不吞异常

    def done(self, result: str = "ok", duration_ms: int | None = None,
             exit_code: int | None = None, notes: str = "",
             complete: bool = True, completion_pct: int = 100):
        if self._ended:
            return  # 幂等: 防止 __exit__ 重复写
        self._ended = True
        if duration_ms is None:
            duration_ms = int((time.time() - self.start_ts) * 1000)
        action_done(self.id, result, duration_ms, exit_code, notes)


@contextmanager
def action(intent: str, **kw):
    """decorator-friendly context manager."""
    a = _Action(intent, **kw)
    a.__enter__()
    try:
        yield a
    finally:
        a.__exit__(*sys.exc_info())


# ---- convenience wrappers (供工具函数直接调用, 不用写 ctx mgr) ----
def log_subprocess(intent: str, args, *, who: str = "Ikaros (self-initiated)",
                   why: str = "", expected_duration_ms: int = 5000):
    """包 subprocess.run: 自动 action.start + end (exit code / 耗时 / fail).

    用法:
        r = log_subprocess("git status", ["git", "status"])
        r = log_subprocess("kill llama", ["taskkill", "/PID", str(pid)], expected_duration_ms=2500)
    """
    target = " ".join(str(a) for a in args)[:200]
    with action(intent, action="terminal.run", target=target, who=who,
                why=why, expected_duration_ms=expected_duration_ms) as a:
        import subprocess as _sp
        try:
            # 用 bytes 模式避免 cp936/utf-8 解码错
            proc = _sp.run(args, capture_output=True, timeout=expected_duration_ms / 1000 + 60)
            out_text = (proc.stdout or b"").decode("utf-8", errors="replace")[:200]
            err_text = (proc.stderr or b"").decode("utf-8", errors="replace")[:200]
            proc.stdout = out_text
            proc.stderr = err_text
            a.done(result="ok" if proc.returncode == 0 else "fail",
                   exit_code=proc.returncode,
                   notes=err_text or out_text)
            return proc
        except _sp.TimeoutExpired:
            a.done(result="timeout",
                   notes=f"timed out after {expected_duration_ms/1000+60}s")
            raise
        except Exception as e:
            a.done(result="fail", notes=f"{type(e).__name__}: {e}")
            raise


def log_file_write(path: str, *, who: str = "Ikaros (self-initiated)",
                   why: str = "", content_size: int = 0):
    """包 file.write: 自动 start + end."""
    with action(f"write {Path(path).name}", action="file.write",
                target=str(path), who=who, why=why,
                expected_duration_ms=1000) as a:
        from pathlib import Path as _P
        # 不实际写 — caller 自己写, 这只 log. 用 yield 模式 (ctx mgr 不能 yield 真资源)
        a.done(result="ok", notes=f"size={content_size}")


def log_file_delete(path: str, *, who: str = "Ikaros (self-initiated)",
                    why: str = ""):
    """包 file.delete: 自动 start + end."""
    with action(f"delete {Path(path).name}", action="file.delete",
                target=str(path), who=who, why=why,
                expected_duration_ms=500) as a:
        a.done(result="ok")


def log_process_kill(pid: int, *, who: str = "Ikaros (self-initiated)",
                     why: str = ""):
    """包 process.kill: 自动 start + end (Windows: taskkill)."""
    with action(f"kill pid {pid}", action="process.kill",
                target=f"pid:{pid}", who=who, why=why,
                expected_duration_ms=2500) as a:
        import subprocess as _sp
        try:
            proc = _sp.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, text=True, timeout=10,
            )
            a.done(result="ok" if proc.returncode == 0 else "fail",
                   exit_code=proc.returncode,
                   notes=(proc.stderr or proc.stdout)[:200])
            return proc
        except Exception as e:
            a.done(result="fail", notes=f"{type(e).__name__}: {e}")
            raise


# ---- 诊断: 找孤儿 (没 END 的 START) ----
def find_orphans(max_age_minutes: int = 60) -> list[dict]:
    """找最近 N 分钟内没 END 的 START. 这就是卡死/崩溃的证据."""
    if not LOG_FILE.exists():
        return []
    orphans: dict[str, dict] = {}
    cutoff_ts = time.time() - max_age_minutes * 60
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = e.get("id")
            if not eid:
                continue
            # 老的忽略
            try:
                e_ts = datetime.fromisoformat(e["ts"]).timestamp()
            except Exception:
                continue
            if e_ts < cutoff_ts:
                continue
            if e.get("type") == "action.start":
                orphans[eid] = e
            elif e.get("type") == "action.end" and eid in orphans:
                del orphans[eid]
    return list(orphans.values())


# ---- CLI ----
def main():
    p = argparse.ArgumentParser(description="ikaros action log (5 维 + 状态)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="记 1 条 START")
    p_start.add_argument("--intent", required=True)
    p_start.add_argument("--action", default="unknown")
    p_start.add_argument("--target", default="")
    p_start.add_argument("--why", default="")
    p_start.add_argument("--who", default=WHO_DEFAULT)
    p_start.add_argument("--expected-duration-ms", type=int, default=5000)

    p_end = sub.add_parser("end", help="记 1 条 END")
    p_end.add_argument("--id", required=True)
    p_end.add_argument("--result", default="ok",
                       choices=["ok", "fail", "timeout", "stuck"])
    p_end.add_argument("--duration-ms", type=int)
    p_end.add_argument("--exit-code", type=int)
    p_end.add_argument("--completion-pct", type=int, default=100)
    p_end.add_argument("--notes", default="")

    p_orphans = sub.add_parser("orphans", help="找没 END 的 action (卡死/崩溃的证据)")
    p_orphans.add_argument("--max-age-minutes", type=int, default=60)

    args = p.parse_args()

    if args.cmd == "start":
        aid = action_start(
            args.intent, action=args.action, target=args.target,
            why=args.why, who=args.who,
            expected_duration_ms=args.expected_duration_ms)
        print(json.dumps({"id": aid}, ensure_ascii=False))
    elif args.cmd == "end":
        action_end(
            args.id, args.result, args.duration_ms,
            args.exit_code, args.completion_pct, args.notes)
        print(json.dumps({"ok": True}, ensure_ascii=False))
    elif args.cmd == "orphans":
        orphans = find_orphans(args.max_age_minutes)
        print(json.dumps({
            "orphans": len(orphans),
            "actions": orphans,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
