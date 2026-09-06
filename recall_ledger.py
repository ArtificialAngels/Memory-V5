"""跨轮去重账本 (OpenViking recall_log 借鉴) — F1.

问题: 同一条 identity / user_trait / 决策在连续多轮被反复注入上下文, 占 token 且
制造"agent 一直在说同一件事"的噪声。OpenViking 用每会话一个 recall_log 记录
"哪些 URI 已在近 N 轮展示过", 召回时跳过。

关键洞察 (ledger.py:126-130): 因预算不足只展示成**裸 URI**(无正文)的记忆**不计入
"已看过"** —— 它输给了预算, 不是读者看过了; 下一轮可正常展示。本模块把这条搬过来:
仅 detail != "uri"(有正文) 的条目才记录为已服务。

持久化: 每会话一个 JSON 文件 data/v5/recall_log_<session>.json, 经 file_store
原子写 + .bak 滚动备份。fail-open: 任何 IO 异常不阻塞检索。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("ikaros.v5.recall_ledger")

_LEDGER_ROOT = Path(__file__).resolve().parent / "data" / "v5"
_MAX_LEDGER_URIS = 500  # 超过则按最旧淘汰 (OpenViking MAX_LEDGER_URIS=500)
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(session_id: str) -> threading.Lock:
    """每会话一把锁 (并发写互斥, 不同会话不串行)."""
    with _LOCKS_GUARD:
        lk = _LOCKS.get(session_id)
        if lk is None:
            lk = threading.Lock()
            _LOCKS[session_id] = lk
        return lk


def _safe_session(session_id: str) -> str:
    """会话 id → 文件名安全片段 (只留字母数字下划线短横线)."""
    keep = "".join(c if (c.isalnum() or c in "_-") else "_" for c in (session_id or "default"))
    return keep or "default"


class RecallLedger:
    """单会话跨轮去重账本.

    结构: {"session": sid, "turn": N, "entries": {id: {"turn": N, "had_body": bool}}}
    """

    def __init__(self, session_id: str = "default", dedup_turns: int = 5):
        self.session_id = session_id or "default"
        self.dedup_turns = max(0, int(dedup_turns))
        self._path = _LEDGER_ROOT / f"recall_log_{_safe_session(self.session_id)}.json"
        self._lock = _lock_for(self.session_id)
        self._state: dict = self._load()

    def _load(self) -> dict:
        """fail-open: 读不到/损坏 → 空 state (不阻塞检索)."""
        try:
            if self._path.is_file():
                import json
                raw = self._path.read_text(encoding="utf-8")
                d = json.loads(raw)
                if isinstance(d, dict) and "entries" in d:
                    d.setdefault("session", self.session_id)
                    d.setdefault("turn", 0)
                    d.setdefault("entries", {})
                    return d
        except Exception as exc:
            logger.debug("recall_ledger: load failed (%s), starting fresh", exc)
        return {"session": self.session_id, "turn": 0, "entries": {}}

    def _save(self) -> None:
        """fail-open: 写失败只 log, 不抛."""
        try:
            from memory_v5.file_store import atomic_write_json
            atomic_write_json(self._path, self._state, make_backup=True, validator=None)
        except Exception as exc:
            logger.warning("recall_ledger: save failed (%s)", exc)

    def advance_turn(self) -> int:
        """推进一轮 (新用户消息到来时调). 返回新 turn."""
        with self._lock:
            self._state["turn"] = int(self._state.get("turn", 0)) + 1
            self._prune()
            self._save()
            return self._state["turn"]

    @property
    def turn(self) -> int:
        return int(self._state.get("turn", 0))

    def cooled_ids(self, dedup_turns: int | None = None) -> set[str]:
        """返回近 dedup_turns 轮内**展示过正文**的记忆 id (应从召回结果排除).

        bare-URI (had_body=False) 不计入 → 下一轮可正常展示 (输给预算 ≠ 读者看过).
        """
        n = self.dedup_turns if dedup_turns is None else max(0, int(dedup_turns))
        if n <= 0:
            return set()
        cur = self.turn
        floor = cur - n + 1
        out: set[str] = set()
        with self._lock:
            for mid, rec in (self._state.get("entries") or {}).items():
                try:
                    t = int(rec.get("turn", 0))
                    had_body = bool(rec.get("had_body", False))
                except Exception:
                    continue
                # 仅"有正文"且在窗口内的才冷却
                if had_body and floor <= t <= cur:
                    out.add(str(mid))
        return out

    def record_served(self, served: list[tuple[str, bool]]) -> int:
        """记录本轮展示的记忆.

        Args:
            served: [(memory_id_str, had_body)] 列表. had_body=False (bare URI) 的
                    条目**不记录** (bare-URI carve-out).
        Returns:
            实际记录条数.
        """
        if not served:
            return 0
        cur = self.turn
        recorded = 0
        with self._lock:
            entries = self._state.setdefault("entries", {})
            for mid, had_body in served:
                if mid is None:
                    continue
                key = str(mid)
                # had_body=False → 不记 (输给预算, 下一轮可正常展示)
                if not had_body:
                    continue
                prev = entries.get(key)
                # 取更近的轮次 (同 id 多次展示, 保留最近)
                if prev is None or int(prev.get("turn", 0)) <= cur:
                    entries[key] = {"turn": cur, "had_body": True}
                    recorded += 1
            self._prune()
            self._save()
        return recorded

    def _prune(self) -> None:
        """超出 MAX_LEDGER_URIS 时按最旧淘汰."""
        entries = self._state.get("entries") or {}
        if len(entries) <= _MAX_LEDGER_URIS:
            return
        # 按 turn 升序保留最新 MAX_LEDGER_URIS 条
        ordered = sorted(entries.items(), key=lambda kv: int((kv[1] or {}).get("turn", 0)))
        excess = len(ordered) - _MAX_LEDGER_URIS
        if excess <= 0:
            return
        for k, _ in ordered[:excess]:
            entries.pop(k, None)

    def reset(self) -> None:
        """清空账本 (会话重置 / 测试用)."""
        with self._lock:
            self._state = {"session": self.session_id, "turn": 0, "entries": {}}
            self._save()


def ledger_for(session_id: str = "default", dedup_turns: int = 5) -> RecallLedger:
    """便捷构造 (短缓存避免连续调用反复读盘)."""
    return RecallLedger(session_id=session_id, dedup_turns=dedup_turns)
