"""Ikaros V5 <-> Hermes memory provider bridge.

Runs INSIDE the Ikaros V5 Python runtime (runtime/portable-python, which has
chromadb). Hermes's ikaros_v5 provider shells out to this script with a JSON
request on stdin and reads a JSON response on stdout. This keeps the heavy
V5 dependencies (chromadb, onnxruntime, etc.) out of hermes-agent's venv.

Request shapes (JSON on stdin):
  {"action": "search", "query": "...", "top_k": 5}
  {"action": "add", "content": "...", "type": "fact", "weight": 0.6, "tags": ""}
  {"action": "activity"}                          # 返回当前活动感知
  {"action": "context_stats"}                     # 返回上下文压缩引擎状态

Response:
  {"ok": true, "results": [...]}          # search
  {"ok": true, "id": 123}                 # add
  {"ok": true, "activity": "..."}         # activity
  {"ok": true, "stats": {...}}            # context_stats
  {"ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# MEM_ROOT = core/v5  (parent of this file's directory core/v5/v5)
MEM_ROOT = Path(__file__).resolve().parent.parent
if str(MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEM_ROOT))


def _do_search(query: str, top_k: int) -> list:
    from v5 import search as v5search
    return v5search.fused_search(query, top_k=top_k)


def _do_add(content: str, type_: str, weight: float, tags: str) -> int:
    from v5 import store as v5store
    return v5store.store(content, type=type_, weight=weight, tags=tags)


def main() -> None:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"bad request: {e}"}))
        return

    action = req.get("action")
    try:
        if action == "search":
            query = (req.get("query") or "").strip()
            top_k = max(1, min(int(req.get("top_k", 5)), 20))
            if not query:
                print(json.dumps({"ok": True, "results": []}))
                return
            results = _do_search(query, top_k)
            print(json.dumps({"ok": True, "results": results}, ensure_ascii=False))
        elif action == "add":
            content = (req.get("content") or "").strip()
            if not content:
                print(json.dumps({"ok": False, "error": "empty content"}))
                return
            type_ = req.get("type", "fact") or "fact"
            weight = float(req.get("weight", 0.6))
            tags = req.get("tags", "") or ""
            mid = _do_add(content, type_, weight, tags)
            print(json.dumps({"ok": True, "id": mid}, ensure_ascii=False))
        elif action == "activity":
            # 返回当前活动感知 (窗口标题 + 活动推断)
            try:
                sys.path.insert(0, str(MEM_ROOT))
                import cogno_5d
                title = cogno_5d._get_foreground_window_title()
                narrative = cogno_5d._get_activity_narrative()
                print(json.dumps({
                    "ok": True,
                    "activity": narrative,
                    "window_title": title,
                    "cached": True,
                }, ensure_ascii=False))
            except Exception as e:
                print(json.dumps({"ok": False, "error": str(e)}))
        elif action == "context_stats":
            # 返回上下文压缩引擎状态
            try:
                sys.path.insert(0, str(MEM_ROOT))
                import cogno_5d
                from v5.rhythm import build_rhythm_block, last_interaction_ts
                from v5.summary import _load_cache as _load_summary_cache
                from v5.profile import load_dislikes, load_preferences

                rhythm_text = build_rhythm_block()
                summary_cache = _load_summary_cache()
                dislikes = load_dislikes()
                prefs = load_preferences()

                # 活动感知
                title = cogno_5d._get_foreground_window_title()
                narrative = cogno_5d._get_activity_narrative()

                # 记忆统计
                from v5 import store as v5store
                _s = v5store.stats()
                total_count = _s.get("total", 0)
                type_breakdown = {k: v["count"] for k, v in _s.get("by_type", {}).items()}

                print(json.dumps({
                    "ok": True,
                    "stats": {
                        "activity": {
                            "narrative": narrative,
                            "window_title": title,
                        },
                        "rhythm": rhythm_text,
                        "summary": {
                            "cached": bool(summary_cache.get("last_summary")),
                            "last_round": summary_cache.get("last_round", -1),
                            "preview": (summary_cache.get("last_summary") or "")[:80],
                        },
                        "profile": {
                            "dislikes": len(dislikes),
                            "preferences": len(prefs),
                        },
                        "memory": {
                            "total": total_count,
                            "by_type": type_breakdown,
                        },
                    }
                }, ensure_ascii=False))
            except Exception as e:
                import traceback
                print(json.dumps({"ok": False, "error": str(e),
                                   "trace": traceback.format_exc()[:500]}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()[:500]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
