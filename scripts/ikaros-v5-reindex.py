"""V5 向量补同步脚本 — 把缺向量的记忆补进 Chroma.

用法: PYTHONPATH=core python core/memory_v5/scripts/ikaros-v5-reindex.py [--dry-run]
扫描 SQLite 全部记忆, 对比 Chroma 现有向量, 缺的重新 embed + add.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
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
sys.path.insert(0, str(ROOT / "core"))

_V5_DB = ROOT / "core" / "memory_v5" / "data" / "v5" / "v5.db"


def main() -> int:
    dry = "--dry-run" in sys.argv
    from memory_v5.search import VectorIndex

    t0 = time.time()
    c = sqlite3.connect(str(_V5_DB))
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT id, content, type, tags, weight FROM memory ORDER BY id"
    )]
    c.close()
    print(f"SQLite 记忆总数: {len(rows)}")

    idx = VectorIndex()
    existing = set(idx._collection.get(limit=1000000)["ids"])
    print(f"Chroma 已有向量: {len(existing)}")

    missing = [r for r in rows if str(r["id"]) not in existing]
    print(f"缺向量: {len(missing)} 条")
    if not missing:
        print("无需补同步 ✅")
        return 0
    if dry:
        print("dry-run: 前 10 条缺失:", [r["id"] for r in missing[:10]])
        return 0

    ok = fail = 0
    for i, r in enumerate(missing):
        try:
            idx.add(r["id"], r["content"], type=r["type"], tags=r["tags"], weight=r["weight"])
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            if fail <= 3:
                print(f"  FAIL id={r['id']}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(missing)} (ok={ok} fail={fail})")
    print(f"完成: 补同步 {ok} 条, 失败 {fail} 条, 耗时 {time.time()-t0:.1f}s")
    print(f"Chroma 现在: {idx.stats()}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
