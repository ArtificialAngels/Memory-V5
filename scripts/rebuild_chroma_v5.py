# bin/rebuild_chroma_v5.py
# 全量重建 V5 向量索引 (chroma 集合 ikaros_v5) — 从 v5.db 重新嵌入所有记忆。
#
# 前置:
#   - embedding 服务 :8587 必须已起 (经 watchdog 拉起,
#     或由人工在真实桌面启动看门狗). 本脚本【不】自己拉起 llama-server,
#     只连接外部已运行的 :8587 服务 (避免沙箱环境差异导致崩溃).
#   - 用 runtime/portable-python/python.exe 运行 (chromadb 在该环境)
#
# 设计:
#   - 复用 v5.search.VectorIndex, 集合 schema/元数据与线上完全一致 (ikaros_v5, cosine)
#   - --wait (默认开): 轮询 :8587 直到 /embedding 可用再开工, 适合"先起服务后跑脚本"
#   - 默认 --clean: 先 delete_collection 再重建, 彻底清除孤儿向量 + 旧模型维度错位
#   - 逐行读 v5.db.memory, 调 VectorIndex.add (task=document) 重嵌入
#   - 超长文本 (>512 字) 触发模型 500 (超上下文窗口), 自动退化为截断前缀嵌入
#   - 报告 成功/跳过 数与最终向量数, 并校验 == v5.db 行数
#
# 用法:
#   E:/Ikaros/runtime/portable-python/python.exe bin/rebuild_chroma_v5.py
#   E:/Ikaros/runtime/portable-python/python.exe bin/rebuild_chroma_v5.py --no-wait
#   E:/Ikaros/runtime/portable-python/python.exe bin/rebuild_chroma_v5.py --dry-run
#   E:/Ikaros/runtime/portable-python/python.exe bin/rebuild_chroma_v5.py --limit 50
import argparse
import logging
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
MEMORY_ROOT = Path(os.environ.get("IKAROS_MEMORY", str(ROOT / "core" / "memory_v5"))).resolve()
DB_PATH = MEMORY_ROOT / "data" / "v5" / "v5.db"

# v5 包在 core/memory_v5 下; 把 core/ 前置进 sys.path 让 `import memory_v5` 生效
sys.path.insert(0, str(ROOT / "core"))
os.chdir(MEMORY_ROOT)

from memory_v5 import search as vs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rebuild_chroma")


def _wait_for_service(timeout: int) -> "list[float] | None":
    """轮询 :8587 /embedding 端点, 直到返回有效向量 (或超时).

    不启动任何进程, 只读外部已运行的服务. 返回探针向量 (含维度) 或 None.
    """
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            v = vs._get_embedding("probe: wait-for-service", task="document")
        except Exception as e:  # 极端: 网络层异常
            v = None
            last_err = str(e)
        if v is not None:
            return v
        left = int(deadline - time.time())
        log.info("  :8587 尚未就绪, 继续等待... (剩余 %ds)", max(left, 0))
        time.sleep(3)
    if last_err:
        log.warning("  末次探测异常: %s", last_err)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="全量重建 V5 向量索引 (chroma ikaros_v5)")
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不写入 chroma")
    ap.add_argument("--no-clean", action="store_true",
                    help="不删旧集合, 仅 upsert (保留孤儿向量, 维度不符会跳过)")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 行 (调试)")
    ap.add_argument("--no-wait", action="store_true",
                    help="不轮询, 立即探测; :8587 未起则直接退出")
    ap.add_argument("--wait-timeout", type=int, default=600,
                    help="等待 :8587 就绪的超时秒数 (默认 600)")
    args = ap.parse_args()

    # 1) 确认 :8587 可达且取维度 (不可达则等待或退出, 绝不自行拉起 llama-server)
    if args.no_wait:
        probe = vs._get_embedding("probe: 向量库重建探针", task="document")
    else:
        probe = _wait_for_service(args.wait_timeout)
    if probe is None:
        log.error("embedding 服务 :8587 在 %ds 内仍不可达 —— 请先经控制面板 start "
                  "拉起 (或人工启动看门狗) 后再运行. 终止.", args.wait_timeout)
        sys.exit(2)
    log.info("embedding 服务 OK (维度=%d)", len(probe))

    # 2) 读 v5.db.memory
    if not DB_PATH.exists():
        log.error("v5.db 不存在: %s", DB_PATH)
        sys.exit(3)
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT id, content, type, tags, weight FROM memory ORDER BY id"
    ).fetchall()
    total = len(rows)
    log.info("v5.db memory 行数=%d", total)
    if args.limit:
        rows = rows[: args.limit]
        log.info("[debug] 仅处理前 %d 行", len(rows))
    if args.dry_run:
        log.info("[dry-run] 不写入 chroma, 完成.")
        return

    # 3) clean: 删旧集合重建 (消除孤儿向量 / 维度错位)
    if not args.no_clean:
        import chromadb

        client = chromadb.PersistentClient(path=str(vs.CHROMA_DIR))
        try:
            client.delete_collection("ikaros_v5")
            log.info("已删除旧集合 ikaros_v5 (清除孤儿向量 + 旧模型维度错位)")
        except Exception as e:  # 集合不存在等
            log.info("旧集合删除跳过: %s", e)

    # 4) 逐行重嵌入
    idx = vs.VectorIndex()
    ok = skip = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        content = (r["content"] or "").strip()
        if not content:
            skip += 1
            continue
        # 重试: 批量负载下 :8587 偶发 HTTP 500 (瞬时抖动), 重试可闭合;
        # 超长文本触发模型 500 (超上下文窗口 ~512 token), 退化为截断前缀嵌入.
        res = False
        last_err = ""
        cuts = [len(content)]
        if len(content) > 512:
            cuts.append(512)
        if len(content) > 256:
            cuts.append(256)
        for attempt in range(1, 4):
            for cut in cuts:
                try:
                    r_add = idx.add(
                        int(r["id"]),
                        content[:cut],
                        type=(r["type"] or "fact"),
                        tags=(r["tags"] or ""),
                        weight=float(r["weight"] or 0.6),
                    )
                    if r_add:
                        res = True
                        break
                    last_err = "idx.add returned False"
                except Exception as e:  # 嵌入异常 (HTTP 500 等)
                    last_err = str(e)
            if res:
                break
            if attempt < 3:
                log.warning("  嵌入重试 id=%s attempt %d: %s",
                            r["id"], attempt, last_err[:80])
                time.sleep(2 * attempt)
        if res:
            ok += 1
        else:
            skip += 1
            log.debug("  跳过 id=%s: %s", r["id"], last_err[:80])
        if i % 200 == 0:
            log.info("进度 %d/%d  ok=%d skip=%d  %.1fs",
                     i, total, ok, skip, time.time() - t0)

    log.info("=== 重建完成: 总=%d 成功=%d 跳过=%d 用时=%.1fs ===",
             total, ok, skip, time.time() - t0)
    log.info("chroma 当前向量数=%d", idx._collection.count())

    # 5) 校验: 向量数应 == v5.db 行数 (无孤儿, 无缺漏)
    final = idx._collection.count()
    if final == total:
        log.info("✔ 校验通过: chroma 向量数(%d) == v5.db 行数(%d)", final, total)
    else:
        log.warning("⚠ 校验偏差: chroma=%d vs v5.db=%d (差异 %d)",
                    final, total, total - final)


if __name__ == "__main__":
    main()
