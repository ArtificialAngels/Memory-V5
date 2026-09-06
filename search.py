# See docs/scripts/core/memory_v5/v5/search.md

from __future__ import annotations

import http.client
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("ikaros.memory.v5.search")

# Inline docs: docs/scripts/core/memory_v5/v5/search.md
_EMBED_LOCK = threading.Lock()
_EMBED_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()
_EMBED_CACHE_MAX = 512

_VI_LOCK = threading.Lock()
_VI: dict = {"instance": None, "dir": None, "ts": 0.0}

# ── 跨进程 Chroma 写锁 (2026-08-02 根因修复) ────────────────────────────
# 多进程 (MCP server × N + watchdog + reflect loop) 并发写同一 Chroma 持久目录
# 会触发 hnsw compactor 冲突 ("Failed to apply logs to the hnsw segment writer")。
# 写前拿文件锁串行化。进程内线程锁 + 进程间文件锁双保险。
_CHROMA_LOCK_FILE = Path(__file__).resolve().parent / "data" / "v5" / ".chroma-write.lock"
_chroma_thread_lock = threading.Lock()


class _chroma_write_lock:
    """上下文管理器: 拿跨进程文件锁 (Windows msvcrt / POSIX fcntl)."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout
        self._fd = None

    def __enter__(self):
        _chroma_thread_lock.acquire()
        _CHROMA_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(str(_CHROMA_LOCK_FILE), "a+b")
        deadline = time.time() + self._timeout
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.time() > deadline:
                        self._fd.close()
                        self._fd = None
                        _chroma_thread_lock.release()
                        raise TimeoutError("chroma write lock timeout")
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            if self._fd is not None:
                if os.name == "nt":
                    import msvcrt
                    try:
                        self._fd.seek(0)
                        msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
        finally:
            _chroma_thread_lock.release()
        return False


def _cache_cfg() -> dict:
    try:
        from memory_v5 import preprocess_config as pc
        return pc.cfg().get("cache", {})
    except Exception:
        return {}


def _cache_enabled() -> bool:
    try:
        return bool(_cache_cfg().get("embedding_enabled", True))
    except Exception:
        return True


# ── Circuit breaker (Task 2.0.2, 2026-08-20) ──────────────────────────────
# 设计: docs/memory_v5-circuit-breaker-design.md
# 状态: closed → open (连续失败 ≥ threshold) → half_open (冷却后探针) → closed
# 线程安全: threading.Lock (与 _EMBED_LOCK / _chroma_thread_lock 同模式)
# 失败分类: 只计网络/OSError/HTTPException; 程序错误 (TypeError 等) 不计入
_CIRCUIT_LOCK = threading.Lock()
_CIRCUIT: dict = {
    "state": "closed",       # "closed" | "open" | "half_open"
    "failure_count": 0,       # resets to 0 on success
    "opened_at": 0.0,         # time.monotonic() when breaker tripped
    "last_failure_reason": "",
}
_NETWORK_EXC = (
    OSError,                  # socket.* + Connection* + TimeoutError (3.10+)
    http.client.HTTPException,
)


def _cb_enabled() -> bool:
    try:
        return bool(_cache_cfg().get("circuit_breaker_enabled", True))
    except Exception:
        return True


def _cb_threshold() -> int:
    try:
        v = _cache_cfg().get("circuit_breaker_threshold", 3)
        return max(1, int(v))
    except Exception:
        return 3


def _cb_reset_seconds() -> float:
    try:
        v = _cache_cfg().get("circuit_breaker_reset_seconds", 30)
        return max(0.0, float(v))
    except Exception:
        return 30.0


def _is_network_error(exc: BaseException) -> bool:
    """只把真正的网络错误计入短路; 程序错误 (TypeError 等) 不计入 (设计 §5.1)。"""
    return isinstance(exc, _NETWORK_EXC)


def _circuit_state() -> dict:
    """读取 (浅拷贝) 状态字典; 给测试/调试用。设计 §2 可观测性。"""
    with _CIRCUIT_LOCK:
        return dict(_CIRCUIT)


def _circuit_reset() -> None:
    """测试用: 重置短路器到 closed。"""
    with _CIRCUIT_LOCK:
        _CIRCUIT["state"] = "closed"
        _CIRCUIT["failure_count"] = 0
        _CIRCUIT["opened_at"] = 0.0
        _CIRCUIT["last_failure_reason"] = ""


def _circuit_is_open() -> bool:
    """短路器是否处于 open (拒服务) 状态。half_open 算"可放行"返回 False。

    副作用: open → half_open 的状态转移在此处原地完成 (设计 §3.2)。
    """
    if not _cb_enabled():
        return False
    with _CIRCUIT_LOCK:
        if _CIRCUIT["state"] == "open":
            if time.monotonic() - _CIRCUIT["opened_at"] >= _cb_reset_seconds():
                _CIRCUIT["state"] = "half_open"
                logger.info("embedding circuit breaker HALF_OPEN: probing")
                return False
            return True
        return False  # closed 或 half_open 均放行


def _circuit_record_success() -> None:
    """成功回调: closed 状态下重置计数; half_open → closed (探针成功)。"""
    with _CIRCUIT_LOCK:
        if _CIRCUIT["state"] == "half_open":
            logger.info("embedding circuit breaker CLOSED: probe succeeded")
        _CIRCUIT["state"] = "closed"
        _CIRCUIT["failure_count"] = 0
        _CIRCUIT["opened_at"] = 0.0
        _CIRCUIT["last_failure_reason"] = ""


def _circuit_record_failure(exc: BaseException) -> None:
    """失败回调: closed 累计失败次数到阈值则跳闸; half_open 失败则立即重开。

    非网络错误不计入 (避免程序 bug 把短路器误跳)。设计 §5.1。
    """
    if not _cb_enabled():
        return
    if not _is_network_error(exc):
        return
    reason = f"{type(exc).__name__}: {exc}"[:200]
    with _CIRCUIT_LOCK:
        if _CIRCUIT["state"] == "half_open":
            # 探针失败 → 立即重新 open, 重置冷却计时 (设计 §3.2)
            _CIRCUIT["state"] = "open"
            _CIRCUIT["opened_at"] = time.monotonic()
            _CIRCUIT["last_failure_reason"] = reason
            logger.warning(
                "embedding circuit breaker RE-OPEN: probe failed (%s)", reason)
            return
        _CIRCUIT["failure_count"] += 1
        _CIRCUIT["last_failure_reason"] = reason
        if _CIRCUIT["failure_count"] >= _cb_threshold():
            _CIRCUIT["state"] = "open"
            _CIRCUIT["opened_at"] = time.monotonic()
            logger.warning(
                "embedding circuit breaker OPEN: %d consecutive failures (last: %s)",
                _CIRCUIT["failure_count"], reason,
            )

MEM_ROOT = Path(__file__).resolve().parent
V5_DATA_DIR = MEM_ROOT / "data" / "v5"
CHROMA_DIR = V5_DATA_DIR / "chroma"

# Embedding service: same as V3, uses :8587 nomic-embed-text
# V3 fix (vector_search.py:36-43) 2026-07-05: use /embedding singular path
EMBED_URL = os.environ.get("IKAROS_EMBED_URL", "http://127.0.0.1:8587/embedding")
EMBED_MODEL = os.environ.get("IKAROS_EMBED_MODEL", "bge-m3")
EMBED_TIMEOUT = 10
USER_AGENT = "ikaros-vector-search-v4/1.0 (curl-compatible)"

# 嵌入连接 (2026-08-19): 模块级单连接复用 (HTTP/1.1 keep-alive),
# 取代 _fetch_embedding 里每 chunk 新建 HTTPConnection (~1.6s → ~30ms 热调)。
#
# 线程安全 (GH audit P1-5, 2026-08-24): http.client.HTTPConnection 不可并发使用,
# dsh web 是多线程 → 模块级单例会在并发检索时竞态损坏 HTTP 响应流。改为
# thread-local: 每线程各自持有一条 keep-alive 连接, 无竞争、保住热调性能。
_embed_conn_tls = threading.local()


def _get_embed_conn(u) -> http.client.HTTPConnection:
    """Return this thread's keep-alive embedding connection (create if needed).

    http.client.HTTPConnection is not safe for concurrent use; a module-level
    singleton raced across threads and corrupted the HTTP response stream. Each
    thread now owns its own persistent connection (keep-alive perf preserved).
    """
    conn = getattr(_embed_conn_tls, "c", None)
    if conn is None or conn.sock is None:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=EMBED_TIMEOUT)
        _embed_conn_tls.c = conn
    return conn


def _reset_embed_conn() -> None:
    """Close + drop this thread's embedding connection (on error / 5xx)."""
    conn = getattr(_embed_conn_tls, "c", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _embed_conn_tls.c = None


def _fetch_embedding(text: str, task: str = "query") -> Optional[list[float]]:
    """Call :8587 embedding service with circuit breaker (Task 2.0.2).

    公开签名/返回契约保持不变: 仍返回 vec 或 None (None = fail-open)。

    短路逻辑 (设计 §3.1):
      1. 若 breaker open: 立即返回 None, 不碰网络 (微秒级)
      2. 否则调 _do_fetch_embedding
         - 成功: breaker.record_success() + 返回 vec
         - 失败: breaker.record_failure(exc) + 返回 None

    V3 -> V4 improvements:
      - Uses relative path (urllib with absolute URI triggers 404, V3 comment recorded)
      - Explicit User-Agent (V3 comment records urllib UA rejection)
      - Logs on failure + returns None, does not swallow

    V5.6 (2026-08-10): chunked embedding for long documents.
      - :8587 llama-server physical batch limit ≈512 tokens; any longer input
        returns HTTP 500 ("input (N tokens) is too large to process").
      - Long memories (conversation transcripts, >~350 CJK chars) silently lost
        their vector sync, leaving them FTS-only and harming recall.
      - Fix: split into ≤350-char chunks, embed each with the task prefix,
        mean-pool the vectors (standard long-text embedding practice).

    历史: 嵌入模型从 nomic-embed-text-v2-moe 切换到 bge-m3 q8_0 (1024 维) ——
      nomic-v2-moe 在 llama.cpp 下输出全零; nomic-v1.5 中文语义弱。
      bge-m3 无需 document 前缀, query 按官方推荐加检索指令。
      (旧 nomic-embed-text-v2-moe 的 task prefixes 已弃用: search_query:/search_document:)

    2026-08-14: 嵌入模型已换 bge-m3 (nomic-v2-moe 在 llama.cpp 下输出全零;
    nomic-v1.5 中文语义弱)。bge-m3 无需 document 前缀, query 按官方推荐加检索指令
    "为这个句子生成表示以用于检索相关文章："。

    2026-08-20 (Task 2.0.2): 嵌入 :8587 不可达时短路 (circuit breaker),
    避免每次调用挂满 10s 超时。设计: docs/memory_v5-circuit-breaker-design.md。
    """
    # 短路器 open → 立即返回 None (设计 §2: 微秒级, <1µs 开销)
    if _circuit_is_open():
        return None
    try:
        vec = _do_fetch_embedding(text, task)
    except Exception as e:
        _circuit_record_failure(e)
        return None
    if vec is None:
        # 静默 None (json 解码失败、HTTP 500 等, 非异常路径) 不计入短路 ——
        # 设计 §5.1: 只计网络/OSError/HTTPException。程序错误 / 数据错误
        # 不应跳闸。这里 _do_fetch_embedding 已吞掉异常并返回 None,
        # 不进入 except 分支, 因此无法获得 exc 类型 → 一律不计。
        # 副作用: 持久性 JSON decode 错误不会跳闸 (后续 Task 排查)。
        return None
    _circuit_record_success()
    return vec


def _do_fetch_embedding(text: str, task: str = "query") -> Optional[list[float]]:
    """Call :8587 embedding service (network implementation, no cache, no breaker).

    由 _fetch_embedding() 包裹调用; 公开 API 仍走 _fetch_embedding。
    本函数保留原签名/语义, 仅被拆出以插入短路器钩子 (Task 2.0.2, 设计 §3.1)。
    """
    if task == "document":
        prefix = ""
    else:
        prefix = "为这个句子生成表示以用于检索相关文章："
    # 单块嵌入: 超长文本分块 (每块 ≤350 字符, 中文 ~500 tokens 内安全)
    text = (text or "").strip()
    if not text:
        return None
    _MAX_CHUNK = 350
    chunks = [text[i:i + _MAX_CHUNK] for i in range(0, len(text), _MAX_CHUNK)] \
        if len(text) > _MAX_CHUNK else [text]

    vectors: list[list[float]] = []
    u = urlparse(EMBED_URL)
    # thread-local keep-alive 连接 (GH audit P1-5: 并发安全)
    conn = _get_embed_conn(u)
    for chunk in chunks:
        payload = (prefix + chunk)[:2000]
        body = json.dumps({"content": payload}).encode("utf-8")
        try:
            conn.request("POST", u.path or "/", body=body, headers={
                "Content-Type": "application/json",
                "Host": u.netloc,
                "User-Agent": USER_AGENT,
            })
            resp = conn.getresponse()
            if resp.status != 200:
                logger.warning("embed HTTP %d for '%s...'", resp.status, chunk[:30])
                _reset_embed_conn()
                return None
            data = json.loads(resp.read().decode("utf-8"))
            # :8587 observed response: list: [{"index":0, "embedding":[[...]]}]
            # Also compatible with dict shapes {"embedding":[[...]]} / {"data":[{"embedding":[...]}]}
            vec = _extract_vector(data)
            if vec is None:
                return None
            vectors.append(vec)
        except Exception as e:
            # 连接可能已断 (服务重启/超时) → 关闭置空, 下次调用重建
            logger.warning("embedding failed: %s", e)
            _reset_embed_conn()
            raise  # 抛给 _fetch_embedding 的 except 触发 _circuit_record_failure

    if len(vectors) == 1:
        return vectors[0]
    # 多块平均池化 (标准长文本嵌入做法)
    import numpy as np
    mean = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    return [float(x) for x in mean]


def _get_embedding(text: str, task: str = "query") -> Optional[list[float]]:
    """Embedding entry with process-level LRU cache (key = task+text[:2000]).

    Cache hit -> skip :8587 network call (~60ms idle savings, ~1s busy, more cold).
    Cache is shared across sessions (watchdog process lives long); capacity cap
    prevents memory bloat.
    """
    if not _cache_enabled():
        return _fetch_embedding(text, task)
    prefix = "search_document: " if task == "document" else "search_query: "
    key = (prefix + text)[:2000]
    with _EMBED_LOCK:
        if key in _EMBED_CACHE:
            _EMBED_CACHE.move_to_end(key)
            return _EMBED_CACHE[key]
    vec = _fetch_embedding(text, task)
    if vec is not None:
        cap = int(_cache_cfg().get("embedding_max", 512))
        with _EMBED_LOCK:
            _EMBED_CACHE[key] = vec
            _EMBED_CACHE.move_to_end(key)
            while len(_EMBED_CACHE) > cap:
                _EMBED_CACHE.popitem(last=False)
    return vec


def _extract_vector(data) -> Optional[list[float]]:
    """Extract single vector (list[float]) from various :8587 response shapes.

    Observed shapes:
      - list:  [{"index":0, "embedding":[[...]]}]   (llama-server /embedding observed)
      - dict:  {"embedding": [[...]]} or {"embedding": [...]}
      - dict:  {"data": [{"embedding": [...]}]}      (OpenAI style)
    """
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                emb = item.get("embedding")
            elif isinstance(item, list):
                emb = item
            else:
                continue
            vec = _coerce_vector(emb)
            if vec is not None:
                return vec
        return None
    if isinstance(data, dict):
        if "embedding" in data:
            return _coerce_vector(data["embedding"])
        if "data" in data and isinstance(data["data"], list) and data["data"]:
            inner = data["data"][0]
            if isinstance(inner, dict) and "embedding" in inner:
                return _coerce_vector(inner["embedding"])
    return None


def _coerce_vector(emb) -> Optional[list[float]]:
    """embedding field may be [[...]] (list wrapping single) or [...]; flatten to list[float]."""
    if isinstance(emb, list) and emb:
        if isinstance(emb[0], list):
            cand = emb[0]
            if cand and isinstance(cand[0], (int, float)):
                return [float(x) for x in cand]
        elif isinstance(emb[0], (int, float)):
            return [float(x) for x in emb]
    return None


class VectorIndex:
    """V5 ChromaDB vector index, synced with v5.store.

    V3 -> V4 improvements:
      - import chromadb moved to __init__ (not module-level), explicit on failure
      - Path uses V5 subdirectory (isolated from V3)
    """

    def __init__(self, persist_dir: Path | None = None):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb not installed. Run via ikaros-mem.bat (uses portable-python) "
                "or: E:\\Ikaros\\runtime\\portable-python\\python.exe -m pip install chromadb"
            ) from e
        self._persist_dir = Path(persist_dir or CHROMA_DIR)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name="ikaros_v5",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorIndex V5: %d vectors in %s",
                    self._collection.count(), self._persist_dir)

    def add(self, memory_id: int, content: str, *,
            type: str = "fact", tags: str = "", weight: float = 0.6) -> bool:
        """Add or update a memory vector.

        跨进程写锁: 多进程 (MCP server × N + watchdog + reflect loop) 并发写同一
        Chroma 实例会触发 hnsw compactor 冲突 ("Failed to apply logs to the hnsw
        segment writer")。写前拿文件锁串行化, 消除该失败源 (2026-08-02 根因修复)。
        """
        embedding = _get_embedding(content, task="document")
        if embedding is None:
            return False
        try:
            with _chroma_write_lock():
                self._collection.upsert(
                    ids=[str(memory_id)],
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[{"type": type, "tags": tags, "weight": weight}],
                )
            return True
        except Exception as e:
            logger.warning("vector add failed: %s", e)
            return False

    def search(self, query: str, top_k: int = 5,
               min_weight: float = 0.0) -> list[dict]:
        """Semantic search, returns [{id, content, type, weight, score}]."""
        embedding = _get_embedding(query, task="query")
        if embedding is None:
            logger.warning("search: embedding failed for '%s...'", query[:30])
            return []
        try:
            n = max(1, min(top_k * 2, self._collection.count() or 1))
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            if not results or not results.get("ids") or not results["ids"][0]:
                return []
            ids0 = results["ids"][0]
            # 防御: chroma 在个别结果上可能返回 None (迁移遗留/异常写入),
            # 不能因单条坏数据让整次查询 abort 并静默返回 [], 否则语义召回直接失效.
            # 注意 chroma 返回均为两层结构 [[...]], 先取内层 [0].
            raw_docs = results.get("documents")
            raw_metas = results.get("metadatas")
            raw_dists = results.get("distances")
            docs0 = raw_docs[0] if raw_docs else [None] * len(ids0)
            metas0 = raw_metas[0] if raw_metas else [None] * len(ids0)
            dists0 = raw_dists[0] if raw_dists else [None] * len(ids0)
            items = []
            for i, mid in enumerate(ids0):
                doc = docs0[i] if i < len(docs0) else None
                if doc is None:
                    continue  # 无内容无法使用, 跳过 (避免 None 落库污染检索结果)
                meta = metas0[i] if (i < len(metas0) and metas0[i] is not None) else {}
                dist = dists0[i] if i < len(dists0) else None
                try:
                    weight = float(meta.get("weight", 0.6))
                except (TypeError, ValueError):
                    weight = 0.6
                if weight < min_weight:
                    continue
                try:
                    dist_f = float(dist) if dist is not None else 1.0
                except (TypeError, ValueError):
                    dist_f = 1.0
                score = max(0.0, min(1.0, 1.0 - dist_f))
                items.append({
                    "id": mid,
                    "content": doc,
                    "type": meta.get("type", "fact"),
                    "weight": weight,
                    "distance": dist_f,
                    "score": score,
                })
                if len(items) >= top_k:
                    break
            return items
        except Exception as e:
            logger.warning("vector search failed: %s", e)
            return []

    def stats(self) -> dict:
        return {
            "total_vectors": self._collection.count(),
            "persist_dir": str(self._persist_dir),
            "embed_model": EMBED_MODEL,
            "embed_url": EMBED_URL,
        }


def get_vector_index(persist_dir: Path | None = None, *, refresh: bool = False):
    """Return cached VectorIndex singleton (performance optimization).

    - Process-level reuse of the same chroma client, avoids reopening every cycle
      (cold start 850ms, warm ~15ms).
    - Auto-refresh every vector_refresh_seconds to pick up memories added by
      other processes (reflection loop, etc.). Same-process adds via the same
      client are immediately visible without refresh.
    - Set cache.vector_index_singleton=false to disable and create fresh each call.
    """
    cfg = _cache_cfg()
    if not cfg.get("vector_index_singleton", True):
        return VectorIndex(persist_dir)
    pdir = str(persist_dir or CHROMA_DIR)
    refresh_s = float(cfg.get("vector_refresh_seconds", 30))
    now = time.time()
    with _VI_LOCK:
        inst = _VI["instance"]
        if (inst is None or _VI["dir"] != pdir or refresh
                or (now - _VI["ts"]) > refresh_s):
            try:
                inst = VectorIndex(persist_dir)
                _VI["instance"] = inst
                _VI["dir"] = pdir
                _VI["ts"] = now
            except Exception:
                # Creation failed: clear cache, let caller handle silently (don't cache bad instance)
                _VI["instance"] = None
                _VI["dir"] = None
                raise
        return _VI["instance"]


# fused_search 已删除 (2026-08-14 P1 检索收敛): 旧双路检索实现, 检索唯一入口收敛到
# memory_retrieval.unified_retrieve。原调用方 dissonance/metacog 已切换。


def entity_graph_search(query: str, top_k: int = 5) -> list[dict]:
    """Entity graph spreading activation search.
    Matches query against entity graph and activates linked episodic memories.
    Returns list of episodic memory dicts with graph_score.

    2026-08-01: 当 preprocess_config.yaml 的 cache.ontology_align_enabled=true 时,
    实体候选走 extensions.ontology_align.find_entity_candidates_fuzzy
    (exact/包含优先 + difflib 模糊补召回, 零 LLM 成本); 默认仍用原精确匹配.
    """
    try:
        # 本体对齐开关: 读 config 的 cache.ontology_align_enabled (fail-open 关)
        use_fuzzy = False
        try:
            from memory_v5 import preprocess_config as pc
            use_fuzzy = bool(pc.cfg().get("cache", {}).get("ontology_align_enabled", False))
        except Exception:
            pass
        if use_fuzzy:
            from memory_v5.extensions.ontology_align import find_entity_candidates_fuzzy
            candidates = find_entity_candidates_fuzzy(query, top_k=3)
        else:
            from memory_v5.entity_graph import find_entity_candidates
            candidates = find_entity_candidates(query)
        if not candidates:
            return []
        seeds = [(c.entity_id, c.similarity) for c in candidates[:5]]
        from memory_v5.entity_graph import spreading_activation_search
        episodic = spreading_activation_search(seeds, top_k=top_k)
        return [{
            "id": m.id, "content": m.summary, "type": "episodic",
            "weight": m.importance, "score": m.graph_score,
            "source": "entity_graph", "detail": m.detail
        } for m in episodic]
    except Exception as e:
        logger.debug("entity_graph search skipped: %s", e)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != "search":
        print("Usage: python v4/search.py search <query>")
        sys.exit(1)
    q = " ".join(sys.argv[2:])
    from memory_v5.memory_retrieval import unified_retrieve
    print(json.dumps(unified_retrieve(q, top_k=5), indent=2, ensure_ascii=False))
