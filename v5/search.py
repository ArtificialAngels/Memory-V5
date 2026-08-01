# See docs/scripts/core/v5/v5/search.md

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.memory.v5.search")

# Inline docs: docs/scripts/core/v5/v5/search.md
_EMBED_LOCK = threading.Lock()
_EMBED_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()
_EMBED_CACHE_MAX = 512

_VI_LOCK = threading.Lock()
_VI: dict = {"instance": None, "dir": None, "ts": 0.0}


def _cache_cfg() -> dict:
    try:
        from v5 import preprocess_config as pc
        return pc.cfg().get("cache", {})
    except Exception:
        return {}


def _cache_enabled() -> bool:
    try:
        return bool(_cache_cfg().get("embedding_enabled", True))
    except Exception:
        return True

MEM_ROOT = Path(__file__).resolve().parent.parent
V5_DATA_DIR = MEM_ROOT / "data" / "v5"
CHROMA_DIR = V5_DATA_DIR / "chroma"

# Embedding service: same as V3, uses :8587 nomic-embed-text
# V3 fix (vector_search.py:36-43) 2026-07-05: use /embedding singular path
EMBED_URL = os.environ.get("IKAROS_EMBED_URL", "http://127.0.0.1:8587/embedding")
EMBED_MODEL = os.environ.get("IKAROS_EMBED_MODEL", "nomic-embed-text-v2-moe")
EMBED_TIMEOUT = 10
USER_AGENT = "ikaros-vector-search-v4/1.0 (curl-compatible)"


def _fetch_embedding(text: str, task: str = "query") -> Optional[list[float]]:
    """Call :8587 embedding service (network implementation, no cache).

    V3 -> V4 improvements:
      - Uses relative path (urllib with absolute URI triggers 404, V3 comment recorded)
      - Explicit User-Agent (V3 comment records urllib UA rejection)
      - Logs on failure + returns None, does not swallow

    nomic-embed-text-v2-moe task prefixes (2026-07-14):
      - task="query"    (semantic search)  -> "search_query: "
      - task="document" (index/re-embed)   -> "search_document: "
      Without prefix, falls to default task, causing query/document vector space
      mismatch and recall distortion.
    """
    import http.client
    from urllib.parse import urlparse

    prefix = "search_document: " if task == "document" else "search_query: "
    payload = (prefix + text)[:2000]
    body = json.dumps({"content": payload}).encode("utf-8")
    try:
        u = urlparse(EMBED_URL)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=EMBED_TIMEOUT)
        conn.request("POST", u.path or "/", body=body, headers={
            "Content-Type": "application/json",
            "Host": u.netloc,
            "User-Agent": USER_AGENT,
        })
        resp = conn.getresponse()
        if resp.status != 200:
            logger.warning("embed HTTP %d for '%s...'", resp.status, text[:30])
            return None
        data = json.loads(resp.read().decode("utf-8"))
        # :8587 observed response: list: [{"index":0, "embedding":[[...]]}]
        # Also compatible with dict shapes {"embedding":[[...]]} / {"data":[{"embedding":[...]}]}
        return _extract_vector(data)
    except Exception as e:
        logger.warning("embedding failed: %s", e)
        return None


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
        """Add or update a memory vector."""
        embedding = _get_embedding(content, task="document")
        if embedding is None:
            return False
        try:
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


def fused_search(query: str, top_k: int = 5) -> list[dict]:
    """Dual-path fusion: FTS5 (keyword) + ChromaDB (semantic) -> merge + deduplicate.

    V3 -> V4: FTS5 via v5.store, vectors via v5.search.
    """
    # V5 package lives in Ikaros-memory/; insert Ikaros-memory not its parent dir
    sys.path.insert(0, str(MEM_ROOT))
    from v5 import store  # noqa: F401

    # 1. FTS5 keyword search
    fts_hits = store.search(query, top_k=top_k, min_weight=0.2)
    fts_results = [{
        "id": str(m.id), "content": m.content, "type": m.type,
        "weight": m.weight, "score": 0.3 * (1.0 / (i + 1)), "source": "fts",
        "pad_p": getattr(m, "pad_p", 0.0), "pad_a": getattr(m, "pad_a", 0.0),
    } for i, m in enumerate(fts_hits)]

    # 2. Vector semantic search
    vec_results: list[dict] = []
    try:
        idx = get_vector_index()
        vec_results = idx.search(query, top_k=top_k)
        for r in vec_results:
            r["score"] = 0.7 * r.get("score", 0)
            r["source"] = "vector"
    except Exception as e:
        logger.warning("vector search skipped: %s", e)

    # 3. Merge and deduplicate by id
    seen: dict[str, dict] = {}
    for r in fts_results + vec_results:
        if r["id"] not in seen:
            seen[r["id"]] = r
        else:
            seen[r["id"]]["score"] += r.get("score", 0)

    merged = sorted(seen.values(), key=lambda x: -x.get("score", 0))
    return merged[:top_k]


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
            from v5 import preprocess_config as pc
            use_fuzzy = bool(pc.cfg().get("cache", {}).get("ontology_align_enabled", False))
        except Exception:
            pass
        if use_fuzzy:
            from v5.extensions.ontology_align import find_entity_candidates_fuzzy
            candidates = find_entity_candidates_fuzzy(query, top_k=3)
        else:
            from v5.entity_graph import find_entity_candidates
            candidates = find_entity_candidates(query)
        if not candidates:
            return []
        seeds = [(c.entity_id, c.similarity) for c in candidates[:5]]
        from v5.entity_graph import spreading_activation_search
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
    print(json.dumps(fused_search(q), indent=2, ensure_ascii=False))
