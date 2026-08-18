# See docs/scripts/core/v5/v5/store.md

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("ikaros.memory.v5.store")

# Inline docs: docs/scripts/core/v5/v5/store.md
MEM_ROOT = Path(__file__).resolve().parent
V5_DATA_DIR = MEM_ROOT / "data" / "v5"
V5_DB_PATH = V5_DATA_DIR / "v5.db"

# R5 (M1): WAL checkpoint 从"每次写前执行"收敛为"每进程首次连接时执行一次"。
#   原实现每次 store() 都 wal_checkpoint(TRUNCATE), WAL 批量写优势被清零;
#   现在交给 SQLite 默认 wal_autocheckpoint (1000 页) + 进程级一次性 checkpoint。
_wal_checkpoint_lock = threading.Lock()
_wal_checkpointed = False

# V5.2 schema: neko memory features merged
# 新增: character(角色隔离), reinforcement/disputation(证据评分),
#       evidence_version(证据版本号), source_memory_id(关联源记忆)
SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'fact',
    tags TEXT DEFAULT '',
    weight REAL NOT NULL DEFAULT 0.6,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL DEFAULT (strftime('%s','now')),
    short_term INTEGER NOT NULL DEFAULT 1,
    long_term INTEGER NOT NULL DEFAULT 0,
    -- V5 emotion fingerprint (PAD model: pleasure / arousal / dominance)
    pad_p REAL NOT NULL DEFAULT 0.0,
    pad_a REAL NOT NULL DEFAULT 0.0,
    pad_d REAL NOT NULL DEFAULT 0.0,
    -- V5.2: character isolation (neko per-character migration)
    character TEXT NOT NULL DEFAULT '',
    -- V5.2: evidence scoring (reinforcement / disputation with half-life)
    reinforcement REAL NOT NULL DEFAULT 0.0,
    disputation REAL NOT NULL DEFAULT 0.0,
    evidence_version INTEGER NOT NULL DEFAULT 0,
    -- V5.2: source memory id chain (for consolidate -> reflection -> persona)
    source_memory_id INTEGER DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(type);
CREATE INDEX IF NOT EXISTS idx_memory_weight ON memory(weight);
CREATE INDEX IF NOT EXISTS idx_memory_last_accessed ON memory(last_accessed);
-- V5.2 indexes created separately after column migration
-- (see conn() ALTER TABLE section)

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content, type, tags,
    content='memory',
    content_rowid='id'
);

-- V3 triggers: FTS5 sync (V4 reused, Phase 4 transition compatible)
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, content, type, tags)
    VALUES (new.id, new.content, new.type, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, type, tags)
    VALUES ('delete', old.id, old.content, old.type, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, type, tags)
    VALUES ('delete', old.id, old.content, old.type, old.tags);
    INSERT INTO memory_fts(rowid, content, type, tags)
    VALUES (new.id, new.content, new.type, new.tags);
END;
"""

# V5.2: Reflections table — migrated from neko ReflectionEngine state machine
REFLECTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    id TEXT PRIMARY KEY,
    character TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    entity TEXT NOT NULL DEFAULT 'master',
    relation_type TEXT NOT NULL DEFAULT 'experience',
    temporal_scope TEXT NOT NULL DEFAULT 'episode',
    status TEXT NOT NULL DEFAULT 'pending',
    importance INTEGER NOT NULL DEFAULT 5,
    source_fact_ids TEXT NOT NULL DEFAULT '[]',
    reinforcement REAL NOT NULL DEFAULT 0.0,
    disputation REAL NOT NULL DEFAULT 0.0,
    event_start_at REAL DEFAULT NULL,
    event_end_at REAL DEFAULT NULL,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    confirmed_at REAL DEFAULT NULL,
    promoted_at REAL DEFAULT NULL,
    merged_into TEXT DEFAULT NULL,
    sub_zero_days INTEGER NOT NULL DEFAULT 0,
    evidence_version INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reflections_char ON reflections(character);
CREATE INDEX IF NOT EXISTS idx_reflections_status ON reflections(status);
CREATE INDEX IF NOT EXISTS idx_reflections_entity ON reflections(entity);
"""

# V5.2: Anti-repeat corpus table — migrated from neko AntiRepeatCorpus
ANTI_REPEAT_SCHEMA = """
CREATE TABLE IF NOT EXISTS anti_repeat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character TEXT NOT NULL DEFAULT '',
    ngram TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'response',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    last_hit_at REAL DEFAULT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_anti_repeat_char ON anti_repeat(character);
CREATE INDEX IF NOT EXISTS idx_anti_repeat_ngram ON anti_repeat(ngram);
"""

# V5.2: User directives table — migrated from neko UserDirectivesManager
USER_DIRECTIVES_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_directives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character TEXT NOT NULL DEFAULT '',
    directive_text TEXT NOT NULL,
    directive_type TEXT NOT NULL DEFAULT 'ban_topic',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    expires_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_directives_char ON user_directives(character);
CREATE INDEX IF NOT EXISTS idx_directives_active ON user_directives(is_active);
"""

# V5.2: Events log table — migrated from neko EventLog (event sourcing)
EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'memory',
    entity_id TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    applied INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_char ON events(character);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_applied ON events(applied);
"""

# Entity graph schema (from Innerlife architecture)
ENTITY_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS eg_entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    embedding_text TEXT NOT NULL DEFAULT '',
    embedding TEXT NOT NULL DEFAULT '[]',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_updated_at INTEGER,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    last_seen_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_eg_entities_type ON eg_entities(type);
CREATE INDEX IF NOT EXISTS idx_eg_entities_name ON eg_entities(canonical_name);

CREATE TABLE IF NOT EXISTS eg_aliases (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_memory_id TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    last_seen_at INTEGER,
    UNIQUE(entity_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_eg_aliases_entity ON eg_aliases(entity_id);

CREATE TABLE IF NOT EXISTS eg_edges (
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    co_occurrence_count INTEGER NOT NULL DEFAULT 0,
    relation_type TEXT NOT NULL DEFAULT 'co_occurrence',
    last_seen_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY(source_entity_id, target_entity_id)
);

CREATE TABLE IF NOT EXISTS eg_episodic (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    detail TEXT DEFAULT '',
    entity_text TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS eg_episodic_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY(memory_id, entity_id)
);

CREATE TABLE IF NOT EXISTS eg_activations (
    episodic_memory_id TEXT NOT NULL,
    activated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    expires_at INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY(episodic_memory_id)
);
"""

# V5.7 (2026-08-14): 类型化项目知识边 (graph-memory 借鉴)
# 连接项目笔记 (v5_project_note) 之间的类型化关系: SOLVES / PREVENTS / CAUSED_BY /
# RELATES_TO, 让 pi 检索时可沿"这个坑怎么解的"扩散。
PROJECT_EDGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relation TEXT NOT NULL DEFAULT 'RELATES_TO',
    weight REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(source_id, target_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_project_edges_source ON project_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_project_edges_target ON project_edges(target_id);
"""


# ─── Connection management (same approach as V3, simplified in V4) ───

# Inline docs: docs/scripts/core/v5/v5/store.md
import threading

_tls = threading.local()


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    """Get a V5 db connection (fresh each time, closed on exit).

    Behavior:
      - Opens a new connection per operation (no thread-local caching, avoids
        implicit read transactions hanging and blocking subsequent writes
        leading to "database is locked")
      - Creates the database on first access
      - Raises on error, does not swallow
      - Auto commits/rollbacks and closes the connection on context exit
    """
    c = getattr(_tls, "c", None)
    if c is not None:
        try:
            c.close()
        except Exception:
            pass
        _tls.c = None

    V5_DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(V5_DB_PATH))
    c.row_factory = sqlite3.Row
    # Multi-process concurrency: watchdog (reflection op) and cloud_chat (store)
    # may access v5.db concurrently. busy_timeout lets the writer wait instead of
    # immediately returning "database is locked"
    try:
        c.execute("PRAGMA busy_timeout=5000")
    except Exception:
        pass
    # WAL mode: write transactions don't block read transactions
    try:
        c.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    # Run all schemas
    c.executescript(SCHEMA)
    c.executescript(REFLECTION_SCHEMA)
    c.executescript(ANTI_REPEAT_SCHEMA)
    c.executescript(USER_DIRECTIVES_SCHEMA)
    c.executescript(EVENTS_SCHEMA)
    c.executescript(ENTITY_GRAPH_SCHEMA)
    c.executescript(PROJECT_EDGES_SCHEMA)
    # V5.7: eg_edges 补 relation_type 列 (幂等; 旧库迁移)
    try:
        c.execute("ALTER TABLE eg_edges ADD COLUMN relation_type TEXT NOT NULL DEFAULT 'co_occurrence'")
    except Exception:
        pass
    # V5: add PAD columns to existing table (idempotent, skip if exists)
    for col in ("pad_p", "pad_a", "pad_d"):
        try:
            c.execute(f"ALTER TABLE memory ADD COLUMN {col} REAL NOT NULL DEFAULT 0.0")
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            pass  # Column already exists
    # V5.2: add character + evidence columns to existing table
    for col in ("character", "reinforcement", "disputation", "evidence_version", "source_memory_id"):
        try:
            if col in ("character",):
                c.execute(f"ALTER TABLE memory ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
            elif col in ("reinforcement", "disputation"):
                c.execute(f"ALTER TABLE memory ADD COLUMN {col} REAL NOT NULL DEFAULT 0.0")
            elif col == "evidence_version":
                c.execute(f"ALTER TABLE memory ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
            elif col == "source_memory_id":
                c.execute(f"ALTER TABLE memory ADD COLUMN {col} INTEGER DEFAULT NULL")
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            pass
    # V5.2: create indexes for new columns (only if columns exist)
    try:
        _info = c.execute("PRAGMA table_info(memory)")
        _existing_cols = {r[1] for r in _info.fetchall()}
        _info.close()
        if "character" in _existing_cols:
            c.execute("CREATE INDEX IF NOT EXISTS idx_memory_character ON memory(character)")
        if "evidence_version" in _existing_cols:
            c.execute("CREATE INDEX IF NOT EXISTS idx_memory_evidence ON memory(evidence_version)")
    except Exception:
        pass
    # V5.3 (2026-08-02): archived 列 — 归档而非删除的转存机制.
    # cleanup 不再物理删除低权重/过期记忆, 而是标记 archived=1 保留在库内
    # (崩溃/误删可恢复; 检索默认排除, 可通过 v5_memory_get 或带参查询取回).
    # 注意: 不能用 `with c.execute(...) as cur` — portable-python 3.13 的
    # sqlite3.Cursor 不支持上下文管理器协议, 会 TypeError 被静默吞掉。
    try:
        _info = c.execute("PRAGMA table_info(memory)")
        _existing_cols = {r[1] for r in _info.fetchall()}
        _info.close()
        if "archived" not in _existing_cols:
            c.execute("ALTER TABLE memory ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
            c.execute("ALTER TABLE memory ADD COLUMN archived_at REAL NOT NULL DEFAULT 0")
            c.execute("CREATE INDEX IF NOT EXISTS idx_memory_archived ON memory(archived)")
            logger.info("V5 store: added archived/archived_at columns (转存机制)")
    except Exception:
        pass
    # R5 (M1): 每进程仅首次连接做一次 WAL checkpoint (收拢帧 + 回收磁盘),
    # 取代原 store() 写路径的每次 wal_checkpoint(TRUNCATE);
    # 后续由 SQLite 默认 wal_autocheckpoint 自动管理。
    global _wal_checkpointed
    if not _wal_checkpointed:
        with _wal_checkpoint_lock:
            if not _wal_checkpointed:
                try:
                    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
                _wal_checkpointed = True
    c.commit()
    logger.info("V5 store: initialized at %s", V5_DB_PATH)
    try:
        yield c
    finally:
        try:
            c.rollback()  # End any incomplete read transactions
        except Exception:
            pass
        try:
            c.close()
        except Exception:
            pass


def close() -> None:
    """Close the current thread's connection (for testing / db switching)."""
    c = getattr(_tls, "c", None)
    if c is not None:
        try:
            c.close()
        except Exception:
            pass
        _tls.c = None


# ─── Core API (V3 compatible) ──────────────────────────────────

@dataclass(frozen=True)
class Memory:
    """V5.2 Memory data class (typed, immutable).

    V3 returned dicts — easy to misspell field names.
    V4 switched to frozen dataclass — IDE hints + immutable.
    V5 adds: pad_p / pad_a / pad_d (emotion fingerprint, default 0.0).
    V5.2 adds: character / reinforcement / disputation / evidence_version / source_memory_id.
    """
    id: int
    content: str
    type: str
    tags: str
    weight: float
    access_count: int
    last_accessed: float
    created: float
    short_term: bool
    long_term: bool
    # V5 emotion fingerprint
    pad_p: float = 0.0
    pad_a: float = 0.0
    pad_d: float = 0.0
    # V5.2 character isolation + evidence scoring
    character: str = ''
    reinforcement: float = 0.0
    disputation: float = 0.0
    evidence_version: int = 0
    source_memory_id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Memory":
        def _r(key: str, default=0.0):
            try:
                return row[key]
            except (IndexError, KeyError):
                return default
        return cls(
            id=_r("id"),
            content=_r("content", ""),
            type=_r("type", "fact"),
            tags=_r("tags", "") or "",
            weight=float(_r("weight", 0.6)),
            access_count=int(_r("access_count", 0)),
            last_accessed=float(_r("last_accessed", 0.0)),
            created=float(_r("created", 0.0)),
            short_term=bool(_r("short_term", 1)),
            long_term=bool(_r("long_term", 0)),
            pad_p=float(_r("pad_p", 0.0)),
            pad_a=float(_r("pad_a", 0.0)),
            pad_d=float(_r("pad_d", 0.0)),
            character=str(_r("character", "")),
            reinforcement=float(_r("reinforcement", 0.0)),
            disputation=float(_r("disputation", 0.0)),
            evidence_version=int(_r("evidence_version", 0)),
            source_memory_id=_r("source_memory_id", None),
        )


def store(content: str, type: str = "fact", weight: float = 0.6,
          tags: str = "", *,  # V5: keyword-only args, don't break V3 callers
          pad_p: float = 0.0, pad_a: float = 0.0, pad_d: float = 0.0,
          character: str = '', reinforcement: float = 0.0, disputation: float = 0.0,
          source_memory_id: int | None = None) -> int:
    """Store a memory, return its id.

    V3 compatible API: same 4 positional params + same int return type.
    V5 additions: pad_p/a/d, keyword-only, default 0.0 (omit to skip emotion).
    V5.2 additions: character, reinforcement, disputation, source_memory_id.

    Concurrency-safe: multiple processes (watchdog + cloud_chat + Hermes Agent)
    may read/write v5.db concurrently. WAL mode with busy_timeout=5000ms.
    If still locked, retry 3 times (1s/3s/5s backoff). Last retry raises.
    """
    import time as _time

    # Validation: catch issues early before retry loop
    from v5.validation import validate_memory, check_and_log
    check_and_log(content, lambda v: validate_memory(
        v, mem_type=type, weight=weight, pad_p=pad_p, pad_a=pad_a, pad_d=pad_d
    ), context="store")

    weight = max(0.0, min(1.0, weight))
    last_err = None
    for attempt in range(4):
        try:
            with conn() as c:
                cur = c.execute(
                    "INSERT INTO memory (content, type, tags, weight, "
                    "pad_p, pad_a, pad_d, character, reinforcement, disputation, "
                    "evidence_version, source_memory_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                    (content, type, tags, weight,
                     pad_p, pad_a, pad_d, character, reinforcement, disputation,
                     source_memory_id),
                )
                c.commit()
                mid = int(cur.lastrowid)
            _sync_vector_best_effort(mid, content, type, tags, weight)
            # V5.2: 事件溯源
            _record_event_best_effort(mid, content, type, character, "memory.created")
            # V5.1: 认知失调检测 — 写入后异步检查矛盾 (仅 fact/preference 类)
            _run_dissonance_detection(content, type)
            return mid
        except sqlite3.OperationalError as e:
            last_err = e
            if "locked" in str(e).lower() and attempt < 3:
                backoff = [1, 3, 5][attempt]
                logger.warning("store: locked, retry %d/3 in %ds", attempt + 1, backoff)
                _time.sleep(backoff)
            else:
                break
    raise RuntimeError(f"store failed after retries: {last_err}") from last_err


# ────────────────────────────────────────────────────────────────────────────
# upsert 写策略 (Phase 1, 2026-08-14)
# 问题: 记忆系统"永远 INSERT"——写入前不查是否已有相似记忆, 同类观察无限堆积
#       (user_trait 579 条雷同的机制性根源), 且"修改记忆"不是一等公民。
# 方案: upsert() 写入口——同类型相似记忆存在则合并强化 (权重取高/内容取长/
#       tags 并集/access+1), 否则新建。情境锚 (context_anchor) 供时间戳与
#       后续召回/检索决策使用。
# ────────────────────────────────────────────────────────────────────────────


def _normalize_probe(content: str) -> str:
    """相似查找探针: 取首标点前的子句 (变体共享的开头).

    观察类记忆 ("哥哥偏好…，…") 的变体共享首子句; 用子句而非固定 14 字符,
    保证探针能作为更短旧内容的子串被 LIKE 命中。
    无标点则回退前 14 字符; 至少 6 字符。
    """
    s = " ".join((content or "").split())
    for sep in "，。；;！？":
        idx = s.find(sep)
        if 6 <= idx <= 24:
            return s[:idx]
    return s[:14]


def _cand_attr(cand, key):
    """候选兼容访问: Memory 行(属性) 或 dict."""
    if isinstance(cand, dict):
        return cand.get(key)
    try:
        return getattr(cand, key)
    except AttributeError:
        return None


def _find_similar(content: str, type: str, threshold: float,
                  top_k: int = 10) -> int | None:
    """找同类型、内容高度相似的既有记忆 (LIKE 子串召回 + difflib 全文本比对).

    用 search_like (LIKE 字节子串) 而非 FTS: FTS5 对连续中文整串探针召回差
    (实测 '主力' 0 命中), LIKE 对中文 100% 命中。返回命中 memory id; 无则 None。
    """
    import difflib
    probe = _normalize_probe(content)
    if not probe:
        return None
    try:
        cands = search_like(probe, top_k=top_k, min_weight=0.0)
    except Exception as exc:  # 不可用/异常时 fail-open (不阻塞写入)
        logger.debug("upsert: similar-search failed (%s)", exc)
        return None
    norm_new = " ".join((content or "").split())
    best_id, best_ratio, best_old = None, 0.0, ""
    for cand in cands or []:
        if (_cand_attr(cand, "type") or "fact") != type:
            continue
        old = " ".join((_cand_attr(cand, "content") or "").split())
        ratio = difflib.SequenceMatcher(None, old, norm_new).ratio()
        if ratio > best_ratio:
            best_ratio, best_id, best_old = ratio, _cand_attr(cand, "id"), old
    if best_id is None:
        return None
    if best_ratio >= threshold:
        return best_id
    # 子串包含判定: 一者是另一者的子串且核心 >= 8 字符 (如旧 "哥哥偏好简短
    # 直接的沟通" ⊂ 新 "…，说人话比修辞更有效") —— 明显同主题但 ratio 被
    # 长度差拉低 (0.69), 应合并。
    if len(best_old) >= 8 and best_old in norm_new:
        return best_id
    if len(norm_new) >= 8 and norm_new in best_old:
        return best_id
    return None


def _merge_into(memory_id: int, content: str, type: str, weight: float,
                tags: str, reinforcement: float = 0.0) -> int:
    """合并强化既有记忆: 内容取更长者, 权重取高者, tags 并集, access+1, last_accessed=now."""
    import time as _time
    with conn() as c:
        row = c.execute(
            "SELECT content, weight, tags FROM memory WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:  # 目标被并发删除: 降级为新建
            return store(content, type=type, weight=weight, tags=tags,
                         reinforcement=reinforcement)
        old_content, old_weight, old_tags = row
        merged_content = old_content if len(old_content or "") >= len(content) else content
        merged_weight = max(float(old_weight or 0.0), float(weight))
        merged_tags = " ".join(dict.fromkeys(f"{old_tags or ''} {tags}".split()))
        # Phase 4 (2026-08-14): 合并即强化 —— 每次合并加固定 reinforcement 增量
        # (被合并越多 → reinforcement 越高 → 检索加分越大; 增量 config 可调)
        try:
            from v5 import preprocess_config as _pc
            _merge_inc = float(_pc.cfg()["memory_retrieval"].get(
                "merge_reinforce_increment", 0.05))
        except Exception:
            _merge_inc = 0.05
        c.execute(
            "UPDATE memory SET content = ?, weight = ?, tags = ?, "
            "  access_count = access_count + 1, last_accessed = ?, "
            "  reinforcement = MIN(1.0, reinforcement + ?) WHERE id = ?",
            (merged_content, merged_weight, merged_tags, _time.time(),
             max(0.0, reinforcement) + _merge_inc, memory_id),
        )
        c.commit()
    _sync_vector_best_effort(memory_id, merged_content, type, merged_tags, merged_weight)
    _record_event_best_effort(memory_id, content[:100], type, "", "memory.updated")
    return memory_id


def upsert(content: str, type: str = "fact", weight: float = 0.6,
           tags: str = "", *, similarity_threshold: float = 0.75,
           reinforcement: float = 0.0, **kw) -> int:
    """写记忆 (推荐写入口): 同类型相似记忆存在则合并强化, 否则新建.

    Phase 1 (2026-08-14):
      - 相似判定: LIKE 子句探针召回同类型候选 + difflib 全文本 ratio / 子串包含
      - 命中 → _merge_into (修改记忆); 未命中 → store() (新建)
      - **带 v5_key: 标签的写入跳过合并** (结构化记录以 key 为显式身份,
        如项目笔记的 kind/domain, 内容相似也不应并表)
    """
    if "v5_key:" in tags:
        return store(content, type=type, weight=weight, tags=tags,
                     reinforcement=reinforcement, **kw)
    existing = _find_similar(content, type, similarity_threshold)
    if existing is not None:
        return _merge_into(existing, content, type, weight, tags, reinforcement)
    return store(content, type=type, weight=weight, tags=tags,
                 reinforcement=reinforcement, **kw)


def _sync_vector_best_effort(memory_id: int, content: str, type: str,
                             tags: str, weight: float) -> bool:
    """Best-effort sync this memory's vector into Chroma after DB write.

    - Only runs when chromadb is available (silently skips otherwise)
    - :8587 unavailable or embedding failed -> returns False, picked up later by
      vector_sync reflection op
    - Writes go through get_vector_index() (process-level singleton), the SAME
      instance retrieval uses -> same-process adds are immediately visible to
      semantic search. A fresh VectorIndex() here would hold a stale snapshot,
      hiding new memories from retrieval for up to vector_refresh_seconds.
    - Thread timeout guard: get_vector_index() init (ChromaLM compactor) may
      hang in unknown C extensions; max block SYNC_TIMEOUT seconds, then
      silently skip to never block store.store()
    """
    _SYNC_TIMEOUT = 10.0  # max seconds to wait for ChromaDB init + add

    # ruff: noqa: BLE001
    _result: list[bool] = []
    _exc: list[Exception] = []

    def _do_sync() -> None:
        try:
            from v5.search import get_vector_index
        except Exception as e:
            logger.warning("vector sync skipped (import): %s", e)
            _result.append(False)
            return
        try:
            # get_vector_index() 返回进程级单例 (检索侧同实例); 创建失败时抛异常,
            # 由下方 except 兜底 -> 静默降级, 与旧 VectorIndex() 行为一致
            idx = get_vector_index()
            ok = idx.add(memory_id, content, type=type, tags=tags, weight=weight)
            if not ok:
                logger.warning("vector sync returned False for id=%s (embedding 不可用, 待 vector_sync op 补录)", memory_id)
            _result.append(ok)
        except Exception as e:
            logger.warning("vector sync failed for id=%s: %s", memory_id, e)
            _result.append(False)

    t = threading.Thread(target=_do_sync, daemon=True)
    t.start()
    t.join(timeout=_SYNC_TIMEOUT)
    if t.is_alive():
        logger.warning("vector sync TIMEOUT for id=%s (>=%ss), skipped", memory_id, _SYNC_TIMEOUT)
        return False
    return _result[0] if _result else False


_DISSONANCE_TYPES = {"fact", "preference"}

def _run_dissonance_detection(content: str, mem_type: str) -> None:
    """写入后异步认知失调检测 (不阻塞 store, 失败静默)."""
    if mem_type not in _DISSONANCE_TYPES or not content or len(content) < 10:
        return
    def _check():
        try:
            from v5.dissonance import detect_dissonance
            result = detect_dissonance(content, mem_type)
            if result.get("conflicts"):
                logger.info("store: dissonance detected (%d conflicts)", len(result["conflicts"]))
        except Exception:
            pass  # 失调检测从不阻塞 stores
    threading.Thread(target=_check, daemon=True).start()


_EVENT_TYPES = {"memory.created", "memory.deleted", "memory.evidence_updated",
                "reflection.synthesized", "reflection.status_changed",
                "reflection.promoted", "user_directive.created"}

def _record_event_best_effort(entity_id: int | str, content: str, entity_type: str,
                               character: str, event_type: str) -> None:
    """异步记录事件日志 (不阻塞 store)."""
    if event_type not in _EVENT_TYPES:
        return
    def _write():
        try:
            with conn() as c:
                c.execute(
                    "INSERT INTO events (character, event_type, entity_type, entity_id, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (character, event_type, entity_type, str(entity_id),
                     json.dumps({"content_preview": content[:100]})),
                )
                c.commit()
        except Exception as e:
            logger.warning("event log write failed (entity=%s %s): %s",
                           entity_type, entity_id, e)
    threading.Thread(target=_write, daemon=True).start()


def get(memory_id: int) -> Memory | None:
    """Fetch a single memory by id. Returns None if not found."""
    with conn() as c:
        row = c.execute("SELECT * FROM memory WHERE id = ?", (memory_id,)).fetchone()
        return Memory.from_row(row) if row else None


def get_batch(memory_ids: list[int]) -> dict[int, Memory]:
    """Fetch multiple memories by id. Returns dict {id: Memory}.

    Used by conversation_tree.get_context() for efficient batch context loading.
    Missing ids are silently omitted from the result.
    """
    if not memory_ids:
        return {}
    with conn() as c:
        placeholders = ",".join("?" * len(memory_ids))
        rows = c.execute(
            f"SELECT * FROM memory WHERE id IN ({placeholders})",
            memory_ids,
        ).fetchall()
    return {row[0]: Memory.from_row(row) for row in rows}


def _sanitize_fts5_query(query: str) -> str:
    """Sanitize a query string for FTS5 MATCH.

    FTS5 treats . : * " ( ) - AND OR NOT as special syntax.
    Wrap each whitespace-separated token in double quotes to make it a
    literal phrase, preventing syntax errors on real-world input
    (file paths, version numbers, etc.).

    V5.6 (2026-08-10): AND → OR for multi-token queries.
      FTS5 default joins quoted phrases with AND; a long natural-language
      query (10+ tokens) then requires EVERY token in one document — near
      certain zero hits. LongMemEval temporal-reasoning 实测: 长句 AND=0
      命中, OR=232 命中 (bm25 排序保证多词命中的文档仍排最前, 召回优先
      且精度不塌)。
    """
    import re as _re
    # Split on whitespace, filter empties
    tokens = [t for t in _re.split(r"\s+", query.strip()) if t]
    if not tokens:
        return ""
    # Wrap each token in double-quotes (FTS5 phrase syntax)
    # Escape internal double-quotes by doubling them (FTS5 escape rule)
    quoted = []
    for t in tokens:
        escaped = t.replace('"', '""')
        quoted.append(f'"{escaped}"')
    # 单 token 无连接符; 多 token 用 OR (召回优先, bm25 排序兜底精度)
    if len(quoted) == 1:
        return quoted[0]
    return " OR ".join(quoted)


def search(query: str, top_k: int = 5, min_weight: float = 0.0,
           character: str = '') -> list[Memory]:
    """FTS5 keyword search (V5.2, character-aware).

    Sanitizes the query to prevent FTS5 syntax errors on real-world input
    (dots, colons, hyphens, etc. in file paths and version numbers).
    """
    fts_query = _sanitize_fts5_query(query)
    if not fts_query:
        return []
    with conn() as c:
        if character:
            rows = c.execute(
                "SELECT m.* FROM memory m "
                "JOIN memory_fts f ON m.id = f.rowid "
                "WHERE memory_fts MATCH ? "
                "  AND m.weight >= ? AND m.character = ? AND m.archived = 0 "
                "ORDER BY bm25(memory_fts) "
                "LIMIT ?",
                (fts_query, min_weight, character, top_k),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT m.* FROM memory m "
                "JOIN memory_fts f ON m.id = f.rowid "
                "WHERE memory_fts MATCH ? "
                "  AND m.weight >= ? AND m.archived = 0 "
                "ORDER BY bm25(memory_fts) "
                "LIMIT ?",
                (fts_query, min_weight, top_k),
            ).fetchall()
    return [Memory.from_row(r) for r in rows]


def search_like(substr: str, top_k: int = 5, min_weight: float = 0.0,
                character: str = '') -> list[Memory]:
    """LIKE 子串查询 (中文 2-gram 拆词后的兜底检索).

    FTS5 unicode61 把连续中文串当单个 token, 拆词后的 2-gram MATCH 基本无效
    (实测 '主力'/'选型' 0 命中). SQLite LIKE 是字节子串匹配, 不依赖 tokenizer,
    对中文 2-gram 100% 命中。仅用于 keyword fallback 弱信号补足, 不替代
    FTS5 主检索。通配符 %/_ 转义防注入/误匹配。
    """
    if not substr or not substr.strip():
        return []
    escaped = substr.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    with conn() as c:
        if character:
            rows = c.execute(
                "SELECT * FROM memory "
                "WHERE content LIKE ? ESCAPE '\\' "
                "  AND weight >= ? AND character = ? AND archived = 0 "
                "ORDER BY weight DESC, id DESC LIMIT ?",
                (pattern, min_weight, character, top_k),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM memory "
                "WHERE content LIKE ? ESCAPE '\\' "
                "  AND weight >= ? AND archived = 0 "
                "ORDER BY weight DESC, id DESC LIMIT ?",
                (pattern, min_weight, top_k),
            ).fetchall()
    return [Memory.from_row(r) for r in rows]


def valid_to_map(ids: list[str], table: str, col: str = "id") -> dict:
    """批量取 valid_to (时效图谱过期过滤用)。

    返回 {str(id): valid_to}。键统一 str, 与检索结果 dict 的 id 字段对齐
    (SQLite 行返回 int id, 不转换会导致 get() 失配、过期过滤静默失效)。

    原位于 extensions.temporal_graph._valid_to_map (2026-08-14 迁移至此,
    供 memory_retrieval 与 temporal_graph 共用, 解开循环依赖)。table 为
    'memory' 或 eg_* 表名 (非 memory 走 entity_graph.eg_conn)。
    """
    if not ids:
        return {}
    # 表名/列名仅内部常量, 但做最小白名单防御 (防未来误传用户输入注入)
    if not table.replace("_", "").isalnum() or not col.replace("_", "").isalnum():
        return {}
    ph = ",".join("?" * len(ids))
    if table == "memory":
        with conn() as c:
            rows = c.execute(
                f"SELECT {col} AS id, valid_to FROM {table} WHERE {col} IN ({ph})",
                ids,
            ).fetchall()
    else:
        from v5.entity_graph import eg_conn
        with eg_conn() as c:
            rows = c.execute(
                f"SELECT {col} AS id, valid_to FROM {table} WHERE {col} IN ({ph})",
                ids,
            ).fetchall()
    return {str(r["id"]): r["valid_to"] for r in rows}


def link_project_edge(source_id, target_id, relation: str = "RELATES_TO",
                      weight: float = 0.5) -> bool:
    """建一条类型化项目边 (幂等: 同 source/target/relation 覆盖 weight)。

    V5.7 (2026-08-14): 连接 v5_project_note 之间的类型化关系
    (SOLVES / PREVENTS / CAUSED_BY / RELATES_TO)。
    """
    if not source_id or not target_id or int(source_id) == int(target_id):
        return False
    relation = (relation or "RELATES_TO").upper()
    try:
        with conn() as c:
            c.execute(
                "INSERT INTO project_edges (source_id, target_id, relation, weight) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source_id, target_id, relation) "
                "DO UPDATE SET weight = excluded.weight",
                (int(source_id), int(target_id), relation, float(weight)),
            )
            c.commit()
        return True
    except Exception as exc:
        logger.warning("link_project_edge failed: %s", exc)
        return False


def get_project_edges(memory_id) -> list[dict]:
    """返回该记忆参与的所有项目边 (作为 source 或 target)。"""
    try:
        with conn() as c:
            rows = c.execute(
                "SELECT source_id, target_id, relation, weight FROM project_edges "
                "WHERE source_id = ? OR target_id = ?",
                (int(memory_id), int(memory_id)),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug("get_project_edges failed: %s", exc)
        return []


def list_all(limit: int = 50, type_filter: str | None = None,
             character: str = '') -> list[Memory]:
    """List memories (for debugging, character-aware)."""
    with conn() as c:
        if type_filter and character:
            rows = c.execute(
                "SELECT * FROM memory WHERE type = ? AND character = ? ORDER BY id DESC LIMIT ?",
                (type_filter, character, limit),
            ).fetchall()
        elif type_filter:
            rows = c.execute(
                "SELECT * FROM memory WHERE type = ? ORDER BY id DESC LIMIT ?",
                (type_filter, limit),
            ).fetchall()
        elif character:
            rows = c.execute(
                "SELECT * FROM memory WHERE character = ? ORDER BY id DESC LIMIT ?",
                (character, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM memory ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [Memory.from_row(r) for r in rows]


def count_like(substr: str, min_weight: float = 0.0,
               character: str = '') -> int:
    """LIKE 子串命中计数 (keyword fallback 的 token 稀有度排序用)."""
    if not substr or not substr.strip():
        return 0
    escaped = substr.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    with conn() as c:
        if character:
            row = c.execute(
                "SELECT COUNT(*) FROM memory "
                "WHERE content LIKE ? ESCAPE '\\' "
                "  AND weight >= ? AND character = ? AND archived = 0",
                (pattern, min_weight, character),
            ).fetchone()
        else:
            row = c.execute(
                "SELECT COUNT(*) FROM memory "
                "WHERE content LIKE ? ESCAPE '\\' "
                "  AND weight >= ? AND archived = 0",
                (pattern, min_weight),
            ).fetchone()
    return int(row[0] if row else 0)


def search_by_time_range(start_ts: float, end_ts: float,
                         limit: int = 10) -> list[Memory]:
    """Search memories by time range (supports cloud_chat time reference resolution).

    The 'created' column stores Unix epoch (strftime('%s','now')).
    start_ts / end_ts are also Unix epoch floats.
    """
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM memory "
            "WHERE created >= ? AND created <= ? "
            "ORDER BY weight DESC, last_accessed DESC "
            "LIMIT ?",
            (start_ts, end_ts, limit),
        ).fetchall()
    return [Memory.from_row(r) for r in rows]


def delete(memory_id: int) -> bool:
    """Delete one memory. Returns True/False."""
    with conn() as c:
        cur = c.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
        c.commit()
        return cur.rowcount > 0


def access(memory_id: int) -> None:
    """Record access + weight +0.05 (same as V3)."""
    with conn() as c:
        c.execute(
            "UPDATE memory SET "
            "  access_count = access_count + 1, "
            "  last_accessed = strftime('%s','now'), "
            "  weight = MIN(1.0, weight + 0.05) "
            "WHERE id = ?",
            (memory_id,),
        )
        c.commit()


def stats() -> dict:
    """v3.stats() compatible API, expanded for V5.2."""
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        by_type = c.execute(
            "SELECT type, COUNT(*), ROUND(AVG(weight), 3) FROM memory GROUP BY type"
        ).fetchall()
        long_term = c.execute(
            "SELECT COUNT(*) FROM memory WHERE long_term = 1"
        ).fetchone()[0]
        avg_weight = float(c.execute("SELECT AVG(weight) FROM memory").fetchone()[0] or 0)
        character_count = c.execute(
            "SELECT character, COUNT(*) FROM memory WHERE character != '' GROUP BY character"
        ).fetchall()
        reflection_count = c.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
        event_count = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        directive_count = c.execute(
            "SELECT COUNT(*) FROM user_directives WHERE is_active = 1"
        ).fetchone()[0]
    return {
        "total": total,
        "long_term": long_term,
        "avg_weight": avg_weight,
        "by_type": {r[0]: {"count": r[1], "avg_weight": r[2]} for r in by_type},
        "by_character": {r[0]: r[1] for r in character_count},
        "reflections": reflection_count,
        "events": event_count,
        "active_directives": directive_count,
        "db_size_bytes": V5_DB_PATH.stat().st_size if V5_DB_PATH.exists() else 0,
        "db_path": str(V5_DB_PATH),
    }


# ─── Evidence scoring (from neko evidence.py) ─────────────
# Half-life constants matching neko's evidence system
EVIDENCE_REIN_HALF_LIFE_DAYS = 14.0    # 强化半衰期 (天)
EVIDENCE_DISP_HALF_LIFE_DAYS = 7.0     # 反驳半衰期 (天)
EVIDENCE_CONFIRMED_THRESHOLD = 0.6     # 确认阈值
EVIDENCE_PROMOTED_THRESHOLD = 1.0      # 晋升阈值

def _halflife_decay(value: float, elapsed_days: float, half_life_days: float) -> float:
    """指数半衰期衰减."""
    if elapsed_days <= 0 or half_life_days <= 0:
        return value
    return value * (0.5 ** (elapsed_days / half_life_days))

def effective_reinforcement(memory_row: dict, now: float | None = None) -> float:
    """带半衰期衰减的有效强化值."""
    if now is None:
        now = time.time()
    created = float(memory_row.get("created", now))
    elapsed_days = (now - created) / 86400.0
    rein = float(memory_row.get("reinforcement", 0.0))
    return _halflife_decay(rein, elapsed_days, EVIDENCE_REIN_HALF_LIFE_DAYS)

def effective_disputation(memory_row: dict, now: float | None = None) -> float:
    """带半衰期衰减的有效反驳值."""
    if now is None:
        now = time.time()
    created = float(memory_row.get("created", now))
    elapsed_days = (now - created) / 86400.0
    disp = float(memory_row.get("disputation", 0.0))
    return _halflife_decay(disp, elapsed_days, EVIDENCE_DISP_HALF_LIFE_DAYS)

def evidence_score(memory_row: dict, now: float | None = None) -> float:
    """证据分数 = effective_reinforcement - effective_disputation."""
    return effective_reinforcement(memory_row, now) - effective_disputation(memory_row, now)

def update_evidence(memory_id: int, delta_rein: float = 0.0, delta_disp: float = 0.0) -> bool:
    """更新记忆的证据分数."""
    try:
        with conn() as c:
            c.execute(
                "UPDATE memory SET reinforcement = reinforcement + ?, "
                "disputation = disputation + ?, "
                "evidence_version = evidence_version + 1 "
                "WHERE id = ?",
                (delta_rein, delta_disp, memory_id),
            )
            c.commit()
            # 记录事件
            _record_event_best_effort(memory_id, "", "memory", "", "memory.evidence_updated")
        return True
    except Exception as e:
        logger.warning("update_evidence failed for id=%s: %s", memory_id, e)
        return False


# ─── CLI (via ikaros-mem.bat v5) ──────────────────────────────

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Ikaros Memory V4")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats").set_defaults(fn=lambda a: print(json.dumps(stats(), indent=2, ensure_ascii=False)))

    p_store = sub.add_parser("store")
    p_store.add_argument("content")
    p_store.add_argument("--type", default="fact")
    p_store.add_argument("--weight", type=float, default=0.6)
    p_store.add_argument("--tags", default="")
    p_store.set_defaults(fn=lambda a: print(store(a.content, a.type, a.weight, a.tags)))

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--min-weight", type=float, default=0.0)
    p_search.set_defaults(fn=lambda a: print(json.dumps(
        [m.__dict__ for m in search(a.query, a.top_k, a.min_weight)],
        indent=2, ensure_ascii=False,
    )))

    p_get = sub.add_parser("get")
    p_get.add_argument("memory_id", type=int)
    p_get.set_defaults(fn=lambda a: print(json.dumps(
        vars(get(a.memory_id)) if get(a.memory_id) else None,
        indent=2, ensure_ascii=False,
    )))

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    main()
