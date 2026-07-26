"""V5.2: Anti-repetition module — migrated from neko AntiRepeatCorpus.

Detects and prevents topic/ngram repetition in AI responses using BM25-style
scoring. Built on V5 store database (anti_repeat table) — no separate files.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from typing import Optional

logger = logging.getLogger("ikaros.v5.anti_repeat")

# ─── Configuration ──────────────────────────────────────────
ANTI_REPEAT_WINDOW_SIZE = 5        # ngram size
ANTI_REPEAT_MAX_CORPUS = 200       # max ngrams per character
ANTI_REPEAT_BM25_THRESHOLD = 0.5   # score threshold to flag as repeat
ANTI_REPEAT_TOP_K_PENALTY = 3      # inject top-K high-scoring ngrams as penalty


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: split into CJK chars and Latin words."""
    tokens = []
    import re
    # CJK characters
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f':
            tokens.append(ch)
    # Latin words
    for word in re.findall(r'[a-zA-Z0-9_]+', text):
        if len(word) > 1:
            tokens.append(word.lower())
    return tokens


def _extract_ngrams(tokens: list[str], n: int = ANTI_REPEAT_WINDOW_SIZE) -> list[str]:
    """Extract n-grams from token list."""
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def record_response(character: str, response_text: str) -> int:
    """Record a response's ngrams into the anti-repeat corpus.

    Args:
        character: Character/role name
        response_text: The AI response text to analyze

    Returns:
        Number of new ngrams recorded
    """
    from v5.store import conn
    tokens = _tokenize(response_text)
    ngrams = _extract_ngrams(tokens)
    if not ngrams:
        return 0

    count = Counter(ngrams)
    recorded = 0
    try:
        with conn() as c:
            # Prune oldest if over limit
            c.execute(
                "DELETE FROM anti_repeat WHERE character = ? "
                "AND id NOT IN (SELECT id FROM anti_repeat WHERE character = ? "
                "  ORDER BY created_at DESC LIMIT ?)",
                (character, character, ANTI_REPEAT_MAX_CORPUS),
            )
            # Upsert each ngram
            for ngram, freq in count.items():
                existing = c.execute(
                    "SELECT id, hit_count FROM anti_repeat "
                    "WHERE character = ? AND ngram = ?",
                    (character, ngram),
                ).fetchone()
                if existing:
                    c.execute(
                        "UPDATE anti_repeat SET weight = weight + ?, "
                        "last_hit_at = strftime('%s','now'), "
                        "hit_count = hit_count + 1 "
                        "WHERE id = ?",
                        (min(freq * 0.1, 1.0), existing[0]),
                    )
                else:
                    c.execute(
                        "INSERT INTO anti_repeat (character, ngram, weight, source_type) "
                        "VALUES (?, ?, ?, 'response')",
                        (character, ngram, min(freq * 0.1, 1.0)),
                    )
                    recorded += 1
            c.commit()
    except Exception as e:
        logger.warning("anti_repeat.record_response failed: %s", e)
    return recorded


def check_repetition(character: str, candidate_text: str) -> dict:
    """Check if candidate_text has high repetition risk.

    Args:
        character: Character/role name
        candidate_text: Candidate response to check

    Returns:
        dict with:
            - score: BM25-style repetition score (0.0 = clean, >threshold = risky)
            - is_repetitive: bool, true if score > threshold
            - top_ngrams: list of top overlapping ngrams
    """
    from v5.store import conn
    tokens = _tokenize(candidate_text)
    ngrams = _extract_ngrams(tokens)
    if not ngrams:
        return {"score": 0.0, "is_repetitive": False, "top_ngrams": []}

    try:
        with conn() as c:
            corpus_rows = c.execute(
                "SELECT ngram, weight, hit_count FROM anti_repeat "
                "WHERE character = ? ORDER BY weight DESC LIMIT 500",
                (character,),
            ).fetchall()
    except Exception as e:
        logger.warning("anti_repeat.check_repetition query failed: %s", e)
        return {"score": 0.0, "is_repetitive": False, "top_ngrams": []}

    if not corpus_rows:
        return {"score": 0.0, "is_repetitive": False, "top_ngrams": []}

    # BM25-style scoring
    corpus = {r[0]: {"weight": r[1], "hit_count": r[2]} for r in corpus_rows}
    total_docs = len(corpus)
    avg_doc_len = sum(v["hit_count"] for v in corpus.values()) / max(total_docs, 1)

    ngram_scores = []
    total_score = 0.0
    for ng in ngrams:
        if ng in corpus:
            w = corpus[ng]["weight"]
            hc = corpus[ng]["hit_count"]
            idf = math.log((total_docs - hc + 0.5) / (hc + 0.5) + 1.0)
            tf = 1.0  # each candidate ngram appears once
            bm25 = idf * (tf * (1.5)) / (tf + 1.5 * (1 - 0.75 + 0.75 * 1.0 / max(avg_doc_len, 1)))
            score = bm25 * w
            ngram_scores.append((ng, score))
            total_score += score

    avg_score = total_score / max(len(ngrams), 1)
    top_ngrams = sorted(ngram_scores, key=lambda x: -x[1])[:ANTI_REPEAT_TOP_K_PENALTY]

    return {
        "score": round(avg_score, 4),
        "is_repetitive": avg_score > ANTI_REPEAT_BM25_THRESHOLD,
        "top_ngrams": [{"ngram": n, "score": round(s, 4)} for n, s in top_ngrams],
    }


def get_penalty_hint(character: str, candidate_text: str) -> str:
    """Get a prompt hint text if repetition risk is high.

    Returns empty string if clean, otherwise a directive for the LLM.
    """
    result = check_repetition(character, candidate_text)
    if not result["is_repetitive"]:
        return ""
    topics = ", ".join(n["ngram"][:30] for n in result["top_ngrams"])
    return (f"[SYSTEM: 检测到话题重复倾向 (score={result['score']:.2f})。"
            f"请转向新话题或提供新视角。最近已讨论过: {topics}]")


def clear(character: str = '') -> int:
    """Clear anti-repeat corpus for a character (or all if character='')."""
    from v5.store import conn
    try:
        with conn() as c:
            if character:
                cur = c.execute("DELETE FROM anti_repeat WHERE character = ?", (character,))
            else:
                cur = c.execute("DELETE FROM anti_repeat")
            c.commit()
            return cur.rowcount
    except Exception as e:
        logger.warning("anti_repeat.clear failed: %s", e)
        return 0


def stats(character: str = '') -> dict:
    """Anti-repeat corpus statistics."""
    from v5.store import conn
    try:
        with conn() as c:
            if character:
                total = c.execute(
                    "SELECT COUNT(*) FROM anti_repeat WHERE character = ?",
                    (character,),
                ).fetchone()[0]
            else:
                total = c.execute("SELECT COUNT(*) FROM anti_repeat").fetchone()[0]
            return {"total_ngrams": total, "character": character or "all"}
    except Exception as e:
        return {"error": str(e), "character": character or "all"}
