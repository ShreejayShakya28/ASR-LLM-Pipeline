# ============================================================
# retriever.py — Two-stage retrieval pipeline.
#
# Stage 1: FAISS cosine search
#           → blended score (cosine × sem_weight + freshness × fresh_weight)
#           → hard filter on min_cosine and days_filter
#           → deduplicate by URL
#
# Stage 2: CrossEncoder reranking over the candidate pool
#
# Staleness is handled by time_decay_score() which exponentially
# down-weights older articles so recent news always ranks higher
# when semantic similarity is equal.
# ============================================================

import math
import sqlite3
import numpy as np
import faiss
from datetime import datetime

from rag.config import (
    INDEX_PATH, DB_PATH,
    DEFAULT_TOP_K, DEFAULT_DAYS_FILTER,
    MIN_COSINE, SEM_WEIGHT, FRESH_WEIGHT, DECAY_RATE,
)
from rag.models import embedding_model, reranker


# ── Freshness scoring ─────────────────────────────────────────

def time_decay_score(date_str: str,
                     decay_rate: float = DECAY_RATE) -> float:
    """
    Returns 1.0 for today's articles, exponentially lower for older ones.
    Formula: e^(-decay_rate * days_old)

    With default decay_rate=0.1:
      - 0 days old  → 1.00
      - 7 days old  → 0.50
      - 14 days old → 0.25
      - 30 days old → 0.05   (near-zero, filtered out by days_filter anyway)
    """
    try:
        days_old = (datetime.now()
                    - datetime.strptime(date_str, '%Y-%m-%d')).days
        return math.exp(-decay_rate * days_old)
    except Exception:
        return 0.5


# ── Main retrieval function ───────────────────────────────────

def retrieve(query:        str,
             top_k:        int   = DEFAULT_TOP_K,
             days_filter:  int   = DEFAULT_DAYS_FILTER,
             min_cosine:   float = MIN_COSINE,
             sem_weight:   float = SEM_WEIGHT,
             fresh_weight: float = FRESH_WEIGHT,
             index_path:   str   = INDEX_PATH,
             db_path:      str   = DB_PATH) -> list[dict]:
    """
    Retrieve the *top_k* most relevant chunks for *query*.

    Returns a list of dicts (sorted by rerank_score desc) with keys:
        text, title, url, date, source,
        cosine_score, freshness_score, final_score, rerank_score
    """
    index  = faiss.read_index(index_path)
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ── Embed + normalise query ───────────────────────────────
    q_vec = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)

    # ── Stage 1: FAISS search (fetch 8× more than needed) ────
    n_probe    = min(top_k * 8, index.ntotal)
    scores, indices = index.search(q_vec, n_probe)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        cosine = float(score)
        if cosine < min_cosine:
            continue            # below relevance threshold — skip

        cursor.execute('SELECT * FROM chunks WHERE chunk_id=?', (int(idx),))
        row = cursor.fetchone()
        if not row:
            continue

        chunk_id, text, title, url, date, source, chunk_index, token_count = row

        # Hard date filter
        try:
            age = (datetime.now()
                   - datetime.strptime(date, '%Y-%m-%d')).days
            if age > days_filter:
                continue
        except Exception:
            pass

        freshness   = time_decay_score(date)
        final_score = sem_weight * cosine + fresh_weight * freshness

        candidates.append({
            'cosine_score'   : round(cosine, 4),
            'freshness_score': round(freshness, 4),
            'final_score'    : round(final_score, 4),
            'text'   : text,
            'title'  : title,
            'url'    : url,
            'date'   : date,
            'source' : source,
        })

    conn.close()

    if not candidates:
        return []

    # Deduplicate by URL, keep highest-scoring chunk per article
    seen: dict = {}
    for c in sorted(candidates, key=lambda x: x['final_score'], reverse=True):
        if c['url'] not in seen:
            seen[c['url']] = c
    deduped = list(seen.values())

    # ── Stage 2: CrossEncoder reranking ──────────────────────
    pairs  = [[query, c['text']] for c in deduped]
    scores = reranker.predict(pairs)
    for i, c in enumerate(deduped):
        c['rerank_score'] = round(float(scores[i]), 4)

    return sorted(deduped,
                  key=lambda x: x['rerank_score'],
                  reverse=True)[:top_k]
