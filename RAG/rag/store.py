# ============================================================
# store.py — FAISS vector index + SQLite metadata store.
#
# Responsibilities:
#   • init_db()          — create table if missing
#   • save_to_index()    — append new embeddings + metadata
#   • load_seen_urls()   — return URLs already indexed
#   • get_next_chunk_id()— for monotonic IDs across runs
# ============================================================

import os
import sqlite3
import numpy as np
import faiss

from rag.config import INDEX_PATH, DB_PATH, INDEX_DIR

# ── DB init ──────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """Create the chunks table if it doesn't already exist."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    INTEGER PRIMARY KEY,
            text        TEXT,
            title       TEXT,
            url         TEXT,
            date        TEXT,
            source      TEXT,
            chunk_index INTEGER,
            token_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ DB ready: {db_path}")


# ── Helpers ───────────────────────────────────────────────────

def load_seen_urls(db_path: str = DB_PATH) -> set:
    """Return every URL already stored — used to skip re-indexing."""
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM chunks")
        seen = {row[0] for row in cursor.fetchall()}
        conn.close()
        return seen
    except Exception:
        return set()


def get_next_chunk_id(db_path: str = DB_PATH) -> int:
    """Return MAX(chunk_id)+1, or 0 if table is empty."""
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(chunk_id) FROM chunks")
        result = cursor.fetchone()[0]
        conn.close()
        return (result + 1) if result is not None else 0
    except Exception:
        return 0


# ── Persistence ───────────────────────────────────────────────

def save_to_index(chunks:      list[dict],
                  embeddings:  np.ndarray,
                  index_path:  str = INDEX_PATH,
                  db_path:     str = DB_PATH) -> None:
    """
    Append *embeddings* to the FAISS flat index and write
    *chunks* metadata to SQLite.

    FAISS index is created (IndexFlatIP) on the first call
    and appended to on subsequent calls — safe for incremental
    daily updates.

    Caller is responsible for L2-normalising embeddings before
    calling this function (so dot-product == cosine similarity).
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    # ── FAISS ────────────────────────────────────────────────
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        # Inner-product on unit vectors == cosine similarity
        index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)
    faiss.write_index(index, index_path)

    # ── SQLite ────────────────────────────────────────────────
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany(
        'INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)',
        [(c['chunk_id'], c['text'], c['title'], c['url'],
          c['date'], c['source'], c['chunk_index'], c['token_count'])
         for c in chunks]
    )
    conn.commit()
    conn.close()

    print(f"✅ FAISS: {index.ntotal} total vectors | "
          f"SQLite: +{len(chunks)} chunks")
