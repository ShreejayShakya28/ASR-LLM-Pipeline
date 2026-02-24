# ============================================================
# store.py
# ============================================================

import os
import sqlite3
import numpy as np
import faiss

from rag.config import (
    INDEX_PATH, DB_PATH, INDEX_DIR,
    FAISS_INDEX_TYPE, FAISS_NLIST, FAISS_NPROBE
)

EMBEDDING_DIM = 384   # all-MiniLM-L6-v2


# ── Guard ─────────────────────────────────────────────────────

def _check_drive():
    # Only raise if path actually points at Drive — allows local runs
    if DB_PATH.startswith('/drive/') and not os.path.exists('/drive/MyDrive'):
        raise RuntimeError(
            "\n❌ Google Drive is not mounted.\n"
            "   Run:  from google.colab import drive; drive.mount('/drive')\n"
            "   Then re-import."
        )

_check_drive()


# ── FAISS index factory ───────────────────────────────────────

def _make_index(dim: int) -> faiss.Index:
    """
    Create the right index type based on config.
    IVFFlat needs training data — caller handles that.
    """
    if FAISS_INDEX_TYPE == 'IVFFlat':
        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, FAISS_NLIST,
                                       faiss.METRIC_INNER_PRODUCT)
        return index
    # Default / fallback
    return faiss.IndexFlatIP(dim)


def _load_or_create_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Load existing index if present; otherwise create and (for IVFFlat)
    train a fresh one using the current batch as training data.
    """
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        # Ensure nprobe is set for IVF indexes on every load
        if hasattr(index, 'nprobe'):
            index.nprobe = FAISS_NPROBE
        return index

    # First run — create
    index = _make_index(embeddings.shape[1])
    if FAISS_INDEX_TYPE == 'IVFFlat':
        if len(embeddings) < FAISS_NLIST:
            print(f"⚠️  Only {len(embeddings)} vectors but nlist={FAISS_NLIST}. "
                  f"Falling back to IndexFlatIP for now.")
            return faiss.IndexFlatIP(embeddings.shape[1])
        print(f"🏋️  Training IVFFlat index on {len(embeddings)} vectors…")
        index.train(embeddings)
        index.nprobe = FAISS_NPROBE
    return index


# ── DB init ───────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('''
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
    # Index on url makes load_seen_urls() fast at millions of rows
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_url ON chunks(url)
    ''')
    conn.commit()
    conn.close()
    print(f"✅ DB ready: {db_path}")


# ── Helpers ───────────────────────────────────────────────────

def load_seen_urls(db_path: str = DB_PATH) -> set:
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT url FROM chunks")
        seen = {row[0] for row in cursor.fetchall()}
        conn.close()
        return seen
    except Exception:
        return set()


def get_next_chunk_id(db_path: str = DB_PATH) -> int:
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

def save_to_index(chunks:     list[dict],
                  embeddings: np.ndarray,
                  index_path: str = INDEX_PATH,
                  db_path:    str = DB_PATH) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)

    index = _load_or_create_index(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    conn = sqlite3.connect(db_path)
    conn.executemany(
        'INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)',
        [
            (c['chunk_id'], c['text'], c['title'], c['url'],
             c['date'], c['source'], c['chunk_index'], c['token_count'])
            for c in chunks
        ]
    )
    conn.commit()
    conn.close()

    print(f"✅ FAISS: {index.ntotal} total vectors | "
          f"SQLite: +{len(chunks)} chunks added")
