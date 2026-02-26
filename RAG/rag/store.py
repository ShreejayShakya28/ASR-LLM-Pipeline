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

EMBEDDING_DIM = 384


# ── Guard ─────────────────────────────────────────────────────

def _check_drive():
    if DB_PATH.startswith('/drive/') and not os.path.exists('/drive/MyDrive'):
        raise RuntimeError(
            "\n❌ Google Drive is not mounted.\n"
            "   Run:  from google.colab import drive; drive.mount('/drive')\n"
            "   Then re-import."
        )

_check_drive()


# ── FAISS index factory ───────────────────────────────────────

def _make_index(dim: int) -> faiss.Index:
    if FAISS_INDEX_TYPE == 'IVFFlat':
        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, FAISS_NLIST,
                                       faiss.METRIC_INNER_PRODUCT)
        return index
    return faiss.IndexFlatIP(dim)


def _load_or_create_index(embeddings: np.ndarray) -> faiss.Index:
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        if hasattr(index, 'nprobe'):
            index.nprobe = FAISS_NPROBE
        return index

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
    conn.execute('CREATE INDEX IF NOT EXISTS idx_url ON chunks(url)')
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


# ── Storage monitor ───────────────────────────────────────────

def storage_report(db_path: str = DB_PATH,
                   index_path: str = INDEX_PATH) -> dict:
    """
    Print and return a full snapshot of what is actually on disk.
    Call this any time to verify state — especially after each batch.
    """
    report = {}

    # SQLite stats
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks")
        report['total_chunks'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT url) FROM chunks")
        report['total_articles'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT source) FROM chunks")
        report['total_sources'] = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM chunks")
        row = cursor.fetchone()
        report['date_range'] = f"{row[0]}  →  {row[1]}"

        # Per-source breakdown
        cursor.execute("""
            SELECT source, COUNT(DISTINCT url) as articles
            FROM chunks
            GROUP BY source
            ORDER BY articles DESC
            LIMIT 15
        """)
        report['by_source'] = cursor.fetchall()

        conn.close()

        db_bytes = os.path.getsize(db_path)
        report['db_size_mb'] = db_bytes / 1_048_576

    except Exception as e:
        report['db_error'] = str(e)

    # FAISS stats
    try:
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            report['faiss_vectors'] = index.ntotal
            faiss_bytes = os.path.getsize(index_path)
            report['faiss_size_mb'] = faiss_bytes / 1_048_576
        else:
            report['faiss_vectors'] = 0
            report['faiss_size_mb'] = 0.0
    except Exception as e:
        report['faiss_error'] = str(e)

    # Total Drive usage
    total_mb = report.get('db_size_mb', 0) + report.get('faiss_size_mb', 0)
    report['total_drive_mb'] = total_mb

    # Print summary
    print("\n" + "=" * 55)
    print("📊  STORAGE REPORT")
    print("=" * 55)
    print(f"  Articles indexed  : {report.get('total_articles', 0):,}")
    print(f"  Total chunks      : {report.get('total_chunks', 0):,}")
    print(f"  FAISS vectors     : {report.get('faiss_vectors', 0):,}")
    print(f"  Date range        : {report.get('date_range', 'n/a')}")
    print(f"  SQLite size       : {report.get('db_size_mb', 0):.2f} MB")
    print(f"  FAISS size        : {report.get('faiss_size_mb', 0):.2f} MB")
    print(f"  Total on Drive    : {total_mb:.2f} MB")
    print("-" * 55)
    if 'by_source' in report:
        print("  Top sources:")
        for src, count in report['by_source']:
            short = src[-55:] if len(src) > 55 else src
            print(f"    {count:>6,} articles  {short}")
    print("=" * 55 + "\n")

    return report


# ── Persistence ───────────────────────────────────────────────

def save_to_index(chunks:     list[dict],
                  embeddings: np.ndarray,
                  index_path: str = INDEX_PATH,
                  db_path:    str = DB_PATH) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)

    # ── FAISS ─────────────────────────────────────────────────
    index = _load_or_create_index(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # ── SQLite ────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    conn.executemany(
        'INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?,?)',
        [
            (c['chunk_id'], c['text'], c['title'], c['url'],
             c['date'], c['source'], c['chunk_index'], c['token_count'])
            for c in chunks
        ]
    )
    conn.commit()

    # Verify the insert actually landed
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_in_db = cursor.fetchone()[0]
    conn.close()

    db_mb    = os.path.getsize(db_path) / 1_048_576
    faiss_mb = os.path.getsize(index_path) / 1_048_576

    print(f"  💾 Saved  → chunks in DB: {total_in_db:,} | "
          f"FAISS vectors: {index.ntotal:,}")
    print(f"  📁 Sizes  → SQLite: {db_mb:.2f} MB | "
          f"FAISS: {faiss_mb:.2f} MB | "
          f"Total: {db_mb + faiss_mb:.2f} MB")
