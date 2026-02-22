# ============================================================
# pipeline.py — Orchestrates the full scrape → index pipeline.
#
# daily_refresh() is the one function you call every session.
# It is intentionally NOT auto-scheduled because Colab runtimes
# disconnect; just call it at the top of your session.
# ============================================================

import faiss
from datetime import datetime

from rag.config        import MAX_PER_FEED
from rag.scraper       import scrape_feeds, test_feeds
from rag.chunker       import chunk_articles
from rag.store         import (init_db, save_to_index,
                               load_seen_urls, get_next_chunk_id)
from rag.models        import embedding_model
from rag.config        import ALL_CANDIDATE_FEEDS


def daily_refresh(feed_urls:     list[str] | None = None,
                  max_per_feed:  int               = MAX_PER_FEED) -> None:
    """
    Full incremental update:
      1. Discover which feeds are alive (if feed_urls is None).
      2. Load URLs already indexed so we skip duplicates.
      3. Scrape → clean → chunk → embed → append to FAISS + SQLite.

    Call this once per Colab session (or whenever you want fresh data).
    There is no background scheduler — Colab runtimes disconnect too
    quickly for that to be reliable. Just run this cell on connect.
    """
    print("=" * 60)
    print(f"🗓️  Daily Refresh — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Step 0: Init DB (idempotent)
    init_db()

    # Step 1: Resolve feed list
    if feed_urls is None:
        feed_urls = test_feeds(ALL_CANDIDATE_FEEDS)

    # Step 2: Skip already-indexed URLs
    seen_urls = load_seen_urls()
    print(f"\n📦 {len(seen_urls)} URLs already indexed — will skip these.\n")

    # Step 3: Scrape new articles
    new_articles = scrape_feeds(feed_urls,
                                max_per_feed=max_per_feed,
                                skip_urls=seen_urls)
    if not new_articles:
        print("\n✅ Nothing new to index — already up to date.")
        return

    # Step 4: Chunk
    start_id   = get_next_chunk_id()
    new_chunks = chunk_articles(new_articles, start_id=start_id)

    # Step 5: Embed
    print("\n🔢 Embedding chunks…")
    texts      = [c['text'] for c in new_chunks]
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(embeddings)   # unit-normalise → cosine via dot-product

    # Step 6: Persist
    save_to_index(new_chunks, embeddings)

    print(f"\n🎉 Done! +{len(new_articles)} articles | +{len(new_chunks)} chunks")
