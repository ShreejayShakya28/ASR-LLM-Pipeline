# ============================================================
# pipeline.py
# ============================================================

import faiss
from datetime import datetime
from rag.config  import MAX_PER_FEED, ALL_CANDIDATE_FEEDS, SITEMAP_SOURCES
from rag.scraper import scrape_feeds, scrape_sitemaps, test_feeds
from rag.chunker import chunk_articles
from rag.store   import (init_db, save_to_index,
                         load_seen_urls, get_next_chunk_id)
from rag.models  import embedding_model


def _embed_and_save(articles: list[dict], label: str = "") -> None:
    """Shared chunk → embed → persist logic."""
    if not articles:
        print(f"\n✅ {label} — nothing new to index.")
        return

    start_id   = get_next_chunk_id()
    new_chunks = chunk_articles(articles, start_id=start_id)

    print(f"\n🔢 Embedding {len(new_chunks)} chunks…")
    texts      = [c['text'] for c in new_chunks]
    embeddings = embedding_model.encode(
        texts,
        batch_size=64,          # bumped from 32 — safe on Colab T4
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(embeddings)
    save_to_index(new_chunks, embeddings)
    print(f"\n🎉 {label} done! "
          f"+{len(articles)} articles | +{len(new_chunks)} chunks")


def daily_refresh(feed_urls:    list[str] | None = None,
                  max_per_feed: int               = MAX_PER_FEED) -> None:
    """
    Incremental RSS update — run this every Colab session.
    Only touches articles published in the last ~30 days (RSS window).
    """
    print("=" * 60)
    print(f"🗓️  Daily Refresh — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    init_db()

    if feed_urls is None:
        feed_urls = test_feeds(ALL_CANDIDATE_FEEDS)

    seen_urls    = load_seen_urls()
    print(f"\n📦 {len(seen_urls)} URLs already indexed — will skip these.\n")

    new_articles = scrape_feeds(feed_urls,
                                max_per_feed=max_per_feed,
                                skip_urls=seen_urls)
    _embed_and_save(new_articles, label="Daily refresh")


def backfill(sitemap_urls: list[str] | None = None,
             batch_size:   int               = 500) -> None:
    """
    One-time historical backfill via sitemaps.
    Scrapes articles from BACKFILL_START_YEAR to BACKFILL_END_YEAR.
    Safe to re-run — already-indexed URLs are skipped.

    batch_size: persist to disk every N articles so a Colab disconnect
                doesn't lose everything scraped so far.
    """
    print("=" * 60)
    print(f"📚 Backfill — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    init_db()

    if sitemap_urls is None:
        sitemap_urls = SITEMAP_SOURCES

    seen_urls    = load_seen_urls()
    print(f"\n📦 {len(seen_urls)} URLs already indexed — will skip.\n")

    new_articles = scrape_sitemaps(sitemap_urls, skip_urls=seen_urls)

    # Save in batches — critical for Colab which disconnects mid-run
    for i in range(0, len(new_articles), batch_size):
        batch = new_articles[i : i + batch_size]
        print(f"\n💾 Saving batch {i // batch_size + 1} "
              f"({len(batch)} articles)…")
        _embed_and_save(batch, label=f"Backfill batch {i // batch_size + 1}")

    print("\n✅ Backfill complete.")
