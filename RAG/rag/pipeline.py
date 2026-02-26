# ============================================================
# pipeline.py
# ============================================================

import faiss
from datetime import datetime
from rag.config  import (MAX_PER_FEED, ALL_CANDIDATE_FEEDS,
    SITEMAP_SOURCES, BACKFILL_START_YEAR,
    BACKFILL_END_YEAR)
from rag.scraper import (scrape_feeds, test_feeds,
    collect_sitemap_urls, scrape_url_batch)
from rag.chunker import chunk_articles
from rag.store   import (init_db, save_to_index, load_seen_urls,
    get_next_chunk_id, storage_report)
from rag.models  import embedding_model


# ── Shared embed + save ───────────────────────────────────────

def _embed_and_save(articles: list[dict], label: str = "") -> int:
    """
    Chunk → embed → persist. Returns number of chunks saved.
    INSERT OR IGNORE in store.py means this is always safe to re-call.
    """
    if not articles:
        return 0

    start_id   = get_next_chunk_id()
    new_chunks = chunk_articles(articles, start_id=start_id)

    if not new_chunks:
        return 0

    print(f"  🔢 Embedding {len(new_chunks)} chunks…")
    texts      = [c['text'] for c in new_chunks]
    embeddings = embedding_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(embeddings)
    save_to_index(new_chunks, embeddings)
    return len(new_chunks)


# ── Daily refresh (RSS — unchanged call signature) ────────────

def daily_refresh(feed_urls:    list[str] | None = None,
                  max_per_feed: int               = MAX_PER_FEED) -> None:
    """
    Incremental RSS update — run every Colab session.
    Call signature is identical to before — notebook cell unchanged.
    """
    print("=" * 60)
    print(f"🗓️  Daily Refresh — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    init_db()

    if feed_urls is None:
        feed_urls = test_feeds(ALL_CANDIDATE_FEEDS)

    seen_urls = load_seen_urls()
    print(f"\n📦 {len(seen_urls):,} URLs already indexed — will skip.\n")

    new_articles = scrape_feeds(feed_urls,
                                max_per_feed=max_per_feed,
                                skip_urls=seen_urls)

    if not new_articles:
        print("\n✅ Nothing new to index — already up to date.")
        storage_report()
        return

    saved = _embed_and_save(new_articles, label="Daily refresh")
    print(f"\n🎉 Daily refresh done! "
        f"+{len(new_articles)} articles | +{saved} chunks")
    storage_report()


# ── Backfill (sitemap — fault-tolerant, year-by-year) ─────────

def backfill(sitemap_urls:  list[str] | None = None,
             articles_per_batch: int          = 3000,
             batch_size:    int | None        = None,
             start_year:    int | None        = None,
             end_year:      int | None        = None) -> None:
    """
    Fault-tolerant historical backfill via sitemaps.

    Architecture:
      For each year (oldest → newest):
        1. Parse sitemaps to collect ALL article URLs for that year
        2. Filter out already-indexed URLs (loaded fresh from DB each time)
        3. Split remaining URLs into batches of articles_per_batch
        4. For each batch: scrape → embed → save → report → next batch
        5. If Colab disconnects, re-running safely skips already-saved URLs

    Args:
        sitemap_urls:       Override default SITEMAP_SOURCES
        articles_per_batch: Scrape and save this many articles before
                            persisting (default 3000 — ~15-20 min per batch)
        start_year:         Override BACKFILL_START_YEAR
        end_year:           Override BACKFILL_END_YEAR
    """
    if batch_size is not None:
        articles_per_batch = batch_size
    print("=" * 60)
    print(f"📚 Backfill — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    init_db()

    if sitemap_urls is None:
        sitemap_urls = SITEMAP_SOURCES
    yr_start = start_year or BACKFILL_START_YEAR
    yr_end   = end_year   or BACKFILL_END_YEAR

    print(f"\n📅 Target range: {yr_start} → {yr_end}")
    print(f"📦 Batch size: {articles_per_batch:,} articles per save\n")

    # Show what's already stored before we begin
    storage_report()

    grand_total_articles = 0
    grand_total_chunks   = 0

    for year in range(yr_start, yr_end + 1):
        print("\n" + "█" * 60)
        print(f"█  YEAR {year}")
        print("█" * 60)

        # Load seen URLs fresh at the start of each year
        # so previous year's saves are already excluded
        seen_urls = load_seen_urls()
        print(f"  📦 {len(seen_urls):,} URLs already in DB (will skip)\n")

        # Step 1 — collect all URLs for this year
        print(f"  🗺️  Collecting sitemap URLs for {year}…")
        year_urls = collect_sitemap_urls(
            sitemap_urls,
            target_year=year,
            skip_urls=seen_urls
        )

        if not year_urls:
            print(f"  ✅ No new URLs found for {year} — already complete.\n")
            continue

        print(f"\n  📋 {len(year_urls):,} new URLs to scrape for {year}")

        # Step 2 — split into batches and process
        batches      = [year_urls[i:i + articles_per_batch]
            for i in range(0, len(year_urls), articles_per_batch)]
        year_articles = 0
        year_chunks   = 0

        for batch_num, batch_urls in enumerate(batches, 1):
            print(f"\n  ┌─ Batch {batch_num}/{len(batches)} "
                f"({len(batch_urls):,} URLs) ─────────────────")

            # Reload seen_urls before each batch — catches mid-year saves
            seen_urls = load_seen_urls()
            batch_urls = [u for u in batch_urls if u not in seen_urls]

            if not batch_urls:
                print(f"  │  All URLs in this batch already indexed — skipping.")
                print(f"  └──────────────────────────────────────────────────")
                continue

            print(f"  │  Scraping {len(batch_urls):,} URLs…")

            # Scrape this batch
            articles = scrape_url_batch(
                batch_urls,
                batch_num=batch_num,
                total_batches=len(batches)
            )

            if not articles:
                print(f"  │  ⚠️  No articles extracted from this batch.")
                print(f"  └──────────────────────────────────────────────────")
                continue

            # Embed and save immediately — before moving to next batch
            print(f"\n  │  💾 Persisting {len(articles):,} articles to Drive…")
            saved_chunks = _embed_and_save(articles)

            year_articles += len(articles)
            year_chunks   += saved_chunks

            print(f"  └─ ✅ Batch {batch_num} complete: "
                f"+{len(articles):,} articles | +{saved_chunks:,} chunks")

            # Storage report after every batch so you can see growth
            storage_report()

        print(f"\n  🏁 Year {year} complete: "
            f"+{year_articles:,} articles | +{year_chunks:,} chunks")

        grand_total_articles += year_articles
        grand_total_chunks   += year_chunks

    # Final report
    print("\n" + "=" * 60)
    print("🎉 BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Total new articles : {grand_total_articles:,}")
    print(f"  Total new chunks   : {grand_total_chunks:,}")
    storage_report()
