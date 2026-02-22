# ============================================================
# scraper.py — Article fetching with two strategies:
#   1. newspaper4k  (best quality, fails on some sites)
#   2. requests + BeautifulSoup  (fallback for blocked sites)
# ============================================================

import re
import time
import requests
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
from newspaper import Article

from rag.config import (
    MIN_WORD_COUNT, REQUEST_DELAY, REQUEST_TIMEOUT,
    SCRAPE_HEADERS, MAX_PER_FEED
)


# ── Low-level fetchers ────────────────────────────────────────

def _fetch_newspaper(url: str) -> str | None:
    """Strategy 1: newspaper4k. Best text quality; fails on some sites."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        return text if len(text.split()) >= MIN_WORD_COUNT else None
    except Exception:
        return None


def _fetch_bs4(url: str) -> str | None:
    """
    Strategy 2: raw requests + BeautifulSoup.
    Works on sites that block newspaper4k (arthasarokar, setopati, …).
    Pulls all <p> tags after stripping nav/footer/ads.
    """
    try:
        resp = requests.get(url, headers=SCRAPE_HEADERS,
                            timeout=REQUEST_TIMEOUT)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Strip non-content tags
        for tag in soup(['nav', 'header', 'footer', 'script',
                         'style', 'aside', 'figure', 'noscript']):
            tag.decompose()

        text = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text.split()) >= MIN_WORD_COUNT else None
    except Exception:
        return None


def fetch_article_text(url: str) -> str | None:
    """
    Try newspaper4k first; fall back to BS4 if it fails or
    returns too little text.
    """
    text = _fetch_newspaper(url)
    if text:
        return text

    text = _fetch_bs4(url)
    if text:
        print("      ♻️  BS4 fallback used")
        return text

    print("      ❌ Both extractors failed")
    return None


# ── Text cleaner ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise whitespace, strip URLs/HTML/junk chars.
    Preserves Devanagari (Nepali) via \\u0900-\\u097F range.
    Drops lines shorter than 4 words.
    """
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Keep: word chars, whitespace, punctuation, Devanagari
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"\u0900-\u097F]', ' ', text)
    lines = [l.strip() for l in text.split('\n') if len(l.split()) >= 4]
    return re.sub(r'\s+', ' ', ' '.join(lines)).strip()


# ── Feed discovery ────────────────────────────────────────────

def test_feeds(feed_list: list[str]) -> list[str]:
    """
    Parse every candidate feed URL and return only those
    that respond with at least one entry.
    Deduplicates the result.
    """
    print("🔍 Testing feeds...\n")
    working = []
    for url in feed_list:
        feed    = feedparser.parse(url)
        entries = len(feed.entries)
        status  = feed.get('status', 0)
        if entries > 0:
            working.append(url)
            print(f"   ✅ {entries:3d} entries | {url}")
        else:
            print(f"   ❌ {status:3d} status  | {url}")

    # Preserve order, deduplicate
    seen, unique = set(), []
    for u in working:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    print(f"\n✅ {len(unique)} working feeds / {len(feed_list)} tested")
    return unique


# ── Main scraper ─────────────────────────────────────────────

def scrape_feeds(feed_urls: list[str],
                 max_per_feed: int = MAX_PER_FEED,
                 skip_urls: set   = None) -> list[dict]:
    """
    Scrape articles from all feed URLs.

    Args:
        feed_urls:    List of RSS/Atom feed URLs to scrape.
        max_per_feed: Max articles to collect per feed.
        skip_urls:    Set of URLs already in the DB (to avoid duplicates).

    Returns:
        List of article dicts with keys:
        title, url, date, text (cleaned), source (feed url).
    """
    if skip_urls is None:
        skip_urls = set()

    articles      = []
    total_skipped = total_failed = 0

    for feed_url in feed_urls:
        print(f"\n📡 {feed_url}")
        feed_ok = feed_fail = 0

        try:
            feed  = feedparser.parse(feed_url)
            count = 0

            for entry in feed.entries:
                if count >= max_per_feed:
                    break

                url   = entry.get('link', '').strip()
                title = entry.get('title', 'No Title').strip()

                if not url or url in skip_urls:
                    total_skipped += 1
                    continue

                try:
                    pub      = entry.published_parsed
                    date_str = datetime(*pub[:6]).strftime('%Y-%m-%d')
                except Exception:
                    date_str = datetime.now().strftime('%Y-%m-%d')

                print(f"   → {title[:65]}…")
                raw = fetch_article_text(url)
                time.sleep(REQUEST_DELAY)

                if raw:
                    cleaned = clean_text(raw)
                    if len(cleaned.split()) >= MIN_WORD_COUNT:
                        articles.append({
                            'title' : title,
                            'url'   : url,
                            'date'  : date_str,
                            'text'  : cleaned,
                            'source': feed_url,
                        })
                        count    += 1
                        feed_ok  += 1
                        print(f"      ✅ {len(cleaned.split())} words")
                    else:
                        feed_fail    += 1
                        total_failed += 1
                else:
                    feed_fail    += 1
                    total_failed += 1

        except Exception as e:
            print(f"   ❌ Feed error: {e}")

        print(f"   📊 {feed_ok} ok | {feed_fail} failed")

    print(f"\n📊 TOTAL — New: {len(articles)} | "
          f"Skipped: {total_skipped} | Failed: {total_failed}")
    return articles
