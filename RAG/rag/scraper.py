# ============================================================
# scraper.py
# ============================================================

import re
import time
import requests
import feedparser
import xml.etree.ElementTree as ET
from datetime import datetime
from bs4 import BeautifulSoup
from newspaper import Article

from rag.config import (
    MIN_WORD_COUNT, REQUEST_DELAY, REQUEST_TIMEOUT,
    SCRAPE_HEADERS, MAX_PER_FEED,
    BACKFILL_START_YEAR, BACKFILL_END_YEAR, SITEMAP_SOURCES
)


# ── Retry-aware GET ───────────────────────────────────────────

def _get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=SCRAPE_HEADERS,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = 'utf-8'
            return resp
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"      ⚠️  GET failed after {retries} tries: {e}")
    return None


# ── Low-level fetchers ────────────────────────────────────────

def _fetch_newspaper(url: str) -> str | None:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        return text if len(text.split()) >= MIN_WORD_COUNT else None
    except Exception:
        return None


def _fetch_bs4(url: str) -> str | None:
    resp = _get(url)
    if not resp:
        return None
    try:
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['nav', 'header', 'footer', 'script',
                         'style', 'aside', 'figure', 'noscript']):
            tag.decompose()
        text = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text.split()) >= MIN_WORD_COUNT else None
    except Exception:
        return None


def fetch_article_text(url: str) -> str | None:
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
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(
        r'[^\w\s\.\,\!\?\;\:\-\'\"\u0900-\u097F\u0964\u0965]', ' ', text
    )
    lines = [l.strip() for l in text.split('\n') if len(l.split()) >= 4]
    return re.sub(r'\s+', ' ', ' '.join(lines)).strip()


# ── Feed discovery ────────────────────────────────────────────

def test_feeds(feed_list: list[str]) -> list[str]:
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

    seen, unique = set(), []
    for u in working:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    print(f"\n✅ {len(unique)} working feeds / {len(feed_list)} tested")
    return unique


# ── Sitemap parsing (URL collection only) ────────────────────

def _parse_sitemap(url: str, target_year: int | None = None) -> list[str]:
    """
    Recursively parse sitemap. Returns flat list of article URLs.
    If target_year is set, only follows child sitemaps containing that year.
    """
    resp = _get(url)
    if not resp:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    ns  = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    tag = root.tag.lower()

    if 'sitemapindex' in tag:
        urls = []
        for sitemap in root.findall('sm:sitemap', ns):
            loc = sitemap.findtext('sm:loc', namespaces=ns)
            if not loc:
                continue
            if target_year is not None:
                if str(target_year) not in loc:
                    continue
            else:
                year_ok = any(
                    str(y) in loc
                    for y in range(BACKFILL_START_YEAR, BACKFILL_END_YEAR + 1)
                )
                if not year_ok:
                    continue
            urls.extend(_parse_sitemap(loc, target_year=target_year))
        return urls

    if 'urlset' in tag:
        return [
            loc.text.strip()
            for loc in root.findall('sm:url/sm:loc', ns)
            if loc.text
        ]

    return []


def collect_sitemap_urls(sitemap_urls: list[str],
                         target_year: int | None = None,
                         skip_urls: set = None) -> list[str]:
    """
    Collect all article URLs from sitemaps without scraping.
    Returns deduplicated list of new URLs not in skip_urls.
    """
    if skip_urls is None:
        skip_urls = set()

    all_urls = []
    seen_in_batch = set()

    for sm_url in sitemap_urls:
        label = f"year={target_year}" if target_year else "all years"
        print(f"\n🗺️  Parsing sitemap ({label}): {sm_url}")
        found = _parse_sitemap(sm_url, target_year=target_year)

        new = []
        for u in found:
            if u not in skip_urls and u not in seen_in_batch:
                new.append(u)
                seen_in_batch.add(u)

        print(f"   → {len(found)} URLs found | {len(new)} new (not yet indexed)")
        all_urls.extend(new)

    return all_urls


def scrape_url_batch(urls: list[str], batch_num: int = 1,
                     total_batches: int = 1) -> list[dict]:
    """
    Scrape a list of URLs and return article dicts.
    This is called per-batch so results are saved before moving to next batch.
    """
    articles = []
    failed   = 0

    for i, url in enumerate(urls, 1):
        print(f"   [{i}/{len(urls)}] {url[:75]}")
        raw = fetch_article_text(url)
        time.sleep(REQUEST_DELAY)

        if raw:
            cleaned = clean_text(raw)
            if len(cleaned.split()) >= MIN_WORD_COUNT:
                articles.append({
                    'title' : url.split('/')[-1].replace('-', ' ').title()[:120],
                    'url'   : url,
                    'date'  : datetime.now().strftime('%Y-%m-%d'),
                    'text'  : cleaned,
                    'source': 'sitemap_backfill',
                })
            else:
                failed += 1
        else:
            failed += 1

    print(f"\n   📊 Batch {batch_num}/{total_batches} — "
          f"✅ {len(articles)} scraped | ❌ {failed} failed")
    return articles


# ── Main RSS scraper ──────────────────────────────────────────

def scrape_feeds(feed_urls: list[str],
                 max_per_feed: int = MAX_PER_FEED,
                 skip_urls: set   = None) -> list[dict]:
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
