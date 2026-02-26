# ============================================================
# config.py — All tuneable constants in one place.
# Change values here; nothing else needs editing.
# ============================================================

# ── Paths ────────────────────────────────────────────────────
INDEX_DIR  = '/drive/MyDrive/Test-1'
INDEX_PATH = f'{INDEX_DIR}/news.faiss'
DB_PATH    = f'{INDEX_DIR}/metadata.db'

# ── Models ───────────────────────────────────────────────────
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
RERANKER_MODEL  = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_MODEL       = 'google/flan-t5-large'

# ── Scraping ─────────────────────────────────────────────────
MAX_PER_FEED     = 100          # higher ceiling for backfill runs
MIN_WORD_COUNT   = 80
REQUEST_DELAY    = 1.0          # slightly more polite for bulk scraping
REQUEST_TIMEOUT  = 15           # longer timeout for older/slower pages
SCRAPE_HEADERS   = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval ────────────────────────────────────────────────
DEFAULT_TOP_K       = 5         # slightly wider retrieval at larger KB scale
DEFAULT_DAYS_FILTER = 1825      # 5 years (5 × 365) — no effective cutoff
MIN_COSINE          = 0.45
SEM_WEIGHT          = 0.7
FRESH_WEIGHT        = 0.3
DECAY_RATE          = 0.02      # gentler decay so 3-year-old articles still surface
                                # original 0.1 would score a 365-day article near zero

# ── Generation ───────────────────────────────────────────────
MAX_NEW_TOKENS  = 150
CONTEXT_CHARS   = 1200

# ── Backfill date range ───────────────────────────────────────
import datetime
BACKFILL_END_YEAR   = datetime.date.today().year
BACKFILL_START_YEAR = BACKFILL_END_YEAR - 5   # go back 5 years

# ── RSS Feeds (live / recent ~30 days) ───────────────────────
ALL_CANDIDATE_FEEDS = [
    # English
    "https://english.onlinekhabar.com/feed",
    "https://www.spotlightnepal.com/feed/",
    "https://nepaleconomicforum.org/feed",
    "https://techmandu.com/feed",
    "https://nepalnews.com/feed",
    "https://newsofnepal.com/feed/",
    "https://www.recordnepal.com/feed",
    "https://nepalipost.com/feed",
    "https://www.nepalitimes.com/feed",
    "https://myrepublica.nagariknetwork.com/feed",
    "https://kathmandupost.com/feed",
    "https://thehimalayantimes.com/feed",
    # Nepali (Devanagari)
    "https://onlinekhabar.com/feed",
    "https://setopati.com/feed",
    "https://ratopati.com/feed",
    "https://ekantipur.com/rss",
    "https://nagariknews.nagariknetwork.com/feed",
    "https://arthasarokar.com/feed",
]

# ── Sitemap URLs (backfill — historical articles beyond 30 days) ──
# These expose full article archives organised by year/month.
# Your backfill.py module should iterate these to collect historical URLs.
SITEMAP_SOURCES = [
    # Kathmandu Post — monthly sitemaps, very deep archive
    "https://kathmandupost.com/sitemap.xml",
    # Online Khabar English
    "https://english.onlinekhabar.com/sitemap.xml",
    # Online Khabar Nepali
    "https://onlinekhabar.com/sitemap.xml",
    # My Republica
    "https://myrepublica.nagariknetwork.com/sitemap.xml",
    # Himalayan Times
    "https://thehimalayantimes.com/sitemap.xml",
    # Setopati
    "https://setopati.com/sitemap.xml",
    # Ratopati
    "https://ratopati.com/sitemap.xml",
    # eKantipur
    "https://ekantipur.com/sitemap.xml",
    # Nepali Times
    "https://www.nepalitimes.com/sitemap.xml",
    # Spotlight Nepal
    "https://www.spotlightnepal.com/sitemap.xml",
    # Record Nepal
    "https://www.recordnepal.com/sitemap.xml",
]

# ── FAISS index type ─────────────────────────────────────────
# At 5-year scale you will have ~2-5M chunks.
# Switch from IndexFlatIP to IndexIVFFlat for tolerable query speed.
# nlist = number of Voronoi cells; 4096 is a good default at this scale.
FAISS_INDEX_TYPE = 'IVFFlat'   # 'Flat' for <500k chunks, 'IVFFlat' beyond
FAISS_NLIST      = 4096        # only used when FAISS_INDEX_TYPE == 'IVFFlat'
FAISS_NPROBE     = 64          # cells to search at query time (speed vs recall tradeoff)
