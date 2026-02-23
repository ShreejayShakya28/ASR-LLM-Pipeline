# ============================================================
# config.py — All tuneable constants in one place.
# Change values here; nothing else needs editing.
# ============================================================

# ── Paths ────────────────────────────────────────────────────
INDEX_DIR  = '/drive/MyDrive/nepal_rag_index'
INDEX_PATH = f'{INDEX_DIR}/news.faiss'
DB_PATH    = f'{INDEX_DIR}/metadata.db'

# ── Models ───────────────────────────────────────────────────
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
RERANKER_MODEL  = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_MODEL       = 'google/flan-t5-large'

# ── Scraping ─────────────────────────────────────────────────
MAX_PER_FEED     = 20
MIN_WORD_COUNT   = 80
REQUEST_DELAY    = 0.5
REQUEST_TIMEOUT  = 10
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
DEFAULT_TOP_K       = 3
DEFAULT_DAYS_FILTER = 30
MIN_COSINE          = 0.45
SEM_WEIGHT          = 0.7
FRESH_WEIGHT        = 0.3
DECAY_RATE          = 0.1

# ── Generation ───────────────────────────────────────────────
MAX_NEW_TOKENS  = 150
CONTEXT_CHARS   = 1200

# ── RSS Feeds ────────────────────────────────────────────────
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
