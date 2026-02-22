# ============================================================
# config.py — All tuneable constants in one place.
# Change values here; nothing else needs editing.
# ============================================================

# ── Paths ────────────────────────────────────────────────────
INDEX_DIR  = '/content/rag_index'
INDEX_PATH = f'{INDEX_DIR}/news.faiss'
DB_PATH    = f'{INDEX_DIR}/metadata.db'

# ── Models ───────────────────────────────────────────────────
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
RERANKER_MODEL  = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_MODEL       = 'google/flan-t5-large'

# ── Scraping ─────────────────────────────────────────────────
MAX_PER_FEED     = 20       # articles scraped per feed per run
MIN_WORD_COUNT   = 80       # discard articles shorter than this
REQUEST_DELAY    = 0.5      # seconds between requests (be polite)
REQUEST_TIMEOUT  = 10       # seconds before giving up on a URL
SCRAPE_HEADERS   = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE    = 500   # target tokens per chunk
CHUNK_OVERLAP = 50    # overlap tokens between consecutive chunks

# ── Retrieval ────────────────────────────────────────────────
DEFAULT_TOP_K       = 3
DEFAULT_DAYS_FILTER = 30    # ignore articles older than N days
MIN_COSINE          = 0.45  # drop candidates below this similarity
SEM_WEIGHT          = 0.7   # weight for cosine score in blended rank
FRESH_WEIGHT        = 0.3   # weight for time-decay score
DECAY_RATE          = 0.1   # controls how fast freshness drops off

# ── Generation ───────────────────────────────────────────────
MAX_NEW_TOKENS  = 150
CONTEXT_CHARS   = 1200      # chars of context fed to LLM

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
