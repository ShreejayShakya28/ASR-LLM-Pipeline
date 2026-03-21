# ============================================================
# config.py — All tuneable constants in one place.
# Change values here; nothing else needs editing.
# ============================================================

# ── Paths ────────────────────────────────────────────────────
INDEX_DIR  = '/drive/MyDrive/RAG'
INDEX_PATH = f'{INDEX_DIR}/Knowledge_Base/news.faiss'
DB_PATH    = f'{INDEX_DIR}/Knowledge_Base/metadata.db'
SLM_PATH   = f'{INDEX_DIR}/SLM_Model/SLM.pth'
REPORT_DIR = f'{INDEX_DIR}/eval_reports'

# ── Models ───────────────────────────────────────────────────
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
RERANKER_MODEL  = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# ── Scraping ─────────────────────────────────────────────────
MAX_PER_FEED     = 100
MIN_WORD_COUNT   = 80
REQUEST_DELAY    = 1.0
REQUEST_TIMEOUT  = 15
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
DEFAULT_TOP_K       = 5
DEFAULT_DAYS_FILTER = 730
MIN_COSINE          = 0.45
SEM_WEIGHT          = 0.7
FRESH_WEIGHT        = 0.3
DECAY_RATE          = 0.02

# ── Generation ───────────────────────────────────────────────
MAX_NEW_TOKENS  = 150
CONTEXT_CHARS   = 1200

# ── Backfill date range ───────────────────────────────────────
import datetime
BACKFILL_END_YEAR   = datetime.date.today().year
BACKFILL_START_YEAR = BACKFILL_END_YEAR - 2

# ── RSS Feeds ────────────────────────────────────────────────
ALL_CANDIDATE_FEEDS = [
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
    "https://onlinekhabar.com/feed",
    "https://setopati.com/feed",
    "https://ratopati.com/feed",
    "https://ekantipur.com/rss",
    "https://nagariknews.nagariknetwork.com/feed",
    "https://arthasarokar.com/feed",
]

# ── Sitemap URLs (backfill) ───────────────────────────────────
SITEMAP_SOURCES = [
    "https://kathmandupost.com/sitemap.xml",
    "https://english.onlinekhabar.com/sitemap.xml",
    "https://onlinekhabar.com/sitemap.xml",
    "https://myrepublica.nagariknetwork.com/sitemap.xml",
    "https://thehimalayantimes.com/sitemap.xml",
    "https://setopati.com/sitemap.xml",
    "https://ratopati.com/sitemap.xml",
    "https://ekantipur.com/sitemap.xml",
    "https://www.nepalitimes.com/sitemap.xml",
    "https://www.spotlightnepal.com/sitemap.xml",
    "https://www.recordnepal.com/sitemap.xml",
]

# ── FAISS index type ─────────────────────────────────────────
FAISS_INDEX_TYPE = 'Flat'
FAISS_NLIST      = 4096
FAISS_NPROBE     = 64
