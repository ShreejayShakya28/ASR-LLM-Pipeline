# ============================================================
# chunker.py — Sentence-aware text chunking.
#
# Key upgrade over the original: language detection so that
# Nepali (Devanagari) articles use indic-nlp-library for
# sentence splitting instead of NLTK (which is English-only).
# ============================================================

import nltk
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Indic NLP (optional — degrades gracefully if missing) ────
try:
    from indicnlp.tokenize import sentence_tokenize as indic_sent
    INDIC_AVAILABLE = True
    print("✅ indic-nlp-library loaded (Nepali tokenization active)")
except ImportError:
    INDIC_AVAILABLE = False
    print("⚠️  indic-nlp-library not found — Nepali will use NLTK fallback")

# ── langdetect (optional) ─────────────────────────────────────
try:
    from langdetect import detect as _detect
    def detect_lang(text: str) -> str:
        try:
            return _detect(text[:500])   # cheap: only look at start
        except Exception:
            return 'en'
except ImportError:
    def detect_lang(text: str) -> str:
        return 'en'


# ── Sentence tokenizer ────────────────────────────────────────

def sentence_tokenize(text: str) -> list[str]:
    """
    Split text into sentences.
    - Nepali (ne) → indic-nlp-library  (handles Devanagari correctly)
    - Everything else → NLTK punkt
    Falls back to NLTK if indic-nlp is unavailable.
    """
    lang = detect_lang(text)
    if lang == 'ne' and INDIC_AVAILABLE:
        return indic_sent.sentence_split(text, lang='ne')
    return nltk.sent_tokenize(text)


# ── Token counting (fast word-based approx) ──────────────────

def _count_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


# ── Core chunker ─────────────────────────────────────────────

def chunk_text(text: str,
               chunk_size: int  = CHUNK_SIZE,
               overlap:    int  = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks of ~chunk_size tokens.
    Overlap carries the last few sentences of the previous chunk
    forward so retrieval never misses context at chunk boundaries.
    """
    sentences     = sentence_tokenize(text)
    chunks        = []
    current       = []
    current_count = 0

    for sentence in sentences:
        s_tokens = _count_tokens(sentence)

        if current_count + s_tokens > chunk_size and current:
            chunks.append(' '.join(current))

            # Build overlap: walk backwards until we hit the budget
            overlap_sents = []
            overlap_count = 0
            for s in reversed(current):
                st = _count_tokens(s)
                if overlap_count + st <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_count += st
                else:
                    break
            current       = overlap_sents
            current_count = overlap_count

        current.append(sentence)
        current_count += s_tokens

    if current:
        chunks.append(' '.join(current))

    return chunks


# ── Article-level chunker ─────────────────────────────────────

def chunk_articles(articles: list[dict],
                   start_id: int = 0) -> list[dict]:
    """
    Chunk every article in *articles* and return a flat list of
    chunk dicts ready for embedding + storage.

    Each chunk dict:
        chunk_id, text, title, url, date, source, chunk_index, token_count
    """
    all_chunks = []
    chunk_id   = start_id

    for article in articles:
        for i, chunk in enumerate(chunk_text(article['text'])):
            all_chunks.append({
                'chunk_id'   : chunk_id,
                'text'       : chunk,
                'title'      : article['title'],
                'url'        : article['url'],
                'date'       : article['date'],
                'source'     : article['source'],
                'chunk_index': i,
                'token_count': _count_tokens(chunk),
            })
            chunk_id += 1

    print(f"✅ {len(all_chunks)} chunks from {len(articles)} articles")
    return all_chunks
