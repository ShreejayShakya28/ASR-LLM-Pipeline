# RAG

## Architecture

```
rag/
  config.py      — all tuneable constants (feeds, paths, thresholds)
  models.py      — load embedding model, reranker, LLM once
  scraper.py     — dual-strategy fetching: newspaper4k → BS4 fallback
  chunker.py     — language-aware sentence splitting (NLTK / indic-nlp)
  store.py       — FAISS index + SQLite metadata persistence
  retriever.py   — cosine search → time-decay blend → CrossEncoder rerank
  generator.py   — context formatting + Flan-T5 generation
  pipeline.py    — daily_refresh() orchestrator
  inference.py   — ask() public interface
notebooks/
  nepal_rag_colab.ipynb — Colab session notebook
```

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It solves a fundamental problem with language models: they are frozen in time at their training cutoff and cannot answer questions about recent events.

The idea is simple:

```
User Question
     ↓
Search a knowledge base for relevant documents
     ↓
Pass those documents as context to a language model
     ↓
Language model answers using only that context
```

This means the system's answers are always grounded in real, recent, verifiable sources — not hallucinated from training data.

The formal mathematical framing is:

```
P(answer | question, documents) = 
    Σ P(answer | question, chunk) × P(chunk | question)
```

Where the system retrieves the most relevant chunks from its knowledge base and conditions the answer generation on those chunks.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                        │
│                                                             │
│  RSS Feeds → Scraper → Cleaner → Chunker → Embedder → Store │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                        │
│                                                             │
│  Query → Embed → FAISS Search → Time-Decay → Reranker       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION PIPELINE                        │
│                                                             │
│  Context + Question → Prompt → Flan-T5 → Answer + Sources   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Google Drive persistence
Index lives at `/drive/MyDrive/nepal_rag_index/` — two files:
```
nepal_rag_index/
├── news.faiss    ← vector embeddings
└── metadata.db   ← article text, titles, URLs, dates (SQLite)
```
`store.py` raises a clear error if Drive isn't mounted before any
import tries to access these paths.

### Staleness handling
Two mechanisms work together:
- **Time-decay score** — `e^(-0.1 × days_old)` exponentially down-weights
  older chunks in the blended retrieval score
- **`days_filter`** (default 30) — hard cutoff that drops articles older
  than N days before reranking

### Scraping strategy
1. `newspaper4k` — best text extraction quality
2. `requests + BeautifulSoup` — fallback for sites that block newspaper4k
   (arthasarokar, setopati, etc.)

### Nepali tokenization
`langdetect` identifies Nepali text; `indic-nlp-library` then splits it
into sentences correctly. English falls back to NLTK. Both degrade
gracefully if the library is missing.

---

## Tuning

| Parameter     | Default | Effect |
|--------------|---------|--------|
| `min_cosine`  | 0.45    | Lower → more recall, less precision |
| `days_filter` | 30      | Higher → older articles included |
| `top_k`       | 3       | More sources = richer context |
| `DECAY_RATE`  | 0.1     | Higher → freshness matters more |

--------

## Component Breakdown

### 1. Data Collection — RSS Scraper

What it does: Fetches real news articles from Nepali news websites automatically.

How it works: RSS (Really Simple Syndication) feeds are structured XML files that news websites publish. Each entry contains a title, URL, and publication date. We use `feedparser` to parse these feeds and `newspaper4k` to extract clean article body text from each URL.

Key design decisions:
- A `skip_urls` set prevents re-fetching articles already in the database
- A `0.5 second` polite delay between requests avoids overloading news servers
- Articles under 100 words are discarded as likely failed extractions or ad fragments
- BS4 fallback kicks in for sites that block `newspaper4k` (arthasarokar, setopati)

News sources tested:
- `english.onlinekhabar.com` — English, reliable, 20 entries
- `spotlightnepal.com` — English, reliable, 20 entries
- `nepalnews.com` — Mixed, 12 entries
- `newsofnepal.com` — Nepali, 12 entries
- `setopati.com` — Nepali, 5 entries

Packages used:
- `feedparser` — RSS parsing
- `newspaper4k` — Article text extraction (replaces deprecated `newspaper3k`)
- `lxml_html_clean` — Required dependency for `newspaper4k` on Python 3.12
- `requests` + `beautifulsoup4` — Fallback scraper for blocked sites

---

### 2. Text Cleaning

What it does: Removes noise from raw article text before it enters the knowledge base.

What gets removed:
- URLs (`http://...`, `www....`)
- HTML tags that slipped through extraction
- Special characters (keeping punctuation and Devanagari `\u0900-\u097F`)
- Lines shorter than 4 words (navigation fragments, ad remnants)
- Multiple consecutive spaces

Why this matters: Dirty text produces poor embeddings. If a chunk is full of "Click here", "Advertisement", "Share on Facebook" fragments, the embedding model will encode those noise signals instead of the actual article meaning — causing irrelevant retrieval results.

Packages used:
- `re` (Python built-in) — Regular expression cleaning

---

### 3. Chunking

What it does: Splits long articles into smaller overlapping pieces that fit within the embedding model's context window.

Strategy used — Sentence-aware chunking:

Rather than splitting blindly at a fixed character count, the chunker:
1. First tokenizes the article into individual sentences using NLTK or indic-nlp-library
2. Groups sentences together until the chunk reaches ~500 tokens
3. When a chunk is full, saves it and starts a new one
4. Carries over the last few sentences into the next chunk (overlap)

Parameters:

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 500 tokens | Fits within MiniLM's context window |
| Overlap | 50 tokens | Preserves context at chunk boundaries |
| Split unit | Sentences | Avoids cutting mid-thought |

Why overlap matters: Without overlap, a sentence spanning the boundary between chunk N and chunk N+1 loses its context on one side. A query about that sentence would retrieve only half the meaning. Overlap ensures continuity.

Token counting: We approximate tokens as `words / 0.75` since most English words tokenize to roughly 1.3 subword tokens.

Packages used:
- `nltk` — Sentence tokenization for English (`sent_tokenize`)
- `indic-nlp-library` — Sentence tokenization for Nepali (Devanagari)
- `langdetect` — Detects language to route to the correct tokenizer

---

### 4. Embedding

What it does: Converts each text chunk into a 384-dimensional numerical vector that captures its semantic meaning.

Model used: `all-MiniLM-L6-v2` from Sentence Transformers

Why this model:
- Only ~80MB in size
- Runs fast on CPU
- Produces 384-dimensional vectors (small enough for fast search)
- Strong semantic similarity performance despite its small size
- Available for free via HuggingFace

How it works: The model is a distilled version of a much larger BERT model, fine-tuned specifically on semantic similarity tasks. Two sentences with similar meanings will produce vectors that are geometrically close to each other in 384-dimensional space, even if the exact words are different.

For example:
- "PM Karki directed election security"
- "Prime Minister tightens polling arrangements"

These produce very similar vectors despite sharing almost no words.

Normalization: All vectors are L2-normalized before storage so that inner product (dot product) is equivalent to cosine similarity. This is required for `faiss.IndexFlatIP`.

Packages used:
- `sentence-transformers` — The embedding library
- `torch` — Backend for model inference
- `numpy` — Vector array operations

---

### 5. Vector Storage — FAISS + SQLite

What it does: Stores all chunk embeddings in a fast searchable index on disk, persisted to Google Drive so the index survives Colab disconnects.

What is FAISS: Facebook AI Similarity Search is a library for efficient nearest-neighbor search in high-dimensional vector spaces. Given a query vector, it finds the most similar stored vectors in milliseconds — even across millions of entries.

Index type used: `IndexFlatIP` (Flat Inner Product)
- "Flat" means exact search — checks every vector, no approximation
- "IP" means Inner Product — equivalent to cosine similarity after normalization
- Best choice for small-to-medium datasets (under 100k vectors)

Similarity formula:

```
similarity(query, chunk) = query · chunk / (|query| × |chunk|)
```

After L2 normalization, this simplifies to just the dot product.

Storage layout:

```
/drive/MyDrive/nepal_rag_index/
├── news.faiss    ← vector index (binary, ~1.5KB per chunk)
└── metadata.db   ← SQLite database (text, title, URL, date)
```

FAISS only stores vectors — not the original text. The SQLite database provides the parallel lookup: FAISS returns index numbers, SQLite maps those numbers back to readable content.

Packages used:
- `faiss-cpu` — Vector index
- `sqlite3` (Python built-in) — Metadata storage

---

### 6. Retrieval — Two-Stage Pipeline

This is the most technically sophisticated part of the system. Retrieval happens in two stages.

#### Stage 1: FAISS Cosine Search with Time-Decay

The query is embedded into a vector and compared against all stored chunk vectors using cosine similarity.

**Minimum score threshold:** Chunks with cosine similarity below `0.45` are immediately rejected. This prevents weakly-related articles from polluting the context.

**Time-decay scoring:** News has a freshness problem — a query about "Nepal elections" should prefer articles from today over articles from 3 months ago, even if the older ones are slightly more semantically similar.

The decay formula:

```
decay_score = e^(-λ × days_old)
```

Where `λ = 0.1` gives a slow decay curve:

| Age | Score |
|---|---|
| Today | 1.000 |
| 3 days | 0.741 |
| 1 week | 0.497 |
| 1 month | 0.050 |
| 3 months | ~0.000 |

**Combined score:**
```
final_score = 0.7 × cosine_score + 0.3 × decay_score
```

This means recency matters but semantic relevance still dominates.

#### Stage 2: CrossEncoder Reranking

The first stage retrieves candidates fast but imprecisely. The reranker scores each (query, chunk) pair together — much more accurately.

**Bi-encoder vs CrossEncoder:**

| | Bi-encoder (Stage 1) | CrossEncoder (Stage 2) |
|---|---|---|
| How | Embeds query and chunk separately | Reads query + chunk together |
| Speed | Very fast | Slower |
| Accuracy | Good | Much better |
| Use case | Candidate retrieval | Final reranking |

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Trained on MS MARCO question-answering dataset
- ~90MB
- Returns a relevance score for each (question, passage) pair
- Negative scores = not relevant, positive scores = relevant

Packages used:
- `sentence-transformers` (CrossEncoder class)
- `faiss-cpu`
- `sqlite3`
- `math` (for exponential decay)

---

### 7. Generation — Flan-T5

What it does: Takes the retrieved context chunks and the user's question, builds a prompt, and generates a grounded answer.

Model used: `google/flan-t5-large`
- Instruction-tuned — understands "Answer using only the articles above"
- ~800MB — fits in Colab free tier RAM
- Encoder-decoder architecture — better at answering than continuing text

Generation parameters:

| Parameter | Value | Reason |
|---|---|---|
| `max_new_tokens` | 150 | Keeps answers concise |
| `temperature` | 0.1 | Very low — factual, deterministic |
| `do_sample` | False | Greedy decoding — most reliable output |
| `repetition_penalty` | 1.5 | Prevents looping/repeating phrases |

Packages used:
- `transformers` — Model loading and tokenization
- `accelerate` — Optimizes loading on different hardware

---

## Package Summary

| Package | Version | Purpose |
|---|---|---|
| `feedparser` | Latest | RSS feed parsing |
| `newspaper4k` | Latest | Article text extraction |
| `lxml_html_clean` | Latest | newspaper4k dependency |
| `requests` | Latest | BS4 fallback scraper |
| `beautifulsoup4` | Latest | BS4 fallback scraper |
| `nltk` | Latest | English sentence tokenization |
| `indic-nlp-library` | Latest | Nepali sentence tokenization |
| `langdetect` | Latest | Language detection |
| `sentence-transformers` | Latest | Bi-encoder embedding + CrossEncoder reranking |
| `faiss-cpu` | Latest | Vector similarity search |
| `transformers` | Latest | Flan-T5 LLM loading |
| `accelerate` | Latest | Optimized model loading |
| `torch` | Latest | ML backend |
| `numpy` | Latest | Vector math |
| `sqlite3` | Built-in | Metadata storage |
| `re` | Built-in | Text cleaning |
| `math` | Built-in | Exponential decay |
