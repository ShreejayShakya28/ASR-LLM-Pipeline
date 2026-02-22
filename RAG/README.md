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
  nepal_rag.py   — Colab session flow (convert with jupytext)
```

## Colab Quick Start

```python
# Cell 1 — Install
!pip install -q -r requirements.txt

# Cell 2 — Clone
!git clone https://github.com/YOUR_USERNAME/nepal-rag.git
import sys; sys.path.insert(0, '/content/nepal-rag')

# Cell 3 — Load models (~2 min first time)
from rag.models import embedding_model

# Cell 4 — Refresh index (run once per session)
from rag.pipeline import daily_refresh
daily_refresh()

# Cell 5 — Ask
from rag.inference import ask
ask("What is the security situation in Nepal?")
ask("नेपालमा के भइरहेको छ?")
```

## Tuning

| Parameter    | Default | Effect |
|-------------|---------|--------|
| `min_cosine` | 0.45    | Lower → more recall, less precision |
| `days_filter`| 30      | Higher → older articles included |
| `top_k`      | 3       | More sources = richer context |
| `DECAY_RATE` | 0.1     | Higher → freshness matters more |


# Theory 
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

## Component Breakdown

### 1. Data Collection — RSS Scraper

**What it does:** Fetches real news articles from Nepali news websites automatically.

**How it works:** RSS (Really Simple Syndication) feeds are structured XML files that news websites publish. Each entry contains a title, URL, and publication date. We use `feedparser` to parse these feeds and `newspaper4k` to extract clean article body text from each URL.

**Key design decisions:**
- A `skip_urls` set prevents re-fetching articles already in the database
- A `0.5 second` polite delay between requests avoids overloading news servers
- Articles under 100 words are discarded as likely failed extractions or ad fragments

**News sources tested:**
- `english.onlinekhabar.com` — English, reliable, 20 entries
- `spotlightnepal.com` — English, reliable, 20 entries
- `nepalnews.com` — Mixed, 12 entries
- `newsofnepal.com` — Nepali, 12 entries
- `setopati.com` — Nepali, 5 entries

**Packages used:**
- `feedparser` — RSS parsing
- `newspaper4k` — Article text extraction (replaces deprecated `newspaper3k`)
- `lxml_html_clean` — Required dependency for `newspaper4k` on Python 3.12

---

### 2. Text Cleaning

**What it does:** Removes noise from raw article text before it enters the knowledge base.

**What gets removed:**
- URLs (`http://...`, `www....`)
- HTML tags that slipped through extraction
- Special characters (keeping punctuation)
- Lines shorter than 4 words (navigation fragments, ad remnants)
- Multiple consecutive spaces

**Why this matters:** Dirty text produces poor embeddings. If a chunk is full of "Click here", "Advertisement", "Share on Facebook" fragments, the embedding model will encode those noise signals instead of the actual article meaning — causing irrelevant retrieval results.

**Packages used:**
- `re` (Python built-in) — Regular expression cleaning

---

### 3. Chunking

**What it does:** Splits long articles into smaller overlapping pieces that fit within the embedding model's context window.

**Strategy used — Sentence-aware chunking:**

Rather than splitting blindly at a fixed character count, the chunker:
1. First tokenizes the article into individual sentences using NLTK
2. Groups sentences together until the chunk reaches ~500 tokens
3. When a chunk is full, saves it and starts a new one
4. Carries over the last few sentences into the next chunk (overlap)

**Parameters:**

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 500 tokens | Fits within MiniLM's context window |
| Overlap | 50 tokens | Preserves context at chunk boundaries |
| Split unit | Sentences | Avoids cutting mid-thought |

**Why overlap matters:** Without overlap, a sentence spanning the boundary between chunk N and chunk N+1 loses its context on one side. A query about that sentence would retrieve only half the meaning. Overlap ensures continuity.

**Token counting:** We approximate tokens as `words / 0.75` since most English words tokenize to roughly 1.3 subword tokens.

**Packages used:**
- `nltk` — Sentence tokenization (`sent_tokenize`)

---

### 4. Embedding

**What it does:** Converts each text chunk into a 384-dimensional numerical vector that captures its semantic meaning.

**Model used:** `all-MiniLM-L6-v2` from Sentence Transformers

**Why this model:**
- Only ~80MB in size
- Runs fast on CPU
- Produces 384-dimensional vectors (small enough for fast search)
- Strong semantic similarity performance despite its small size
- Available for free via HuggingFace

**How it works:** The model is a distilled version of a much larger BERT model, fine-tuned specifically on semantic similarity tasks. Two sentences with similar meanings will produce vectors that are geometrically close to each other in 384-dimensional space, even if the exact words are different.

For example:
- "PM Karki directed election security" 
- "Prime Minister tightens polling arrangements"

These produce very similar vectors despite sharing almost no words.

**Normalization:** All vectors are L2-normalized before storage so that inner product (dot product) is equivalent to cosine similarity. This is required for `faiss.IndexFlatIP`.

**Packages used:**
- `sentence-transformers` — The embedding library
- `torch` — Backend for model inference
- `numpy` — Vector array operations

---

### 5. Vector Storage — FAISS

**What it does:** Stores all chunk embeddings in a fast searchable index on disk.

**What is FAISS:** Facebook AI Similarity Search is a library for efficient nearest-neighbor search in high-dimensional vector spaces. Given a query vector, it finds the most similar stored vectors in milliseconds — even across millions of entries.

**Index type used:** `IndexFlatIP` (Flat Inner Product)
- "Flat" means exact search — checks every vector, no approximation
- "IP" means Inner Product — equivalent to cosine similarity after normalization
- Best choice for small-to-medium datasets (under 100k vectors)

**Similarity formula:**
```
similarity(query, chunk) = query · chunk / (|query| × |chunk|)
```

After L2 normalization, this simplifies to just the dot product.

**Storage layout:**
```
/content/rag_index/
├── news.faiss    ← vector index (binary, ~1.5KB per chunk)
└── metadata.db   ← SQLite database (text, title, URL, date)
```

FAISS only stores vectors — not the original text. The SQLite database provides the parallel lookup: FAISS returns index numbers, SQLite maps those numbers back to readable content.

**Packages used:**
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

**Result:** The reranker correctly demoted the Nepal-India economic cooperation article when asked about security, pushing the actual security articles to the top.

**Packages used:**
- `sentence-transformers` (CrossEncoder class)
- `faiss-cpu`
- `sqlite3`
- `math` (for exponential decay)

---

**Generation parameters:**


| Parameter          | Value | Reason                                 |
| ------------------ | ----- | -------------------------------------- |
| max_new_tokens     | 150   | Keeps answers concise                  |
| temperature        | 0.1   | Very low — factual, deterministic      |
| do_sample          | False | Greedy decoding — most reliable output |
| repetition_penalty | 1.5   | Prevents looping/repeating phrases     |

**Packages used:**
- `transformers` — Model loading and tokenization
- `accelerate` — Optimizes loading on different hardware

---

### 8. Daily Refresh + Scheduler

**What it does:** Automatically keeps the knowledge base fresh by scraping new articles every 24 hours.

**Incremental update strategy:**
1. Load all URLs already stored in SQLite
2. Scrape feeds but skip any URL already seen
3. Chunk + embed only the new articles
4. **Append** new vectors to the existing FAISS index — never rebuild from scratch
5. Insert new metadata into SQLite

This is critical for a news system. Rebuilding the entire index daily would be wasteful and slow. Appending is instant regardless of how large the existing index is.

**Scheduler:** Runs on a Python `threading.Thread` as a daemon in the background. Colab stays interactive while the scheduler runs silently, waking every 24 hours to refresh.

## Future Improvements

| Improvement | Impact | Difficulty |
|---|---|---|
| Upgrade to Mistral 7B | Much better answer quality | Medium |
| Multilingual embedding model | Nepali article support | Easy |
| BM25 hybrid search | Better keyword matching | Medium |
| Simple Gradio UI | User-friendly interface | Easy |
| Persistent deployment (VM) | 24/7 availability | Medium |
| Query caching | Faster repeated queries | Easy |

---

## Package Summary

| Package | Version | Purpose |
|---|---|---|
| `feedparser` | Latest | RSS feed parsing |
| `newspaper4k` | Latest | Article text extraction |
| `lxml_html_clean` | Latest | newspaper4k dependency |
| `nltk` | Latest | Sentence tokenization |
| `sentence-transformers` | Latest | Bi-encoder embedding + CrossEncoder reranking |
| `faiss-cpu` | Latest | Vector similarity search |
| `transformers` | Latest | Flan-T5 LLM loading |
| `accelerate` | Latest | Optimized model loading |
| `torch` | Latest | ML backend |
| `numpy` | Latest | Vector math |
| `sqlite3` | Built-in | Metadata storage |
| `re` | Built-in | Text cleaning |
| `math` | Built-in | Exponential decay |
| `threading` | Built-in | Background scheduler |

---

