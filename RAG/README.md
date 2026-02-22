# Nepal News RAG

Retrieval-Augmented Generation over live Nepali news feeds.
Supports both English and Nepali (Devanagari) articles.

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

## Key Design Decisions

### Why no background scheduler?
Colab free-tier runtimes disconnect after ~90 min of inactivity
(12 hr max). A `threading.Thread` with `time.sleep(86400)` is
dead code in Colab. Just call `daily_refresh()` at session start —
it skips URLs already indexed so repeated calls are fast.

### Staleness handling
Two mechanisms work together:
1. **Time-decay score** — `e^(-0.1 * days_old)` exponentially
   down-weights older chunks in the blended retrieval score.
2. **`days_filter`** (default 30) — hard cut that drops anything
   older than N days before reranking.

### Scraping strategy
1. `newspaper4k` — best text extraction quality.
2. `requests + BeautifulSoup` — fallback for sites that block
   newspaper4k (arthasarokar, setopati, etc.).

### Nepali tokenization
`langdetect` identifies Nepali text; `indic-nlp-library` then
splits it into sentences correctly. English falls back to NLTK.
Both degrade gracefully if the library is missing.

## Tuning

| Parameter    | Default | Effect |
|-------------|---------|--------|
| `min_cosine` | 0.45    | Lower → more recall, less precision |
| `days_filter`| 30      | Higher → older articles included |
| `top_k`      | 3       | More sources = richer context |
| `DECAY_RATE` | 0.1     | Higher → freshness matters more |
