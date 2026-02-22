# ============================================================
# nepal_rag.ipynb  (represented as .py for version control)
# Convert to notebook: jupytext --to notebook nepal_rag.py
#
# SESSION FLOW (run top to bottom each time you open Colab):
#   Cell 1 — install deps
#   Cell 2 — clone repo & set path
#   Cell 3 — refresh index with today's news
#   Cell 4 — ask questions
# ============================================================

# %% [markdown]
# # Nepal News RAG
# Run all cells top-to-bottom each session. The refresh cell
# skips articles already indexed, so re-running is safe and fast.

# %% Cell 1 — Install
# -----------------------------------------------------------------
# Run once per Colab session (packages reset on reconnect).
# -----------------------------------------------------------------
# !pip install -q -r requirements.txt
# print("✅ Dependencies installed")

# %% Cell 2 — Clone repo + add to path
# -----------------------------------------------------------------
# !git clone https://github.com/YOUR_USERNAME/nepal-rag.git
# import sys
# sys.path.insert(0, '/content/nepal-rag')
# print("✅ Repo on path")

# %% Cell 3 — Load models (heavy, ~2 min; cached after first run)
# -----------------------------------------------------------------
# Models are loaded when rag.models is first imported.
# This import triggers the prints you see below.
# -----------------------------------------------------------------
# from rag.models import embedding_model, reranker, tokenizer, llm
# print("✅ Models ready")

# %% Cell 4 — Daily refresh (run once per session)
# -----------------------------------------------------------------
# Scrapes new articles, skips already-indexed URLs,
# embeds + appends to FAISS. Fast after the first run.
#
# NO background scheduler — Colab runtimes disconnect too quickly.
# Just run this cell when you open the notebook.
# -----------------------------------------------------------------
# from rag.pipeline import daily_refresh
# daily_refresh()   # uses ALL_CANDIDATE_FEEDS from config.py

# %% Cell 5 — Ask questions
# -----------------------------------------------------------------
# from rag.inference import ask
#
# ask("What is happening with Nepal elections?")
# ask("What is Prime Minister Karki doing?")
# ask("What is the security situation in Nepal?")
# ask("नेपालमा के भइरहेको छ?")   # Nepali works too

# %% Cell 6 — Tune retrieval (optional)
# -----------------------------------------------------------------
# If you get "No relevant articles found", try widening the net:
#
# ask("Nepal economy", min_cosine=0.35, days_filter=60)
#
# If answers are off-topic, tighten it:
#
# ask("Nepal economy", min_cosine=0.55, days_filter=7)
