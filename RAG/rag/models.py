# ============================================================
# models.py — Load retrieval models only.
# Import `embedding_model`, `reranker` from here everywhere.
# ============================================================

from sentence_transformers import SentenceTransformer, CrossEncoder
from rag.config import EMBEDDING_MODEL, RERANKER_MODEL

print("📦 Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"   ✅ {EMBEDDING_MODEL}")

print("📦 Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)
print(f"   ✅ {RERANKER_MODEL}")

# Kept as None so any existing file that imports these won't crash
tokenizer = None
llm       = None

print("\n✅ Retrieval models ready.")
