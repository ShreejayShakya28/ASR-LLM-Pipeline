# ============================================================
# models.py — Load all ML models once.
# Import `embedding_model`, `reranker`, `tokenizer`, `llm`
# from here everywhere else.
# ============================================================

print("📦 Loading embedding model...")
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rag.config import EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"   ✅ {EMBEDDING_MODEL}")

print("📦 Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)
print(f"   ✅ {RERANKER_MODEL}")

print("📦 Loading LLM (this takes ~1 min)...")
tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
llm       = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
print(f"   ✅ {LLM_MODEL}")

print("\n✅ All models ready.")
