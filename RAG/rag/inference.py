# ============================================================
# inference.py — Public ask() interface.
#
# This is the only file you need to import in a notebook
# once setup is done:
#
#   from rag.inference import ask
#   ask("What is the PM doing this week?")
# ============================================================

from rag.config    import DEFAULT_TOP_K, DEFAULT_DAYS_FILTER, MIN_COSINE
from rag.retriever import retrieve
from rag.generator import build_context, generate_answer


def ask(question:    str,
        top_k:       int   = DEFAULT_TOP_K,
        days_filter: int   = DEFAULT_DAYS_FILTER,
        min_cosine:  float = MIN_COSINE) -> str | None:
    """
    Full RAG pipeline in one call:
      Question → FAISS retrieval → CrossEncoder rerank → LLM → Answer

    Args:
        question:    Natural language question in English or Nepali.
        top_k:       Number of source chunks to use as context.
        days_filter: Ignore articles older than this many days.
        min_cosine:  Drop candidates below this cosine similarity.
                     Lower = more recall, higher = more precision.

    Returns:
        Generated answer string, or None if no relevant articles found.
    """
    print(f"\n🔍 {question}\n")

    results = retrieve(question,
                       top_k=top_k,
                       days_filter=days_filter,
                       min_cosine=min_cosine)

    if not results:
        print("❌ No relevant articles found in the last "
              f"{days_filter} days.\n"
              "   Try: lower min_cosine (e.g. 0.35) or "
              "increase days_filter (e.g. 60).")
        return None

    print(f"📰 Top {len(results)} sources:")
    for r in results:
        print(f"   [{r['rerank_score']:+.2f}] {r['title']} "
              f"({r['date']})  cosine={r['cosine_score']}")

    context = build_context(results)
    print("\n🤖 Generating answer…\n")
    answer  = generate_answer(question, context)

    print(f"💬 {answer}\n")
    print("📚 Sources:")
    for i, r in enumerate(results, start=1):
        print(f"   [{i}] {r['title']}")
        print(f"        {r['url']}")
        print(f"        {r['date']} | rerank={r['rerank_score']}")

    return answer
