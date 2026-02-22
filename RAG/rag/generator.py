# ============================================================
# generator.py — Build a context string and run the LLM.
# ============================================================

from rag.config import MAX_NEW_TOKENS, CONTEXT_CHARS
from rag.models import tokenizer, llm


def build_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered source block
    that the LLM can cite in its answer.
    """
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"Title  : {r['title']}\n"
            f"Date   : {r['date']}\n"
            f"URL    : {r['url']}\n"
            f"Content: {r['text']}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(question: str,
                    context:  str,
                    max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Feed the question + top-N context snippets into Flan-T5
    and return a grounded 2-3 sentence answer.
    """
    prompt = (
        "Based on these Nepal news articles:\n\n"
        f"{context[:CONTEXT_CHARS]}\n\n"
        "Answer in 2-3 sentences using only the articles above:\n"
        f"{question}"
    )
    inputs  = tokenizer(prompt, return_tensors="pt",
                        max_length=512, truncation=True)
    outputs = llm.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.5,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
