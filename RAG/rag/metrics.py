# ============================================================
# metrics.py — Evaluation framework for RAG comparison.
#
# Measures five dimensions for each (model × rag_mode) combo:
#   1. Faithfulness      — n_supported_sentences / n_total_sentences
#   2. Answer Relevance  — cosine_sim(answer, question)
#   3. Context Relevance — |relevant_chunks| / |total_chunks|
#   4. Latency           — embed + search + prompt_build + generate (s)
#   5. Coverage          — answered vs refused rate
#
# Usage:
#   from rag.metrics import run_eval, print_report, save_report
#   results = run_eval(eval_questions)
#   print_report(results)
#   save_report(results, "/drive/MyDrive/eval_report.json")
# ============================================================

import time
import json
import datetime
import re
import numpy as np
from typing import Optional

# ── Lazy imports so metrics.py loads even before models are ready ──
_embedding_model = None
_SLM_PATH_ADDED  = False          # guard so sys.path is mutated only once

def _get_embedder():
    global _embedding_model
    if _embedding_model is None:
        from rag.models import embedding_model
        _embedding_model = embedding_model
    return _embedding_model

def _ensure_slm_path():
    """Add SLM dir to sys.path exactly once."""
    global _SLM_PATH_ADDED
    if not _SLM_PATH_ADDED:
        import sys
        SLM_DIR = "/content/ASR-LLM-Pipeline/SLM"
        if SLM_DIR not in sys.path:
            sys.path.insert(0, SLM_DIR)
        _SLM_PATH_ADDED = True


# ──────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────

def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    a = vec_a / (np.linalg.norm(vec_a) + 1e-10)
    b = vec_b / (np.linalg.norm(vec_b) + 1e-10)
    return float(np.dot(a, b))


def _split_sentences(text: str) -> list:
    """
    Lightweight sentence splitter that works without NLTK.
    Splits on '.', '!', '?' followed by whitespace or end-of-string.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# ──────────────────────────────────────────────────────────────────
# Metric 1 — Faithfulness
# n_supported / n_total  (academic RAGAS definition)
#
# A sentence in the answer is "supported" if its cosine similarity
# to the BEST matching source chunk exceeds FAITH_THRESHOLD.
# This approximates the LLM-judge approach without API cost.
# ──────────────────────────────────────────────────────────────────

FAITH_THRESHOLD = 0.30   # sentences above this are considered grounded


def faithfulness_score(answer: str, source_chunks: list) -> float:
    """
    Faithfulness = n_supported_sentences / n_total_sentences.

    Each sentence of the answer is embedded and compared against every
    source chunk; it is "supported" if max cosine ≥ FAITH_THRESHOLD.

    Args:
        answer:        Generated answer string.
        source_chunks: List of retrieval result dicts (must have 'text' key).

    Returns:
        Float in [0, 1].  1.0 = every sentence is grounded in sources.
    """
    if not answer or not source_chunks:
        return 0.0

    embedder   = _get_embedder()
    sentences  = _split_sentences(answer)
    if not sentences:
        return 0.0

    # Pre-encode all chunks once
    chunk_texts = [c.get("text", "") for c in source_chunks if c.get("text", "")]
    if not chunk_texts:
        return 0.0
    chunk_vecs = embedder.encode(chunk_texts, normalize_embeddings=True,
                                 batch_size=16, show_progress_bar=False)

    supported = 0
    for sent in sentences:
        sent_vec = embedder.encode(sent, normalize_embeddings=True)
        best     = max(_cosine(sent_vec, cv) for cv in chunk_vecs)
        if best >= FAITH_THRESHOLD:
            supported += 1

    return round(supported / len(sentences), 4)


# ──────────────────────────────────────────────────────────────────
# Metric 2 — Answer Relevance
# cosine_sim(answer_embedding, question_embedding)
# ──────────────────────────────────────────────────────────────────

def answer_relevance_score(answer: str, question: str) -> float:
    """
    AnswerRelevance(A, q) = cosine_sim(embed(A), embed(q)).

    Measures whether the answer addresses the question, regardless of
    factual correctness.  High score = on-topic answer.

    Returns float in [-1, 1]; typically 0.4–0.9 for good answers.
    """
    if not answer or not question:
        return 0.0
    embedder = _get_embedder()
    q_vec = embedder.encode(question, normalize_embeddings=True)
    a_vec = embedder.encode(answer,   normalize_embeddings=True)
    return round(_cosine(q_vec, a_vec), 4)


# ──────────────────────────────────────────────────────────────────
# Metric 3 — Context Relevance
# |relevant_chunks| / |total_chunks|
#
# A chunk is "relevant" if cosine_sim(chunk, question) ≥ CONTEXT_THRESHOLD.
# ──────────────────────────────────────────────────────────────────

CONTEXT_THRESHOLD = 0.30   # chunks above this are considered relevant


def context_relevance_score(question: str, source_chunks: list) -> float:
    """
    ContextRelevance(C, q) = |C_relevant| / |C|.

    Measures retrieval precision — how much of the pulled context is
    actually about the question.  Low score = over-retrieval / noise.

    Returns float in [0, 1].
    """
    if not source_chunks:
        return 0.0
    embedder  = _get_embedder()
    q_vec     = embedder.encode(question, normalize_embeddings=True)

    relevant = 0
    for chunk in source_chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        c_vec = embedder.encode(text, normalize_embeddings=True)
        if _cosine(q_vec, c_vec) >= CONTEXT_THRESHOLD:
            relevant += 1

    return round(relevant / len(source_chunks), 4)


# ──────────────────────────────────────────────────────────────────
# Single-question runners (one per condition)
# Each runner returns a standardised dict with ALL metric fields.
# ──────────────────────────────────────────────────────────────────

def _make_empty_result(question, t_embed=0.0, t_search=0.0,
                       t_prompt=0.0, t_generate=0.0) -> dict:
    """Return a no-answer result with zero metrics."""
    return {
        "answered":          False,
        "answer":            None,
        "faithfulness":      0.0,
        "answer_relevance":  0.0,
        "context_relevance": 0.0,
        "latency_embed":     t_embed,
        "latency_search":    t_search,
        "latency_prompt":    t_prompt,
        "latency_generate":  t_generate,
        "latency_total":     t_embed + t_search + t_prompt + t_generate,
        "sources":           [],
        "question":          question,
    }


def _run_flan_rag(question: str, top_k: int, days_filter: int,
                  min_cosine: float) -> dict:
    """Flan-T5 + RAG retrieval — four-stage latency breakdown."""
    from rag.retriever import retrieve
    from rag.generator import build_context, generate_answer
    from rag.models    import embedding_model

    # ── Stage 1: embed query ──────────────────────────────────────
    t0      = time.perf_counter()
    _       = embedding_model.encode(question, normalize_embeddings=True)
    t_embed = time.perf_counter() - t0

    # ── Stage 2: FAISS + rerank search ───────────────────────────
    t1      = time.perf_counter()
    results = retrieve(question, top_k=top_k,
                       days_filter=days_filter, min_cosine=min_cosine)
    t_search = time.perf_counter() - t1

    if not results:
        return _make_empty_result(question, t_embed, t_search)

    # ── Stage 3: build context prompt ────────────────────────────
    t2      = time.perf_counter()
    context = build_context(results)
    t_prompt = time.perf_counter() - t2

    # ── Stage 4: LLM generate ────────────────────────────────────
    t3      = time.perf_counter()
    answer  = generate_answer(question, context)
    t_gen   = time.perf_counter() - t3

    faith   = faithfulness_score(answer, results)
    ar      = answer_relevance_score(answer, question)
    cr      = context_relevance_score(question, results)

    return {
        "answered":          True,
        "answer":            answer,
        "faithfulness":      faith,
        "answer_relevance":  ar,
        "context_relevance": cr,
        "latency_embed":     round(t_embed,   3),
        "latency_search":    round(t_search,  3),
        "latency_prompt":    round(t_prompt,  3),
        "latency_generate":  round(t_gen,     3),
        "latency_total":     round(t_embed + t_search + t_prompt + t_gen, 3),
        "sources":           [{"title": r["title"], "url": r["url"],
                               "date": r["date"]} for r in results],
        "question":          question,
    }


def _run_flan_norag(question: str) -> dict:
    """Flan-T5 without retrieval — raw model knowledge only."""
import rag.models as _m
tokenizer = _m.tokenizer
llm       = _m.llm
    from rag.config import MAX_NEW_TOKENS

    # embed stage still happens conceptually (we just skip search)
    embedder = _get_embedder()
    t0       = time.perf_counter()
    _        = embedder.encode(question, normalize_embeddings=True)
    t_embed  = time.perf_counter() - t0

    # prompt build
    t2      = time.perf_counter()
    prompt  = f"Answer this question about Nepal in 2-3 sentences:\n{question}"
    inputs  = tokenizer(prompt, return_tensors="pt",
                        max_length=512, truncation=True)
    t_prompt = time.perf_counter() - t2

    # generate
    t3 = time.perf_counter()
    outputs = llm.generate(
        inputs["input_ids"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.5,
    )
    answer  = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    t_gen   = time.perf_counter() - t3

    # no source chunks → faithfulness and context_relevance are 0
    ar = answer_relevance_score(answer, question)

    return {
        "answered":          bool(answer),
        "answer":            answer,
        "faithfulness":      0.0,   # no sources to be faithful to
        "answer_relevance":  ar,
        "context_relevance": 0.0,   # no context retrieved
        "latency_embed":     round(t_embed,  3),
        "latency_search":    0.0,
        "latency_prompt":    round(t_prompt, 3),
        "latency_generate":  round(t_gen,    3),
        "latency_total":     round(t_embed + t_prompt + t_gen, 3),
        "sources":           [],
        "question":          question,
    }


def _run_slm_rag(question: str, slm_model, slm_config,
                 device: str, top_k: int, days_filter: int,
                 min_cosine: float) -> dict:
    """Custom GPT-2 SLM + RAG retrieval."""
    _ensure_slm_path()
    from inference     import run_inference as slm_infer
    from rag.retriever import retrieve
    from rag.generator import build_context
    from rag.config    import CONTEXT_CHARS
    from rag.models    import embedding_model

    # Stage 1: embed
    t0      = time.perf_counter()
    _       = embedding_model.encode(question, normalize_embeddings=True)
    t_embed = time.perf_counter() - t0

    # Stage 2: search
    t1       = time.perf_counter()
    results  = retrieve(question, top_k=top_k,
                        days_filter=days_filter, min_cosine=min_cosine)
    t_search = time.perf_counter() - t1

    if not results:
        return _make_empty_result(question, t_embed, t_search)

    # Stage 3: build prompt
    t2          = time.perf_counter()
    context     = build_context(results)
    instruction = (
        "Based on the following Nepal news articles, answer the question "
        "in 2-3 sentences using only the provided sources."
    )
    input_text  = f"Articles:\n{context[:CONTEXT_CHARS]}\n\nQuestion: {question}"
    t_prompt    = time.perf_counter() - t2

    # Stage 4: generate
    t3     = time.perf_counter()
    answer = slm_infer(
        model=slm_model, config=slm_config,
        instruction=instruction, input_text=input_text,
        device=device, max_new_tokens=256,
    )
    t_gen  = time.perf_counter() - t3

    faith = faithfulness_score(answer, results)
    ar    = answer_relevance_score(answer, question)
    cr    = context_relevance_score(question, results)

    return {
        "answered":          bool(answer and answer.strip()),
        "answer":            answer,
        "faithfulness":      faith,
        "answer_relevance":  ar,
        "context_relevance": cr,
        "latency_embed":     round(t_embed,   3),
        "latency_search":    round(t_search,  3),
        "latency_prompt":    round(t_prompt,  3),
        "latency_generate":  round(t_gen,     3),
        "latency_total":     round(t_embed + t_search + t_prompt + t_gen, 3),
        "sources":           [{"title": r["title"], "url": r["url"],
                               "date": r["date"]} for r in results],
        "question":          question,
    }


def _run_slm_norag(question: str, slm_model, slm_config,
                   device: str) -> dict:
    """Custom GPT-2 SLM without retrieval."""
    _ensure_slm_path()
    from inference  import run_inference as slm_infer

    embedder = _get_embedder()
    t0       = time.perf_counter()
    _        = embedder.encode(question, normalize_embeddings=True)
    t_embed  = time.perf_counter() - t0

    t3     = time.perf_counter()
    answer = slm_infer(
        model=slm_model, config=slm_config,
        instruction=question, input_text="",
        device=device, max_new_tokens=256,
    )
    t_gen  = time.perf_counter() - t3

    ar = answer_relevance_score(answer, question)

    return {
        "answered":          bool(answer and answer.strip()),
        "answer":            answer,
        "faithfulness":      0.0,
        "answer_relevance":  ar,
        "context_relevance": 0.0,
        "latency_embed":     round(t_embed, 3),
        "latency_search":    0.0,
        "latency_prompt":    0.0,
        "latency_generate":  round(t_gen,   3),
        "latency_total":     round(t_embed + t_gen, 3),
        "sources":           [],
        "question":          question,
    }


# ──────────────────────────────────────────────────────────────────
# Main eval runner
# ──────────────────────────────────────────────────────────────────

CONDITIONS = ["flan_rag", "flan_norag", "slm_rag", "slm_norag"]


def run_eval(
    questions:   list,
    slm_model    = None,
    slm_config   = None,
    slm_device:  str   = "cuda",
    top_k:       int   = 5,
    days_filter: int   = 730,
    min_cosine:  float = 0.45,
    conditions:  list  = None,
) -> dict:
    """
    Run all conditions over every question and collect metrics.

    Args:
        questions:   List of question strings.
        slm_model:   Loaded GPT-2 model (required for slm_* conditions).
        slm_config:  Config dict from load_model() (required for slm_*).
        slm_device:  'cuda' or 'cpu'.
        top_k:       Chunks retrieved per question.
        days_filter: Days lookback window.
        min_cosine:  Minimum cosine threshold.
        conditions:  Subset of CONDITIONS to run (default = all four).

    Returns:
        Nested dict: results[condition][question] = metric_dict
    """
    if conditions is None:
        conditions = CONDITIONS

    results = {c: {} for c in conditions}
    n = len(questions)

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{n}] {q[:80]}")

        for cond in conditions:
            try:
                if cond == "flan_rag":
                    r = _run_flan_rag(q, top_k, days_filter, min_cosine)
                elif cond == "flan_norag":
                    r = _run_flan_norag(q)
                elif cond == "slm_rag":
                    r = _run_slm_rag(q, slm_model, slm_config,
                                     slm_device, top_k, days_filter, min_cosine)
                elif cond == "slm_norag":
                    r = _run_slm_norag(q, slm_model, slm_config, slm_device)
                else:
                    raise ValueError(f"Unknown condition: {cond}")

                results[cond][q] = r

            except Exception as e:
                print(f"  💥 [{cond}] ERROR: {e}")
                results[cond][q] = {
                    **_make_empty_result(q),
                    "error": str(e),
                }

    return results


# ──────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────

def _safe_mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(float(np.mean(vals)), 4) if vals else 0.0


def _agg(results: dict, cond: str) -> dict:
    rows     = list(results.get(cond, {}).values())
    if not rows:
        return {}

    # Exclude errored rows from latency averages
    clean    = [r for r in rows if "error" not in r]
    answered = [r for r in rows if r.get("answered")]

    coverage = len(answered) / len(rows) * 100 if rows else 0.0

    avg_words = (
        float(np.mean([len((r.get("answer") or "").split()) for r in answered]))
        if answered else 0.0
    )

    return {
        "coverage_pct":              round(coverage, 1),
        "answered":                  len(answered),
        "total":                     len(rows),
        # Faithfulness: only meaningful for RAG conditions
        "faithfulness_mean":         _safe_mean([r["faithfulness"]      for r in answered]),
        # Answer relevance: all answered rows
        "answer_relevance_mean":     _safe_mean([r["answer_relevance"]  for r in answered]),
        # Context relevance: RAG conditions only (0 for no-RAG)
        "context_relevance_mean":    _safe_mean([r["context_relevance"] for r in answered]),
        # Latency breakdown (exclude errored rows)
        "latency_embed_mean":        _safe_mean([r["latency_embed"]     for r in clean]),
        "latency_search_mean":       _safe_mean([r["latency_search"]    for r in clean]),
        "latency_prompt_mean":       _safe_mean([r["latency_prompt"]    for r in clean]),
        "latency_generate_mean":     _safe_mean([r["latency_generate"]  for r in clean]),
        "latency_total_mean":        _safe_mean([r["latency_total"]     for r in clean]),
        "avg_answer_words":          round(avg_words, 1),
    }


def aggregate(results: dict) -> dict:
    return {cond: _agg(results, cond)
            for cond in results if results[cond]}


# ──────────────────────────────────────────────────────────────────
# Print report (kept for quick Colab inspection)
# ──────────────────────────────────────────────────────────────────

_LABEL = {
    "flan_rag":   "Flan-T5  + RAG",
    "flan_norag": "Flan-T5  (no RAG)",
    "slm_rag":    "SLM GPT2 + RAG",
    "slm_norag":  "SLM GPT2 (no RAG)",
}


def print_report(results: dict) -> None:
    agg      = aggregate(results)
    active   = [c for c in results if results[c]]   # only conditions with data
    W        = 90

    print("\n" + "═" * W)
    print("  RAG EVALUATION REPORT — Nepal News System")
    print("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("═" * W)

    header = (f"  {'Model':24s} {'Cover%':>7} {'Faith':>7} "
              f"{'AnsRel':>7} {'CtxRel':>7} "
              f"{'T-emb':>7} {'T-srch':>7} {'T-gen':>7} {'T-tot':>7} {'Words':>6}")
    print(header)
    print("  " + "─" * (W - 2))

    for cond in active:
        s = agg[cond]
        if not s:
            continue
        label = _LABEL.get(cond, cond)
        print(
            f"  {label:24s}"
            f"{s['coverage_pct']:>7.1f}"
            f"{s['faithfulness_mean']:>7.3f}"
            f"{s['answer_relevance_mean']:>7.3f}"
            f"{s['context_relevance_mean']:>7.3f}"
            f"{s['latency_embed_mean']:>7.2f}"
            f"{s['latency_search_mean']:>7.2f}"
            f"{s['latency_generate_mean']:>7.2f}"
            f"{s['latency_total_mean']:>7.2f}"
            f"{s['avg_answer_words']:>6.0f}"
        )

    print("  " + "─" * (W - 2))
    print("\n  Metric guide:")
    print("  • Cover%  — % questions answered (not refused)")
    print("  • Faith   — n_supported_sentences / n_total  (grounding in sources)")
    print("  • AnsRel  — cosine_sim(answer, question)     (on-topic-ness)")
    print("  • CtxRel  — relevant_chunks / total_chunks   (retrieval precision)")
    print("  • T-emb/srch/gen/tot — latency per stage (seconds)")
    print("  • Words   — average answer word count")
    print()

    # Per-question faithfulness + answer_relevance
    all_q = list(next(iter(results.values())).keys())
    col_w = max(45, min(60, max(len(q) for q in all_q) + 2))
    print(f"\n  {'Question':{col_w}}", end="")
    for cond in active:
        print(f"  {_LABEL.get(cond, cond)[:16]:>16}", end="")
    print()
    print("  " + "─" * (col_w + 18 * len(active)))

    for q in all_q:
        short_q = (q[:col_w - 2] + "..") if len(q) > col_w else q
        print(f"  {short_q:{col_w}}", end="")
        for cond in active:
            row = results[cond].get(q, {})
            if row.get("answered"):
                tag = f"{row['faithfulness']:.2f}/{row['answer_relevance']:.2f}"
            else:
                tag = "N/A"
            print(f"  {tag:>16}", end="")
        print()

    print()


# ──────────────────────────────────────────────────────────────────
# Save / load JSON
# ──────────────────────────────────────────────────────────────────

def save_report(results: dict, path: str) -> None:
    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "aggregate":    aggregate(results),
        "per_question": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Report saved → {path}")


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
