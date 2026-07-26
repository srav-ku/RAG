from app.retrieval.hybrid_search import hybrid_retrieve
from app.generation.prompt import build_prompt
from app.generation.llm import generate_text
from app.generation.citation import format_citations

# Threshold validated against real test data (Step 20): relevant questions
# scored 0.0163-0.1910, irrelevant scored 0.0000-0.0001 on the reranker scale.
RELEVANCE_THRESHOLD = 0.005


def answer_question(query: str, top_k: int = 5) -> dict:
    """
    Full query pipeline: hybrid retrieve (semantic + keyword + rerank),
    relevance check, grounded generation, citations.
    """
    best_chunks = hybrid_retrieve(query, top_k=top_k)

    if not best_chunks:
        return {
            "answer": "I don't have enough information to answer that - no documents have been ingested yet, or nothing matched your question.",
            "sources": "No sources.",
            "was_answered": False
        }

    top_score = best_chunks[0].distance
    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer that based on the uploaded documents.",
            "sources": "No sources.",
            "was_answered": False
        }

    prompt = build_prompt(query, best_chunks)
    answer = generate_text(prompt)
    sources = format_citations(best_chunks)

    return {
        "answer": answer,
        "sources": sources,
        "was_answered": True
    }
