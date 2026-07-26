from app.retrieval.hybrid_search import hybrid_retrieve
from app.generation.prompt import build_prompt
from app.generation.llm import generate_text
from app.generation.citation import format_citations

# Threshold history: originally 0.005, validated only against the T4TSA
# audit report. Found via testing (Step 29) that short/dense documents
# like a resume produce systematically lower absolute reranker scores,
# even for correct matches (resume relevant ~0.0012 vs T4TSA relevant
# ~0.0163+). Lowered to 0.0008 - safely above both documents' irrelevant
# scores (~0.0000-0.0001) while staying below both documents' relevant
# scores. Still just 2 data points - may need revisiting with more
# document variety.
RELEVANCE_THRESHOLD = 0.0008


def answer_question(query: str, top_k: int = 5, document_ids: list[str] = None) -> dict:
    best_chunks = hybrid_retrieve(query, top_k=top_k, document_ids=document_ids)

    if not best_chunks:
        return {
            "answer": "I don't have enough information to answer that - no matching documents found, or nothing has been ingested yet.",
            "sources": "No sources.",
            "was_answered": False
        }

    top_score = best_chunks[0].distance
    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer that based on the selected document(s).",
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
