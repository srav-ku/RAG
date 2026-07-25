from app.retrieval.retriever import retrieve
from app.retrieval.reranker import rerank
from app.generation.prompt import build_prompt
from app.generation.llm import generate_text
from app.generation.citation import format_citations

# Threshold validated against real test data (Step 20 investigation):
# relevant questions scored 0.0163-0.1910, irrelevant scored 0.0000-0.0001
# on our BAAI/bge-reranker-base model. 0.005 sits clearly in the gap.
# NOTE: this is tuned to this specific reranker model and our current
# small test document - revisit if either changes significantly.
RELEVANCE_THRESHOLD = 0.005


def answer_question(query: str, top_k: int = 5, rerank_top_n: int = 3) -> dict:
    """
    Full query pipeline: retrieve, rerank, check relevance, generate a
    grounded answer, and attach citations.
    """
    initial_results = retrieve(query, top_k=top_k)

    if not initial_results:
        return {
            "answer": "I don't have enough information to answer that - no documents have been ingested yet, or nothing matched your question.",
            "sources": "No sources.",
            "was_answered": False
        }

    reranked_results = rerank(query, initial_results)

    top_score = reranked_results[0].distance
    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer that based on the uploaded documents.",
            "sources": "No sources.",
            "was_answered": False
        }

    best_chunks = reranked_results[:rerank_top_n]

    prompt = build_prompt(query, best_chunks)
    answer = generate_text(prompt)
    sources = format_citations(best_chunks)

    return {
        "answer": answer,
        "sources": sources,
        "was_answered": True
    }
