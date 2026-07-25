from app.retrieval.retriever import retrieve
from app.retrieval.reranker import rerank
from app.generation.prompt import build_prompt
from app.generation.llm import generate_text
from app.generation.citation import format_citations

# If the top reranked chunk scores below this, we treat it as "not relevant enough"
# and skip calling the LLM entirely, rather than risk a hallucinated answer.
RELEVANCE_THRESHOLD = 0.1


def answer_question(query: str, top_k: int = 5, rerank_top_n: int = 3) -> dict:
    """
    Full query pipeline: retrieve, rerank, check relevance, generate a
    grounded answer, and attach citations.

    Returns a dict with: answer, sources, was_answered (bool)
    """
    # Step 1: initial broad retrieval
    initial_results = retrieve(query, top_k=top_k)

    if not initial_results:
        return {
            "answer": "I don't have enough information to answer that - no documents have been ingested yet, or nothing matched your question.",
            "sources": "No sources.",
            "was_answered": False
        }

    # Step 2: rerank for precision
    reranked_results = rerank(query, initial_results)

    # Step 3: relevance check - the deterministic hallucination guard
    top_score = reranked_results[0].distance  # remember: after reranking, higher = more relevant
    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer that based on the uploaded documents.",
            "sources": "No sources.",
            "was_answered": False
        }

    # Step 4: keep only the top N most relevant chunks for the prompt
    # (we retrieved more than we need, to give the reranker good options to choose from)
    best_chunks = reranked_results[:rerank_top_n]

    # Step 5: build the grounded prompt and generate
    prompt = build_prompt(query, best_chunks)
    answer = generate_text(prompt)

    # Step 6: format citations from the same chunks we actually used
    sources = format_citations(best_chunks)

    return {
        "answer": answer,
        "sources": sources,
        "was_answered": True
    }
