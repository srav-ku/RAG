from app.retrieval.retriever import retrieve, RetrievedChunk
from app.retrieval.keyword_search import keyword_search
from app.retrieval.reranker import rerank


def hybrid_retrieve(query: str, top_k: int = 10, rrf_k: int = 60) -> list[RetrievedChunk]:
    """
    Combines semantic + keyword search via RRF to build a candidate pool,
    then uses the cross-encoder reranker for final, accurate ordering.
    RRF here is just for building a good candidate SET, not final ranking.
    """
    semantic_results = retrieve(query, top_k=top_k * 2)
    keyword_results = keyword_search(query, top_k=top_k * 2)

    chunk_lookup = {}
    rrf_scores = {}

    for rank, chunk in enumerate(semantic_results):
        key = f"{chunk.document_id}_{chunk.chunk_index}"
        chunk_lookup[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank + 1)

    for rank, result in enumerate(keyword_results):
        meta = result["metadata"]
        key = f"{meta.get('document_id', '')}_{meta.get('chunk_index', -1)}"
        if key not in chunk_lookup:
            chunk_lookup[key] = RetrievedChunk(
                text=result["text"], distance=0.0,
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", ""),
                page_numbers=meta.get("page_numbers", ""),
                chunk_index=meta.get("chunk_index", -1)
            )
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank + 1)

    # Union of candidates from both methods - this is our pool, not final order
    candidate_keys = list(rrf_scores.keys())
    candidates = [chunk_lookup[k] for k in candidate_keys]

    # Let the cross-encoder make the real final judgment
    final_ranked = rerank(query, candidates)
    return final_ranked[:top_k]
