from rank_bm25 import BM25Okapi
from app.retrieval.vector_store import get_all_chunks_for_documents


def build_bm25_index(document_ids: list[str] = None):
    """Builds a BM25 index from chunks, optionally scoped to specific documents."""
    all_data = get_all_chunks_for_documents(document_ids)

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    if not documents:
        return None, [], []

    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return bm25, documents, metadatas


def keyword_search(query: str, top_k: int = 10, document_ids: list[str] = None) -> list[dict]:
    """Searches chunks using BM25, optionally scoped to specific documents."""
    bm25, documents, metadatas = build_bm25_index(document_ids)

    if bm25 is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_results = list(zip(documents, metadatas, scores))
    scored_results.sort(key=lambda x: x[2], reverse=True)

    top_results = scored_results[:top_k]

    return [
        {"text": text, "metadata": meta, "bm25_score": float(score)}
        for text, meta, score in top_results
    ]
