from rank_bm25 import BM25Okapi
from app.retrieval.vector_store import get_collection


def build_bm25_index():
    """
    Builds a BM25 keyword search index from every chunk currently in ChromaDB.
    Returns the index plus the matching documents/metadatas, so results can
    be mapped back to full chunk info afterward.
    """
    collection = get_collection()
    all_data = collection.get()

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    # BM25 expects each document pre-split into words ("tokenized")
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return bm25, documents, metadatas


def keyword_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Searches all stored chunks using BM25 keyword matching (not embeddings).
    Returns a list of dicts with text, metadata, and bm25 score.
    """
    bm25, documents, metadatas = build_bm25_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair each document with its score, sort by score descending
    scored_results = list(zip(documents, metadatas, scores))
    scored_results.sort(key=lambda x: x[2], reverse=True)

    top_results = scored_results[:top_k]

    return [
        {"text": text, "metadata": meta, "bm25_score": float(score)}
        for text, meta, score in top_results
    ]
