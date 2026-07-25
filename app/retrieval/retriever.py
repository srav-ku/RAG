from dataclasses import dataclass
from app.ingestion.embedder import embed_texts
from app.retrieval.vector_store import query_vectors


@dataclass
class RetrievedChunk:
    """A single search result: the chunk text plus where it came from and how relevant it is."""
    text: str
    distance: float          # lower = more similar
    document_id: str
    filename: str
    page_numbers: str        # comma-separated string, e.g. "3,4"
    chunk_index: int


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Embeds a user's question and searches ChromaDB for the most similar chunks.
    Returns a clean list of RetrievedChunk objects, not ChromaDB's raw format.
    """
    query_embedding = embed_texts([query])[0]
    raw_results = query_vectors(query_embedding, top_k=top_k)

    # ChromaDB returns everything as lists-of-lists (one outer list per query embedding;
    # we only ever send one query at a time, so we always index [0])
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    results = []
    for doc_text, meta, dist in zip(documents, metadatas, distances):
        results.append(RetrievedChunk(
            text=doc_text,
            distance=dist,
            document_id=meta.get("document_id", ""),
            filename=meta.get("filename", ""),
            page_numbers=meta.get("page_numbers", ""),
            chunk_index=meta.get("chunk_index", -1)
        ))

    return results
