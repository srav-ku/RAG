from dataclasses import dataclass
from app.ingestion.embedder import embed_texts
from app.retrieval.vector_store import query_vectors


@dataclass
class RetrievedChunk:
    text: str
    distance: float
    document_id: str
    filename: str
    page_numbers: str
    chunk_index: int


def retrieve(query: str, top_k: int = 5, document_ids: list[str] = None) -> list[RetrievedChunk]:
    """
    Embeds a query and searches for similar chunks, optionally scoped
    to specific document_ids only.
    """
    query_embedding = embed_texts([query])[0]
    raw_results = query_vectors(query_embedding, top_k=top_k, document_ids=document_ids)

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
