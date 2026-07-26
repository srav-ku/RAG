import chromadb
from app.config import CONFIG

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CONFIG.chroma_dir)
        _collection = _client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def add_chunks(doc_id: str, chunk_texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    collection = get_collection()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunk_texts))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunk_texts, metadatas=metadatas)


def query_vectors(query_embedding: list[float], top_k: int = 5, document_ids: list[str] = None):
    """
    Searches for similar chunks. If document_ids is provided, only searches
    within those specific documents (metadata filtering) instead of the
    whole collection.
    """
    collection = get_collection()

    where_filter = None
    if document_ids:
        # ChromaDB's filter syntax: match any document_id in this list
        where_filter = {"document_id": {"$in": document_ids}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )
    return results


def get_all_chunks_for_documents(document_ids: list[str] = None) -> dict:
    """
    Returns all chunks (documents + metadatas), optionally filtered to
    specific document_ids. Used by keyword (BM25) search, which needs
    the full text rather than a similarity query.
    """
    collection = get_collection()

    if document_ids:
        where_filter = {"document_id": {"$in": document_ids}}
        return collection.get(where=where_filter)
    else:
        return collection.get()
