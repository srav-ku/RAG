import chromadb
from app.config import CONFIG

# Singleton - same "load once, reuse" pattern as the embedder
_client = None
_collection = None


def get_collection():
    """
    Returns the ChromaDB collection, connecting to the persistent
    Drive-backed database on first call and reusing that connection after.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CONFIG.chroma_dir)
        # get_or_create: safe to call even if it already exists - won't wipe data
        _collection = _client.get_or_create_collection(name="document_chunks")
    return _collection


def add_chunks(doc_id: str, chunk_texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    """
    Stores chunks in ChromaDB.

    Args:
        doc_id: parent document's ID, used to build unique chunk IDs.
        chunk_texts: the actual text of each chunk.
        embeddings: the corresponding vector for each chunk.
        metadatas: extra info per chunk (e.g. page numbers, chunk index).
    """
    collection = get_collection()

    # ChromaDB needs a unique string ID per entry - we build one from doc_id + position
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunk_texts))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas
    )


def query_vectors(query_embedding: list[float], top_k: int = 5):
    """
    Finds the most similar stored chunks to a given query vector.
    Returns ChromaDB's raw result dict (ids, documents, metadatas, distances).
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results
