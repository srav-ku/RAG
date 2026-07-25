import chromadb
from app.config import CONFIG

_client = None
_collection = None


def get_collection():
    """
    Returns the ChromaDB collection, connecting to the persistent
    Drive-backed database on first call and reusing that connection after.
    Explicitly configured to use cosine similarity, matching how our
    embedding model (bge-small-en-v1.5) is designed to be compared.
    """
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


def query_vectors(query_embedding: list[float], top_k: int = 5):
    collection = get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results
