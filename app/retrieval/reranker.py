import torch
from sentence_transformers import CrossEncoder
from app.config import CONFIG
from app.retrieval.retriever import RetrievedChunk

# Singleton, same pattern as the embedder
_reranker_model = None


def get_reranker() -> CrossEncoder:
    """Returns the reranker model, loading it once and reusing it after."""
    global _reranker_model
    if _reranker_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Reranker] Loading {CONFIG.reranker_model} on {device}...")
        _reranker_model = CrossEncoder(CONFIG.reranker_model, device=device)
        print("[Reranker] Model loaded.")
    return _reranker_model


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Re-scores and reorders retrieved chunks based on how relevant each one
    actually is to the specific query, using a cross-encoder model.
    """
    if not chunks:
        return []

    model = get_reranker()

    # Build (query, chunk_text) pairs - the cross-encoder scores each pair together
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)

    # Attach the new scores to each chunk by overwriting `distance`
    # (repurposed here as "rerank score" - higher now means MORE relevant,
    # the opposite direction from ChromaDB's distance, so take care downstream)
    for chunk, score in zip(chunks, scores):
        chunk.distance = float(score)

    # Sort by score descending - most relevant first
    reranked = sorted(chunks, key=lambda c: c.distance, reverse=True)
    return reranked
