import torch
from sentence_transformers import CrossEncoder
from app.config import CONFIG
from app.retrieval.retriever import RetrievedChunk

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
    Re-scores and reorders retrieved chunks using a cross-encoder.
    bge-reranker-base's CrossEncoder.predict() already returns scores
    normalized to a 0-1 range internally (sigmoid applied by the library
    itself for single-label models) - we use that directly, no extra
    normalization needed.
    """
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.distance = float(score)

    reranked = sorted(chunks, key=lambda c: c.distance, reverse=True)
    return reranked
