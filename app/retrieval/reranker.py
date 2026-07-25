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
    Raw model output is an unbounded logit, so we apply a sigmoid to
    convert it into a stable, comparable 0-1 relevance score.
    """
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(query, chunk.text) for chunk in chunks]

    # apply_softmax=False keeps raw logits; we normalize manually below
    raw_scores = model.predict(pairs)

    # Sigmoid: squashes any real number into a clean 0-1 range.
    # sigmoid(x) = 1 / (1 + e^-x)
    normalized_scores = [1 / (1 + torch.exp(torch.tensor(-s))) for s in raw_scores]

    for chunk, score in zip(chunks, normalized_scores):
        chunk.distance = float(score)

    reranked = sorted(chunks, key=lambda c: c.distance, reverse=True)
    return reranked
