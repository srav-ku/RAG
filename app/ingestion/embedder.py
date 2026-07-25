import torch
from sentence_transformers import SentenceTransformer
from app.config import CONFIG

# Singleton - starts empty, gets filled in on first use
_model = None


def get_embedder() -> SentenceTransformer:
    """
    Returns the embedding model, loading it into GPU memory on first call
    and reusing that same loaded instance on every call after.
    """
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] Loading {CONFIG.embedding_model} on {device}...")
        _model = SentenceTransformer(CONFIG.embedding_model, device=device)
        print("[Embedder] Model loaded.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Converts a list of text strings into a list of embedding vectors.
    Processes them together as a batch for speed, rather than one at a time.
    """
    if not texts:
        return []

    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)

    # Convert from numpy array to plain Python lists (easier to store/serialize later)
    return embeddings.tolist()
