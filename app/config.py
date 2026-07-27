import os

class Config:
    project_root = "/content/RAG"

    data_dir = "/content/drive/MyDrive/rag_platform_data"
    upload_dir = os.path.join(data_dir, "uploads")
    chroma_dir = os.path.join(data_dir, "chroma")
    sqlite_path = os.path.join(data_dir, "registry.db")

    # If True, model downloads are cached to Drive so they persist across
    # Colab sessions (saves ~9+ min on every fresh session for the 7B LLM).
    # Costs significant Drive storage (~13GB+ for all 3 models combined).
    # Set to False for users who don't want to use their Drive quota this way.
    use_drive_model_cache = True
    model_cache_dir = os.path.join(data_dir, "model_cache")

    embedding_model = "BAAI/bge-small-en-v1.5"
    llm_model = "Qwen/Qwen2.5-7B-Instruct"
    reranker_model = "BAAI/bge-reranker-base"

CONFIG = Config()
