import os

class Config:
    # Root folder where our code lives (inside Colab, temporary)
    project_root = "/content/RAG"

    # Where we'll store data — for now, still inside Colab.
    # We'll move this to Google Drive in the next task so it survives restarts.
    data_dir = os.path.join(project_root, "data")
    upload_dir = os.path.join(data_dir, "uploads")
    chroma_dir = os.path.join(data_dir, "chroma")
    sqlite_path = os.path.join(data_dir, "registry.db")

    # Model names — one place to change if we upgrade later
    embedding_model = "BAAI/bge-small-en-v1.5"
    llm_model = "Qwen/Qwen2.5-1.5B-Instruct"
    reranker_model = "BAAI/bge-reranker-base"

CONFIG = Config()
