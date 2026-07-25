import os

class Config:
    # Code lives here (synced with GitHub)
    project_root = "/content/RAG"

    # Data lives on Google Drive - survives Colab restarts
    data_dir = "/content/drive/MyDrive/rag_platform_data"
    upload_dir = os.path.join(data_dir, "uploads")
    chroma_dir = os.path.join(data_dir, "chroma")
    sqlite_path = os.path.join(data_dir, "registry.db")

    # Model names
    embedding_model = "BAAI/bge-small-en-v1.5"
    llm_model = "Qwen/Qwen2.5-1.5B-Instruct"
    reranker_model = "BAAI/bge-reranker-base"

CONFIG = Config()
