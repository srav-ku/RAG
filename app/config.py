import os

class Config:
    project_root = "/content/RAG"

    data_dir = "/content/drive/MyDrive/rag_platform_data"
    upload_dir = os.path.join(data_dir, "uploads")
    chroma_dir = os.path.join(data_dir, "chroma")
    sqlite_path = os.path.join(data_dir, "registry.db")

    embedding_model = "BAAI/bge-small-en-v1.5"
    llm_model = "Qwen/Qwen2.5-7B-Instruct"
    reranker_model = "BAAI/bge-reranker-base"

CONFIG = Config()
