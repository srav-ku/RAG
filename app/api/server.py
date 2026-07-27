import sys
sys.path.append("/content/RAG")

import os
from app.config import CONFIG
if CONFIG.use_drive_model_cache:
    os.makedirs(CONFIG.model_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = CONFIG.model_cache_dir

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Optional

from app.pipeline.ingest_pipeline import ingest_document
from app.pipeline.query_pipeline import answer_question
from app.pipeline.delete_pipeline import delete_document
from app.storage.db import init_db, list_documents

init_db()

app = FastAPI(title="Document Intelligence API")

# CORS: allows our frontend (served from the same or a different origin)
# to actually call these endpoints from browser JavaScript. Without this,
# browsers block cross-origin requests by default as a security measure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/documents")
def get_documents():
    """Returns all ingested documents with their metadata."""
    docs = list_documents()
    return {"documents": docs}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Accepts a file upload, runs it through the ingestion pipeline."""
    supported = (".pdf", ".txt", ".md", ".docx")
    if not file.filename.lower().endswith(supported):
        return {"status": "error", "message": f"Unsupported file type. Supported: {', '.join(supported)}"}

    file_bytes = await file.read()
    result = ingest_document(file_bytes, file.filename)
    return result


@app.post("/api/ask")
def ask(question: str = Form(...), document_ids: Optional[str] = Form(None)):
    """
    Answers a question, optionally scoped to specific documents.
    document_ids arrives as a comma-separated string from the form
    (simpler than JSON for this case) - empty/None means search all.
    """
    doc_id_list = document_ids.split(",") if document_ids else None
    result = answer_question(question, document_ids=doc_id_list)
    return result


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str):
    """Deletes a document and all its associated data."""
    result = delete_document(doc_id)
    return result


# Serve the frontend's static files (HTML/CSS/JS) directly from FastAPI,
# so the whole app - frontend and backend - is reachable from one URL.
app.mount("/", StaticFiles(directory="/content/RAG/app/frontend", html=True), name="frontend")
