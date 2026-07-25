import uuid
from app.storage.fs import save_upload, compute_hash
from app.storage.db import insert_document, update_document_status, get_document_by_hash
from app.ingestion.parser import parse_document
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts
from app.retrieval.vector_store import add_chunks
from app.ingestion.parser import PageText


def ingest_document(file_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline: takes raw uploaded file bytes and filename,
    processes them completely, and returns a summary dict.

    Returns a dict with keys: doc_id, status, message, page_count, chunk_count
    """
    # Step 1: save the file to Drive first (we need it on disk to hash and parse it)
    saved_path = save_upload(file_bytes, filename)
    file_hash = compute_hash(saved_path)

    # Step 2: duplicate detection - if we've seen this exact content before, stop here
    existing = get_document_by_hash(file_hash)
    if existing is not None:
        return {
            "doc_id": existing["id"],
            "status": "duplicate",
            "message": f"This file was already uploaded as '{existing['filename']}'.",
            "page_count": existing["page_count"],
            "chunk_count": 0
        }

    # Step 3: register this as a new document, starting as 'pending'
    doc_id = str(uuid.uuid4())
    insert_document(doc_id, filename=filename, file_hash=file_hash, file_path=saved_path)

    # Step 4: parse the document into pages (with automatic OCR fallback)
    pages = parse_document(saved_path)

    # Step 5: clean each page's text before chunking
    cleaned_pages = [PageText(page_num=p.page_num, text=clean_text(p.text)) for p in pages]

    # Step 6: chunk the cleaned pages, preserving page number tracking
    chunks = chunk_document(cleaned_pages)

    if not chunks:
        update_document_status(doc_id, status="failed", page_count=len(pages))
        return {
            "doc_id": doc_id,
            "status": "failed",
            "message": "No text could be extracted from this document.",
            "page_count": len(pages),
            "chunk_count": 0
        }

    # Step 7: embed all chunks in one batch call
    chunk_texts = [c.text for c in chunks]
    embeddings = embed_texts(chunk_texts)

    # Step 8: build metadata for each chunk (needed for citations later)
    metadatas = [
        {
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": c.chunk_index,
            "page_numbers": ",".join(str(p) for p in c.page_numbers)  # ChromaDB needs simple types, not lists
        }
        for c in chunks
    ]

    # Step 9: store everything in ChromaDB
    add_chunks(doc_id=doc_id, chunk_texts=chunk_texts, embeddings=embeddings, metadatas=metadatas)

    # Step 10: mark this document as fully processed
    update_document_status(doc_id, status="processed", page_count=len(pages))

    return {
        "doc_id": doc_id,
        "status": "processed",
        "message": f"Successfully processed {filename}.",
        "page_count": len(pages),
        "chunk_count": len(chunks)
    }
