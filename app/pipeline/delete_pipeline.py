from app.storage.db import get_document_by_id, delete_document_record
from app.storage.fs import delete_file
from app.retrieval.vector_store import get_collection


def delete_document(doc_id: str) -> dict:
    """
    Fully removes a document: its chunks from ChromaDB, its record from
    SQLite, and its raw file from Drive. Mirrors the three-place storage
    pattern from ingestion (Step 14), just in reverse.
    """
    doc = get_document_by_id(doc_id)
    if doc is None:
        return {"success": False, "message": "Document not found."}

    # Remove chunks from ChromaDB - find all chunk IDs belonging to this doc
    collection = get_collection()
    all_data = collection.get(where={"document_id": doc_id})
    chunk_ids = all_data["ids"]
    if chunk_ids:
        collection.delete(ids=chunk_ids)

    # Remove the raw file from Drive
    delete_file(doc["file_path"])

    # Remove the SQLite record
    delete_document_record(doc_id)

    return {"success": True, "message": f"Deleted '{doc['filename']}' ({len(chunk_ids)} chunks removed)."}
