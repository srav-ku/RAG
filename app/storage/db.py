import sqlite3
from app.config import CONFIG


def get_connection():
    """
    Opens a connection to our SQLite database file (on Drive).
    row_factory makes query results behave like dictionaries (access by
    column name) instead of plain unlabeled tuples - much easier to work with.
    """
    conn = sqlite3.connect(CONFIG.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the documents table if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            file_path TEXT NOT NULL,
            page_count INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            uploaded_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_document(doc_id: str, filename: str, file_hash: str, file_path: str) -> None:
    """Registers a new document as 'pending' - before any processing has happened."""
    import datetime

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents (id, filename, file_hash, file_path, status, uploaded_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
    """, (doc_id, filename, file_hash, file_path, datetime.datetime.now().isoformat()))

    conn.commit()
    conn.close()


def update_document_status(doc_id: str, status: str, page_count: int = None) -> None:
    """Updates a document's status (e.g. to 'processed' or 'failed') once ingestion finishes."""
    conn = get_connection()
    cursor = conn.cursor()

    if page_count is not None:
        cursor.execute(
            "UPDATE documents SET status = ?, page_count = ? WHERE id = ?",
            (status, page_count, doc_id)
        )
    else:
        cursor.execute(
            "UPDATE documents SET status = ? WHERE id = ?",
            (status, doc_id)
        )

    conn.commit()
    conn.close()


def list_documents() -> list[dict]:
    """Returns all registered documents, most recently uploaded first."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()

    # Convert sqlite3.Row objects into plain dicts - easier to use elsewhere
    return [dict(row) for row in rows]


def get_document_by_hash(file_hash: str) -> dict | None:
    """Looks up a document by its file hash - used for duplicate detection."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None
