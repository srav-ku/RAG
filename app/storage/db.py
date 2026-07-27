import sqlite3
from app.config import CONFIG


def get_connection():
    conn = sqlite3.connect(CONFIG.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
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
    conn = get_connection()
    cursor = conn.cursor()
    if page_count is not None:
        cursor.execute("UPDATE documents SET status = ?, page_count = ? WHERE id = ?", (status, page_count, doc_id))
    else:
        cursor.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
    conn.commit()
    conn.close()


def list_documents() -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_document_by_hash(file_hash: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_document_by_id(doc_id: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_document_record(doc_id: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
