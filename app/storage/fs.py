import os
import hashlib
import uuid
from app.config import CONFIG


def save_upload(file_bytes: bytes, original_filename: str) -> str:
    """
    Saves uploaded file bytes to Drive with a unique filename
    (to avoid collisions if two different users upload files with the same name).

    Returns the full saved file path.
    """
    _, ext = os.path.splitext(original_filename)
    unique_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(CONFIG.upload_dir, unique_name)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return save_path


def compute_hash(filepath: str) -> str:
    """
    Computes the SHA256 hash of a file's contents, reading it in small
    chunks rather than loading the whole file into memory at once.
    """
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        # Read 8KB at a time until the file is exhausted
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def delete_file(filepath: str) -> None:
    """Deletes a file from disk, if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)
