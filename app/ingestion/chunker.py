from dataclasses import dataclass, field
import tiktoken
from app.ingestion.parser import PageText

_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """Represents one chunk of text, ready for embedding."""
    text: str
    chunk_index: int
    token_count: int
    page_numbers: list[int] = field(default_factory=list)


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[Chunk]:
    """
    Splits raw text into overlapping chunks, measured in tokens.
    Reduced from 300 to 150 tokens (Step 21 investigation): larger chunks
    were found to dilute embedding relevance when unrelated content (e.g.
    a trailing sentence + a distinct data section) landed in the same chunk.
    Smaller chunks keep each one more topically focused.
    """
    if not text:
        return []

    tokens = _encoding.encode(text)
    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_token_ids = tokens[start:end]
        chunk_str = _encoding.decode(chunk_token_ids)

        chunks.append(Chunk(text=chunk_str, chunk_index=index, token_count=len(chunk_token_ids)))

        index += 1
        start += (chunk_size - overlap)

    return chunks


def chunk_document(pages: list[PageText], chunk_size: int = 150, overlap: int = 30) -> list[Chunk]:
    """Chunks a full document while tracking which page(s) each chunk came from."""
    if not pages:
        return []

    all_tokens = []
    token_page_map = []

    for page in pages:
        page_tokens = _encoding.encode(page.text + " ")
        all_tokens.extend(page_tokens)
        token_page_map.extend([page.page_num] * len(page_tokens))

    chunks = []
    start = 0
    index = 0

    while start < len(all_tokens):
        end = start + chunk_size
        chunk_token_ids = all_tokens[start:end]
        chunk_str = _encoding.decode(chunk_token_ids)
        pages_in_chunk = sorted(set(token_page_map[start:end]))

        chunks.append(Chunk(
            text=chunk_str,
            chunk_index=index,
            token_count=len(chunk_token_ids),
            page_numbers=pages_in_chunk
        ))

        index += 1
        start += (chunk_size - overlap)

    return chunks
