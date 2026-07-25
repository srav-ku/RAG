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


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    """Splits raw text into overlapping chunks, measured in tokens. Page-agnostic."""
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


def chunk_document(pages: list[PageText], chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    """
    Chunks a full document while tracking which page(s) each chunk came from.

    Strategy: encode each page's tokens separately first, keeping a running
    map of "which page does token N belong to" - then slide the chunking
    window over ALL tokens at once, reading page numbers directly from
    that map. No text-searching involved, so no ambiguity possible.
    """
    if not pages:
        return []

    all_tokens = []       # every token in the document, in order
    token_page_map = []   # token_page_map[i] = which page token i came from

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

        # Directly read which pages this exact token range covers - no guessing
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
