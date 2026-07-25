from dataclasses import dataclass
import tiktoken

# We reuse one tokenizer encoding across all calls - loading it is a bit slow,
# so we don't want to reload it every single time we chunk something.
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """Represents one chunk of text, ready for embedding."""
    text: str
    chunk_index: int   # position of this chunk within the document (0, 1, 2...)
    token_count: int   # how many tokens this chunk actually contains


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    """
    Splits text into overlapping chunks, measured in TOKENS (not words/characters).

    Args:
        text: the cleaned text to split.
        chunk_size: target number of tokens per chunk.
        overlap: number of tokens repeated at the start of each chunk
                  from the end of the previous one, to preserve context.
    """
    if not text:
        return []

    # Convert the whole text into a list of token IDs
    tokens = _encoding.encode(text)

    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_token_ids = tokens[start:end]

        # Convert these token IDs back into readable text
        chunk_str = _encoding.decode(chunk_token_ids)

        chunks.append(Chunk(
            text=chunk_str,
            chunk_index=index,
            token_count=len(chunk_token_ids)
        ))

        index += 1
        # Move the window forward, but step back by `overlap` tokens
        # so the next chunk repeats the tail end of this one
        start += (chunk_size - overlap)

    return chunks
