from app.retrieval.retriever import RetrievedChunk


def format_citations(chunks: list[RetrievedChunk]) -> str:
    """
    Formats retrieved chunks into a readable citation list, grouped by
    document so the same file doesn't get listed multiple times.
    """
    if not chunks:
        return "No sources."

    # Group page numbers by filename
    sources_by_file = {}
    for chunk in chunks:
        filename = chunk.filename or "Unknown document"
        pages = chunk.page_numbers.split(",") if chunk.page_numbers else []

        if filename not in sources_by_file:
            sources_by_file[filename] = set()
        sources_by_file[filename].update(p.strip() for p in pages if p.strip())

    # Build the formatted output
    lines = []
    for filename, pages in sources_by_file.items():
        sorted_pages = sorted(pages, key=lambda p: int(p) if p.isdigit() else 0)
        pages_str = ", ".join(sorted_pages)
        lines.append(f"- {filename} (page{'s' if len(sorted_pages) != 1 else ''} {pages_str})")

    return "\n".join(lines)
