import fitz  # PyMuPDF
from dataclasses import dataclass

@dataclass
class PageText:
    """Holds the extracted text for a single page."""
    page_num: int   # 1-indexed, so page 1 is "page 1", not "page 0"
    text: str

def parse_pdf(filepath: str) -> list[PageText]:
    """
    Extracts text from a PDF, one page at a time.
    Safe for large PDFs because we never hold the whole document's
    rendered content in memory - only one page at a time during the loop.
    """
    pages = []

    with fitz.open(filepath) as doc:
        print(f"[Parser] PDF has {len(doc)} pages. Starting extraction...")

        for index, page in enumerate(doc):
            page_text = page.get_text()
            pages.append(PageText(page_num=index + 1, text=page_text))

            # Progress feedback every 25 pages - important for large docs
            if (index + 1) % 25 == 0:
                print(f"[Parser] Processed {index + 1}/{len(doc)} pages...")

    print(f"[Parser] Done. Extracted {len(pages)} pages.")
    return pages
