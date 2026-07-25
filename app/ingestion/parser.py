import fitz  # PyMuPDF
from dataclasses import dataclass
from app.ingestion.ocr import is_scanned, run_ocr

@dataclass
class PageText:
    """Holds the extracted text for a single page."""
    page_num: int
    text: str
    was_ocr: bool = False  # tracks whether this page needed OCR (useful for debugging/stats)


def parse_pdf_smart(filepath: str) -> list[PageText]:
    """
    Extracts text from a PDF, page by page.
    For each page: tries normal text extraction first.
    If the page looks scanned (too little text), falls back to OCR automatically.
    """
    pages = []

    with fitz.open(filepath) as doc:
        print(f"[Parser] PDF has {len(doc)} pages. Starting extraction...")

        for index, page in enumerate(doc):
            text = page.get_text()
            used_ocr = False

            if is_scanned(text):
                text = run_ocr(page)
                used_ocr = True

            pages.append(PageText(page_num=index + 1, text=text, was_ocr=used_ocr))

            if (index + 1) % 25 == 0:
                print(f"[Parser] Processed {index + 1}/{len(doc)} pages...")

    ocr_count = sum(1 for p in pages if p.was_ocr)
    print(f"[Parser] Done. {len(pages)} pages total, {ocr_count} needed OCR.")
    return pages


def parse_document(filepath: str) -> list[PageText]:
    """
    Single entry point for parsing any supported document type.
    Currently supports: PDF.
    Future file types (.txt, .md, .docx) will be routed here too.
    """
    if filepath.lower().endswith(".pdf"):
        return parse_pdf_smart(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
