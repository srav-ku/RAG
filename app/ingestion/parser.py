import fitz
import docx
from dataclasses import dataclass
from app.ingestion.ocr import is_scanned, run_ocr


@dataclass
class PageText:
    page_num: int
    text: str
    was_ocr: bool = False


def parse_pdf_smart(filepath: str) -> list[PageText]:
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


def parse_txt(filepath: str) -> list[PageText]:
    """Reads a plain text file as a single 'page' - no page concept for TXT."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"[Parser] TXT file loaded, {len(text)} characters.")
    return [PageText(page_num=1, text=text)]


def parse_markdown(filepath: str) -> list[PageText]:
    """
    Reads a Markdown file as a single 'page'. We keep the raw markdown
    syntax (headers, bullets) rather than stripping it - the text still
    reads fine for chunking/embedding purposes, and stripping adds
    complexity for little benefit at our scale.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"[Parser] Markdown file loaded, {len(text)} characters.")
    return [PageText(page_num=1, text=text)]


def parse_docx(filepath: str) -> list[PageText]:
    """
    Reads a Word document. DOCX has no reliable 'page' concept in the
    file format itself (page breaks depend on rendering, not fixed
    markers), so we treat each PARAGRAPH as one unit and combine them
    all into a single logical page, same as TXT/MD.
    """
    document = docx.Document(filepath)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    print(f"[Parser] DOCX file loaded, {len(paragraphs)} paragraphs.")
    return [PageText(page_num=1, text=full_text)]


def parse_document(filepath: str) -> list[PageText]:
    """
    Single entry point for parsing any supported document type.
    Supports: PDF, TXT, MD, DOCX.
    """
    lower_path = filepath.lower()

    if lower_path.endswith(".pdf"):
        return parse_pdf_smart(filepath)
    elif lower_path.endswith(".txt"):
        return parse_txt(filepath)
    elif lower_path.endswith(".md"):
        return parse_markdown(filepath)
    elif lower_path.endswith(".docx"):
        return parse_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
