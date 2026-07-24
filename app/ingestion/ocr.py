import fitz  # PyMuPDF - also used to render pages as images
import pytesseract
from PIL import Image
import io

def is_scanned(text: str, min_char_threshold: int = 50) -> bool:
    """Heuristic to detect if a page is likely a scanned image with no real text."""
    clean = text.strip()
    return len(clean) < min_char_threshold


def run_ocr(pdf_page) -> str:
    """
    Runs OCR on a single PyMuPDF page object.
    Renders the page as an image, then extracts text from that image.
    """
    # Render the page as a pixel image (like taking a screenshot of it)
    pix = pdf_page.get_pixmap(dpi=200)

    # Convert PyMuPDF's image format into a standard image format (PIL)
    # that pytesseract knows how to read
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))

    # Run Tesseract OCR on the image, get back plain text
    text = pytesseract.image_to_string(image)
    return text
