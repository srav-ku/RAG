import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw extracted text.

    Steps:
    1. Normalize unicode characters (e.g. fancy quotes -> standard quotes).
    2. Collapse all whitespace (newlines, tabs, multiple spaces) into single spaces.
    3. Strip leading/trailing whitespace.
    """
    if not text:
        return ""

    # Step 1: Normalize unicode to a consistent form
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Replace any run of whitespace (space, tab, newline) with one space
    text = re.sub(r"\s+", " ", text)

    # Step 3: Trim leading/trailing space
    text = text.strip()

    return text
