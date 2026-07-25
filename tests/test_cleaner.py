import sys
sys.path.append("/content/RAG")
from app.ingestion.cleaner import clean_text

def test_collapses_whitespace():
    messy = "This  is\n\na   test."
    assert clean_text(messy) == "This is a test."

def test_strips_leading_trailing_space():
    assert clean_text("   hello world   ") == "hello world"

def test_empty_string_returns_empty():
    assert clean_text("") == ""

def test_none_like_falsy_input():
    assert clean_text(None) == ""
