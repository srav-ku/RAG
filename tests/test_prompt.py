import sys
sys.path.append("/content/RAG")
from app.generation.prompt import build_prompt
from app.retrieval.retriever import RetrievedChunk

def make_chunk(text="Sample text.", filename="doc.pdf", pages="1"):
    return RetrievedChunk(
        text=text, distance=0.9, document_id="d1",
        filename=filename, page_numbers=pages, chunk_index=0
    )

def test_prompt_contains_question():
    prompt = build_prompt("What is the policy?", [make_chunk()])
    assert "What is the policy?" in prompt

def test_prompt_contains_context_text():
    prompt = build_prompt("Q?", [make_chunk(text="Unique policy sentence here.")])
    assert "Unique policy sentence here." in prompt

def test_prompt_labels_sources_numbered():
    chunks = [make_chunk(text="First."), make_chunk(text="Second.")]
    prompt = build_prompt("Q?", chunks)
    assert "Source 1" in prompt
    assert "Source 2" in prompt

def test_prompt_has_dont_guess_instruction():
    prompt = build_prompt("Q?", [make_chunk()])
    assert "do not guess" in prompt.lower()

def test_empty_chunks_still_builds_valid_prompt():
    prompt = build_prompt("Q?", [])
    assert "Q?" in prompt
    assert "Context:" in prompt
