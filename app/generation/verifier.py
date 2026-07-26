from app.generation.llm import generate_text
from app.retrieval.retriever import RetrievedChunk


def context_supports_detailed_answer(query: str, context: str) -> bool:
    """
    Asks the LLM a narrow yes/no question: does this single piece of
    context contain the SPECIFIC details needed to answer, or does it
    only mention the topic in passing?
    """
    check_prompt = f"""Context:
{context}

Question: Does the context above contain SPECIFIC step-by-step details, exact instructions, exact numbers, or precise configuration values that would answer this question: "{query}"

Answer with only one word: YES or NO."""

    response = generate_text(check_prompt, max_new_tokens=5)
    return "yes" in response.lower()


def any_chunk_supports_answer(query: str, chunks: list[RetrievedChunk]) -> bool:
    """
    Checks each retrieved chunk INDIVIDUALLY rather than combining them
    into one noisy blob (combining was found to dilute the judgment -
    Step 26 investigation: same chunk scored True alone, False when
    mixed with 4 unrelated chunks). Passes if ANY single chunk supports
    a detailed answer.
    """
    for chunk in chunks:
        if context_supports_detailed_answer(query, chunk.text):
            return True
    return False
