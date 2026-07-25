from app.retrieval.retriever import RetrievedChunk


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Builds a grounded prompt: instructions + retrieved context + the user's question.
    Designed to minimize hallucination by explicitly restricting the model
    to only the provided context, and telling it what to do if the answer isn't there.
    """
    # Build the context block from retrieved chunks, numbered for clarity
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[Source {i} - {chunk.filename}, page(s) {chunk.page_numbers}]\n{chunk.text}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions using ONLY the context provided below.

Rules:
- Only use information from the context below to answer.
- If the context does not contain enough information to answer the question, say "I don't have enough information to answer that" - do not guess or use outside knowledge.
- Be concise and direct.
- If you use information from a specific source, you may refer to it by its source number (e.g. "Source 1").

Context:
{context_str}

Question: {query}

Answer:"""

    return prompt
