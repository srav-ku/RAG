from app.retrieval.retriever import RetrievedChunk


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Builds a grounded prompt with hardened anti-fabrication instructions.
    Explicitly addresses the case where context mentions a topic but lacks
    the specific details being asked for (found via testing - Step 25).
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[Source {i} - {chunk.filename}, page(s) {chunk.page_numbers}]\n{chunk.text}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions using ONLY the context provided below.

Rules:
- Only use information from the context below to answer.
- If the context does not contain enough information to answer the question, say "I don't have enough information to answer that" - do not guess or use outside knowledge.
- IMPORTANT: If the context only mentions that something exists or is used, but does NOT provide the specific details being asked for (such as step-by-step instructions, exact numbers, or configuration details), you MUST say the context does not include those specific details. Do NOT invent, assume, or fill in plausible-sounding specifics that are not explicitly written in the context.
- Be concise and direct.
- If you use information from a specific source, you may refer to it by its source number (e.g. "Source 1").

Context:
{context_str}

Question: {query}

Answer:"""

    return prompt
