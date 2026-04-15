"""
services/prompt_builder.py — Stage 3
Assembles the grounding prompt from retrieved chunks + user query.
This is pure Python — no external library needed.
The grounding instruction is what makes this RAG, not just a chatbot.
"""

from models import RAGContext


def build_prompt(ctx: RAGContext) -> RAGContext:
    """
    Takes a RAGContext with retrieved_chunks and query populated.
    Returns the same context with the prompt field filled.
    """

    if not ctx.retrieved_chunks:
        ctx.error = "Cannot build prompt — no chunks were retrieved."
        return ctx

    # Format each chunk as a numbered context block
    context_blocks = ""
    for i, chunk in enumerate(ctx.retrieved_chunks, 1):
        context_blocks += f"[{i}]\n{chunk.strip()}\n\n"

    ctx.prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
Do not use outside knowledge. If the answer is not found in the context, respond with:
"I don't know based on the provided documents."

Context:
{context_blocks.strip()}

Question: {ctx.query}

Answer:"""

    return ctx