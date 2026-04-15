"""
services/retriever.py — Stage 2
Embeds the query using MiniLM (same model as ingest.py),
then queries ChromaDB for the top-k most similar chunks.
Both the embedding model and chroma collection are loaded
once at startup and passed in.
"""

from models import RAGContext


def embed_and_retrieve(ctx: RAGContext, embed_model, collection, top_k: int = 5) -> RAGContext:
    """
    Takes a RAGContext with ctx.query populated.
    Returns the same context with retrieved_chunks and sources filled.
    embed_model: loaded SentenceTransformer instance
    collection:  loaded ChromaDB collection instance
    """

    if not ctx.query:
        ctx.error = "Cannot retrieve — query is empty. Check transcription stage."
        return ctx

    try:
        # Embed the query into the same 384-dim space used during ingestion
        query_vector = embed_model.encode([ctx.query])[0].tolist()

        # Query ChromaDB — READ ONLY, never writes
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack results — ChromaDB wraps everything in an outer list
        docs = results["documents"][0]       # List[str]
        metas = results["metadatas"][0]      # List[dict]

        if not docs:
            ctx.error = "ChromaDB returned no results. Was ingest.py run successfully?"
            return ctx

        ctx.retrieved_chunks = docs
        ctx.sources = list({m.get("source", "unknown") for m in metas})

    except Exception as e:
        ctx.error = f"Retrieval failed: {str(e)}"

    return ctx