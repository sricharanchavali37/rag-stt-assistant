"""
test_text_query.py
Tests the RAG pipeline WITHOUT audio — sends text directly.
Bypasses Whisper so you can verify retrieval + generation work
before worrying about audio recording.

Usage: python test_text_query.py
"""

import asyncio
from sentence_transformers import SentenceTransformer
import chromadb

from config import settings
from models import RAGContext
from services.retriever import embed_and_retrieve
from services.prompt_builder import build_prompt
from services.generator import generate


async def test_pipeline(query_text: str):
    print(f"\n{'='*50}")
    print(f"Query: {query_text}")
    print('='*50)

    # Load models (same as startup in main.py)
    print("\n[1/4] Loading embedding model ...")
    embed_model = SentenceTransformer(settings.embed_model_name)

    print("[2/4] Connecting to ChromaDB ...")
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_collection(name=settings.collection_name)
    print(f"      {collection.count()} chunks available")

    # Build context manually (skipping Whisper)
    ctx = RAGContext(query=query_text, transcription=query_text)

    print("[3/4] Retrieving relevant chunks ...")
    ctx = embed_and_retrieve(ctx, embed_model, collection, top_k=settings.top_k_results)
    if ctx.error:
        print(f"❌ Retrieval error: {ctx.error}")
        return

    print(f"      Retrieved {len(ctx.retrieved_chunks)} chunks from: {ctx.sources}")

    print("[4/4] Building prompt and generating answer ...")
    ctx = build_prompt(ctx)
    ctx = await generate(ctx)

    if ctx.error:
        print(f"❌ Generation error: {ctx.error}")
        return

    print(f"\n{'─'*50}")
    print(f"ANSWER:\n{ctx.answer}")
    print(f"{'─'*50}")
    print(f"Sources: {ctx.sources}")
    print(f"Chunks used: {len(ctx.retrieved_chunks)}\n")


if __name__ == "__main__":
    # Change this to any question answerable from your PDF
    asyncio.run(test_pipeline("what is this document about"))