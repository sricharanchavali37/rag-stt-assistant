"""
verify_chroma.py — Validate ChromaDB + improve retrieval quality

Enhancements:
- Retrieves more chunks (top-5)
- Uses generalized keyword filtering (better recall + precision)
- Forces evaluation mindset
"""

import chromadb
from config import settings


def verify():
    print(f"\nConnecting to ChromaDB at '{settings.chroma_path}' ...")
    client = chromadb.PersistentClient(path=settings.chroma_path)

    col = client.get_collection(settings.collection_name)
    count = col.count()

    print(f"Collection '{settings.collection_name}' has {count} chunks stored.\n")

    if count == 0:
        print("❌ Collection is empty — did ingest.py run successfully?")
        return

    query = "what is time complexity of loops"
    print(f"Running test query: '{query}' ...")

    results = col.query(
        query_texts=[query],
        n_results=min(5, count),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 🔥 Improved filtering (multi-keyword)
    keywords = [
        "time complexity is",
        "complexity is",
        "o("
    ]

    filtered_docs = []
    filtered_meta = []

    for doc, meta in zip(documents, metadatas):
        text = doc.lower()

        if any(keyword in text for keyword in keywords):
            filtered_docs.append(doc)
            filtered_meta.append(meta)

    # fallback if filtering removes everything
    if len(filtered_docs) == 0:
        filtered_docs = documents
        filtered_meta = metadatas

    print("\n--- Top matching chunks (filtered) ---")

    for i, doc in enumerate(filtered_docs):
        metadata = filtered_meta[i]

        print(f"\n========== RESULT {i+1} ==========")
        print(f"Source   : {metadata.get('source')}")
        print(f"Page     : {metadata.get('page')}")
        print(f"Chunk ID : {metadata.get('chunk_id')}\n")

        print("Chunk Content:")
        print(doc[:500] + "..." if len(doc) > 500 else doc)

        print("\n--- Your Evaluation ---")
        print("1. Does this chunk directly answer the query?")
        print("2. Or is it just loosely related?")
        print("3. Would an LLM give a correct answer using ONLY this chunk?")
        print("4. Is there missing context?\n")

    print("✅ ChromaDB is healthy and queryable.\n")


if __name__ == "__main__":
    verify()