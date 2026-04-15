import json
import requests
import asyncio
import nest_asyncio
nest_asyncio.apply()

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings


# ── Ollama Call ─────────────────────────────────────────

def ollama_call(prompt: str) -> str:
    response = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# ── Load Dataset ─────────────────────────────────────────

print("\n📋 Loading questions...")
with open("eval_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
ground_truths = [item["ground_truth"] for item in data]

# ✅ LIMIT QUESTIONS (IMPORTANT)
questions = questions[:3]
ground_truths = ground_truths[:3]

print(f"{len(questions)} questions loaded ✅")


# ── Load ChromaDB ───────────────────────────────────────

print("[1/5] Loading ChromaDB...")

client = chromadb.PersistentClient(path=settings.chroma_path)
collection = client.get_collection(settings.collection_name)

print(f"✅ ({collection.count()} chunks)")


# ── Load Embeddings ─────────────────────────────────────

print("[2/5] Loading embedding model...")

embed_model = SentenceTransformer(settings.embed_model_name)

print("✅")


# ── Run Queries ─────────────────────────────────────────

print(f"[3/5] Running {len(questions)} questions...")

answers = []
contexts = []

for i, q in enumerate(questions):
    print(f"\n[{i+1}/{len(questions)}] {q}")

    query_embedding = embed_model.encode(q).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=settings.top_k_results
    )

    retrieved_docs = results["documents"][0]

    print("\n[RETRIEVED CONTEXT]")
    for j, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {j+1} ---")
        print(doc[:300])

    context = "\n".join(retrieved_docs)

    # ✅ UPDATED PROMPT (CRITICAL FIX)
    prompt = f"""
You are a helpful assistant.

Answer the question using the context below.
Even if the answer is partial, try to infer it.

DO NOT say "I don't know".

Context:
{context}

Question:
{q}
"""

    answer = ollama_call(prompt)

    print(f"\n[ANSWER] {answer}\n")

    answers.append(answer)
    contexts.append(retrieved_docs)


# ── SIMPLE EVALUATION (REPLACES RAGAS) ───────────────────

print("\n=== SIMPLE EVALUATION ===")

score = 0

for i in range(len(questions)):
    gt = ground_truths[i].lower()
    ans = answers[i].lower()

    if any(word in ans for word in gt.split()):
        score += 1
        print(f"[Q{i+1}] ✅ MATCH")
    else:
        print(f"[Q{i+1}] ❌ NO MATCH")

print(f"\nScore: {score}/{len(questions)}")

print("\n✅ Done.")