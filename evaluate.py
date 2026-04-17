import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer

from config import settings


# ── Ollama Call (WITH TIMEOUT + SAFE) ─────────────────────

def ollama_call(prompt: str) -> str:
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=300   # ✅ prevents freezing
        )
        return response.json().get("response", "").strip()

    except Exception as e:
        return f"ERROR: {e}"


# ── Simple Evaluation (BETTER VERSION) ───────────────────

def simple_score(gt, ans):
    gt_words = set(gt.lower().split())
    ans_words = set(ans.lower().split())

    if len(gt_words) == 0:
        return 0

    overlap = gt_words.intersection(ans_words)
    return len(overlap) / len(gt_words)


# ── Load Dataset ─────────────────────────────────────────

print("\n📋 Loading questions...")

with open("eval_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
ground_truths = [item["ground_truth"] for item in data]

# ✅ LIMIT (DON’T BE STUPID AND RUN 12)
questions = questions[:3]
ground_truths = ground_truths[:3]

print(f"{len(questions)} questions loaded ✅")


# ── Load ChromaDB ───────────────────────────────────────

print("[1/3] Loading ChromaDB...")

client = chromadb.PersistentClient(path=settings.chroma_path)
collection = client.get_collection(settings.collection_name)

print(f"✅ ({collection.count()} chunks)")


# ── Load Embeddings ─────────────────────────────────────

print("[2/3] Loading embedding model...")

embed_model = SentenceTransformer(settings.embed_model_name)

print("✅")


# ── Run Queries ─────────────────────────────────────────

print(f"[3/3] Running {len(questions)} questions...")

answers = []
contexts = []

for i, q in enumerate(questions):
    print(f"\n==============================")
    print(f"[Q{i+1}] {q}")

    # Encode query
    query_embedding = embed_model.encode(q).tolist()

    # ✅ REDUCED CONTEXT (CRITICAL)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    retrieved_docs = results["documents"][0]

    # 🔍 DEBUG RETRIEVAL
    print("\n[RETRIEVED CONTEXT]")
    for j, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {j+1} ---")
        print(doc[:300])

    context = "\n".join(retrieved_docs)

    # ✅ BETTER PROMPT (NO “I DON’T KNOW”)
    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.
Even if incomplete, try to infer the best possible answer.

Context:
{context}

Question:
{q}
"""

    answer = ollama_call(prompt)

    print(f"\n[ANSWER]\n{answer}")

    answers.append(answer)
    contexts.append(retrieved_docs)


# ── Evaluation ─────────────────────────────────────────

print("\n==============================")
print("📊 SIMPLE EVALUATION")
print("==============================")

total_score = 0

for i in range(len(questions)):
    score = simple_score(ground_truths[i], answers[i])

    print(f"\nQ{i+1}")
    print(f"GT: {ground_truths[i]}")
    print(f"ANS: {answers[i]}")
    print(f"SCORE: {score:.2f}")

    total_score += score

avg_score = total_score / len(questions)

print("\n------------------------------")
print(f"Average Score: {avg_score:.2f}")
print("------------------------------")

print("\n✅ Done.")