import os
import glob
import re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from config import settings
from models import DocumentChunk


# ── Text Cleaning ─────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("\u200b", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Load PDFs ─────────────────────────────────────────────

def load_pdfs(docs_path: str):
    documents = []
    pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDFs found in docs folder")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()

                if text:
                    documents.append({
                        "text": clean_text(text),
                        "source": filename,
                        "page": page_num + 1
                    })

        print(f"[loader] {filename} → {len(pdf.pages)} pages")

    return documents


# ── Chunking ─────────────────────────────────────────────

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])

        for i, text in enumerate(split_texts):
            chunk_id = f"{doc['source']}_p{doc['page']}_chunk_{i}"

            chunks.append(
                DocumentChunk(
                    text=text,
                    source=doc["source"],
                    chunk_id=chunk_id
                )
            )

    print(f"[chunker] {len(chunks)} chunks created")

    # 🔥 DEBUG
    if len(chunks) > 0:
        print("\n[SAMPLE CHUNK]")
        print(chunks[0].text[:500])

    # 🔥 sanity check
    if len(chunks) < 20:
        print("⚠️ WARNING: Too few chunks — retrieval will be weak")

    return chunks


# ── Embedding ────────────────────────────────────────────

def embed_chunks(chunks):
    print("[embedder] loading model...")

    model = SentenceTransformer(settings.embed_model_name)

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for c, e in zip(chunks, embeddings):
        c.embedding = e.tolist()

    print("[embedder] done")
    return chunks


# ── Store in Chroma ───────────────────────────────────────

def persist_to_chroma(chunks):
    client = chromadb.PersistentClient(path=settings.chroma_path)

    try:
        client.delete_collection(settings.collection_name)
        print("[chroma] old collection deleted")
    except:
        print("[chroma] fresh start")

    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids, embeddings, docs, metas = [], [], [], []

    for c in chunks:
        ids.append(c.chunk_id)
        embeddings.append(c.embedding)
        docs.append(c.text)

        page = c.chunk_id.split("_p")[1].split("_")[0]

        metas.append({
            "source": c.source,
            "page": int(page),
            "chunk_id": c.chunk_id
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metas
    )

    print(f"[chroma] {collection.count()} chunks stored")


# ── MAIN ─────────────────────────────────────────────────

def main():
    print("\n=== INGEST START ===\n")

    docs = load_pdfs(settings.docs_path)
    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)
    persist_to_chroma(chunks)

    print("\n=== INGEST DONE ===\n")


if __name__ == "__main__":
    main()