"""
main.py — Phase 3 entry point
FastAPI app with lifespan startup: loads all models ONCE.
Registers the /query and /health routes.
NEVER imports ingest.py — hard rule.
"""

from contextlib import asynccontextmanager
import whisper
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from config import settings
from routers.query import router as query_router


# ── Lifespan: load all heavy resources once at startup ───────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE when the server starts.
    Loads Whisper, MiniLM, and ChromaDB into app.state
    so every request can reuse them without reloading.
    """

    print(f"\n[startup] Loading Whisper model '{settings.whisper_model}' ...")
    app.state.whisper_model = whisper.load_model(settings.whisper_model)
    print(f"[startup] Whisper ready.")

    print(f"[startup] Loading embedding model '{settings.embed_model_name}' ...")
    app.state.embed_model = SentenceTransformer(settings.embed_model_name)
    print(f"[startup] Embedding model ready.")

    print(f"[startup] Connecting to ChromaDB at '{settings.chroma_path}' ...")
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    app.state.collection = chroma_client.get_collection(name=settings.collection_name)
    count = app.state.collection.count()
    print(f"[startup] ChromaDB ready — {count} chunks loaded.")

    print(f"\n✅ All models loaded. Server is ready.\n")

    yield  # app runs here — everything after yield runs on shutdown

    print("\n[shutdown] Cleaning up ...")


# ── App init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG STT Assistant",
    description="Speech-to-text + Retrieval Augmented Generation using Whisper + ChromaDB + Ollama",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(query_router)


@app.get("/health")
async def health():
    """
    Health check endpoint — used by Docker in Phase 4.
    Returns ok if all models loaded successfully.
    """
    return JSONResponse({
        "status": "ok",
        "app": settings.app_name,
        "ollama_model": settings.ollama_model,
        "whisper_model": settings.whisper_model,
        "embed_model": settings.embed_model_name,
    })


@app.get("/")
async def root():
    return JSONResponse({
        "message": "RAG STT Assistant is running.",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query  (multipart audio_file)"
    })