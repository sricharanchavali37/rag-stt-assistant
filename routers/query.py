"""
routers/query.py
Defines the POST /query endpoint.
All pipeline logic lives in services/ — this file is only routing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from services.transcriber import transcribe
from services.retriever import embed_and_retrieve
from services.prompt_builder import build_prompt
from services.generator import generate
from models import RAGContext
from config import settings

router = APIRouter()


@router.post("/query")
async def query_endpoint(
    request: Request,
    audio_file: UploadFile = File(...),
):
    """
    Main pipeline endpoint.
    Accepts a multipart audio file upload.
    Runs all 4 stages in sequence and returns a grounded JSON answer.
    """

    # Read the uploaded audio into memory
    audio_bytes = await audio_file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # Pull the preloaded models from app state (set in main.py lifespan)
    whisper_model = request.app.state.whisper_model
    embed_model   = request.app.state.embed_model
    collection    = request.app.state.collection

    # ── Stage 1: Speech → Text ──────────────────────────────────────────────
    ctx = transcribe(audio_bytes, audio_file.filename or "audio.wav", whisper_model)
    if ctx.error:
        return JSONResponse({"error": ctx.error, "stage": "transcription"}, status_code=500)

    # ── Stage 2: Text → Vector → Top-k Chunks ───────────────────────────────
    ctx = embed_and_retrieve(ctx, embed_model, collection, top_k=settings.top_k_results)
    if ctx.error:
        return JSONResponse({"error": ctx.error, "stage": "retrieval"}, status_code=500)

    # ── Stage 3: Build Grounding Prompt ─────────────────────────────────────
    ctx = build_prompt(ctx)
    if ctx.error:
        return JSONResponse({"error": ctx.error, "stage": "prompt_builder"}, status_code=500)

    # ── Stage 4: LLM Generation ──────────────────────────────────────────────
    ctx = await generate(ctx)
    if ctx.error:
        return JSONResponse({"error": ctx.error, "stage": "generation"}, status_code=500)

    # ── Response ─────────────────────────────────────────────────────────────
    return JSONResponse({
        "transcription": ctx.transcription,
        "query":         ctx.query,
        "answer":        ctx.answer,
        "sources":       ctx.sources,
        "chunks_used":   len(ctx.retrieved_chunks),
    })