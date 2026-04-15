"""
services/generator.py — Stage 4
Sends the assembled prompt to Ollama via httpx.
Uses async httpx so FastAPI can handle other requests while waiting.
stream: false — waits for the full response before returning.
"""

import httpx
from models import RAGContext
from config import settings


async def generate(ctx: RAGContext) -> RAGContext:
    """
    Takes a RAGContext with the prompt field populated.
    Calls Ollama's /api/generate endpoint.
    Returns the same context with the answer field filled.
    """

    if not ctx.prompt:
        ctx.error = "Cannot generate — prompt is empty. Check prompt_builder stage."
        return ctx

    payload = {
        "model": settings.ollama_model,
        "prompt": ctx.prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "").strip()

            if not answer:
                ctx.error = "Ollama returned an empty response."
                return ctx

            ctx.answer = answer

    except httpx.ConnectError:
        ctx.error = (
            f"Cannot reach Ollama at {settings.ollama_base_url}. "
            "Is 'ollama serve' running?"
        )
    except httpx.TimeoutException:
        ctx.error = "Ollama request timed out after 120 seconds."
    except Exception as e:
        ctx.error = f"Generation failed: {str(e)}"

    return ctx