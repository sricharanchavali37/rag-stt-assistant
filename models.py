from dataclasses import dataclass, field
from typing import Optional, List

# --- Phase 1 + Phase 3: the bundle that flows through the online pipeline ---
@dataclass
class RAGContext:
    query: str = ""
    transcription: str = ""
    retrieved_chunks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    prompt: str = ""
    answer: str = ""
    error: Optional[str] = None

# --- Phase 2: typed chunk for the ingestion pipeline ---
@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: str
    embedding: Optional[List[float]] = None