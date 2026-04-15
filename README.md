\# RAG STT Assistant



A local, fully offline \*\*Retrieval-Augmented Generation\*\* system with \*\*Speech-to-Text\*\* input.

Ask questions by voice — get grounded answers from your own documents.



Built with: Whisper · ChromaDB · MiniLM · Ollama (llama3:8b) · FastAPI



\---



\## Architecture

Audio (WAV/MP3)

→ Whisper STT          (speech → text)

→ MiniLM Embedder      (text → float32\[384])

→ ChromaDB Retriever   (vector → top-5 chunks)

→ Prompt Builder       (chunks + query → grounded prompt)

→ Ollama llama3:8b     (prompt → answer)

→ FastAPI Response     (answer → JSON)


## Project Structure

rag-stt-assistant/

├── main.py                  # FastAPI app + startup

├── ingest.py                # Offline PDF ingestion pipeline

├── evaluate.py              # RAG evaluation script

├── config.py                # All settings in one place

├── models.py                # RAGContext + DocumentChunk dataclasses

├── services/

│   ├── transcriber.py       # Whisper STT

│   ├── retriever.py         # MiniLM embed + ChromaDB query

│   ├── prompt\_builder.py    # Grounding prompt assembly

│   └── generator.py        # Ollama LLM call

├── routers/

│   └── query.py             # POST /query endpoint

├── docs/                    # Put your PDFs here

├── chroma\_db/               # Auto-created by ingest.py

├── eval/

│   ├── eval\_dataset.json    # 12 evaluation questions

│   └── ragas\_results.json   # Evaluation output

├── Dockerfile

├── docker-compose.yml

└── requirements.txt

