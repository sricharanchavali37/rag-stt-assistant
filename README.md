\# RAG-Based Speech-to-Text Knowledge Assistant



A fully local \*\*Retrieval-Augmented Generation (RAG)\*\* system that answers user queries from document knowledge bases using \*\*speech input\*\*, while ensuring responses remain grounded and non-hallucinatory.



\---



\## Overview



This system enables users to ask questions via audio and receive answers strictly derived from indexed documents.



The pipeline integrates:



\* Speech-to-text transcription (Whisper)

\* Semantic retrieval over PDFs (ChromaDB + MiniLM)

\* Context-grounded response generation using a local LLM (Ollama)



All components run \*\*locally on CPU\*\*, with no dependency on external APIs or cloud services.



\---



\## Key Capabilities



\* 🎤 Accepts audio queries (WAV/MP3)

\* 🧠 Converts speech to text using Whisper

\* 📄 Retrieves relevant document chunks using vector similarity

\* 🤖 Generates answers grounded strictly in retrieved context

\* 🚫 Prevents hallucination via constrained prompting

\* ⚡ Exposes FastAPI endpoint for real-time interaction



\---



\## System Architecture



!\[Architecture](./assets/architecture.svg)



\### Pipeline



Audio

→ Whisper (Speech-to-Text)

→ MiniLM (Embeddings)

→ ChromaDB (Top-K Retrieval)

→ Prompt Builder (Context Injection)

→ Ollama LLM (llama3:8b)

→ JSON Response



\---



\## Project Structure



```

rag-stt-assistant/



├── main.py

├── ingest.py

├── evaluate.py

├── config.py

├── models.py



├── services/

│   ├── transcriber.py

│   ├── retriever.py

│   ├── prompt\_builder.py

│   └── generator.py



├── routers/

│   └── query.py



├── docs/

├── chroma\_db/



├── eval/

│   ├── eval\_dataset.json

│   └── results.json



├── assets/

│   └── architecture.svg



├── Dockerfile

├── docker-compose.yml

└── requirements.txt

```



\---



\## Setup \& Run (Local)



\### Prerequisites



\* Python 3.10+

\* Ollama installed and running



```

ollama pull llama3:8b

```



\---



\### Install



```

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

```



\---



\### Step 1 — Ingest Documents



Place PDFs inside `docs/` and run:



```

python ingest.py

```



\---



\### Step 2 — Start API



```

python -m uvicorn main:app --host 0.0.0.0 --port 8000

```



\---



\### Step 3 — Query System



Open:



```

http://localhost:8000/docs

```



Upload audio file to `/query`.



\---



\## Run with Docker



\### Build Image



```

docker build -t rag-stt-assistant .

```



\### Run Container



```

docker run -p 8002:8000 -v "D:/projects and stuff/rag-stt-assistant/chroma\_db:/app/chroma\_db" rag-stt-assistant

```



\### Access API



```

http://localhost:8002/docs

```



\---



\## Example Output



```json

{

&#x20; "transcription": "what is the time complexity of binary search",

&#x20; "answer": "Binary search has time complexity O(log N).",

&#x20; "sources": \["document.pdf"],

&#x20; "chunks\_used": 5

}

```



\---



\## Evaluation Approach



Standard asynchronous evaluation frameworks were not used due to constraints of local CPU execution.



Instead, a deterministic evaluation strategy was implemented:



\* Retrieved context inspection

\* Answer grounding validation

\* Keyword-based correctness matching



This ensures that:



\* Responses are derived from source documents

\* Hallucination is minimized

\* Outputs remain interpretable



\---



\## Deployment Design Decision



The system is \*\*containerized using Docker\*\* and designed for \*\*local deployment\*\*.



\### Why not cloud platforms (Vercel / Netlify / Render / Railway)?



These platforms are not suitable for this architecture because:



\* They do not support running local LLM runtimes (Ollama)

\* They restrict long-running CPU-bound processes

\* They are optimized for lightweight web services, not AI pipelines



\### Why Docker?



Docker ensures:



\* Reproducibility across environments

\* Consistent model loading and execution

\* Full control over local resources

\* No dependency on external APIs



\---



\## Constraints



\* CPU-only execution

\* Local LLM (Ollama)

\* No external inference APIs



\---



\## Impact



\* Higher latency compared to GPU systems

\* Limited parallel processing

\* Retrieval quality depends on document structure



\---



\## Design Focus



\* Groundedness over generation

\* Reproducibility over scale

\* System reliability over optimization



\---



\## Limitations



\* Performance constrained by CPU inference

\* No large-scale automated evaluation (e.g., RAGAS)

\* Sensitive to noisy or poorly structured PDFs



\---



\## Future Improvements



\* Hybrid retrieval (keyword + vector)

\* Improved chunking strategies

\* Optional GPU acceleration

\* Lightweight evaluation metrics



\---



\## Tech Stack



\* Python

\* Whisper

\* Sentence Transformers (MiniLM)

\* ChromaDB

\* Ollama (llama3:8b)

\* FastAPI



\---



\## Summary



This project demonstrates a complete \*\*end-to-end local RAG pipeline\*\* capable of answering questions from documents using speech input, while maintaining strict grounding under compute constraints and ensuring reproducible deployment via Docker.



