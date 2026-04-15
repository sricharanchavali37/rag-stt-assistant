from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "RAG STT Assistant"
    debug: bool = False
    chroma_path: str = "./chroma_db"
    collection_name: str = "documents"
    docs_path: str = "./docs"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_model_name: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    whisper_model: str = "base"
    top_k_results: int = 5
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()