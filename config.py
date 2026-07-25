import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    hf_token: str = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or ""
    hf_llm_repo_id: str = os.getenv("HF_LLM_REPO_ID", "meta-llama/Llama-3.2-3B-Instruct")
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", 5))


settings = Settings()
