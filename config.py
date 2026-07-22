
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()


class Settings:
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    hf_llm_repo_id = os.getenv("HF_LLM_REPO_ID", "Qwen/Qwen2.5-7B-Instruct")
    hf_embedding_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
    top_k_results = int(os.getenv("TOP_K_RESULTS", 6))


settings = Settings()
