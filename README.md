# AI Resume Screener & Career Advisor — Streamlit + LangChain + Hugging Face

Same RAG concept as before, rebuilt on a fully open-source-friendly stack:

- **UI**: Streamlit (single-page app, three tabs)
- **RAG orchestration**: LangChain (LCEL chains, prompt templates, text splitter)
- **Embeddings**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2`, run
  **locally** — no API calls, no cost
- **LLM**: a hosted Hugging Face model (default: `mistralai/Mistral-7B-Instruct-v0.3`)
  called via the free-tier HF Inference API
- **Vector store**: Chroma, persisted locally, one collection per resume

## Project structure

```
resume-rag-streamlit/
├── app.py               # Streamlit UI (upload, screener, advisor chat, improver)
├── rag_pipeline.py       # LangChain RAG core: embeddings, Chroma, HF LLM, chains
├── document_parser.py    # PDF/DOCX/TXT text extraction
├── config.py              # Env var settings
├── requirements.txt
├── .env.example
└── data/chroma/           # local vector DB storage (created at runtime)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and add your HUGGINGFACEHUB_API_TOKEN
# (free token: https://huggingface.co/settings/tokens)

streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How it works

1. **Upload** a resume in the sidebar → text is extracted, split into
   ~1000-token chunks (150-token overlap) via LangChain's
   `RecursiveCharacterTextSplitter`, embedded locally with a
   sentence-transformers model, and stored in a Chroma collection scoped to
   that resume's generated ID.
2. **Screener tab**: your job description is used as the retrieval query,
   pulling back the resume chunks most relevant to that specific role, then
   scored by the LLM into a structured JSON result (match score, strengths,
   gaps, skill breakdown).
3. **Advisor tab**: a chat interface. Each question is used to retrieve
   relevant resume chunks (if a resume is active) so advice is grounded in
   the person's real background; otherwise it falls back to general career
   guidance.
4. **Improver tab**: retrieves the full resume text (not just top-k chunks)
   and asks the LLM for section-by-section, actionable feedback.

## Swapping models

- **Different HF LLM**: change `HF_LLM_REPO_ID` in `.env` to any
  instruction-tuned model available on the HF Inference API (e.g.
  `HuggingFaceH4/zephyr-7b-beta`, `meta-llama/Meta-Llama-3-8B-Instruct`).
  Larger/gated models may need a Pro HF account or a dedicated Inference
  Endpoint.
- **Run the LLM fully locally** (no API, no token needed): replace
  `HuggingFaceEndpoint` in `rag_pipeline.py` with a local
  `transformers` pipeline via LangChain's `HuggingFacePipeline`, e.g.:
  ```python
  from langchain_huggingface import HuggingFacePipeline
  llm = HuggingFacePipeline.from_model_id(
      model_id="google/flan-t5-large",
      task="text2text-generation",
  )
  ```
  This trades API dependency for local compute/GPU requirements.
- **Different embeddings**: change `HF_EMBEDDING_MODEL` in `.env` to any
  sentence-transformers model, e.g. `BAAI/bge-small-en-v1.5` for stronger
  retrieval quality at a bit more compute cost.

## Notes

- HF instruct models don't have a guaranteed JSON output mode (unlike
  OpenAI's `response_format`), so `rag_pipeline.py` extracts the first
  `{...}` block from the model's response and parses it. If a model ignores
  the JSON instruction, you'll see a clear error — try a stronger instruct
  model if that happens often.
- This is a local, single-user scaffold (Streamlit session state, no auth).
  For multi-user deployment, move resume/session state out of
  `st.session_state` and into a real datastore keyed by user ID.
