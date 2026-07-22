"""RAG core built with LangChain + Hugging Face.

- Embeddings: local sentence-transformers model via HuggingFaceEmbeddings
  (no API calls, runs on CPU).
- Vector store: Chroma, one collection per resume (namespaced by resume_id)
  so retrieval never mixes chunks from different resumes.
- LLM: a hosted Hugging Face model called through HuggingFaceEndpoint
  (Inference API), wrapped in ChatHuggingFace for chat-style prompting.
"""

import json
import re
import uuid

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import settings

_embeddings = None
_llm = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)
    return _embeddings


def get_llm() -> ChatHuggingFace:
    global _llm
    if _llm is None:
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_llm_repo_id,
            huggingfacehub_api_token=settings.hf_token,
            max_new_tokens=1024,
            temperature=0.3,
        )
        _llm = ChatHuggingFace(llm=endpoint)
    return _llm


def _collection_name(resume_id: str) -> str:
    return f"resume_{resume_id}"


def _get_store(resume_id: str) -> Chroma:
    return Chroma(
        collection_name=_collection_name(resume_id),
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def index_resume(resume_text: str) -> tuple[str, int]:
    """Chunks + embeds a resume into a fresh Chroma collection. Returns
    (resume_id, num_chunks)."""
    resume_id = uuid.uuid4().hex[:12]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(resume_text)
    if not chunks:
        raise ValueError("No extractable text found in resume.")

    docs = [
        Document(page_content=chunk, metadata={"resume_id": resume_id, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    store = _get_store(resume_id)
    store.add_documents(docs)

    return resume_id, len(chunks)


def resume_exists(resume_id: str) -> bool:
    try:
        store = _get_store(resume_id)
        return store._collection.count() > 0
    except Exception:
        return False


def retrieve_relevant_chunks(resume_id: str, query: str, k: int | None = None) -> list[str]:
    if not resume_exists(resume_id):
        raise ValueError(f"Resume '{resume_id}' not found. Upload it first.")

    store = _get_store(resume_id)
    results = store.similarity_search(query, k=k or settings.top_k_results)
    return [doc.page_content for doc in results]


def get_full_resume_text(resume_id: str) -> str:
    if not resume_exists(resume_id):
        raise ValueError(f"Resume '{resume_id}' not found. Upload it first.")

    store = _get_store(resume_id)
    data = store.get()
    pairs = sorted(
        zip(data["metadatas"], data["documents"]),
        key=lambda p: p[0]["chunk_index"],
    )
    return "\n".join(doc for _, doc in pairs)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    """HF instruct models don't have a guaranteed JSON mode, so pull the
    first {...} block out of the response and parse it, with a clear error
    if the model didn't cooperate."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON. Raw output: {raw[:300]}")
    return json.loads(match.group(0))


SCREEN_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert technical recruiter. Evaluate how well the candidate
matches the job below, using ONLY the resume excerpts provided. Never invent
experience that isn't mentioned.

JOB DESCRIPTION:
{job_description}

RELEVANT RESUME EXCERPTS:
{resume_context}

Respond with ONLY a JSON object, no other text, matching exactly this schema:
{{
  "overall_match_score": <integer 0-100>,
  "summary": "<2-3 sentence overview>",
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "skill_breakdown": [{{"skill": "...", "present_in_resume": true, "note": "..."}}],
  "recommendation": "<Strong Match | Possible Match | Weak Match — one sentence why>"
}}"""
)


def screen_resume(resume_id: str, job_description: str) -> dict:
    chunks = retrieve_relevant_chunks(resume_id, query=job_description, k=8)
    resume_context = "\n---\n".join(chunks)

    chain = SCREEN_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"job_description": job_description, "resume_context": resume_context})

    result = _parse_json_response(raw)
    result["resume_id"] = resume_id
    return result


IMPROVE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert resume writer and career coach. Review the full
resume text below and give concrete, actionable improvement feedback.
Focus on: quantifying impact with metrics, strong action verbs, clarity,
removing filler, and ATS-friendliness.{role_line}

RESUME TEXT:
{resume_text}

Respond with ONLY a JSON object, no other text, matching exactly this schema:
{{
  "overall_feedback": "<2-3 sentence high-level assessment>",
  "suggestions": [
    {{"section": "...", "issue": "...", "suggestion": "..."}}
  ]
}}
Provide at least 4 suggestions covering different sections."""
)


def improve_resume(resume_id: str, target_role: str | None = None) -> dict:
    resume_text = get_full_resume_text(resume_id)
    role_line = f"\nTARGET ROLE: {target_role}" if target_role else ""

    chain = IMPROVE_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"resume_text": resume_text, "role_line": role_line})

    result = _parse_json_response(raw)
    result["resume_id"] = resume_id
    return result


ADVISOR_SYSTEM = """You are a warm, knowledgeable career advisor. You help with
career planning, skill development, job search strategy, and interview prep.
If resume context is given, ground your advice in that person's real
experience rather than generic tips. Keep answers concise and practical."""

ADVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ADVISOR_SYSTEM),
        ("human", "{chat_history}\n\nRESUME CONTEXT (if any):\n{resume_context}\n\nQuestion: {question}"),
    ]
)


def ask_advisor(question: str, resume_id: str | None, chat_history: list[dict]) -> tuple[str, bool]:
    used_resume_context = False
    resume_context = "(none provided)"

    if resume_id and resume_exists(resume_id):
        chunks = retrieve_relevant_chunks(resume_id, query=question, k=5)
        if chunks:
            used_resume_context = True
            resume_context = "\n---\n".join(chunks)

    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history[-6:])

    chain = ADVISOR_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {"question": question, "resume_context": resume_context, "chat_history": history_str}
    )

    return answer, used_resume_context
