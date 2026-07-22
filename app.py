import os
# Silence ChromaDB's internal telemetry connection errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Patch PyTorch class inspection conflict with Streamlit's file watcher
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

import streamlit as st

from document_parser import extract_text
from rag_pipeline import index_resume, screen_resume, improve_resume, ask_advisor

st.set_page_config(page_title="AI Resume Screener & Career Advisor", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "resume_id" not in st.session_state:
    st.session_state.resume_id = None
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}

# ---------------------------------------------------------------------------
# Sidebar: resume upload (shared across all tabs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload Resume")
    uploaded_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        if st.button("Index Resume", type="primary", use_container_width=True):
            with st.spinner("Extracting text and building embeddings..."):
                try:
                    text = extract_text(uploaded_file.name, uploaded_file.getvalue())
                    if not text.strip():
                        st.error("No extractable text found in this file.")
                    else:
                        resume_id, num_chunks = index_resume(text)
                        st.session_state.resume_id = resume_id
                        st.session_state.resume_filename = uploaded_file.name
                        st.session_state.chat_history = []
                        st.success(f"Indexed into {num_chunks} chunks.")
                except Exception as e:
                    st.error(f"Failed to index resume: {e}")

    if st.session_state.resume_id:
        st.info(f"**Active resume:** {st.session_state.resume_filename}\n\n`{st.session_state.resume_id}`")
        if st.button("Clear resume", use_container_width=True):
            st.session_state.resume_id = None
            st.session_state.resume_filename = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.warning("No resume indexed yet.")

    st.divider()
    st.caption(
        "Stack: **Streamlit** UI · **LangChain** RAG orchestration · "
        "**Hugging Face** embeddings (local) + LLM (Inference API) · **Chroma** vector store."
    )

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_screen, tab_advisor, tab_improve = st.tabs(
    ["🎯 Resume Screener", "💬 Career Advisor", "✨ Resume Improver"]
)

# --- Tab 1: Screener -------------------------------------------------------
with tab_screen:
    st.subheader("Screen your resume against a job description")

    job_description = st.text_area(
        "Paste the job description",
        height=200,
        placeholder="We are looking for a Python backend engineer with FastAPI, AWS...",
    )

    if st.button("Run Screening", type="primary"):
        if not st.session_state.resume_id:
            st.error("Upload and index a resume first (see sidebar).")
        elif not job_description.strip():
            st.error("Paste a job description first.")
        else:
            with st.spinner("Retrieving relevant resume sections and scoring..."):
                try:
                    result = screen_resume(st.session_state.resume_id, job_description)

                    score = result.get("overall_match_score", 0)
                    st.metric("Overall Match Score", f"{score}/100")
                    st.progress(min(max(score, 0), 100) / 100)

                    st.markdown(f"**Summary:** {result.get('summary', '')}")
                    st.markdown(f"**Recommendation:** {result.get('recommendation', '')}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**✅ Strengths**")
                        for s in result.get("strengths", []):
                            st.markdown(f"- {s}")
                    with col2:
                        st.markdown("**⚠️ Gaps**")
                        for g in result.get("gaps", []):
                            st.markdown(f"- {g}")

                    st.markdown("**Skill breakdown**")
                    for item in result.get("skill_breakdown", []):
                        icon = "✅" if item.get("present_in_resume") else "❌"
                        st.markdown(f"{icon} **{item.get('skill')}** — {item.get('note', '')}")

                except Exception as e:
                    st.error(f"Screening failed: {e}")

# --- Tab 2: Advisor chat ----------------------------------------------------
with tab_advisor:
    st.subheader("Ask your career advisor")
    st.caption(
        "Grounded in your uploaded resume when one is active — otherwise general career advice."
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about roles, skills, interview prep, career paths...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, used_context = ask_advisor(
                        question=question,
                        resume_id=st.session_state.resume_id,
                        chat_history=st.session_state.chat_history[:-1],
                    )
                    st.markdown(answer)
                    if used_context:
                        st.caption("📎 Grounded in your resume")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Advisor failed: {e}")

# --- Tab 3: Improver ---------------------------------------------------------
with tab_improve:
    st.subheader("Get resume improvement suggestions")

    target_role = st.text_input("Target role (optional)", placeholder="e.g. Senior Backend Engineer")

    if st.button("Get Suggestions", type="primary"):
        if not st.session_state.resume_id:
            st.error("Upload and index a resume first (see sidebar).")
        else:
            with st.spinner("Analyzing resume..."):
                try:
                    result = improve_resume(st.session_state.resume_id, target_role or None)

                    st.markdown(f"**Overall feedback:** {result.get('overall_feedback', '')}")
                    st.markdown("---")

                    for s in result.get("suggestions", []):
                        with st.expander(f"📌 {s.get('section', 'General')}"):
                            st.markdown(f"**Issue:** {s.get('issue', '')}")
                            st.markdown(f"**Suggestion:** {s.get('suggestion', '')}")

                except Exception as e:
                    st.error(f"Improvement analysis failed: {e}")
