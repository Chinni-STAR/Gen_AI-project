import os
# Disable TensorFlow import in Transformers
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
# Silence ChromaDB internal telemetry connection warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Patch PyTorch class inspection conflict with Streamlit file watcher
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

import streamlit as st

from config import settings
from document_parser import extract_text
from rag_pipeline import (
    index_resume,
    screen_resume,
    improve_resume,
    ask_advisor,
    get_rag_diagnostics,
)

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener & Career Advisor (RAG Engine)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look and clear visual hierarchy
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .suggestion-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .missing-skill-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 4px;
    }
    .matched-skill-box {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "resume_id" not in st.session_state:
    st.session_state.resume_id = None
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = None
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "screener_result" not in st.session_state:
    st.session_state.screener_result = None
if "rag_trace" not in st.session_state:
    st.session_state.rag_trace = None

# ---------------------------------------------------------------------------
# Sidebar: Upload & Technical Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Resume Ingestion")
    st.caption("Step 1: Upload candidate document")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        help="Supports searchable text PDFs, Word documents, and plain text files.",
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index Resume", type="primary", use_container_width=True):
            with st.spinner("Extracting text, chunking, and embedding into ChromaDB..."):
                try:
                    raw_bytes = uploaded_file.getvalue()
                    extracted_text = extract_text(uploaded_file.name, raw_bytes)
                    resume_id, num_chunks, chunks = index_resume(extracted_text)

                    st.session_state.resume_id = resume_id
                    st.session_state.resume_filename = uploaded_file.name
                    st.session_state.chat_history = []
                    st.session_state.screener_result = None
                    st.session_state.rag_trace = None

                    st.success(f"Indexed **{uploaded_file.name}** into **{num_chunks} vector chunks**!")
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    if st.session_state.resume_id:
        st.divider()
        st.markdown(f"**Active Resume:** `{st.session_state.resume_filename}`")
        st.caption(f"Vector Collection ID: `{st.session_state.resume_id}`")
        if st.button("🗑️ Clear Active Resume", use_container_width=True):
            st.session_state.resume_id = None
            st.session_state.resume_filename = None
            st.session_state.chat_history = []
            st.session_state.screener_result = None
            st.session_state.rag_trace = None
            st.rerun()
    else:
        st.info("No active resume loaded. Upload a file above to begin.")



    if not settings.hf_token:
        st.warning(
            "⚠️ **API Token Missing**: `HUGGINGFACEHUB_API_TOKEN` is not set. Add it to your `.env` file to enable LLM scoring."
        )

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">AI Resume Screener & RAG Career Advisor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation engine for automated resume screening, skill gap analysis, and interview preparation.</div>',
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab_screen, tab_advisor, tab_improve = st.tabs(
    ["🎯 Resume Screener & Evaluation", "💬 Follow-up Chat & Advisor", "✨ Resume Improver"]
)

# ===========================================================================
# TAB 1: RESUME SCREENER & RAG VISUALIZER
# ===========================================================================
with tab_screen:
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("1. Target Job Description")
        job_input = st.text_area(
            "Paste Job Description here:",
            value=st.session_state.job_description,
            height=260,
            placeholder="e.g. Seeking a Senior Python Engineer with 3+ years experience in FastAPI, Docker, PostgreSQL, and cloud deployments...",
        )
        st.session_state.job_description = job_input

        run_screener = st.button("⚡ Run RAG Screening & Evaluation", type="primary", use_container_width=True)

    with col_right:
        st.subheader("2. Candidate Evaluation")
        if run_screener:
            if not st.session_state.resume_id:
                st.error("Please upload and index a resume first using the left sidebar.")
            elif not job_input.strip():
                st.error("Please paste a target Job Description to screen against.")
            else:
                with st.spinner("Executing RAG Pipeline: Retrieving vector chunks & scoring..."):
                    try:
                        result = screen_resume(st.session_state.resume_id, job_input)
                        trace = get_rag_diagnostics(st.session_state.resume_id, job_input)

                        st.session_state.screener_result = result
                        st.session_state.rag_trace = trace
                    except Exception as e:
                        st.error(f"Screening failed: {str(e)}")

        if st.session_state.screener_result:
            res = st.session_state.screener_result
            fit_score = res.get("fit_score", 0)

            # Fit Score Gauge Metric
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                st.metric("Fit Score", f"{fit_score} / 100")
            with m_col2:
                st.markdown(f"**Recommendation:** {res.get('recommendation', 'N/A')}")
                st.progress(fit_score / 100.0)

            st.markdown(f"**Executive Summary:** {res.get('summary', '')}")

    # Results Breakdown Section (if results exist)
    if st.session_state.screener_result:
        res = st.session_state.screener_result
        st.divider()

        # Matched vs Missing Skills Columns
        s_col1, s_col2 = st.columns(2, gap="large")

        with s_col1:
            st.markdown("### ✅ Matched Skills")
            matched = res.get("matched_skills", [])
            if matched:
                for skill in matched:
                    st.markdown(f'<div class="matched-skill-box"><b>✓</b> {skill}</div>', unsafe_allow_html=True)
            else:
                st.info("No explicit skills matched directly.")

        with s_col2:
            st.markdown("### ❌ Missing / Required Skills")
            missing = res.get("missing_skills", [])
            if missing:
                for skill in missing:
                    st.markdown(f'<div class="missing-skill-box"><b>✗</b> {skill}</div>', unsafe_allow_html=True)
            else:
                st.success("No critical skill gaps identified.")

        # Exactly Three Actionable Suggestions
        st.divider()
        st.markdown("### 🎯 3 Actionable Suggestions to Improve Resume")
        suggestions = res.get("suggestions", [])
        for idx, sugg in enumerate(suggestions[:3], 1):
            st.markdown(
                f'<div class="suggestion-box"><b>Suggestion {idx}:</b> {sugg}</div>',
                unsafe_allow_html=True,
            )

        # Skill Breakdown Table
        if res.get("skill_breakdown"):
            with st.expander("📊 Detailed Skill Breakdown Table"):
                for item in res.get("skill_breakdown", []):
                    icon = "✅" if item.get("present_in_resume") else "❌"
                    st.markdown(f"{icon} **{item.get('skill')}** — {item.get('note', '')}")

    # =======================================================================
    # INTERACTIVE RAG WORKFLOW VISUALIZER (INTERVIEW DEMO)
    # =======================================================================
    if st.session_state.rag_trace:
        trace = st.session_state.rag_trace
        st.divider()

        with st.expander("🔍 **Interactive RAG Workflow & Vector DB Inspector (Click to Demonstrate to Interviewer)**", expanded=False):


            rag_tab1, rag_tab2, rag_tab3, rag_tab4, rag_tab5 = st.tabs(
                [
                    "1. Chunking",
                    "2. Vector Embeddings",
                    "3. ChromaDB Vector Store",
                    "4. Similarity Retrieval",
                    "5. Augmented Prompt",
                ]
            )

            with rag_tab1:
                st.markdown(f"**Text Splitter Configuration:** `RecursiveCharacterTextSplitter(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})`")
                st.markdown(f"**Total Document Chunks Created:** `{trace['total_chunks']}`")
                for c in trace["all_chunks"]:
                    st.text_area(
                        f"Chunk #{c['index']} ({c['length']} chars):",
                        value=c["content"],
                        height=100,
                        key=f"chunk_preview_{c['index']}",
                    )

            with rag_tab2:
                st.markdown(f"**Embedding Model:** `{trace['embedding_model']}` (runs locally on CPU)")
                st.markdown(f"**Embedding Vector Dimensions:** `{trace['embedding_dimension']}` float32 dimensions")
                st.markdown("**Sample High-Dimensional Dense Vector Slice:**")
                st.code(f"Dense Vector Sample (first 6 values): {trace['sample_vector_slice']} ...", language="python")

            with rag_tab3:
                st.markdown(f"**ChromaDB Persist Directory:** `{settings.chroma_persist_dir}`")
                st.markdown(f"**Collection Name:** `resume_{trace['resume_id']}`")
                st.markdown(f"**Stored Vector Records Count:** `{trace['total_chunks']}`")

            with rag_tab4:
                st.markdown(f"**Retrieval Query:** `{st.session_state.job_description[:100]}...`")
                st.markdown(f"**Top-{settings.top_k_results} Vector Cosine Distance Search Results:**")
                for item in trace["retrieved_results"]:
                    st.markdown(f"📍 **Chunk Index #{item['chunk_index']}** (Distance Metric: `{item['distance']}`)")
                    st.info(item["content"])

            with rag_tab5:
                st.markdown("**Final Context-Augmented Prompt Injected into LLM:**")
                st.code(trace["formatted_prompt"], language="markdown")

# ===========================================================================
# TAB 2: FOLLOW-UP CHAT & ADVISOR
# ===========================================================================
with tab_advisor:
    st.subheader("💬 Interactive Candidate & Job Preparation Advisor")
    st.caption(
        "Ask follow-up questions about candidate strengths, interview strategies, missing qualification mitigations, or targeted technical questions."
    )

    if st.session_state.resume_id and st.session_state.job_description:
        st.success(
            f"📎 **Active RAG Context:** Grounded in `{st.session_state.resume_filename}` and current Job Description."
        )
    elif st.session_state.resume_id:
        st.info(f"📎 **Active RAG Context:** Grounded in uploaded resume `{st.session_state.resume_filename}`.")
    else:
        st.warning("⚠️ No active resume uploaded. Chat will provide general career coaching advice.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Chat Input
    user_query = st.chat_input("e.g. How can the candidate address the missing Kubernetes requirement in an interview?")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context & generating response..."):
                try:
                    answer, used_resume = ask_advisor(
                        question=user_query,
                        resume_id=st.session_state.resume_id,
                        job_description=st.session_state.job_description,
                        chat_history=st.session_state.chat_history[:-1],
                    )
                    st.markdown(answer)
                    if used_resume:
                        st.caption("📎 Answer grounded in candidate's vector database chunks.")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Advisor error: {str(e)}")

# ===========================================================================
# TAB 3: RESUME IMPROVER
# ===========================================================================
with tab_improve:
    st.subheader("✨ Comprehensive Resume Optimizer & Role Tailoring Engine")
    st.caption("Generates section-by-section rewrite suggestions showing EXACTLY what to add, modify, or rephrase to make your resume 100% suitable for your target role.")

    if st.session_state.job_description and st.session_state.job_description.strip():
        st.success("🎯 **Target Role Alignment:** Recommendations will analyze your resume against the active Job Description.")

    target_role_input = st.text_input(
        "Target Role Title:",
        placeholder="e.g. Senior Backend Engineer / Full Stack Developer",
        help="Specify the target job title to customize resume recommendations.",
    )

    if st.button("✨ Analyze Full Resume & Generate Role-Tailored Enhancements", type="primary"):
        if not st.session_state.resume_id:
            st.error("Please upload and index a resume in the sidebar first.")
        else:
            with st.spinner("Analyzing resume against target role requirements..."):
                try:
                    improve_res = improve_resume(
                        resume_id=st.session_state.resume_id,
                        target_role=target_role_input or None,
                        job_description=st.session_state.job_description or None,
                    )

                    st.markdown(f"### Overall Role Suitability Assessment\n{improve_res.get('overall_feedback', '')}")
                    st.divider()

                    st.markdown("### Actionable Section Recommendations")
                    for suggestion in improve_res.get("suggestions", []):
                        sec = suggestion.get("section", "General")
                        with st.expander(f"📌 **{sec}**", expanded=True):
                            st.markdown(f"**Identified Weakness / Gap for Role:** {suggestion.get('issue', '')}")
                            st.markdown(f"**Exact Recommendation / Addition for Suitability:** {suggestion.get('suggestion', '')}")
                except Exception as e:
                    st.error(f"Improvement engine error: {str(e)}")
