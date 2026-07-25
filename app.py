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
        color: #1E293B;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
    }
    .missing-skill-box {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        color: #7F1D1D;
        font-weight: 600;
        font-size: 0.98rem;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .matched-skill-box {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        color: #14532D;
        font-weight: 600;
        font-size: 0.98rem;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
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
if "improve_result" not in st.session_state:
    st.session_state.improve_result = None
if "target_role_preset" not in st.session_state:
    st.session_state.target_role_preset = ""

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
                    st.session_state.improve_result = None

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
            st.session_state.improve_result = None
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
st.markdown('<div class="main-title">AI Resume Screener & Career Advisor</div>', unsafe_allow_html=True)
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
# ===========================================================================
# TAB 2: FOLLOW-UP CHAT & ADVISOR
# ===========================================================================
with tab_advisor:
    st.subheader("💬 Interactive Career Advisor & Interview Coach")
    st.caption(
        "Get real-time, personalized guidance grounded in your resume vector index and target job requirements."
    )

    # Interactive Guide Expander for Easy Understanding
    with st.expander("💡 **How to Use this AI Advisor (Quick Guide)**", expanded=False):
        st.markdown(
            """
            - 🎯 **Tailored Job Matching**: Ask about key qualifications and matching strengths for the active job.
            - ⚠️ **Addressing Gaps**: Get strategic answers and sample interview scripts for missing skills.
            - 💬 **Interview Preparation**: Practice expected technical & behavioral interview questions.
            - ⚡ **Instant One-Click Prompts**: Click any quick action button below to start immediately!
            - 🧠 **Vector RAG Grounding**: Every answer retrieves relevant context chunks from your uploaded resume.
            """
        )

    # Active Context Indicator & Controls
    col_status, col_clear = st.columns([3, 1])
    with col_status:
        if st.session_state.resume_id and st.session_state.job_description:
            st.success(
                f"📎 **Active RAG Grounding:** Grounded in `{st.session_state.resume_filename}` & Target Job Description."
            )
        elif st.session_state.resume_id:
            st.info(f"📎 **Active RAG Grounding:** Grounded in uploaded resume `{st.session_state.resume_filename}`.")
        else:
            st.warning("⚠️ **General Mode:** Upload a resume in the sidebar to enable full RAG-grounded insights.")

    with col_clear:
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Reset conversation history"):
                st.session_state.chat_history = []
                st.rerun()

    # Quick Question Action Buttons
    st.markdown("##### ⚡ Quick Prompt Shortcuts")
    prompt_to_process = None

    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        if st.button("🎯 Top Strengths", use_container_width=True, help="Analyze matching strengths"):
            prompt_to_process = "What are my top 3 matching strengths and key qualifications for this position?"
    with qcol2:
        if st.button("⚠️ Handle Skill Gaps", use_container_width=True, help="Strategy to address missing requirements"):
            prompt_to_process = "How can I effectively address missing skills or requirements during an interview?"
    with qcol3:
        if st.button("💬 Likely Questions", use_container_width=True, help="Targeted technical & behavioral questions"):
            prompt_to_process = "What targeted technical and behavioral interview questions should I prepare for?"
    with qcol4:
        if st.button("📝 Resume Bullets", use_container_width=True, help="Role-tailored resume bullet point scripts"):
            prompt_to_process = "Can you provide role-tailored bullet point scripts to highlight my key achievements?"

    st.divider()

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_query = st.chat_input("Ask any question about your resume, job fit, interview prep, or career advice...")
    if user_query:
        prompt_to_process = user_query

    # Process Query (from input box or quick prompt button)
    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector index & generating advisor guidance..."):
                try:
                    answer, used_resume = ask_advisor(
                        question=prompt_to_process,
                        resume_id=st.session_state.resume_id,
                        job_description=st.session_state.job_description,
                        chat_history=st.session_state.chat_history[:-1],
                    )
                    st.markdown(answer)
                    if used_resume:
                        st.caption("📎 *Answer grounded in retrieved candidate vector database chunks.*")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Advisor error: {str(e)}")
        st.rerun()

# ===========================================================================
# TAB 3: RESUME IMPROVER
# ===========================================================================
with tab_improve:
    st.subheader("✨ Comprehensive Resume Optimizer & Role Tailoring Engine")
    st.caption(
        "Generates section-by-section rewrite suggestions showing EXACTLY what to add, modify, or rephrase to make your resume 100% suitable for your target role."
    )

    # Interactive Guide Expander for Easy Understanding
    with st.expander("💡 **How to Use Resume Optimizer (Quick Guide)**", expanded=False):
        st.markdown(
            """
            - 🎯 **Target Role Alignment**: Select or type your desired target job title to receive customized recommendations.
            - 🔍 **Full Resume Analysis**: Evaluates your uploaded resume text against target role keywords and ATS best practices.
            - 📝 **Copy-Pasteable Rewrites**: Provides exact text for Professional Summary, Skills, Work Experience, and ATS formatting.
            - ⚡ **Role Preset Shortcuts**: Click any target role preset button below to quickly populate target job titles!
            """
        )

    # Active Alignment Banner
    if st.session_state.job_description and st.session_state.job_description.strip():
        st.success("🎯 **Target Role Alignment:** Recommendations will analyze your resume against the active Job Description.")

    # Target Role Quick Selection Shortcuts
    st.markdown("##### ⚡ Quick Target Role Shortcuts")
    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    with rcol1:
        if st.button("💻 Backend Engineer", use_container_width=True, help="Set target role to Senior Backend Engineer"):
            st.session_state.target_role_preset = "Senior Backend Engineer"
    with rcol2:
        if st.button("🌐 Full Stack Dev", use_container_width=True, help="Set target role to Full Stack Developer"):
            st.session_state.target_role_preset = "Full Stack Developer"
    with rcol3:
        if st.button("📊 Data & AI Engineer", use_container_width=True, help="Set target role to Data Scientist / AI Engineer"):
            st.session_state.target_role_preset = "Data Scientist / AI Engineer"
    with rcol4:
        if st.button("☁️ DevOps & Cloud", use_container_width=True, help="Set target role to DevOps / Cloud Engineer"):
            st.session_state.target_role_preset = "DevOps / Cloud Engineer"

    target_role_input = st.text_input(
        "Target Role Title:",
        value=st.session_state.target_role_preset,
        placeholder="e.g. Senior Backend Engineer / Full Stack Developer",
        help="Specify the target job title to customize resume recommendations.",
    )

    if st.button("✨ Analyze Full Resume & Generate Role-Tailored Enhancements", type="primary", use_container_width=True):
        if not st.session_state.resume_id:
            st.error("Please upload and index a resume in the sidebar first.")
        else:
            with st.spinner("Analyzing full resume text against target role requirements..."):
                try:
                    improve_res = improve_resume(
                        resume_id=st.session_state.resume_id,
                        target_role=target_role_input or None,
                        job_description=st.session_state.job_description or None,
                    )
                    st.session_state.improve_result = improve_res
                except Exception as e:
                    st.error(f"Improvement engine error: {str(e)}")

    # Render Persistent Improvement Results if present
    if st.session_state.improve_result:
        res = st.session_state.improve_result
        st.divider()

        st.markdown("### 📊 Overall Role Suitability Assessment")
        st.info(res.get("overall_feedback", "Evaluation complete."))

        st.markdown("### 📌 Actionable Section-by-Section Recommendations")
        suggestions = res.get("suggestions", [])
        for idx, suggestion in enumerate(suggestions, 1):
            sec = suggestion.get("section", "General")
            with st.expander(f"📌 **{idx}. {sec}**", expanded=True):
                st.markdown(f"**Identified Weakness / Gap for Role:**\n{suggestion.get('issue', '')}")
                st.markdown(f"**Exact Recommendation / Copy-Pasteable Phrasing:**\n{suggestion.get('suggestion', '')}")
