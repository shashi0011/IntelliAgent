import os
import shutil
import time
from html import escape

import streamlit as st

from agents.ingestion import IngestionAgent
from config.settings import DEFAULT_EMBEDDING_PROVIDER, DEFAULT_LLM_PROVIDER, logger
from graph.workflow import run_pipeline
from vector_store.hybrid_retriever import load_hybrid_retriever

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="IntelliAgent",
    page_icon="IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* ── Main content ── */
    .block-container {
        max-width: 920px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }

    /* ── Sidebar container ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 2.5rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }

    /* ── Sidebar identity block ── */
    .sb-brand {
        margin-bottom: 1.4rem;
    }
    .sb-brand h2 {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0 0 0.15rem;
        line-height: 1.2;
    }
    .sb-brand p {
        color: rgba(128, 128, 128, 0.9);
        font-size: 0.82rem;
        margin: 0;
        line-height: 1.4;
    }

    /* ── Section divider + label ── */
    .sb-section {
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        margin-top: 0.6rem;
        padding-top: 0.7rem;
    }
    .sb-label {
        color: rgba(128, 128, 128, 0.75);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }

    /* ── Document pill ── */
    .doc-pill {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 0.4rem;
        font-size: 0.84rem;
        line-height: 1.4;
        margin: 0.2rem 0;
        overflow-wrap: anywhere;
        padding: 0.38rem 0.55rem;
    }

    /* ── File uploader ── */
    div[data-testid="stFileUploader"] section {
        padding: 0.4rem;
    }

    /* ── Main intro ── */
    .intro {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.2rem;
        padding-bottom: 0.9rem;
    }
    .intro h1 {
        font-size: 1.75rem;
        line-height: 1.15;
        margin: 0;
    }
    .intro p {
        color: rgba(128, 128, 128, 0.9);
        font-size: 0.97rem;
        margin: 0.35rem 0 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def clear_uploaded_files() -> None:
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def stream_words(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            filename = source.get("filename", "Unknown")
            page = source.get("page", "Unknown")
            snippet = source.get("snippet", "")
            st.markdown(f"**{index}. {filename} — page {page}**")
            st.caption(snippet)


# ── Session state ──────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None

retriever = load_hybrid_retriever(top_k=5)
if retriever is not None and not st.session_state.ingested_files:
    st.session_state.ingested_files = ["Previously ingested data"]


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:

    # Brand
    st.markdown(
        """
        <div class="sb-brand">
            <h2>IntelliAgent</h2>
            <p>Document chat — upload PDFs or TXT files and ask questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload section
    st.markdown(
        "<div class='sb-section'><div class='sb-label'>Upload documents</div></div>",
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_file_names = [f.name for f in uploaded_files]
        already_ingested = all(n in st.session_state.ingested_files for n in new_file_names)

        if already_ingested:
            st.success("Already indexed.")
        else:
            if st.button("Process documents", use_container_width=True):
                with st.spinner("Indexing…"):
                    saved_paths = []
                    for file in uploaded_files:
                        path = os.path.join(UPLOAD_DIR, file.name)
                        with open(path, "wb") as fh:
                            fh.write(file.getvalue())
                        saved_paths.append(path)
                    try:
                        agent = IngestionAgent(chunk_size=800, chunk_overlap=150)
                        agent.ingest(saved_paths)
                        st.session_state.ingested_files = new_file_names
                        st.success("Indexed.")
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Ingestion failed")
                        st.error(f"Ingestion failed: {exc}")

    # Active documents section
    st.markdown(
        "<div class='sb-section'><div class='sb-label'>Active documents</div></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.ingested_files:
        for name in st.session_state.ingested_files:
            st.markdown(
                f"<div class='doc-pill'>{escape(name)}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed yet.")

    # Actions section
    st.markdown(
        "<div class='sb-section'><div class='sb-label'>Actions</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Clear database", use_container_width=True):
        from vector_store.faiss_store import clear_vector_store
        clear_vector_store()
        clear_uploaded_files()
        st.session_state.ingested_files = []
        st.session_state.chat_history = []
        st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="intro">
        <h1>IntelliAgent</h1>
        <p>Upload documents from the sidebar, then ask questions with grounded citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.ingested_files:
    st.info("Upload and process PDF or TXT files from the sidebar to begin.")
else:
    # Render chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("sources", []))

    if not st.session_state.chat_history:
        st.info("Your document index is ready. Ask your first question below.")

    user_query = st.chat_input("Ask about your documents…")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    simple_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                    ][-10:]

                    result = run_pipeline(user_query, simple_history)
                    response_text = result.get("response", "No response generated.")
                    sources = result.get("sources", [])
                    retry_count = result.get("retry_count", 0)

                    st.session_state.last_run = {
                        "retrieved_sources": len(sources),
                        "retry_count": retry_count,
                        "answer_chars": len(response_text),
                    }

                except Exception as exc:
                    logger.exception("Pipeline error")
                    response_text = f"Failed to generate an answer: {exc}"
                    sources = []
                    retry_count = 0
                    st.session_state.last_run = {
                        "error": str(exc),
                        "retrieved_sources": 0,
                        "retry_count": 0,
                    }

            streamed_response = st.write_stream(stream_words(response_text))

            if retry_count > 0:
                st.caption(
                    f"Self-corrected after {retry_count} "
                    f"revision{'s' if retry_count > 1 else ''}."
                )

            show_sources(sources)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": streamed_response,
                "sources": sources,
                "retry_count": retry_count,
            }
        )