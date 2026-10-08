"""Run: python -m streamlit run app.py"""
import os
from pathlib import Path
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv
from src.embed_store import EmbeddingStore
from src.retrieve import retrieve, format_context
from src.generate import generate_answer, DEFAULT_MODEL
from src.service import index_uploads, error_message

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="Ask Your PDFs", page_icon="📄", layout="wide")
st.title("📄 Ask Your PDFs")
st.caption("Answers from your PDFs, with numbered page and chunk references.")
st.caption("Each question is searched independently. Include the topic again in follow-up questions.")

if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    try:
        root = Path(os.getenv("RAG_STORAGE_DIR", str(BASE_DIR / "chroma_db")))
        st.session_state.store = EmbeddingStore(
            persist_dir=str(root / st.session_state.workspace_id))
    except Exception as exc:
        st.error(error_message(exc))
        st.stop()
store = st.session_state.store

with st.sidebar:
    st.header("Documents")
    st.caption(f"LLM: {os.getenv('GROQ_MODEL') or DEFAULT_MODEL}")
    st.caption(f"Embeddings: {os.getenv('EMBEDDING_BACKEND', 'cloud')}")
    st.caption("Cloud mode sends document text to your embedding provider and retrieved excerpts to Groq.")
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "").startswith("your_"):
        st.warning("Set GROQ_API_KEY in .env before asking questions.")
    if os.getenv("EMBEDDING_BACKEND", "cloud") == "cloud" and (
        not os.getenv("COHERE_API_KEY") or os.getenv("COHERE_API_KEY", "").startswith("your_")
    ):
        st.warning("Set COHERE_API_KEY in .env before indexing (cloud embeddings).")
    k = st.slider("Sources per answer (top-k)", 1, 8, 4)
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    st.caption("Up to 10 PDFs, 20 MB / 200 pages each. Selectable text only; no OCR.")
    st.caption("Index documents replaces this session's entire index with the selected PDFs.")
    if st.button("Index documents", disabled=not uploaded):
        try:
            with st.spinner("Extracting, chunking and indexing…"):
                added = index_uploads(store, uploaded)
            st.session_state.messages = []
            st.success(f"Indexed {added} chunks from {len(uploaded)} PDF(s). Chat reset.")
            st.info("Pages without extractable text are skipped. Review scans separately.")
        except Exception as exc:
            st.error(error_message(exc))
            st.info("The previous active index has been kept.")
    if st.button("Clear index"):
        try:
            store.clear()
            st.session_state.messages = []
            st.success("Active index and chat cleared.")
        except Exception as exc:
            st.error(error_message(exc))
    st.caption(f"Chunks in index: {store.count}")
    st.caption("Each browser session gets a separate workspace. A new session starts empty.")


def render_sources(sources):
    with st.expander("Retrieved sources"):
        for i, source in enumerate(sources, 1):
            st.caption(f"[{i}] {source['doc_name']} — page {source['page']}, chunk {source['chunk_index'] + 1}")
            st.text(source["text"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask a question about your documents…", max_chars=2000)
if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    sources = []
    try:
        if store.count == 0:
            answer = "Upload and index a PDF first."
        else:
            with st.spinner("Retrieving and generating…"):
                sources = retrieve(store, question, k=k)
                answer = generate_answer(question, format_context(sources), source_count=len(sources))
    except Exception as exc:
        answer = "⚠️ " + error_message(exc)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            render_sources(sources)
