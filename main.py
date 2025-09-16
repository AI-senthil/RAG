import os
import tempfile
from typing import List, Dict, Any

import streamlit as st

from vector_db import (
    get_or_create_collection,
    ingest_pdfs_to_chroma,
    query_collection,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_DIR,
    DEFAULT_EMBEDDING_MODEL,
)


# Preserve simple UI text and flow while upgrading backend logic
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer questions. "
    "If you don't know, say so."
)


@st.cache_resource(show_spinner=False)
def _load_gemini_client():
    import google.generativeai as genai

    api_key = "AIzaSyDymBOyfvJ7BArfVAeKmbcauvPvDTJB5Ys"
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="gemini-2.0-flash")


@st.cache_resource(show_spinner=False)
def _get_client_and_collection():
    client, collection = get_or_create_collection(
        persist_directory=DEFAULT_DB_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    )
    return client, collection


def _format_context(results: Dict[str, Any]) -> str:
    docs: List[str] = results.get("documents", [[]])[0]
    metas: List[Dict[str, Any]] = results.get("metadatas", [[]])[0]
    parts: List[str] = []
    for i, d in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        src = os.path.basename(meta.get("source", ""))
        page = meta.get("page", "?")
        parts.append(f"[Source: {src} | Page {page}]\n{d}".strip())
    return "\n\n---\n\n".join(parts)


def _gemini_answer(model, question: str, context: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or ""
    except Exception as e:
        return f"[Model error] {e}"


# Streamlit UI (simple layout)
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.sidebar.header("Upload PDF")

client, collection = _get_client_and_collection()

uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
if uploaded_file:
    st.sidebar.write(f"**Uploaded:** {uploaded_file.name}")
    if st.sidebar.button("Submit"):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, uploaded_file.name)
            with open(dest, "wb") as f:
                f.write(uploaded_file.read())
            with st.spinner("Processing PDF and updating knowledge base..."):
                ingest_pdfs_to_chroma(
                    pdf_folder_path=tmpdir,
                    persist_directory=DEFAULT_DB_DIR,
                    collection_name=DEFAULT_COLLECTION_NAME,
                    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                )
        st.sidebar.success("PDF processed and added to knowledge base!")

# Optional: maintenance controls
with st.sidebar:
    if st.button("Reset current collection (this model)"):
        try:
            client.delete_collection(DEFAULT_COLLECTION_NAME)
        except Exception:
            pass
        _get_client_and_collection.clear()
        client, collection = _get_client_and_collection()
        st.success("Collection reset.")

# Main UI
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h2 style='text-align:center;'>RAG Chatbot</h2>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background-color:#e8f0fe; color:#174ea6; padding:20px; border-radius:8px; margin-bottom:4px;'><b>You:</b> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background-color:#d9ead3; color:#000000; padding:20px; border-radius:8px; margin-bottom:8px;'><b>Bot:</b> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )

    # Clear the text input safely before the widget is created on rerun
    if st.session_state.get("clear_user_input", False):
        st.session_state.user_input = ""
        st.session_state.clear_user_input = False

    user_input = st.text_input("Ask a question", key="user_input")
    ask_button = st.button("Ask")
    if ask_button and user_input:
        with st.spinner("Retrieving context..."):
            results = query_collection(collection=collection, query_text=user_input, n_results=5)
            context = _format_context(results)

        with st.spinner("Generating answer..."):
            model = _load_gemini_client()
            if not model:
                answer = "Please set GEMINI_API_KEY or GOOGLE_API_KEY in your environment to enable answers."
            else:
                answer = _gemini_answer(model, user_input, context)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "bot", "content": answer})
        st.session_state["clear_user_input"] = True
        st.rerun()

# System prompt and final response display
with st.sidebar:
    st.markdown("---")
    st.markdown(f"<b>System Prompt:</b> {SYSTEM_PROMPT}", unsafe_allow_html=True)
    if st.session_state.get("chat_history"):
        st.markdown("---")
        st.markdown(
            f"<b>Final Response:</b> {st.session_state['chat_history'][-1]['content']}",
            unsafe_allow_html=True,
        )
