"""
Streamlit UI for the RAG Documentation Assistant.
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load env vars (.env must contain GROQ_API_KEY)
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from rag_engine import (
    DOCS_DIR,
    get_or_create_vector_store,
    ingest_documents,
    get_conversation_chain,
    get_pdf_files,
    get_pdf_stats,
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="📚 Doc RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background: #f8f9fa;
        border-left: 4px solid #4b6cb7;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "show_retrieval" not in st.session_state:
    st.session_state.show_retrieval = True
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False


# ---------------------------------------------------------------------------
# Cached functions for performance
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="📊 Analyzing PDFs...")
def get_cached_pdf_stats(pdf_paths_tuple, enable_ocr=False):
    """Cache PDF stats to avoid re-computing on every rerun."""
    return get_pdf_stats(list(pdf_paths_tuple), enable_ocr=enable_ocr)


# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📂 Document Management")

    # --- API key check ---
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        st.warning("⚠️ Set your `GROQ_API_KEY` in the `.env` file to get started.")

    # --- Embedding model selection ---
    st.markdown("**Embedding Model:**")
    model_label = st.selectbox(
        "Choose embedding model",
        list(EMBEDDING_MODELS.keys()),
        index=list(EMBEDDING_MODELS.values()).index(DEFAULT_EMBEDDING_MODEL),
        key="embedding_model_select",
    )
    embedding_model = EMBEDDING_MODELS[model_label]

    # --- Show retrieval results toggle ---
    st.session_state.show_retrieval = st.checkbox(
        "Show retrieved chunks before answer",
        value=st.session_state.show_retrieval,
        help="Display the document chunks found by embedding search before generating the LLM answer.",
    )

    # --- OCR toggle for scanned PDFs ---
    enable_ocr = st.checkbox(
        "Enable OCR for scanned PDFs",
        value=False,
        help="Use OCR (pytesseract) for image-only or scanned PDFs. Slower but handles PDFs with no text layer.",
    )

    # --- Upload PDFs ---
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files. They will be saved to the docs/ folder.",
    )
    if uploaded_files:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_files:
            dest = DOCS_DIR / uploaded.name
            dest.write_bytes(uploaded.getvalue())
        st.success(f"✅ Saved {len(uploaded_files)} file(s) to `docs/`.")

    # --- List loaded docs and PDF stats ---
    existing_pdfs = get_pdf_files()
    if existing_pdfs:
        st.markdown("**Loaded documents:**")
        # Use cached stats to avoid blocking UI on every rerun
        pdf_stats = get_cached_pdf_stats(tuple(existing_pdfs), enable_ocr=enable_ocr)
        for stat in pdf_stats:
            method_icon = "🔧" if stat.get("method", "").startswith("error") else "✓"
            quality_display = (
                f" (Q:{stat.get('quality_score', 0):.2f})"
                if "quality_score" in stat
                else ""
            )
            st.markdown(
                f"{method_icon} `{stat['file']}` — {stat['n_pages']} pages, "
                f"{stat['n_chars']} chars · _({stat.get('method', 'unknown')}{quality_display})_"
            )
            # Show warnings if any
            if stat.get("warnings"):
                for warning in stat["warnings"]:
                    st.caption(f"  ⚠️ {warning}")
        # Warn if any PDF has 0 pages or 0 chars or low quality
        for stat in pdf_stats:
            if stat["n_pages"] == 0 or stat["n_chars"] < 50:
                st.warning(
                    f"⚠️ `{stat['file']}`: No readable text/pages. PDF may be scanned or encrypted. Try enabling OCR."
                )
            elif stat.get("quality_score", 1.0) < 0.3:
                st.warning(
                    f"⚠️ `{stat['file']}`: Low quality score ({stat['quality_score']:.2f}). Consider enabling OCR if scanned."
                )
    else:
        st.info("No PDFs found. Upload files or place them in the `docs/` folder.")

    st.divider()

    # --- Clear all indexes ---
    if st.button("🧹 Clear All Indexes", use_container_width=True):
        import shutil, chromadb as _chromadb

        # 1. Release the in-memory connection first
        st.session_state.vector_store = None
        st.session_state.chain = None

        # 2. Delete via ChromaDB client API (avoids Windows file-lock errors)
        chroma_dir = DOCS_DIR.parent / "chroma_db"
        try:
            _client = _chromadb.PersistentClient(path=str(chroma_dir))
            for col in _client.list_collections():
                _client.delete_collection(col.name)
            del _client
        except Exception:
            pass

        # 3. Best-effort folder removal (may stay if files still locked)
        if chroma_dir.exists():
            try:
                shutil.rmtree(chroma_dir)
            except PermissionError:
                pass

        st.session_state.messages = []
        st.session_state.vector_store_loaded = False
        get_cached_pdf_stats.clear()
        st.success("All indexes cleared. Please re-index your documents.")
        st.rerun()

    # --- Index / re-index ---
    if st.button("🔄 Index Documents", use_container_width=True):
        if not existing_pdfs and not uploaded_files:
            st.error("No PDF files to index.")
        else:
            # Release current connection — build_vector_store() clears old
            # collection internally via the ChromaDB client API (no lock errors)
            st.session_state.vector_store = None
            st.session_state.chain = None

            with st.spinner("🔄 Indexing documents — this may take a few moments..."):
                progress_placeholder = st.empty()
                try:
                    progress_placeholder.info("📄 Extracting text from PDFs...")
                    st.session_state.vector_store, chunks = ingest_documents(
                        model_name=embedding_model, enable_ocr=enable_ocr
                    )
                    progress_placeholder.info("🧠 Building conversation chain...")
                    st.session_state.chain = get_conversation_chain(
                        st.session_state.vector_store, all_chunks=chunks
                    )
                    st.session_state.messages = []
                    st.session_state.vector_store_loaded = True
                    progress_placeholder.empty()
                    st.success(
                        f"✅ Documents indexed successfully! Model: {model_label}"
                    )
                    get_cached_pdf_stats.clear()
                except Exception as exc:
                    progress_placeholder.empty()
                    st.error(f"Indexing failed: {exc}")

    # --- Clear chat ---
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

    st.divider()
    st.caption("Built with LangChain · Groq · ChromaDB · Streamlit")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="main-header">📚 Documentation Assistant</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Ask questions about your documents and get instant, accurate answers.</p>',
    unsafe_allow_html=True,
)

# Auto-load vector store on first run if docs exist (lazy loading - only when needed)
if (
    not st.session_state.vector_store_loaded
    and st.session_state.vector_store is None
    and existing_pdfs
):
    # Check if vector store exists without loading it
    chroma_dir = DOCS_DIR.parent / "chroma_db"
    if chroma_dir.exists():
        st.info(
            "💡 Vector store detected. It will be loaded when you ask your first question."
        )
    else:
        st.info("👉 Click **'🔄 Index Documents'** in the sidebar to get started.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box"><strong>{src["file"]}</strong> '
                        f"(page {src['page']})<br/>{src['snippet']}</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents …"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Guard: no chain yet - load vector store lazily on first query
    if st.session_state.chain is None:
        if st.session_state.vector_store is None:
            # Try to load existing vector store
            chroma_dir = DOCS_DIR.parent / "chroma_db"
            if chroma_dir.exists() and existing_pdfs:
                with st.spinner("🔄 Loading knowledge base for the first time..."):
                    try:
                        st.session_state.vector_store, chunks = (
                            get_or_create_vector_store()
                        )
                        st.session_state.chain = get_conversation_chain(
                            st.session_state.vector_store, all_chunks=chunks
                        )
                        st.session_state.vector_store_loaded = True
                        assistant_msg = None  # fall through to query
                    except Exception as exc:
                        assistant_msg = f"❌ Could not load vector store: {exc}"
            else:
                assistant_msg = "⚠️ Please upload and **index** your documents first using the sidebar."
        else:
            with st.spinner("🧠 Initializing conversation chain..."):
                st.session_state.chain = get_conversation_chain(
                    st.session_state.vector_store, all_chunks=None
                )
            assistant_msg = None  # fall through to query

    if st.session_state.chain is not None:
        with st.chat_message("assistant"):
            # Build chat_history as LangChain messages from session
            chat_history = []
            for m in st.session_state.messages[:-1]:  # exclude current user msg
                if m["role"] == "user":
                    chat_history.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    chat_history.append(AIMessage(content=m["content"]))

            # Step 1: Show retrieved chunks if enabled
            if st.session_state.show_retrieval and st.session_state.vector_store:
                with st.spinner("🔍 Searching documents …"):
                    try:
                        retriever = st.session_state.vector_store.as_retriever(
                            search_type="mmr",
                            search_kwargs={"k": 5, "fetch_k": 10},
                        )
                        retrieved_docs = retriever.invoke(prompt)

                        st.markdown(
                            "#### 🔍 Retrieved Chunks (Embedding Search Results)"
                        )
                        st.caption(
                            f"Found {len(retrieved_docs)} relevant chunks from embeddings:"
                        )

                        for idx, doc in enumerate(retrieved_docs, 1):
                            meta = doc.metadata
                            file_name = Path(meta.get("source", "unknown")).name
                            page_num = meta.get("page", "?")
                            content = (
                                doc.page_content[:500] + "…"
                                if len(doc.page_content) > 500
                                else doc.page_content
                            )

                            with st.expander(
                                f"📄 Chunk {idx}: {file_name} (page {page_num})",
                                expanded=(idx <= 2),
                            ):
                                st.text(content)
                                st.caption(f"Length: {len(doc.page_content)} chars")

                        st.divider()
                    except Exception as e:
                        st.warning(f"Could not retrieve chunks: {e}")

            # Step 2: Generate answer with LLM
            with st.spinner("💭 Generating answer with Groq LLM …"):
                try:
                    result = st.session_state.chain.invoke(
                        {"question": prompt, "chat_history": chat_history}
                    )
                    assistant_msg = result["answer"]

                    # Collect sources
                    sources = []
                    seen = set()
                    for doc in result.get("source_documents", []):
                        meta = doc.metadata
                        key = (meta.get("source", ""), meta.get("page", 0))
                        if key not in seen:
                            seen.add(key)
                            sources.append(
                                {
                                    "file": Path(meta.get("source", "unknown")).name,
                                    "page": meta.get("page", "?"),
                                    "snippet": (
                                        doc.page_content[:300] + "…"
                                        if len(doc.page_content) > 300
                                        else doc.page_content
                                    ),
                                }
                            )

                    st.markdown(assistant_msg)

                    if sources:
                        with st.expander("📄 Sources"):
                            for src in sources:
                                st.markdown(
                                    f'<div class="source-box"><strong>{src["file"]}</strong> '
                                    f"(page {src['page']})<br/>{src['snippet']}</div>",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg,
                            "sources": sources,
                        }
                    )

                except Exception as exc:
                    err = f"❌ Error: {exc}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
    else:
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_msg}
        )
