"""
RAG Engine — handles PDF ingestion, chunking, embedding, vector storage,
and retrieval-augmented generation using modern LangChain (LCEL).
"""

import os
import glob
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage

from pdf_reader_v2 import extract_pdf, load_pdfs_to_langchain


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
EMBEDDING_MODELS = {
    "MultiQA MPNet (QA-tuned, recommended)": "multi-qa-mpnet-base-dot-v1",
    "MPNet (large, accurate)": "all-mpnet-base-v2",
    "MiniLM (fast, small)": "all-MiniLM-L6-v2",
}
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a precise technical documentation assistant. Your answers must be \
grounded ONLY in the retrieved context below.

RULES:
1. If the answer requires steps, ALWAYS number them clearly (Step 1, Step 2 ...).
2. If the answer contains a list of items (e.g. prerequisites, components, \
versions), present them as a formatted bullet list or table.
3. After your answer, add a "Source:" line citing the document name and page \
number for every fact you state.
4. If the context does not contain enough information to answer, say exactly: \
"The documentation does not cover this. Try re-indexing or checking the source \
PDF directly."
5. NEVER guess, infer, or use outside knowledge.
6. Keep answers concise — do not repeat the question.

Retrieved context:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def get_pdf_files() -> list[str]:
    """Return a list of PDF file paths from the docs/ directory."""
    return glob.glob(str(DOCS_DIR / "*.pdf"))


def get_pdf_stats(pdf_paths: list[str], enable_ocr: bool = False) -> list[dict]:
    """Return stats for each PDF: file, n_pages, n_chars, extraction_method, quality_score."""
    stats = []

    for pdf_path in pdf_paths:
        try:
            result = extract_pdf(pdf_path, enable_ocr=enable_ocr, log_results=False)
            stats.append(
                {
                    "file": os.path.basename(pdf_path),
                    "n_pages": result.page_count,
                    "n_chars": len(result.text),
                    "method": result.engine_used,
                    "quality_score": round(result.quality_score, 3),
                    "warnings": result.warnings,
                }
            )
        except Exception as e:
            stats.append(
                {
                    "file": os.path.basename(pdf_path),
                    "n_pages": 0,
                    "n_chars": 0,
                    "method": f"error: {e}",
                    "quality_score": 0.0,
                    "warnings": [str(e)],
                }
            )
    return stats


def load_and_split_pdfs(
    pdf_paths: list[str] | None = None, enable_ocr: bool = False
) -> list:
    """Load PDFs using quality-scored multi-engine extraction and split into chunks."""
    if pdf_paths is None:
        pdf_paths = get_pdf_files()

    if not pdf_paths:
        return []

    print(f"\n📚 Loading {len(pdf_paths)} PDF(s) with quality-scored extraction...")

    # Use improved multi-engine extractor
    all_docs = load_pdfs_to_langchain(
        pdf_paths, enable_ocr=enable_ocr, show_progress=True
    )

    if not all_docs:
        print("⚠️  No documents loaded successfully.")
        return []

    print(f"✓ Loaded {len(all_docs)} page chunks total")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"✓ Split into {len(chunks)} chunks\n")

    return chunks


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """Return the HuggingFace embedding model (downloaded & cached locally)."""
    if model_name is None:
        model_name = DEFAULT_EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Create (or overwrite) a Chroma vector store from document chunks.
    Deletes any existing collection first so stale data is never mixed in.
    """
    import chromadb

    # Delete the existing collection via the client API (no file-lock issues)
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        existing = [c.name for c in client.list_collections()]
        for col_name in existing:
            client.delete_collection(col_name)
        del client  # release the client so LangChain can reopen it
    except Exception:
        pass  # directory doesn't exist yet — that's fine

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma | None:
    """Load an existing Chroma vector store from disk, if available."""
    if not CHROMA_DIR.exists():
        return None
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def get_llm() -> ChatGroq:
    """Instantiate the Groq-hosted LLM."""
    return ChatGroq(
        model_name=GROQ_MODEL,
        temperature=0.2,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """\
You are a search query optimizer for a technical documentation search engine.

Given the user question below, output 3 search queries that will retrieve the \
best matching documentation chunks. Make them specific, use technical terms, \
and vary the phrasing.

Return ONLY the 3 queries, one per line, no numbering, no explanations.

User question: {question}"""


def _rewrite_query(question: str, llm) -> list[str]:
    """Use the LLM to expand the user question into 3 better search queries."""
    try:
        response = llm.invoke(_REWRITE_PROMPT.format(question=question))
        lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
        # Return original + rewrites (deduplicated)
        queries = [question] + lines[:3]
        return list(dict.fromkeys(queries))  # preserve order, deduplicate
    except Exception:
        return [question]  # graceful fallback


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_instance = None  # module-level cache — loaded once


def _get_reranker():
    """Lazy-load the cross-encoder reranker (cached after first call)."""
    global _reranker_instance
    if _reranker_instance is None:
        from sentence_transformers import CrossEncoder

        _reranker_instance = CrossEncoder(_RERANKER_MODEL)
    return _reranker_instance


def _rerank(question: str, docs: list, top_k: int = 5) -> list:
    """Re-score docs with a cross-encoder and return top_k by score."""
    if not docs:
        return docs
    try:
        reranker = _get_reranker()
        pairs = [(question, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except Exception:
        return docs[:top_k]  # graceful fallback


# ---------------------------------------------------------------------------
# Hybrid retrieval (BM25 keyword + vector semantic)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall of in on at to for with by "
    "from and or but not that this it its what which who how".split()
)


def _tokenize(text: str) -> list[str]:
    import re

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _hybrid_retrieve(
    vector_store: Chroma, all_chunks: list, question: str, k: int = 6
) -> list:
    """
    Hybrid retrieval: top BM25 keyword results guaranteed, then fill remaining
    slots with MMR vector results not already selected.

    Strategy:
    - BM25 top-3  →  guaranteed in output (exact keyword match wins)
    - Vector top-k → fill remaining slots (semantic coverage)
    - Deduplicate by content
    """
    from rank_bm25 import BM25Okapi

    # --- BM25 over all chunks ---
    corpus = [_tokenize(doc.page_content) for doc in all_chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(question)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranked = sorted(
        range(len(all_chunks)), key=lambda i: bm25_scores[i], reverse=True
    )

    bm25_slots = min(3, k)
    bm25_top = [all_chunks[i] for i in bm25_ranked[:bm25_slots]]

    # --- Vector MMR for remaining slots ---
    remaining = k - bm25_slots
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": remaining + bm25_slots, "fetch_k": 40, "lambda_mult": 0.7},
    )
    vector_docs = vector_retriever.invoke(question)

    seen = {doc.page_content for doc in bm25_top}
    vector_fill = []
    for doc in vector_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            vector_fill.append(doc)
        if len(vector_fill) >= remaining:
            break

    return bm25_top + vector_fill


def _multi_query_retrieve(
    vector_store: Chroma, all_chunks: list, queries: list[str], k: int = 20
) -> list:
    """
    Run hybrid retrieval for each rewritten query, pool all candidates,
    deduplicate by content, return up to k unique docs.
    """
    seen = set()
    pooled = []
    for q in queries:
        candidates = _hybrid_retrieve(vector_store, all_chunks, q, k=6)
        for doc in candidates:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                pooled.append(doc)
    return pooled[:k]


def _format_docs(docs: list) -> str:
    """Join retrieved document contents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_conversation_chain(vector_store: Chroma, all_chunks: list | None = None):
    """
    Production RAG chain:
      User question
        → Query Rewriter (LLM, 3 variants)
        → Multi-query Hybrid Retrieval (BM25 + MMR vector, pooled)
        → Cross-Encoder Reranker (top 5)
        → LLM answer with citations
    """
    llm = get_llm()

    def _run_chain(inputs: dict) -> dict:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # 1. Query rewriting — expand into 3 better queries
        queries = _rewrite_query(question, llm)

        # 2. Multi-query hybrid retrieval — pool up to 20 candidates
        if all_chunks:
            candidates = _multi_query_retrieve(vector_store, all_chunks, queries, k=20)
        else:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7},
            )
            candidates = retriever.invoke(question)

        # 3. Cross-encoder reranker — pick best 5 from candidates
        docs = _rerank(question, candidates, top_k=5)

        # 4. Build prompt and generate
        prompt_value = QA_PROMPT.invoke(
            {
                "context": _format_docs(docs),
                "question": question,
                "chat_history": chat_history,
            }
        )
        ai_message = llm.invoke(prompt_value)

        return {
            "answer": ai_message.content,
            "source_documents": docs,
        }

    return RunnableLambda(_run_chain)


# ---------------------------------------------------------------------------
# High-level helpers used by the UI
# ---------------------------------------------------------------------------


def ingest_documents(
    pdf_paths: list[str] | None = None, model_name: str = None, enable_ocr: bool = False
) -> tuple[Chroma, list]:
    """Full pipeline: load PDFs ➜ chunk ➜ embed ➜ store.
    Returns (vector_store, chunks) so caller can build a hybrid retriever.
    """
    chunks = load_and_split_pdfs(pdf_paths, enable_ocr=enable_ocr)
    if not chunks:
        raise ValueError(
            "No document chunks were produced. Add PDFs to the docs/ folder."
        )
    embeddings = get_embeddings(model_name)
    vector_store = build_vector_store(chunks, embeddings)
    return vector_store, chunks


def get_or_create_vector_store(
    model_name: str = None, enable_ocr: bool = False
) -> tuple[Chroma, list]:
    """Return the existing vector store or create one from docs/.
    Returns (vector_store, chunks) tuple.
    """
    embeddings = get_embeddings(model_name)
    vs = load_vector_store(embeddings)
    if vs and vs._collection.count() > 0:
        # Re-load chunks from disk for BM25
        chunks = load_and_split_pdfs(enable_ocr=enable_ocr)
        return vs, chunks
    return ingest_documents(model_name=model_name, enable_ocr=enable_ocr)
