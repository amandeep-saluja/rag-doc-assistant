# 📚 RAG Documentation Assistant

A **production-grade Retrieval-Augmented Generation (RAG)** chatbot for querying technical PDF documentation, powered by **Groq**, **LangChain**, **ChromaDB**, and **Streamlit**.

Features query rewriting, hybrid BM25+vector retrieval, cross-encoder reranking, and QA-tuned embeddings — so you get precise, structured answers with page citations instead of vague summaries.

---

## Features

- 🔍 **Semantic search** across your PDF documentation
- 🤖 **Conversational AI** with chat memory (Groq / Llama 3.3 70B)
- 📄 **Source citations** with page numbers for every answer
- 📂 **Upload PDFs** directly from the UI or drop them in `docs/`
- ⚡ **Fast inference** via Groq's LPU hardware
- 💾 **Persistent vector store** (ChromaDB) — index once, query many times
- **✨ NEW:** **Quality-scored PDF extraction** with automatic engine selection
- **✨ NEW:** **4-engine fallback** (PyMuPDF → pdfplumber → pypdf → pdfminer.six)
- **✨ NEW:** **OCR support** for scanned PDFs (pytesseract)
- **✨ NEW:** **Smart text cleaning** with garbled text detection
- **✨ NEW:** **Detailed quality metrics** and extraction logs

---

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

_Note: OCR support (pytesseract, pdf2image) is included but requires Tesseract binary installation separately if needed._

### 2. Add your Groq API key

Edit the `.env` file:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free key at <https://console.groq.com>.

### 3. Add PDF documents

Place your PDF files in the `docs/` folder (or upload them through the UI).

### 4. Run the app

```bash
# Option A — launch Streamlit UI (also indexes docs automatically)
python main.py

# Option B — index documents from CLI first
python main.py --ingest
```

The Streamlit app will open at **http://localhost:8501**.

---

## Project Structure

```
rag-doc-assistant/
├── app.py                  # Streamlit chat UI with quality metrics
├── rag_engine.py           # RAG pipeline (load → chunk → embed → retrieve → answer)
├── pdf_reader_v2.py        # Production-grade PDF extractor with quality scoring
├── pdf_reader.py           # Legacy extractor (still available)
├── main.py                 # CLI entry point
├── docs/                   # Place your PDF files here (not committed to git)
├── chroma_db/              # Auto-generated vector store (not committed to git)
├── .env                    # GROQ_API_KEY lives here (not committed to git)
├── compare_extractors.py   # Compare old vs new extractor on any PDF
├── test_engines.py         # Test all extraction engines on any PDF
└── README.md
```

---

## How It Works

### PDF Extraction (Enhanced!)

The system now uses a **quality-scored multi-engine approach**:

1. **Try 4 engines** in priority order:
   - **PyMuPDF** (fitz) — Fast, handles 90% of PDFs
   - **pdfplumber** — Better table extraction, layout-aware
   - **pypdf** — Simple, reliable fallback
   - **pdfminer.six** — Most compatible, handles edge cases

2. **Score each result** (0-1 scale) based on:
   - Word density (healthy word length distribution)
   - Garbled text ratio (non-ASCII/control characters)
   - Whitespace distribution (normal prose ~15% spaces)

3. **Select best engine** automatically based on quality score

4. **OCR fallback** (optional): If text quality < 0.2, tries pytesseract

### RAG Pipeline

1. **Chunking** — `RecursiveCharacterTextSplitter` breaks pages into ~1,500-char overlapping chunks.
2. **Embedding** — `multi-qa-mpnet-base-dot-v1` (QA-tuned HuggingFace model) converts chunks to dense vectors.
3. **Storage** — Vectors are persisted in a **ChromaDB** collection on disk.
4. **Query Rewriting** — The LLM generates 3 phrasing variants of your question before retrieval.
5. **Hybrid Retrieval** — BM25 keyword search (guaranteed top-3) + MMR vector search, pooled across all query variants (~20 candidates).
6. **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` re-scores all candidates and selects the top 5.
7. **Generation** — **Groq (Llama 3.3 70B)** generates a structured answer grounded in the reranked context.

---

## Configuration

| Setting                | Default                            | Location                                  |
| ---------------------- | ---------------------------------- | ----------------------------------------- |
| LLM model              | `llama-3.3-70b-versatile`          | `rag_engine.py → GROQ_MODEL`              |
| Embedding model        | `multi-qa-mpnet-base-dot-v1`       | `rag_engine.py → DEFAULT_EMBEDDING_MODEL` |
| Chunk size             | 1,500 chars                        | `rag_engine.py → CHUNK_SIZE`              |
| Chunk overlap          | 300 chars                          | `rag_engine.py → CHUNK_OVERLAP`           |
| Rewriter queries       | 3 variants per question            | `rag_engine.py → _rewrite_query()`        |
| Retrieval candidates   | ~20 (hybrid BM25 + vector, pooled) | `rag_engine.py → _multi_query_retrieve()` |
| Reranker top-k         | 5                                  | `rag_engine.py → _rerank()`               |
| **✨ PDF engines**     | **4-engine fallback**              | `pdf_reader_v2.py`                        |
| **✨ Quality scoring** | **Automatic (0–1)**                | `pdf_reader_v2.py → _score_text()`        |

---

## ✨ Production RAG Pipeline

This project implements the full production RAG pattern — not just vector search, but every layer needed to get _correct, cited, structured_ answers from your documents:

- **Query Rewriting** — LLM expands each question into 3 search variants to bridge the phrasing gap between user language and technical documentation
- **Hybrid Retrieval** — BM25 keyword search guarantees exact-match results, while vector MMR search covers semantic similarity; results from all query variants are pooled
- **Cross-Encoder Reranking** — A `ms-marco-MiniLM-L-6-v2` cross-encoder re-scores all ~20 candidates jointly, picking the top 5 with highest relevance
- **QA-tuned Embeddings** — `multi-qa-mpnet-base-dot-v1` maps questions directly onto the passages that answer them (vs. general-purpose similarity models)
- **Structured Prompt** — Forces numbered steps, bullet tables, page citations, and an explicit "don't guess" guard

### Engine Quality Comparison

Example: `technical_manual.pdf` (131 pages)

| Engine      | Characters  | Quality Score | Selected?  |
| ----------- | ----------- | ------------- | ---------- |
| **pymupdf** | **124,183** | **0.901**     | ✅ **YES** |
| pdfplumber  | 181,428     | 0.422         | No         |
| pypdf       | 111,648     | 0.409         | No         |
| pdfminer    | 126,140     | 0.900         | No         |

### Try the Utility Scripts

```bash
# Compare old vs new extractor on your PDFs
python compare_extractors.py

# Test all engines with quality scores
python test_engines.py
```

---

## UI Features

### Sidebar Controls

- **Embedding Model Selector** — Choose from 3 HuggingFace models:
  - MultiQA MPNet (QA-tuned, **recommended default**)
  - MPNet (large, accurate)
  - MiniLM (fast, small)

- **Show Retrieved Chunks** — Preview document chunks before LLM answer generation

- **✨ Enable OCR** — Use pytesseract for scanned/image-only PDFs

- **Upload PDFs** — Drag & drop multiple files

- **PDF Stats** — View extraction quality metrics:

  ```
  ✓ manual.pdf — 64 pages, 329,354 chars · (pdfplumber Q:0.85)
  ```

- **Clear All Indexes** — Remove vector store (requires re-indexing)

- **Index Documents** — Build/rebuild vector store with selected embedding model

- **Clear Chat** — Reset conversation history

---

## Troubleshooting

### Low PDF Extraction Quality

**Symptoms:** Quality score < 0.3, missing text

**Solutions:**

1. Enable "OCR for scanned PDFs" checkbox
2. Check `extraction_log.json` for per-file diagnostics
3. Manually test engines with `test_engines.py`

### OCR Not Working

**Requirements:**

1. Install Tesseract binary:
   - Windows: `choco install tesseract` or [download](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `apt-get install tesseract-ocr`

2. Verify installation: `tesseract --version`

### Empty Vector Store

**Symptoms:** "No documents loaded" error

**Solutions:**

1. Ensure PDFs are in `docs/` folder
2. Check PDF stats for extraction errors
3. Try clearing indexes and re-indexing
4. Enable OCR if PDFs are scanned
