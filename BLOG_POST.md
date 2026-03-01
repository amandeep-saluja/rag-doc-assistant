# 🚀 From Broken RAG to Production-Grade: What Actually Changed

> _The story of how a chatbot went from returning 218-character chapter headings to delivering precise, numbered, cited technical answers — and every engineering decision that got us there._

---

## The Starting Point: "It Kinda Worked"

I built a RAG (Retrieval-Augmented Generation) chatbot to answer questions from a set of technical PDF documentation. The stack looked reasonable on paper:

- **Groq** (`llama-3.3-70b-versatile`) for the LLM
- **ChromaDB** for vector storage
- **LangChain** for orchestration
- **Streamlit** for the UI
- `all-MiniLM-L6-v2` for embeddings

And then I asked it a simple question:

> **"What are the prerequisites for installing this software?"**

It returned this:

```
Prerequisites

Chapter 3 — System Requirements

See page 10 for full details.
```

**218 characters. A chapter heading. Not an answer.**

I had built a very expensive table of contents navigator.

---

## What Was Actually Wrong (All 5 Layers)

After deep debugging, I found the system was broken at _every layer_ of the pipeline. Not one thing — five things.

---

### ❌ Problem 1: Chunks Were Heading-Only Fragments

**Root cause:** The PDF text was loaded as one giant string and split on `\n\n`. Result? Every chapter title, section header, and table-of-contents entry became its own chunk.

The chunk ranked #1 for "prerequisites" was literally:

```
Prerequisites

Chapter 3
```

**Fix:** Switched to **page-by-page extraction** using PyMuPDF (`fitz`). Each page becomes one candidate chunk, averaging **953 characters** of real content vs. the previous 80.

---

### ❌ Problem 2: Table of Contents Pages Were Poisoning Retrieval

Every document has a Table of Contents. Every TOC has lines like:

```
Prerequisites ................ 10
System Requirements ........... 12
Installation Steps ............ 15
```

BM25 keyword search loved those pages. Search for "prerequisites" → TOC page ranked #1 every time, because it mentioned "Prerequisites" right there.

**Fix:** Built a `_is_toc_or_boilerplate()` filter that discards any page where:

- More than 45% of lines end in a page number (dead giveaway for TOC)
- Two or more date patterns exist (revision history pages)

This removed ~104 garbage pages before they ever touched the index.

---

### ❌ Problem 3: Short Heading Pages Still Slipped Through

Some pages were just section openers:

```
CHAPTER 3

System Requirements and Prerequisites
```

Too short to be a TOC (no trailing numbers), but too useless to answer any question. These were 150–500 character fragments that ranked well on keyword match but had zero information density.

**Fix:** Any page under **600 characters** gets merged forward into the next page. Any page under **150 characters** gets discarded entirely. Section-opening pages (matching `CHAPTER|SECTION|Appendix` patterns) are always merged forward.

---

### ❌ Problem 4: The Embedding Model Wasn't Built for QA

`all-MiniLM-L6-v2` is a general-purpose sentence similarity model. It's optimized for _"these two sentences mean the same thing"_ — not for _"this passage answers this question."_

When you ask "What are the prerequisites for installing this software?", the vector store should retrieve the passage containing the prerequisites table. But `all-MiniLM-L6-v2` doesn't know that a question relates to an answer — it just measures generic semantic overlap.

**Fix:** Switched to **`multi-qa-mpnet-base-dot-v1`** — a model specifically fine-tuned on question-answer pairs. It learns to map a _question_ onto the _passage that answers it_, not just onto passages that _sound similar_.

---

### ❌ Problem 5: Single Query, No Reranking, Weak Prompt

The original pipeline:

1. Take the user's question exactly as typed
2. Do one vector search
3. Take top-3 results
4. Dump into the LLM

This fails for three reasons:

- Users don't phrase queries the way technical documents phrase answers
- Top-3 vector results may not include the best chunk (especially if it's short)
- No verification step — garbage in, garbage out

---

## The Production Fix: 4 New Layers

### ✅ Fix 4: Query Rewriting

Before hitting the retrieval system, the LLM generates **3 differently-phrased search variants** of the user's question:

```
User asks: "What are the prerequisites for installing this software?"

LLM generates:
  1. "software system requirements and dependencies"
  2. "required components before installation"
  3. "prerequisite software versions and minimum specs"
```

Each variant runs through the full retrieval pipeline. This dramatically increases the chance of finding the right chunk — even if the user's phrasing doesn't match the document's phrasing.

**Why it works:** Technical documents say _"System Requirements: version 3.16 SP1 or later"_. Users say _"what do I need to install this?"_. Query rewriting bridges that gap.

---

### ✅ Fix 5: Hybrid Retrieval (Vector + BM25, Guaranteed Slots)

Pure vector search misses exact matches. Pure BM25 (keyword) misses semantic meaning. The production system uses both:

- **BM25 (keyword search)** gets guaranteed **top-3 slots** — if a chunk contains the exact keywords, it always makes it through
- **Vector search** fills the remaining slots with semantically similar chunks
- Results from all 3+ query variants are **pooled and deduplicated**

This gives ~20 candidate chunks for the next step.

---

### ✅ Fix 6: Cross-Encoder Reranking

A bi-encoder (like ChromaDB's vector search) encodes query and document separately, then compares vectors. It's fast, but coarse.

A **cross-encoder** reads the query and document _together_, like a reading comprehension model. It's slower, but far more accurate at judging relevance.

Model used: **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (fine-tuned on MS MARCO, ~90MB)

The cross-encoder scores all ~20 candidates jointly and picks the **top 5** for the LLM. This is the step that eliminated the last remaining false positives.

---

### ✅ Fix 7: Production-Grade System Prompt

Old prompt:

```
You are a helpful assistant. Use the context to answer the question.
```

New prompt:

```
You are a precise technical documentation assistant.

RULES:
1. If the answer requires steps, ALWAYS number them clearly (Step 1, Step 2...).
2. If the answer contains a list of items, present them as bullet list or table.
3. After your answer, add a "Source:" line citing document name and page number.
4. If context doesn't contain enough info, say exactly:
   "The documentation does not cover this..."
5. NEVER guess, infer, or use outside knowledge.
```

The LLM is the same (`llama-3.3-70b-versatile`). The instructions changed everything about _how_ it uses context.

---

## Before vs. After

| Dimension               | Before                                  | After                                   |
| ----------------------- | --------------------------------------- | --------------------------------------- |
| **Embedding model**     | `all-MiniLM-L6-v2` (general similarity) | `multi-qa-mpnet-base-dot-v1` (QA-tuned) |
| **Chunk size**          | 80 chars avg (heading fragments)        | 953 chars avg (full page content)       |
| **TOC/boilerplate**     | Indexed and ranked                      | Filtered out before indexing            |
| **Query strategy**      | Single query, exact user phrasing       | 3 rewritten variants, pooled            |
| **Retrieval**           | Top-3 vector only                       | Hybrid BM25+Vector, ~20 candidates      |
| **Reranking**           | None                                    | Cross-encoder on all 20 candidates      |
| **Final context**       | Top 3 (unverified)                      | Top 5 (cross-encoder verified)          |
| **Answer format**       | Unstructured prose                      | Numbered steps, tables, page citations  |
| **Hallucination guard** | None                                    | Explicit "don't guess" instruction      |

---

## The Result

**Question:** _"What are the prerequisites for installing this software?"_

**Before:**

```
Prerequisites

Chapter 3 — System Requirements
```

**After:**

```
To install the software, the following prerequisites must be met:

| Component              | Required Version       |
|------------------------|------------------------|
| Platform               | 3.16 SP1 or later      |
| Core Module            | 6.6 SP5 or later       |
| Integration Layer      | 9.0 or later           |
| Java (JRE)             | 11 or later            |
| Database               | MS SQL Server 2019+    |

Source: Installation_Guide.pdf, Page 10
```

---

## The Full Production Pipeline (Diagram)

```
User Question
     │
     ▼
┌─────────────────────┐
│   Query Rewriting   │  ← LLM generates 3 phrasing variants
└─────────────────────┘
     │ 3 queries
     ▼
┌─────────────────────┐
│  Hybrid Retrieval   │  ← BM25 (top-3 guaranteed) + Vector per query
│  (per query)        │
└─────────────────────┘
     │ ~20 pooled, deduplicated candidates
     ▼
┌─────────────────────┐
│  Cross-Encoder      │  ← ms-marco-MiniLM-L-6-v2 scores each jointly
│  Reranking          │
└─────────────────────┘
     │ Top 5 verified chunks
     ▼
┌─────────────────────┐
│  LLM Generation     │  ← llama-3.3-70b + production system prompt
│  (Groq)             │
└─────────────────────┘
     │
     ▼
Structured Answer with Page Citations
```

---

## Tech Stack

| Component      | Library / Model                                    |
| -------------- | -------------------------------------------------- |
| LLM            | Groq `llama-3.3-70b-versatile`                     |
| Embeddings     | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| Vector DB      | ChromaDB (persistent)                              |
| BM25           | `rank_bm25.BM25Okapi`                              |
| Reranker       | `cross-encoder/ms-marco-MiniLM-L-6-v2`             |
| PDF Extraction | PyMuPDF + pdfplumber + pypdf fallback chain        |
| Orchestration  | LangChain LCEL                                     |
| UI             | Streamlit                                          |

---

## Key Lessons

1. **Garbage in, garbage out** — fix the chunking before anything else. No amount of fancy retrieval fixes bad chunks.
2. **TOC pages will destroy your BM25 scores** — always filter boilerplate before indexing.
3. **Embedding model choice matters enormously** — general similarity ≠ question-answer relevance.
4. **Query rewriting is free** — one extra LLM call with a short prompt, massive retrieval improvement.
5. **The cross-encoder is the quality gate** — it's the step that eliminates the last false positives.
6. **Your prompt is half the product** — the same LLM gives radically different answer quality with a structured vs. generic prompt.

---

## What's Next

- **Metadata filtering** — tag chunks by document type (installation, operations, user guide) and filter before retrieval
- **Streaming responses** — progressive answer display for long technical answers
- **Automatic re-index detection** — detect when the embedding model changes and prompt for re-indexing
- **Graph RAG** — entity extraction across documents for relationship-aware answers

---

_Built with: Python 3.12 · LangChain · ChromaDB · Groq · Streamlit · sentence-transformers_

_If this helped you, share it — every RAG system I've seen has at least 3 of these 5 problems._
