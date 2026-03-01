"""
Production-grade multi-engine PDF extractor with quality scoring.
Engine priority: PyMuPDF → pdfplumber → pypdf → pdfminer → OCR (optional)
Inspired by rag-pdf-chatbot skill with >95% content coverage.
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of PDF extraction with quality metadata."""

    text: str
    engine_used: str
    page_count: int
    quality_score: float
    warnings: List[str] = field(default_factory=list)


def _score_text(text: str) -> float:
    """
    Score extraction quality (0–1). Higher = better.
    Checks word density, garbled text ratio, and whitespace distribution.
    """
    if not text or len(text) < 50:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    # Word density: avg word length (3-10 chars is healthy)
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_density_score = (
        min(avg_word_len / 7, 1.0)
        if avg_word_len <= 7
        else max(0, 1 - (avg_word_len - 7) / 10)
    )

    # Garbled text heuristic: ratio of non-ASCII or control chars
    non_ascii = sum(
        1 for c in text if ord(c) > 127 or (ord(c) < 32 and c not in "\n\t\r")
    )
    garble_penalty = min(non_ascii / max(len(text), 1), 1.0)

    # Space-to-char ratio (no spaces = garbled or merged words)
    space_ratio = text.count(" ") / max(len(text), 1)
    space_score = min(space_ratio / 0.15, 1.0)  # ~15% spaces is normal prose

    return word_density_score * 0.4 + (1 - garble_penalty) * 0.4 + space_score * 0.2


def _clean_text(text: str) -> str:
    """Normalize extracted text."""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable control chars (keep \n \t)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def extract_with_pymupdf(pdf_path: str) -> Optional[str]:
    """Primary engine: PyMuPDF (fitz). Best for most PDFs."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            # Use "blocks" mode for better layout ordering
            blocks = page.get_text("blocks", sort=True)
            page_text = "\n".join(
                b[4] for b in blocks if b[6] == 0  # type 0 = text block
            )
            pages.append(page_text)
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"PyMuPDF failed on {pdf_path}: {e}")
        return None


def extract_with_pdfplumber(pdf_path: str) -> Optional[str]:
    """Secondary engine: pdfplumber. Good for tables and structured PDFs."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract tables first, then remaining text
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    rows = [
                        " | ".join(str(cell or "") for cell in row)
                        for row in table
                        if row
                    ]
                    table_texts.append("\n".join(rows))

                # Extract text excluding table bounding boxes
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                combined = text
                if table_texts:
                    combined += "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(table_texts)
                pages.append(combined)
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"pdfplumber failed on {pdf_path}: {e}")
        return None


def extract_with_pypdf(pdf_path: str) -> Optional[str]:
    """Third fallback: pypdf. Simple but reliable."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"pypdf failed on {pdf_path}: {e}")
        return None


def extract_with_pdfminer(pdf_path: str) -> Optional[str]:
    """Fourth fallback: pdfminer.six. Most compatible, handles edge cases."""
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        from pdfminer.layout import LAParams

        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=True,
        )
        return pdfminer_extract(pdf_path, laparams=laparams)
    except ImportError:
        logger.warning(
            "pdfminer.six not installed. Install with: pip install pdfminer.six"
        )
        return None
    except Exception as e:
        logger.warning(f"pdfminer failed on {pdf_path}: {e}")
        return None


def extract_with_ocr(pdf_path: str, dpi: int = 300) -> Optional[str]:
    """OCR fallback for scanned/image-only PDFs. Requires pytesseract + pdf2image."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(pdf_path, dpi=dpi)
        pages = [pytesseract.image_to_string(img, lang="eng") for img in images]
        return "\n\n".join(pages)
    except ImportError:
        logger.warning(
            "OCR skipped: install pdf2image and pytesseract for scanned PDF support."
        )
        return None
    except Exception as e:
        logger.warning(f"OCR failed on {pdf_path}: {e}")
        return None


def extract_pdf(
    pdf_path: str,
    enable_ocr: bool = False,
    ocr_threshold: float = 0.2,
    log_results: bool = True,
) -> ExtractionResult:
    """
    Extract text from a PDF using a multi-engine fallback pipeline.
    Returns the best result across all engines based on quality scoring.

    Args:
        pdf_path: Path to PDF file
        enable_ocr: Enable OCR for scanned PDFs
        ocr_threshold: If best engine scores below this, try OCR
        log_results: Log which engine was chosen and its score

    Returns:
        ExtractionResult with text, engine used, and quality metadata
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Get page count for metadata
    page_count = 0
    try:
        import fitz

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        pass

    warnings = []
    results = {}

    # Run all engines, collect scored results
    engines = [
        ("pymupdf", extract_with_pymupdf),
        ("pdfplumber", extract_with_pdfplumber),
        ("pypdf", extract_with_pypdf),
        ("pdfminer", extract_with_pdfminer),
    ]

    for name, extractor in engines:
        raw = extractor(pdf_path)
        if raw:
            cleaned = _clean_text(raw)
            score = _score_text(cleaned)
            results[name] = (cleaned, score)
            logger.debug(
                f"{name} score for {path.name}: {score:.3f} ({len(cleaned)} chars)"
            )

    # Pick best result
    if results:
        best_engine = max(results, key=lambda k: results[k][1])
        best_text, best_score = results[best_engine]
    else:
        best_engine = "none"
        best_text = ""
        best_score = 0.0
        warnings.append("All text engines failed")

    # OCR fallback for low-quality or scanned PDFs
    if enable_ocr and best_score < ocr_threshold:
        warnings.append(f"Low text quality ({best_score:.2f}), attempting OCR...")
        ocr_text = extract_with_ocr(pdf_path)
        if ocr_text:
            ocr_cleaned = _clean_text(ocr_text)
            ocr_score = _score_text(ocr_cleaned)
            if ocr_score > best_score:
                best_engine = "ocr"
                best_text = ocr_cleaned
                best_score = ocr_score
                warnings.append(f"OCR improved score to {ocr_score:.2f}")

    if log_results:
        logger.info(
            f"{path.name} → engine={best_engine}, score={best_score:.3f}, chars={len(best_text)}"
        )

    return ExtractionResult(
        text=best_text,
        engine_used=best_engine,
        page_count=page_count,
        quality_score=best_score,
        warnings=warnings,
    )


def batch_extract(
    pdf_dir: str,
    enable_ocr: bool = False,
    save_log: str = "extraction_log.json",
) -> Dict[str, ExtractionResult]:
    """
    Extract text from all PDFs in a directory.
    Returns dict of {filename: ExtractionResult}.
    Saves an extraction log for debugging.
    """
    from tqdm import tqdm

    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    results = {}
    log_entries = []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        result = extract_pdf(str(pdf_path), enable_ocr=enable_ocr)
        results[pdf_path.name] = result
        log_entries.append(
            {
                "file": pdf_path.name,
                "engine": result.engine_used,
                "quality_score": round(result.quality_score, 3),
                "char_count": len(result.text),
                "page_count": result.page_count,
                "warnings": result.warnings,
            }
        )

    if save_log:
        with open(save_log, "w") as f:
            json.dump(log_entries, f, indent=2)
        logger.info(f"Extraction log saved to {save_log}")

    return results


def load_pdfs_to_langchain(
    pdf_paths: List[str],
    enable_ocr: bool = False,
    show_progress: bool = True,
):
    """
    Load PDFs and convert to LangChain Documents with quality-scored extraction.
    PROPERLY extracts page-by-page for better chunking and retrieval.

    Args:
        pdf_paths: List of PDF file paths
        enable_ocr: Enable OCR for scanned PDFs
        show_progress: Print progress messages

    Returns:
        List of LangChain Document objects (one per PDF page)
    """
    from langchain_core.documents import Document
    import fitz  # PyMuPDF for page-by-page extraction

    all_docs = []

    for i, path in enumerate(pdf_paths, 1):
        if show_progress:
            print(f"📄 Processing {i}/{len(pdf_paths)}: {Path(path).name} ...", end=" ")

        try:
            # Use PyMuPDF for reliable page-by-page extraction
            doc = fitz.open(str(path))
            page_count = len(doc)
            total_chars = 0

            # --- Pass 1: extract raw text per page ---
            raw_pages = []  # list of (page_num_1indexed, text)
            for page_num in range(page_count):
                page = doc[page_num]

                # Try blocks first (better layout preservation)
                blocks = page.get_text("blocks", sort=True)
                page_text = "\n".join(
                    b[4]
                    for b in blocks
                    if len(b) > 6 and b[6] == 0  # type 0 = text block
                )

                # Fallback to simple text if blocks are empty
                if not page_text.strip():
                    page_text = page.get_text("text", sort=True)

                page_text = _clean_text(page_text)
                total_chars += len(page_text)
                raw_pages.append((page_num + 1, page_text))

            doc.close()

            # --- Pass 2: filter and merge pages ---
            MIN_STANDALONE = 600  # pages shorter than this get merged forward
            MIN_DISCARD = 150  # pages shorter than this get discarded entirely

            import re as _re

            _HEADING_ONLY = _re.compile(
                r"^(CHAPTER|Chapter|SECTION|Appendix)\b", _re.MULTILINE
            )

            def _is_toc_or_boilerplate(text: str) -> bool:
                """Return True for table-of-contents and revision-history pages.
                These contain keywords but no real content, poisoning BM25 results.
                """
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                if len(lines) < 3:
                    return False
                # TOC heuristic: >50% of lines end with a page number
                page_num_lines = sum(1 for l in lines if _re.search(r"\d+\s*$", l))
                if page_num_lines / len(lines) > 0.45:
                    return True
                # Revision history heuristic: many date-like entries
                date_lines = sum(
                    1
                    for l in lines
                    if _re.search(
                        r"\b(January|February|March|April|May|June|July|August|"
                        r"September|October|November|December)\b.*\d{4}",
                        l,
                    )
                )
                if date_lines >= 2 and len(lines) < 30:
                    return True
                return False

            merged_pages = []  # list of (page_num, merged_text)
            carry = ""  # text carried forward from a short page
            carry_page = None

            for page_num, text in raw_pages:
                stripped = text.strip()
                if not stripped or len(stripped) < MIN_DISCARD:
                    # Too short to be useful — discard (page numbers, blank pages)
                    carry = ""
                    carry_page = None
                    continue

                # Discard pure TOC and revision-history pages entirely
                if _is_toc_or_boilerplate(stripped):
                    continue

                is_short = len(stripped) < MIN_STANDALONE
                is_heading = bool(_HEADING_ONLY.match(stripped))

                if is_short or is_heading:
                    # Chapter divider or short page — carry forward, don't emit yet
                    carry = (carry + "\n\n" + stripped).strip() if carry else stripped
                    carry_page = carry_page if carry_page is not None else page_num
                else:
                    # Normal content page — prepend any carried heading
                    if carry:
                        merged_text = carry + "\n\n" + stripped
                        emit_page = carry_page
                        carry = ""
                        carry_page = None
                    else:
                        merged_text = stripped
                        emit_page = page_num
                    merged_pages.append((emit_page, merged_text))

            # Flush any remaining carry (e.g. a heading at the very end)
            if carry:
                merged_pages.append((carry_page, carry))

            # --- Pass 3: emit LangChain Documents ---
            for emit_page, merged_text in merged_pages:
                all_docs.append(
                    Document(
                        page_content=merged_text,
                        metadata={
                            "source": str(path),
                            "page": emit_page,
                            "file_name": Path(path).name,
                            "extraction_method": "pymupdf",
                            "page_chars": len(merged_text),
                        },
                    )
                )

            if show_progress:
                quality_score = _score_text(
                    "\n".join([d.page_content for d in all_docs[-page_count:]])
                )
                print(
                    f"✓ {page_count} pages, {total_chars:,} chars, engine=pymupdf, score={quality_score:.2f}"
                )

        except Exception as e:
            if show_progress:
                print(f"✗ ERROR: {e}")
            continue

    return all_docs
