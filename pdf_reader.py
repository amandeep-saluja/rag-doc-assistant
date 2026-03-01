"""
Advanced PDF reading with multiple extraction strategies.
Tries multiple methods to extract text accurately from PDFs.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

# PDF readers
import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader

warnings.filterwarnings("ignore")


class PDFDocument:
    """Represents a parsed PDF with metadata."""

    def __init__(self, file_path: str, pages: List[Dict]):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.pages = pages
        self.total_pages = len(pages)
        self.total_chars = sum(len(p.get("text", "")) for p in pages)
        self.method_used = pages[0].get("method", "unknown") if pages else "unknown"

    def get_page_text(self, page_num: int) -> str:
        """Get text for a specific page (0-indexed)."""
        if 0 <= page_num < len(self.pages):
            return self.pages[page_num].get("text", "")
        return ""

    def get_all_text(self) -> str:
        """Get all text concatenated."""
        return "\n\n".join(p.get("text", "") for p in self.pages)

    def to_langchain_docs(self):
        """Convert to LangChain Document format."""
        from langchain_core.documents import Document

        docs = []
        for i, page in enumerate(self.pages):
            if page.get("text", "").strip():
                docs.append(
                    Document(
                        page_content=page["text"],
                        metadata={
                            "source": self.file_path,
                            "page": i + 1,
                            "file_name": self.file_name,
                            "extraction_method": page.get("method", "unknown"),
                        },
                    )
                )
        return docs


class AdvancedPDFReader:
    """
    Multi-strategy PDF reader that tries multiple extraction methods.
    Prioritizes: PyMuPDF (fast) -> pdfplumber (layout-aware) -> pypdf (fallback)
    """

    def __init__(self, prefer_layout: bool = True):
        """
        Args:
            prefer_layout: If True, uses pdfplumber first (better layout preservation).
                          If False, uses PyMuPDF first (faster).
        """
        self.prefer_layout = prefer_layout

    def read_pdf(self, file_path: str) -> PDFDocument:
        """
        Read PDF using the best available method.
        Returns PDFDocument with extracted text and metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        # Try methods in order
        methods = (
            [self._read_with_pdfplumber, self._read_with_pymupdf, self._read_with_pypdf]
            if self.prefer_layout
            else [
                self._read_with_pymupdf,
                self._read_with_pdfplumber,
                self._read_with_pypdf,
            ]
        )

        last_error = None
        for method in methods:
            try:
                pages = method(file_path)
                if pages and self._has_meaningful_text(pages):
                    return PDFDocument(file_path, pages)
            except Exception as e:
                last_error = e
                continue

        # If all methods fail or return empty
        if last_error:
            raise Exception(
                f"All PDF extraction methods failed for {file_path}: {last_error}"
            )
        else:
            # Return empty doc rather than fail
            print(f"⚠️ Warning: No text extracted from {file_path}")
            return PDFDocument(file_path, [{"text": "", "page": 1, "method": "failed"}])

    def _has_meaningful_text(self, pages: List[Dict]) -> bool:
        """Check if extracted pages have meaningful text (not just whitespace)."""
        total_chars = sum(len(p.get("text", "").strip()) for p in pages)
        return total_chars > 50  # At least 50 characters across all pages

    def _read_with_pymupdf(self, file_path: str) -> List[Dict]:
        """Fast extraction with PyMuPDF (fitz)."""
        doc = fitz.open(file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text(
                "text", sort=True
            )  # sort=True for better reading order

            # If text is empty, try "blocks" extraction
            if not text.strip():
                blocks = page.get_text("blocks")
                text = "\n".join(block[4] for block in blocks if len(block) > 4)

            pages.append(
                {
                    "text": text,
                    "page": page_num + 1,
                    "method": "pymupdf",
                }
            )

        doc.close()
        return pages

    def _read_with_pdfplumber(self, file_path: str) -> List[Dict]:
        """Layout-aware extraction with pdfplumber (preserves tables/structure)."""
        pages = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text with layout
                text = page.extract_text(layout=True) or ""

                # If empty, try without layout
                if not text.strip():
                    text = page.extract_text(layout=False) or ""

                # Try extracting tables if text is still sparse
                if len(text.strip()) < 100:
                    tables = page.extract_tables()
                    if tables:
                        table_text = "\n\n".join(
                            "\n".join(
                                " | ".join(str(cell) for cell in row) for row in table
                            )
                            for table in tables
                        )
                        text = text + "\n\n" + table_text

                pages.append(
                    {
                        "text": text,
                        "page": i + 1,
                        "method": "pdfplumber",
                    }
                )

        return pages

    def _read_with_pypdf(self, file_path: str) -> List[Dict]:
        """Fallback extraction with pypdf."""
        reader = PdfReader(file_path)
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(
                {
                    "text": text,
                    "page": i + 1,
                    "method": "pypdf",
                }
            )

        return pages

    def compare_methods(self, file_path: str) -> Dict[str, PDFDocument]:
        """
        Extract with all methods and return comparison.
        Useful for debugging which method works best for a specific PDF.
        """
        results = {}

        for name, method in [
            ("pymupdf", self._read_with_pymupdf),
            ("pdfplumber", self._read_with_pdfplumber),
            ("pypdf", self._read_with_pypdf),
        ]:
            try:
                pages = method(file_path)
                results[name] = PDFDocument(file_path, pages)
            except Exception as e:
                results[name] = None
                print(f"  {name}: FAILED - {e}")

        return results


def batch_read_pdfs(
    pdf_paths: List[str],
    prefer_layout: bool = True,
    show_progress: bool = True,
) -> List[PDFDocument]:
    """
    Read multiple PDFs with progress tracking.

    Args:
        pdf_paths: List of PDF file paths
        prefer_layout: Use layout-aware extraction (slower but better)
        show_progress: Print progress messages

    Returns:
        List of PDFDocument objects
    """
    reader = AdvancedPDFReader(prefer_layout=prefer_layout)
    documents = []

    for i, path in enumerate(pdf_paths, 1):
        if show_progress:
            print(f"📄 Reading {i}/{len(pdf_paths)}: {Path(path).name} ...", end=" ")

        try:
            doc = reader.read_pdf(path)
            documents.append(doc)

            if show_progress:
                print(
                    f"✓ {doc.total_pages} pages, {doc.total_chars} chars ({doc.method_used})"
                )

        except Exception as e:
            if show_progress:
                print(f"✗ ERROR: {e}")
            continue

    return documents


# Convenience function for direct LangChain integration
def load_pdfs_to_langchain(pdf_paths: List[str], prefer_layout: bool = True):
    """
    Load PDFs and convert directly to LangChain Documents.

    Args:
        pdf_paths: List of PDF file paths
        prefer_layout: Use layout-aware extraction

    Returns:
        List of LangChain Document objects
    """
    docs = batch_read_pdfs(pdf_paths, prefer_layout=prefer_layout, show_progress=True)

    all_langchain_docs = []
    for doc in docs:
        all_langchain_docs.extend(doc.to_langchain_docs())

    return all_langchain_docs
