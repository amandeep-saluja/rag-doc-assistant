"""
Comparison between old and new PDF extractors.
Run this on any PDF in the docs/ folder to see quality improvements.

Usage:
    python compare_extractors.py               # uses first PDF found in docs/
    python compare_extractors.py myfile.pdf    # uses a specific file
"""

import sys
from pathlib import Path
from pdf_reader import AdvancedPDFReader
from pdf_reader_v2 import extract_pdf

# Accept a CLI argument or fall back to the first PDF in docs/
if len(sys.argv) > 1:
    test_pdf = sys.argv[1]
else:
    pdf_files = sorted(Path("docs").glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in docs/ — add a PDF and re-run.")
        sys.exit(1)
    test_pdf = str(pdf_files[0])
    print(f"No file specified — using: {test_pdf}")

if Path(test_pdf).exists():
    print("=" * 70)
    print("PDF EXTRACTION COMPARISON")
    print("=" * 70)

    # Old method (pdf_reader.py)
    print("\n📦 OLD EXTRACTOR (pdf_reader.py)")
    print("-" * 70)
    try:
        old_reader = AdvancedPDFReader(prefer_layout=True)
        old_doc = old_reader.read_pdf(test_pdf)
        old_chars = old_doc.total_chars
        old_method = old_doc.method_used
        print(f"Engine:     {old_method}")
        print(f"Pages:      {old_doc.total_pages}")
        print(f"Characters: {old_chars:,}")
        print(f"Quality:    Not measured")
    except Exception as e:
        print(f"ERROR: {e}")
        old_chars = 0

    # New method (pdf_reader_v2.py)
    print("\n✨ NEW EXTRACTOR (pdf_reader_v2.py - with skill optimizations)")
    print("-" * 70)
    try:
        new_result = extract_pdf(test_pdf, enable_ocr=False, log_results=False)
        new_chars = len(new_result.text)
        print(f"Engine:     {new_result.engine_used}")
        print(f"Pages:      {new_result.page_count}")
        print(f"Characters: {new_chars:,}")
        print(f"Quality:    {new_result.quality_score:.3f} / 1.000")
        if new_result.warnings:
            print(f"Warnings:   {', '.join(new_result.warnings)}")
        else:
            print(f"Warnings:   None")
    except Exception as e:
        print(f"ERROR: {e}")
        new_chars = 0

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    if old_chars > 0 and new_chars > 0:
        improvement = ((new_chars - old_chars) / old_chars) * 100
        print(f"\nCharacter count:")
        print(f"  Old: {old_chars:,}")
        print(f"  New: {new_chars:,}")
        print(f"  Change: {improvement:+.1f}%")

        if improvement > 0:
            print(f"\n✅ New extractor extracted {improvement:.1f}% MORE content!")
        elif improvement < 0:
            print(f"\n⚠️ New extractor extracted {abs(improvement):.1f}% LESS content")
        else:
            print("\n➡️ Same content extracted by both methods")

    print("\n✨ NEW FEATURES:")
    print("  • 4-engine fallback (PyMuPDF → pdfplumber → pypdf → pdfminer)")
    print("  • Quality scoring (0-1 scale)")
    print("  • Smart text cleaning and normalization")
    print("  • OCR support for scanned PDFs")
    print("  • Detailed warnings and metadata")
    print("  • Batch extraction with JSON logs")

    print("\n" + "=" * 70)
else:
    print(f"❌ Test PDF not found: {test_pdf}")
    print("Place a PDF in the docs/ folder to run comparison")
