"""
Test the PDF extraction quality — verifies that chunks are full pages,
not tiny header-only fragments.

Usage:
    python test_extraction_fix.py               # uses first PDF found in docs/
    python test_extraction_fix.py myfile.pdf    # uses a specific file
"""

import sys
from pdf_reader_v2 import load_pdfs_to_langchain
from pathlib import Path

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
    print("TESTING FIXED PDF EXTRACTION")
    print("=" * 70)

    docs = load_pdfs_to_langchain([test_pdf], enable_ocr=False, show_progress=True)

    print(f"\n📊 RESULTS:")
    print(f"Total documents (pages): {len(docs)}")

    print(f"\n📄 SAMPLE CHUNKS (first 3 pages):")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n--- Page {doc.metadata['page']} ---")
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Preview: {doc.page_content[:300]}...")
        print(f"Metadata: {doc.metadata}")

    # Check if chunks have substantial content
    char_counts = [len(doc.page_content) for doc in docs]
    avg_chars = sum(char_counts) / len(char_counts)
    print(f"\n📈 STATISTICS:")
    print(f"Average chars per page: {avg_chars:.0f}")
    print(f"Min chars: {min(char_counts)}")
    print(f"Max chars: {max(char_counts)}")

    # Check for tiny chunks (bad)
    tiny_chunks = [d for d in docs if len(d.page_content) < 100]
    if tiny_chunks:
        print(f"\n⚠️  WARNING: {len(tiny_chunks)} pages with <100 chars")
    else:
        print(f"\n✅ All pages have substantial content!")

    print("\n" + "=" * 70)
else:
    print(f"❌ Test PDF not found: {test_pdf}")
