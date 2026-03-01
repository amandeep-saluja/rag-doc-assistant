"""Compare all extraction engines on a PDF with quality scores.

Usage:
    python test_engines.py               # uses first PDF found in docs/
    python test_engines.py myfile.pdf    # uses a specific file
"""

import sys
from pathlib import Path
from pdf_reader_v2 import (
    extract_with_pymupdf,
    extract_with_pdfplumber,
    extract_with_pypdf,
    extract_with_pdfminer,
    _score_text,
    _clean_text,
)

if len(sys.argv) > 1:
    pdf = sys.argv[1]
else:
    pdf_files = sorted(Path("docs").glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in docs/ — add a PDF and re-run.")
        sys.exit(1)
    pdf = str(pdf_files[0])
    print(f"No file specified — using: {pdf}")

engines = [
    ("pymupdf", extract_with_pymupdf),
    ("pdfplumber", extract_with_pdfplumber),
    ("pypdf", extract_with_pypdf),
    ("pdfminer", extract_with_pdfminer),
]

print("=" * 70)
print("ALL ENGINES COMPARISON")
print("=" * 70)

results = []
for name, extractor in engines:
    raw = extractor(pdf)
    if raw:
        cleaned = _clean_text(raw)
        score = _score_text(cleaned)
        chars = len(cleaned)
        results.append((name, chars, score))
        print(f"{name:12} | chars={chars:7,} | quality={score:.3f}")
    else:
        print(f"{name:12} | FAILED")

print("\n" + "=" * 70)
print("BEST ENGINE (by quality score)")
print("=" * 70)

if results:
    best = max(results, key=lambda x: x[2])
    print(f"\n✅ {best[0]} wins with quality score {best[2]:.3f}")
    print(f"   Characters: {best[1]:,}")
