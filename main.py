"""
RAG Documentation Assistant — CLI entry point.

Usage:
    python main.py            Launch the Streamlit UI
    python main.py --ingest   Index PDFs in docs/ without starting the UI
"""

import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    if "--ingest" in sys.argv:
        from rag_engine import ingest_documents, get_pdf_files

        pdfs = get_pdf_files()
        if not pdfs:
            print("❌ No PDF files found in docs/ — add some and try again.")
            sys.exit(1)
        print(f"📄 Found {len(pdfs)} PDF(s). Indexing …")
        ingest_documents()
        print("✅ Indexing complete! Vector store saved to chroma_db/.")
    else:
        # Launch Streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=True,
        )


if __name__ == "__main__":
    main()
