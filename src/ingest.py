"""Extract selectable PDF text, preserving physical 1-based page numbers."""
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200


def extract_pages(pdf_path) -> list[dict]:
    """Accept a path or a seekable binary stream. Image-only PDFs need OCR."""
    if isinstance(pdf_path, (str, Path)):
        path = Path(pdf_path)
        if path.stat().st_size > MAX_PDF_BYTES:
            raise ValueError("PDF exceeds the 20 MB limit.")
        source = str(path)
    else:
        source = pdf_path
        source.seek(0, 2)
        if source.tell() > MAX_PDF_BYTES:
            raise ValueError("PDF exceeds the 20 MB limit.")
        source.seek(0)
    try:
        reader = PdfReader(source)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("Password-protected PDF: upload an unlocked copy.")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError("PDF exceeds the 200-page demo limit.")
        pages = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    except (PdfReadError, OSError, KeyError, TypeError) as exc:
        raise ValueError("Could not read this PDF; upload a valid PDF file.") from exc
    if not pages:
        raise ValueError("No extractable text. Scanned/image-only PDFs need OCR (not included).")
    return pages


def extract_documents(pdf_paths: list) -> list[dict]:
    docs = []
    for path in pdf_paths:
        docs.extend({"doc_name": Path(path).name, **page} for page in extract_pages(path))
    return docs
