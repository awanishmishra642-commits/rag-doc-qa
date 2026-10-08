"""Testable upload orchestration. PDFs stay in memory; no user-controlled paths."""
import os
from io import BytesIO
from pathlib import PurePosixPath
from src.chunk import chunk_document
from src.ingest import extract_pages

MAX_FILES = 10
MAX_CHUNKS = 2000


def prepare_uploads(uploads) -> list[dict]:
    if not uploads or len(uploads) > MAX_FILES:
        raise ValueError("Select between 1 and 10 PDFs.")
    names, chunks = set(), []
    for upload in uploads:
        name = PurePosixPath(upload.name.replace("\\", "/")).name
        if not name or not name.lower().endswith(".pdf"):
            raise ValueError("Every upload must have a .pdf filename.")
        if name.casefold() in names:
            raise ValueError("Duplicate PDF filenames: rename one before indexing.")
        names.add(name.casefold())
        pages = extract_pages(BytesIO(upload.getvalue()))
        for page in pages:
            chunks.extend(chunk_document({"doc_name": name, **page},
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))))
            if len(chunks) > MAX_CHUNKS:
                raise ValueError("Demo limit: 2,000 chunks. Upload fewer/smaller PDFs.")
    return chunks


def index_uploads(store, uploads):
    chunks = prepare_uploads(uploads)
    return store.replace_chunks(chunks)


def error_message(exc):
    from groq import AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
    from src.embeddings import EmbeddingError
    if isinstance(exc, AuthenticationError):
        return "Groq authentication failed. Check GROQ_API_KEY in .env."
    if isinstance(exc, RateLimitError):
        return "Groq rate limit/quota reached. Check your console limits and retry later."
    if isinstance(exc, APIConnectionError):
        return "Could not reach Groq. Check your internet connection and retry."
    if isinstance(exc, APIStatusError):
        return f"Groq request failed (HTTP {exc.status_code}). Check model access and account limits."
    if isinstance(exc, (EmbeddingError, ValueError, RuntimeError)):
        return str(exc)
    return "Operation failed. Check the PDF, configuration and local disk permissions, then retry."
