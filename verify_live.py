"""Opt-in small live acceptance check; uses your configured providers and quota.
Run: python verify_live.py --live (test dependencies required for sample PDF).
"""
import argparse
from pathlib import Path
import tempfile
from dotenv import load_dotenv
from src.ingest import extract_documents
from src.chunk import chunk_document
from src.embed_store import EmbeddingStore
from src.retrieve import retrieve, format_context
from src.generate import generate_answer, NOT_FOUND
from src.service import error_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make real embedding and Groq calls.")
    args = parser.parse_args()
    if not args.live:
        parser.print_help()
        return 0
    load_dotenv(Path(__file__).resolve().parent / ".env")
    from tests.make_sample_pdf import make_sample_pdf
    folder = Path(tempfile.mkdtemp(prefix="rag-live-"))
    print(f"Sample PDF and temporary test index: {folder}")
    print("This check uses your configured embedding provider and Groq quota.")
    try:
        docs = extract_documents([make_sample_pdf(str(folder))])
        chunks = [chunk for doc in docs for chunk in chunk_document(doc)]
        store = EmbeddingStore(str(folder / "index"))
        store.replace_chunks(chunks)
        for query, expected, page in [
            ("What is the efficiency of the SolarX-2000?", "23.8", 1),
            ("Who founded the company?", "Priya Sharma", 3),
            ("What was the company's revenue in 2025?", NOT_FOUND, None),
        ]:
            sources = retrieve(store, query)
            if page is not None and (not sources or sources[0]["page"] != page):
                raise RuntimeError(f"Retrieval check failed: expected page {page} ranked first.")
            answer = generate_answer(query, format_context(sources), source_count=len(sources))
            print(f"\nQ: {query}\nA: {answer}")
            if expected not in answer:
                raise RuntimeError("Answer did not contain the expected test fact/refusal; review the sources.")
        print("\nPASS: small live sample. This is not a general accuracy guarantee.")
        return 0
    except Exception as exc:
        print("FAIL:", error_message(exc))
        return 1
    finally:
        print("After this process exits, you can delete the temporary folder printed above.")


if __name__ == "__main__":
    raise SystemExit(main())
