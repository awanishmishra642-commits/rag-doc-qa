"""Offline regression tests: real PDF extraction, splitter and persistent Chroma.
Embedding and generation network calls use deterministic doubles, NOT real AI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import pytest
from pypdf import PdfWriter
from src.chunk import chunk_document, split_text
from src.ingest import extract_pages, extract_documents
from src.embed_store import EmbeddingStore
from src.embeddings import CloudEmbeddings, EmbeddingError, validate_vectors
from src.retrieve import retrieve, format_context
from src.generate import generate_answer, build_messages, NOT_FOUND
from src.service import index_uploads, prepare_uploads
from tests.fakes import FakeEmbeddings, fake_client
from tests.make_sample_pdf import make_sample_pdf


@pytest.fixture
def store(tmp_path):
    return EmbeddingStore(str(tmp_path / "db"), embedder=FakeEmbeddings())


def record(text="solar efficiency 23.8%", name="sample.pdf", page=1, index=0):
    return dict(doc_name=name, page=page, chunk_index=index, text=text)


def test_full_offline_pipeline(tmp_path, store):
    path = make_sample_pdf(str(tmp_path))
    docs = extract_documents([path])
    assert [d["page"] for d in docs] == [1, 2, 3]
    assert all(d["doc_name"] == "sample.pdf" for d in docs)
    chunks = [chunk for doc in docs for chunk in chunk_document(doc)]
    assert store.add_chunks(chunks) == len(chunks)
    results = retrieve(store, "solar efficiency", 4)
    assert results[0]["page"] == 1 and "23.8%" in results[0]["text"]
    assert retrieve(store, "company founded", 2)[0]["page"] == 3
    context = format_context(results)
    assert "[1]" in context and "page 1, chunk 1" in context
    client, captured = fake_client()
    assert "23.8%" in generate_answer("efficiency?", context, client=client)
    assert context in captured["messages"][1]["content"]
    assert "untrusted" in build_messages("q", context)[0]["content"]


def test_real_overlap():
    text = "".join(chr(0x400 + i) for i in range(2300))
    parts = split_text(text, 1000, 200)
    assert all(len(p) <= 1000 for p in parts)
    assert parts[0][-200:] == parts[1][:200]
    recovered = parts[0] + "".join(part[200:] for part in parts[1:])
    assert recovered == text


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 101)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_blank_text():
    assert split_text("  \n") == []


def test_blank_pages_preserve_number(tmp_path):
    sample = make_sample_pdf(str(tmp_path))
    from pypdf import PdfReader
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_page(PdfReader(sample).pages[0])
    output = BytesIO()
    writer.write(output)
    assert extract_pages(output)[0]["page"] == 2


def test_bad_and_scanned_pdf():
    with pytest.raises(ValueError):
        extract_pages(BytesIO(b"not a pdf"))
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="OCR"):
        extract_pages(output)


def test_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="Password"):
        extract_pages(output)


def test_empty_index_and_k(store):
    assert store.search("question") == []
    with pytest.raises(ValueError):
        store.search(" ")
    with pytest.raises(ValueError):
        store.search("q", 0)
    store.add_chunks([record()])
    assert len(store.search("solar", 8)) == 1


def test_reindex_removes_stale_chunks(store):
    store.add_chunks([record(index=0), record(page=2), record(name="other.pdf")])
    store.add_chunks([record("new solar fact")])
    assert store.count == 2
    texts = store.collection.get()["documents"]
    assert "new solar fact" in texts
    assert not any(m["doc_name"] == "sample.pdf" and m["page"] == 2
                   for m in store.collection.get()["metadatas"])


def test_selected_corpus_replaces_removed_documents(store):
    store.replace_chunks([record(), record(name="other.pdf")])
    store.replace_chunks([record()])
    assert store.count == 1


def test_persistence_and_clear(tmp_path, store):
    store.replace_chunks([record()])
    reopened = EmbeddingStore(str(tmp_path / "db"), embedder=FakeEmbeddings())
    assert reopened.count == 1
    assert reopened.search("solar")[0]["page"] == 1
    reopened.clear()
    assert reopened.count == 0
    assert EmbeddingStore(str(tmp_path / "db"), embedder=FakeEmbeddings()).count == 0


def test_failed_embedding_keeps_previous_index(store):
    store.replace_chunks([record()])
    old_name = store.collection.name
    def fail(texts):
        raise EmbeddingError("provider offline")
    store.embedder.encode = fail
    with pytest.raises(EmbeddingError):
        store.replace_chunks([record("new text")])
    assert store.count == 1 and store.collection.name == old_name
    assert json.loads(store.manifest.read_text())["collection"] == old_name


def test_failed_late_batch_keeps_old_index(store):
    store.replace_chunks([record()])
    original = store.embedder.encode
    calls = 0
    def fail_second(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise EmbeddingError("quota")
        return original(texts)
    store.embedder.encode = fail_second
    with pytest.raises(EmbeddingError):
        store.replace_chunks([record(index=i) for i in range(65)])
    assert store.count == 1
    assert len(store.client.list_collections()) == 1


def test_different_embedding_identity_rejected(tmp_path, store):
    store.replace_chunks([record()])
    other = FakeEmbeddings()
    other.identity = "different-model"
    with pytest.raises(ValueError, match="configuration changed"):
        EmbeddingStore(str(tmp_path / "db"), embedder=other)


def test_distinct_workspaces(tmp_path):
    a = EmbeddingStore(str(tmp_path / "a"), embedder=FakeEmbeddings())
    b = EmbeddingStore(str(tmp_path / "b"), embedder=FakeEmbeddings())
    a.add_chunks([record()])
    assert a.count == 1 and b.count == 0
    b.clear()
    assert a.count == 1


def test_cosine_configuration(store):
    assert store.collection.configuration["hnsw"]["space"] == "cosine"


def test_duplicate_chunk_ids_rejected(store):
    with pytest.raises(ValueError, match="Duplicate"):
        store.replace_chunks([record(), record()])
    assert store.count == 0


@pytest.mark.parametrize("values", [[[float("nan"), 1]], [[0, 0]], [[[1, 2]]], [[1], [2]], []])
def test_invalid_embeddings(values):
    with pytest.raises(EmbeddingError):
        validate_vectors(values, 1)


def test_cloud_adapter_batches_without_local_model():
    import httpx
    batches = []
    def handler(request):
        payload = json.loads(request.content)
        batches.append(payload)
        return httpx.Response(200, json={"embeddings": {"float": [[1, 2, 3]] * len(payload["texts"])}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = CloudEmbeddings(token="test-only", client=client)
        assert len(embedder.encode(["text"] * 65)) == 65
        embedder.encode_query(["question"])
    assert [len(b["texts"]) for b in batches] == [64, 1, 1]
    assert batches[0]["truncate"] == "NONE"
    assert batches[0]["input_type"] == "search_document"
    assert batches[2]["input_type"] == "search_query"


def test_cloud_missing_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="COHERE_API_KEY"):
        CloudEmbeddings().encode(["text"])


@pytest.mark.parametrize("code", [400, 401, 403, 429, 500])
def test_cloud_error_redacted(code):
    import httpx
    def handler(request):
        return httpx.Response(code, json={"message": "secret-token"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EmbeddingError) as error:
            CloudEmbeddings(token="test-only", client=client).encode(["x"])
    assert "secret-token" not in str(error.value)


@pytest.mark.parametrize("answer", ["Unsupported statement.", "Fact [0].", "Fact [2].", None, ""])
def test_generation_rejects_bad_answers(answer):
    client, _ = fake_client(answer)
    with pytest.raises(RuntimeError):
        generate_answer("q", "[1] (Source: a.pdf, page 1)\nfact", client=client, source_count=1)


def test_generation_no_context_never_calls_api():
    assert generate_answer("question", "", client=object()) == NOT_FOUND


def test_generation_refusal_and_truncation():
    client, _ = fake_client(NOT_FOUND)
    assert generate_answer("q", "context", client=client) == NOT_FOUND
    client, _ = fake_client("Fact [1].", finish_reason="length")
    with pytest.raises(RuntimeError, match="length"):
        generate_answer("q", "context", client=client, source_count=1)


def test_upload_names_and_failed_index(tmp_path, store):
    payload = open(make_sample_pdf(str(tmp_path)), "rb").read()
    upload = SimpleNamespace(name="../../sample.pdf", getvalue=lambda: payload)
    assert prepare_uploads([upload])[0]["doc_name"] == "sample.pdf"
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_uploads([upload, upload])
    store.replace_chunks([record()])
    bad = SimpleNamespace(name="bad.pdf", getvalue=lambda: b"invalid")
    with pytest.raises(ValueError):
        index_uploads(store, [upload, bad])
    assert store.count == 1



def test_pdf_limits(monkeypatch):
    import src.ingest as ingest
    monkeypatch.setattr(ingest, "MAX_PDF_BYTES", 3)
    with pytest.raises(ValueError, match="20 MB"):
        extract_pages(BytesIO(b"1234"))
    monkeypatch.setattr(ingest, "MAX_PDF_BYTES", 100000)
    monkeypatch.setattr(ingest, "MAX_PAGES", 1)
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="200-page"):
        extract_pages(output)


def test_groq_sdk_request_and_auth_error():
    import httpx
    from groq import Groq, AuthenticationError
    from src.service import error_message
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Efficiency is 23.8% [1]."}}],
        })
    with Groq(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        answer = generate_answer("efficiency?", "context", client=client, source_count=1)
    assert "23.8%" in answer and seen[0]["max_completion_tokens"] == 1024
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com"))
    error = AuthenticationError("secret-bearing-body", response=response, body={})
    assert "authentication" in error_message(error).lower()
    assert "secret" not in error_message(error)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
