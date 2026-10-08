"""Streamlit script/UI tests without network calls or browser automation."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.embed_store import EmbeddingStore
from tests.fakes import FakeEmbeddings

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_missing_keys_and_empty_question_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("COHERE_API_KEY", "")
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert len(app.warning) == 2
    assert app.button[0].disabled
    app.chat_input[0].set_value("What is the efficiency?").run()
    assert not app.exception
    assert "Upload and index" in app.session_state.messages[-1]["content"]
    app.button[1].click().run()
    assert not app.exception and app.session_state.messages == []


def test_cited_answer_history_and_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path / "unused"))
    store = EmbeddingStore(str(tmp_path / "test"), embedder=FakeEmbeddings())
    store.replace_chunks([dict(doc_name="sample.pdf", page=1, chunk_index=0,
                               text="Solar efficiency is 23.8%.")])
    monkeypatch.setattr("src.generate.generate_answer", lambda *a, **kw: "Efficiency is 23.8% [1].")
    app = AppTest.from_file(APP)
    app.session_state["store"] = store
    app.run(timeout=30)
    app.chat_input[0].set_value("solar efficiency?").run()
    assert not app.exception
    assert "23.8% [1]" in app.session_state.messages[-1]["content"]
    assert any("[1] sample.pdf" in c.value and "chunk 1" in c.value for c in app.caption)
    app.run()
    assert len(app.session_state.messages) == 2
    def fail(*a, **kw):
        raise RuntimeError("Please retry.")
    monkeypatch.setattr("src.generate.generate_answer", fail)
    app.chat_input[0].set_value("another question").run()
    assert not app.exception
    assert "Please retry" in app.session_state.messages[-1]["content"]
