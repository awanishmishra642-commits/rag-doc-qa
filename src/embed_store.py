"""Persistent cosine index with explicit embeddings and staged corpus replacement."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
import chromadb
from src.embeddings import configured_embedder, validate_vectors


class EmbeddingStore:
    def __init__(self, persist_dir="chroma_db", collection_name="documents", embedder=None):
        self.embedder = embedder or configured_embedder()
        self.path = Path(persist_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        # Hash caller's name to keep manifest filenames safe.
        key = hashlib.sha256(collection_name.encode()).hexdigest()[:24]
        self.manifest = self.path / f"active-{key}.json"
        self._collection_name = f"docs-{key}"
        if self.manifest.exists():
            state = json.loads(self.manifest.read_text(encoding="utf-8"))
            if state["embedding_identity"] != self.embedder.identity:
                raise ValueError("Embedding configuration changed. Start a new workspace or restore the old configuration.")
            self._collection_name = state["collection"]
        self.collection = self._open(self._collection_name)

    def _open(self, name):
        collection = self.client.get_or_create_collection(
            name=name, embedding_function=None,
            metadata={"embedding_identity": self.embedder.identity},
            configuration={"hnsw": {"space": "cosine"}},
        )
        if collection.metadata.get("embedding_identity") != self.embedder.identity:
            raise ValueError("Stored embedding model differs from the configured model.")
        return collection

    def _encode(self, texts, query=False):
        encoder = getattr(self.embedder, "encode_query", self.embedder.encode) if query else self.embedder.encode
        return validate_vectors(encoder(texts), len(texts))

    def replace_chunks(self, chunks: list[dict]) -> int:
        """Replace the COMPLETE active corpus. Failed staging leaves old index active.

        One writer per workspace. A manifest switches only after every batch succeeds.
        """
        records, ids = [], set()
        for chunk in chunks:
            if not chunk["text"].strip():
                raise ValueError("Cannot index an empty chunk.")
            metadata = {key: chunk[key] for key in ("doc_name", "page", "chunk_index")}
            cid = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
            if cid in ids:
                raise ValueError("Duplicate document/page/chunk IDs. Rename PDFs with identical names.")
            ids.add(cid)
            records.append((cid, chunk["text"], metadata))
        staged = self._open("docs-" + uuid4().hex)
        try:
            for start in range(0, len(records), 64):
                batch = records[start:start + 64]
                texts = [r[1] for r in batch]
                staged.add(ids=[r[0] for r in batch], documents=texts,
                           metadatas=[r[2] for r in batch], embeddings=self._encode(texts))
            temp = self.manifest.with_suffix(".tmp")
            temp.write_text(json.dumps({"collection": staged.name,
                                       "embedding_identity": self.embedder.identity}), encoding="utf-8")
            os.replace(temp, self.manifest)
        except Exception:
            self.client.delete_collection(staged.name)
            raise
        old_name = self.collection.name
        self.collection = staged
        self._collection_name = staged.name
        try:
            self.client.delete_collection(old_name)
        except Exception:
            # Cleanup failure must not invalidate a successfully committed corpus.
            pass
        return len(records)

    def add_chunks(self, chunks: list[dict]) -> int:
        """Compatibility method: replace supplied document names, keep other documents."""
        if not chunks:
            return 0
        names = {c["doc_name"] for c in chunks}
        old = self.collection.get(include=["documents", "metadatas"])
        keep = [{**meta, "text": text} for text, meta in zip(old["documents"], old["metadatas"])
                if meta["doc_name"] not in names]
        self.replace_chunks(keep + chunks)
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not query.strip():
            raise ValueError("Question cannot be empty.")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        count = self.count
        if not count:
            return []
        result = self.collection.query(query_embeddings=self._encode([query], query=True),
                                       n_results=min(k, count),
                                       include=["documents", "metadatas", "distances"])
        return [{"text": text, **meta, "distance": distance}
                for text, meta, distance in zip(result["documents"][0],
                    result["metadatas"][0], result["distances"][0])]

    def clear(self):
        self.replace_chunks([])

    @property
    def count(self):
        return self.collection.count()
