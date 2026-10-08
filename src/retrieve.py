"""Retrieval and numbered source blocks; citations match the UI exactly."""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.embed_store import EmbeddingStore


def retrieve(store: "EmbeddingStore", query: str, k: int = 4) -> list[dict]:
    return store.search(query, k=k)


def format_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (Source: {r['doc_name']}, page {r['page']}, chunk {r['chunk_index'] + 1})\n{r['text']}"
        for i, r in enumerate(results, 1)
    )
