"""Cloud embeddings by default; local CPU inference is an explicit opt-in."""
import os
import numpy as np


class EmbeddingError(RuntimeError):
    pass


def validate_vectors(values, count: int) -> list[list[float]]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("Embedding provider returned an invalid vector array.") from exc
    if (array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0
            or not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any()):
        raise EmbeddingError("Expected one finite, non-zero sentence vector per input.")
    return array.tolist()


class CloudEmbeddings:
    """Cohere v2 embed with a free evaluation key, subject to provider limits."""
    def __init__(self, model=None, token=None, client=None):
        self.model = model or os.getenv("EMBEDDING_MODEL") or "embed-english-light-v3.0"
        self.identity = f"cohere:{self.model}:search_document-query:v1"
        self._token = token
        self._client = client

    def encode(self, texts):
        return self._encode(texts, "search_document")

    def encode_query(self, texts):
        return self._encode(texts, "search_query")

    def _encode(self, texts, input_type):
        import httpx
        from contextlib import nullcontext
        token = self._token or os.getenv("COHERE_API_KEY", "").strip()
        if not token or token.startswith("your_"):
            raise EmbeddingError("Set COHERE_API_KEY in .env using a free evaluation key.")
        output = []
        manager = nullcontext(self._client) if self._client is not None else httpx.Client(timeout=60)
        try:
            with manager as client:
                for start in range(0, len(texts), 64):
                    batch = texts[start:start + 64]
                    response = client.post(
                        "https://api.cohere.com/v2/embed",
                        headers={"Authorization": f"Bearer {token}"},
                        json={"model": self.model, "texts": batch,
                              "input_type": input_type, "embedding_types": ["float"],
                              "truncate": "NONE"},
                    )
                    response.raise_for_status()
                    output.extend(validate_vectors(response.json()["embeddings"]["float"], len(batch)))
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (401, 403):
                message = "Cohere authentication/access failed. Check COHERE_API_KEY."
            elif code == 429:
                message = "Cohere rate limit/quota reached. Check evaluation limits and retry later."
            elif code in (400, 422):
                message = "Cohere rejected the embedding input/model. Reduce CHUNK_SIZE or question length and check EMBEDDING_MODEL."
            else:
                message = f"Cohere embedding request failed (HTTP {code}). Retry later."
            raise EmbeddingError(message) from exc
        except httpx.RequestError as exc:
            raise EmbeddingError("Could not reach Cohere. Check your connection and retry.") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("Cohere returned an invalid embedding response.") from exc
        return output


class LocalEmbeddings:
    def __init__(self, model=None):
        self.name = model or os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.identity = f"local:{self.name}"
        self._model = None

    def encode(self, texts):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError("Local mode needs requirements-local.txt; it runs AI on your CPU.") from exc
            self._model = SentenceTransformer(self.name, device="cpu")
        # Fail explicitly instead of silently truncating long chunks/questions.
        lengths = self._model.tokenizer(texts, truncation=False)["input_ids"]
        if any(len(tokens) > self._model.max_seq_length for tokens in lengths):
            raise EmbeddingError("Embedding input exceeds model token limit. Reduce CHUNK_SIZE in .env or use a shorter question.")
        return validate_vectors(self._model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False), len(texts))


def configured_embedder():
    mode = os.getenv("EMBEDDING_BACKEND", "cloud").strip().lower()
    if mode == "cloud":
        return CloudEmbeddings()
    if mode == "local":
        return LocalEmbeddings()
    raise EmbeddingError("EMBEDDING_BACKEND must be cloud or local.")
