"""Page-preserving recursive character splitting; sizes are characters."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    if not text.strip():
        return []
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""], length_function=len,
    ).split_text(text)


def chunk_document(doc: dict, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    return [dict(doc_name=doc["doc_name"], page=doc["page"], chunk_index=i, text=text)
            for i, text in enumerate(split_text(doc["text"], chunk_size, chunk_overlap))]
