from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .tracing import observe_span


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@observe_span(name="chunking.chunk_docs")
def chunk_docs(
    docs: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split raw docs into overlapping text chunks.

    Each doc must have keys: ``id`` (str), ``text`` (str),
    and optionally ``metadata`` (dict).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Chunk] = []
    for doc in docs:
        texts = splitter.split_text(doc["text"])
        for i, text in enumerate(texts):
            chunks.append(
                Chunk(
                    id=f"{doc['id']}_{i}",
                    text=text,
                    metadata=doc.get("metadata", {}),
                )
            )
    return chunks
