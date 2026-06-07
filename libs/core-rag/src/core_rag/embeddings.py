from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk
from .tracing import observe_span

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        )
    return _model


@observe_span(name="embeddings.embed_chunks")
def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, np.ndarray]]:
    """Return (chunk, vector) pairs. Vector dim determined by EMBED_MODEL."""
    model = _get_model()
    texts = [c.text for c in chunks]
    vectors: np.ndarray = model.encode(texts, show_progress_bar=False)
    return list(zip(chunks, vectors))
