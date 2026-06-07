from __future__ import annotations

import os
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import Chunk
from .tracing import observe_span
from .types import RetrievedDoc

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.environ.get("VECTOR_DB_URL") or "http://localhost:6333"
        )
    return _client


def ensure_collection(
    collection: str,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> None:
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )


def upsert(collection: str, points: list[PointStruct]) -> None:
    _get_client().upsert(collection_name=collection, points=points)


def upsert_chunks(
    collection: str,
    embedded: list[tuple[Chunk, np.ndarray]],
) -> None:
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={"text": chunk.text, **chunk.metadata},
        )
        for chunk, vec in embedded
    ]
    _get_client().upsert(collection_name=collection, points=points)


@observe_span(name="retrieval.retrieve")
def retrieve(
    query_vector: list[float],
    collection: str,
    top_k: int = 10,
) -> list[RetrievedDoc]:
    results = _get_client().search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        RetrievedDoc(
            id=str(r.id),
            text=r.payload.get("text", "") if r.payload else "",
            score=r.score,
            metadata={k: v for k, v in (r.payload or {}).items() if k != "text"},
        )
        for r in results
    ]
