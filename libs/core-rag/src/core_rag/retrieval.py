from __future__ import annotations

import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from .tracing import observe_span

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.environ.get("VECTOR_DB_URL", "http://localhost:6333")
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


@observe_span(name="retrieval.retrieve")
def retrieve(
    query_vector: list[float],
    collection: str,
    top_k: int = 10,
) -> list[ScoredPoint]:
    return _get_client().search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
    )
