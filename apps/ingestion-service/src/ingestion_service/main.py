from __future__ import annotations

import os
import uuid

from qdrant_client.models import PointStruct

from core_rag import chunking, embeddings, retrieval
from core_rag.tracing import observe_span

COLLECTION = os.environ.get("COLLECTION_NAME", "documents")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "384"))


def load_raw_docs() -> list[dict]:
    """Load source documents for ingestion.

    Each returned dict must have:
        id       (str)  — unique document identifier
        text     (str)  — raw text content
        metadata (dict) — arbitrary key/value pairs stored in the vector DB

    Override this to pull from S3, a database, a filesystem, an API, etc.
    """
    raise NotImplementedError("Implement load_raw_docs for your data source")


@observe_span(name="ingestion.run")
def run_ingestion() -> None:
    docs = load_raw_docs()
    chunks = chunking.chunk_docs(docs)
    embedded = embeddings.embed_chunks(chunks)

    retrieval.ensure_collection(COLLECTION, VECTOR_SIZE)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={"text": chunk.text, **chunk.metadata},
        )
        for chunk, vec in embedded
    ]
    retrieval.upsert(COLLECTION, points)
    print(f"Ingested {len(points)} chunks into '{COLLECTION}'")


if __name__ == "__main__":
    run_ingestion()
