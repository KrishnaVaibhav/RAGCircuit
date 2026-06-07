from unittest.mock import MagicMock, patch

import pytest
from ingestion_service.main import load_raw_docs, run_ingestion


def test_load_raw_docs_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        load_raw_docs()


def test_run_ingestion_calls_full_pipeline():
    docs = [{"id": "1", "text": "hello world", "metadata": {}}]
    fake_chunks = [MagicMock(), MagicMock()]
    fake_embedded = [(MagicMock(), MagicMock()), (MagicMock(), MagicMock())]

    with (
        patch("ingestion_service.main.load_raw_docs", return_value=docs),
        patch("ingestion_service.main.chunking.chunk_docs", return_value=fake_chunks) as mock_chunk,
        patch(
            "ingestion_service.main.embeddings.embed_chunks", return_value=fake_embedded
        ) as mock_embed,
        patch("ingestion_service.main.retrieval.ensure_collection") as mock_ensure,
        patch("ingestion_service.main.retrieval.upsert_chunks") as mock_upsert,
    ):
        run_ingestion()

    mock_chunk.assert_called_once_with(docs)
    mock_embed.assert_called_once_with(fake_chunks)
    mock_ensure.assert_called_once()
    mock_upsert.assert_called_once()


def test_run_ingestion_passes_embedded_to_upsert():
    docs = [{"id": "doc1", "text": "text", "metadata": {}}]
    fake_chunks = [MagicMock()]
    fake_embedded = [(MagicMock(), MagicMock())]

    with (
        patch("ingestion_service.main.load_raw_docs", return_value=docs),
        patch("ingestion_service.main.chunking.chunk_docs", return_value=fake_chunks),
        patch("ingestion_service.main.embeddings.embed_chunks", return_value=fake_embedded),
        patch("ingestion_service.main.retrieval.ensure_collection"),
        patch("ingestion_service.main.retrieval.upsert_chunks") as mock_upsert,
    ):
        run_ingestion()

    # upsert_chunks receives the collection name and the embedded pairs
    args = mock_upsert.call_args
    assert args[0][1] is fake_embedded
