from core_rag.chunking import Chunk, chunk_docs


def test_single_short_doc_produces_one_chunk():
    docs = [{"id": "d1", "text": "Hello world."}]
    chunks = chunk_docs(docs)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "Hello world."


def test_long_doc_splits_into_multiple_chunks():
    docs = [{"id": "d1", "text": "word " * 300}]
    chunks = chunk_docs(docs, chunk_size=64, chunk_overlap=0)
    assert len(chunks) > 1


def test_chunk_ids_carry_doc_prefix():
    docs = [{"id": "doc-42", "text": "word " * 200}]
    chunks = chunk_docs(docs, chunk_size=64, chunk_overlap=0)
    assert all(c.id.startswith("doc-42_") for c in chunks)


def test_metadata_propagated_to_chunks():
    docs = [{"id": "d1", "text": "hello", "metadata": {"source": "wiki"}}]
    chunks = chunk_docs(docs)
    assert chunks[0].metadata == {"source": "wiki"}


def test_empty_input_returns_empty_list():
    assert chunk_docs([]) == []


def test_multiple_docs_chunk_independently():
    docs = [
        {"id": "a", "text": "word " * 200},
        {"id": "b", "text": "word " * 200},
    ]
    chunks = chunk_docs(docs, chunk_size=64, chunk_overlap=0)
    a_chunks = [c for c in chunks if c.id.startswith("a_")]
    b_chunks = [c for c in chunks if c.id.startswith("b_")]
    assert len(a_chunks) > 0
    assert len(b_chunks) > 0
