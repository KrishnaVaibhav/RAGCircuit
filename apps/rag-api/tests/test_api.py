from unittest.mock import MagicMock, patch

from core_rag.types import RetrievedDoc
from fastapi.testclient import TestClient
from rag_api.api import app

client = TestClient(app)


def _make_doc(doc_id: str = "doc-1", text: str = "some text") -> RetrievedDoc:
    return RetrievedDoc(id=doc_id, text=text, score=0.9)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_answer():
    mock_vector = MagicMock()
    mock_vector.tolist.return_value = [0.1] * 384
    doc = _make_doc("doc-1", "relevant context")

    with (
        patch("rag_api.api.embed_chunks", return_value=[(None, mock_vector)]),
        patch("rag_api.api.retrieval.retrieve", return_value=[doc]),
        patch("rag_api.api.ranking.rank", return_value=[doc]),
        patch("rag_api.api.prompts.build_prompt", return_value="the prompt"),
        patch("rag_api.api.llm.generate", return_value="the answer"),
    ):
        resp = client.post("/chat", json={"question": "What is RAG?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "the answer"
    assert data["context_ids"] == ["doc-1"]


def test_chat_context_ids_match_ranked_docs():
    mock_vector = MagicMock()
    mock_vector.tolist.return_value = [0.0] * 384
    docs = [_make_doc("a"), _make_doc("b"), _make_doc("c")]

    with (
        patch("rag_api.api.embed_chunks", return_value=[(None, mock_vector)]),
        patch("rag_api.api.retrieval.retrieve", return_value=docs),
        patch("rag_api.api.ranking.rank", return_value=docs),
        patch("rag_api.api.prompts.build_prompt", return_value="p"),
        patch("rag_api.api.llm.generate", return_value="ans"),
    ):
        resp = client.post("/chat", json={"question": "q?"})

    assert resp.json()["context_ids"] == ["a", "b", "c"]


def test_chat_missing_question_returns_422():
    resp = client.post("/chat", json={})
    assert resp.status_code == 422
