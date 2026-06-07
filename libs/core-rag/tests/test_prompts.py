from core_rag.prompts import RAG_TEMPLATE, build_prompt
from core_rag.types import RetrievedDoc


def _make_doc(text: str) -> RetrievedDoc:
    return RetrievedDoc(id="doc-id", text=text, score=1.0)


def test_question_appears_in_prompt():
    prompt = build_prompt("What is RAG?", [_make_doc("RAG stands for retrieval-augmented generation.")])
    assert "What is RAG?" in prompt


def test_context_text_appears_in_prompt():
    prompt = build_prompt("q?", [_make_doc("The sky is blue.")])
    assert "The sky is blue." in prompt


def test_multiple_context_docs_all_included():
    docs = [_make_doc("Fact one."), _make_doc("Fact two.")]
    prompt = build_prompt("q?", docs)
    assert "Fact one." in prompt
    assert "Fact two." in prompt


def test_empty_context_still_includes_question():
    prompt = build_prompt("What is 2+2?", [])
    assert "What is 2+2?" in prompt


def test_prompt_uses_rag_template_structure():
    prompt = build_prompt("q?", [_make_doc("ctx")])
    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt


def test_doc_with_empty_text_still_includes_question():
    doc = RetrievedDoc(id="doc-id", text="", score=1.0)
    prompt = build_prompt("q?", [doc])
    assert "Question: q?" in prompt
