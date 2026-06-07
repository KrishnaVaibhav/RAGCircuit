from __future__ import annotations

from .types import RetrievedDoc

RAG_TEMPLATE = """\
Answer the question using only the provided context. \
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, context_docs: list[RetrievedDoc]) -> str:
    context = "\n\n".join(d.text for d in context_docs)
    return RAG_TEMPLATE.format(context=context, question=question)
