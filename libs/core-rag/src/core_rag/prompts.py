from __future__ import annotations

from qdrant_client.models import ScoredPoint

RAG_TEMPLATE = """\
Answer the question using only the provided context. \
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, context_docs: list[ScoredPoint]) -> str:
    context = "\n\n".join(
        d.payload.get("text", "") for d in context_docs if d.payload
    )
    return RAG_TEMPLATE.format(context=context, question=question)
