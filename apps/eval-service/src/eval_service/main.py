from __future__ import annotations

import os

import httpx
from core_rag.tracing import langfuse, observe_span

RAG_API_URL = os.environ.get("RAG_API_URL") or "http://localhost:8000"

# Populate with question/ground_truth pairs to evaluate against.
EVAL_QUESTIONS: list[dict] = [
    # {"question": "What is RAGCircuit?", "ground_truth": "A RAG platform."},
]


@observe_span(name="eval.run")
def run_eval() -> None:
    results = []
    with httpx.Client(base_url=RAG_API_URL, timeout=120) as client:
        for item in EVAL_QUESTIONS:
            resp = client.post("/chat", json={"question": item["question"]})
            resp.raise_for_status()
            data = resp.json()
            results.append(
                {
                    "question": item["question"],
                    "answer": data["answer"],
                    "ground_truth": item.get("ground_truth"),
                    "context_ids": data.get("context_ids", []),
                }
            )

    # TODO: compute RAGAS metrics (faithfulness, answer_relevancy, context_recall)
    # and log scores to Langfuse via langfuse.score(...)
    print(f"Evaluated {len(results)} questions")
    langfuse.flush()


if __name__ == "__main__":
    run_eval()
