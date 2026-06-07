from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from core_rag import llm, prompts, ranking, retrieval
from core_rag.chunking import Chunk
from core_rag.embeddings import embed_chunks
from core_rag.tracing import observe_span

app = FastAPI(title="RAG API", version="0.1.0")

COLLECTION = os.environ.get("COLLECTION_NAME", "documents")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    context_ids: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@observe_span(name="rag-api.chat")
def chat(request: ChatRequest) -> ChatResponse:
    [(_, query_vector)] = embed_chunks(
        [Chunk(id="query", text=request.question)]
    )
    retrieved = retrieval.retrieve(
        query_vector.tolist(), COLLECTION, top_k=request.top_k * 2
    )
    ranked = ranking.rank(request.question, retrieved, top_k=request.top_k)
    prompt = prompts.build_prompt(request.question, ranked)
    answer = llm.generate(prompt)
    return ChatResponse(
        answer=answer,
        context_ids=[str(r.id) for r in ranked],
    )
