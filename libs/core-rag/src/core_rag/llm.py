from __future__ import annotations

import os

import httpx

from .tracing import observe_span


@observe_span(name="llm.generate")
def generate(prompt: str, model: str | None = None) -> str:
    """Call Ollama-compatible HTTP API and return generated text."""
    url = os.environ.get("LLM_URL") or "http://localhost:11434"
    resolved_model = model or os.environ.get("LLM_MODEL") or "llama3.2"
    response = httpx.post(
        f"{url}/api/generate",
        json={"model": resolved_model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]
