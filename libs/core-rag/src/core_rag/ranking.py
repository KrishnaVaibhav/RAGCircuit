from __future__ import annotations

from .tracing import observe_span
from .types import RetrievedDoc


@observe_span(name="ranking.rank")
def rank(
    query: str,
    candidates: list[RetrievedDoc],
    top_k: int = 5,
) -> list[RetrievedDoc]:
    """Re-rank retrieval candidates.

    Default implementation: pass-through, score-ordered from Qdrant.
    Replace with a cross-encoder or MMR here without touching other modules.
    """
    return candidates[:top_k]
