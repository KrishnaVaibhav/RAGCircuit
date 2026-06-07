from __future__ import annotations

from qdrant_client.models import ScoredPoint

from .tracing import observe_span


@observe_span(name="ranking.rank")
def rank(
    query: str,
    candidates: list[ScoredPoint],
    top_k: int = 5,
) -> list[ScoredPoint]:
    """Re-rank retrieval candidates.

    Default implementation: pass-through, score-ordered from Qdrant.
    Replace with a cross-encoder or MMR here without touching other modules.
    """
    return candidates[:top_k]
