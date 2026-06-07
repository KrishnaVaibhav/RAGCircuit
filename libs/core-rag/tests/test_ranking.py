from core_rag.types import RetrievedDoc


def _make_doc(score: float) -> RetrievedDoc:
    return RetrievedDoc(id="doc-id", text="some text", score=score)


def test_rank_returns_top_k():
    from core_rag.ranking import rank

    candidates = [_make_doc(float(i)) for i in range(10)]
    result = rank("q", candidates, top_k=3)
    assert len(result) == 3


def test_rank_preserves_order():
    from core_rag.ranking import rank

    candidates = [_make_doc(float(i)) for i in range(5)]
    result = rank("q", candidates, top_k=5)
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=False) or scores == [float(i) for i in range(5)]


def test_rank_fewer_candidates_than_top_k():
    from core_rag.ranking import rank

    candidates = [_make_doc(1.0), _make_doc(2.0)]
    result = rank("q", candidates, top_k=10)
    assert len(result) == 2


def test_rank_empty_candidates():
    from core_rag.ranking import rank

    assert rank("q", [], top_k=5) == []
