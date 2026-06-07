from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
