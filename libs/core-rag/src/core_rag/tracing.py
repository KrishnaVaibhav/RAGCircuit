from __future__ import annotations

import os

from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
)

observe_span = observe
