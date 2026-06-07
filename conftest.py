import os

# Set dummy credentials before any package imports so Langfuse
# and other clients initialise without raising on missing env vars.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "test")
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")
os.environ.setdefault("VECTOR_DB_URL", "http://localhost:6333")
os.environ.setdefault("LLM_URL", "http://localhost:11434")
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
