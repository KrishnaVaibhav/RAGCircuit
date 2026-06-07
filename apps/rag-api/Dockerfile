FROM python:3.11-slim

WORKDIR /app

COPY libs/core-rag /app/libs/core-rag
COPY apps/rag-api /app/apps/rag-api

RUN pip install --no-cache-dir -e /app/libs/core-rag && \
    pip install --no-cache-dir -e /app/apps/rag-api

EXPOSE 8000
CMD ["uvicorn", "rag_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
