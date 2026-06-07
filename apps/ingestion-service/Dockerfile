FROM python:3.11-slim

WORKDIR /app

COPY libs/core-rag /app/libs/core-rag
COPY apps/ingestion-service /app/apps/ingestion-service

RUN pip install --no-cache-dir -e /app/libs/core-rag && \
    pip install --no-cache-dir -e /app/apps/ingestion-service

CMD ["python", "apps/ingestion-service/src/ingestion_service/main.py"]
