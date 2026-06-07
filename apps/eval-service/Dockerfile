FROM python:3.11-slim

WORKDIR /app

COPY libs/core-rag /app/libs/core-rag
COPY apps/eval-service /app/apps/eval-service

RUN pip install --no-cache-dir -e /app/libs/core-rag && \
    pip install --no-cache-dir -e /app/apps/eval-service

CMD ["python", "apps/eval-service/src/eval_service/main.py"]
