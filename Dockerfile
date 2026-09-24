FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/model_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so the first request does not have to.
# (Before COPY . . so code changes do not invalidate this cached layer.)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/model_cache')"

COPY . .

EXPOSE 8000

# Render sets $PORT; fall back to 8000 for local runs.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}