FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Descarga el modelo de embeddings al construir, no al arrancar
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')" || true

# Cloud Run asigna el puerto en $PORT. 1 worker: el índice vive en memoria del proceso.
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --worker-class gthread --threads 4 --timeout 600
