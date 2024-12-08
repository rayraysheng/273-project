FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade chromadb

# Create directory for persistence
RUN mkdir -p /data && chmod -R 777 /data

WORKDIR /app

# Run the Chroma app directly using uvicorn
EXPOSE 8000
CMD ["sh", "-c", "uvicorn chromadb.app:app --host 0.0.0.0 --port ${CHROMA_DB_PORT:-8000}"]
