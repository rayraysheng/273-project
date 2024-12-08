FROM python:3.10-slim

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Chroma DB
RUN pip install chromadb

# Expose the port for Chroma API
EXPOSE 8000

# Run the Chroma server using the environment variable for the port
CMD ["sh", "-c", "chromadb --host 0.0.0.0 --port ${CHROMA_DB_PORT:-8000} --persist-dir /data"]
