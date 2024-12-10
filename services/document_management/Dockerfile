# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Upgrade pip and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --upgrade chromadb

# Copy application code
COPY ./app /app

# Expose the application port
EXPOSE ${DOCUMENT_MANAGEMENT_PORT:-8000}

# Set default environment variables
ENV DOCUMENT_MANAGEMENT_PORT=8000

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${DOCUMENT_MANAGEMENT_PORT:-8000} --reload"]

