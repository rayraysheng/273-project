# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create and set permissions for Chroma DB directory
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Copy application code
COPY ./app /app

# Expose port (optional, for documentation purposes; runtime handled by env)
EXPOSE ${DOCUMENT_MANAGEMENT_PORT}

# Set environment variables for runtime
ENV DOCUMENT_MANAGEMENT_PORT=8000

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${DOCUMENT_MANAGEMENT_PORT:-8000} --reload"]

