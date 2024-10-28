# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app /app

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${DOCUMENT_MANAGEMENT_PORT}"]
