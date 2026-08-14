# Dockerfile for Google Cloud Run.
# Cloud Run injects a PORT environment variable (defaults to 8080) that the
# container must listen on — using shell form so $PORT is expanded at
# container start, with a fallback default for local testing.

FROM python:3.11-slim

WORKDIR /app

# System libraries required by opencv-python-headless and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

# --timeout 120: EfficientNet-B4 inference + Grad-CAM's backward pass on CPU
# can take longer than gunicorn's 30s default worker timeout.
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
