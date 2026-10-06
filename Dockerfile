# Dockerfile for a persistent backend service (e.g. an Oracle Cloud Always
# Free VM running Docker). Listens on $PORT (default 8080) via shell form so
# it's still overridable, but on a VM you typically just map this container
# port to a host port with `-p 8080:8080` (see docker-compose.yml).

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
