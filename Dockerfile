FROM python:3.11-slim-bookworm

# Install audio system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    pulseaudio \
    alsa-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py loop.py controller.py config.py ./

# Environment variables with defaults
ENV PYTHONUNBUFFERED=1 \
    SAMPLE_RATE=16000 \
    MODEL_NAME=small \
    WHISPER_DEVICE_PREFERENCE=cpu

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD curl -f http://localhost:5000/transcripts || exit 1

CMD ["python", "controller.py"]
