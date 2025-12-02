"""
Centralized configuration for convo-mapper.
All settings can be overridden via environment variables.
"""

import os

# =========================
# AUDIO CONFIGURATION
# =========================

# Target sample rate for speech-to-text
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# Recorder buffer size (frames per read)
# macOS CoreAudio limits blocksize <= 512
REC_BLOCKSIZE = int(os.getenv("REC_BLOCKSIZE", "512"))

# Chunk sizes for audio processing
SYSTEM_CHUNK_SECONDS = float(os.getenv("SYSTEM_CHUNK_SECONDS", "3.0"))
MIC_CHUNK_SECONDS = float(os.getenv("MIC_CHUNK_SECONDS", "4.0"))


# =========================
# WHISPER MODEL CONFIGURATION
# =========================

# Whisper model size: "tiny", "base", "small", "medium", "large-v3"
MODEL_NAME = os.getenv("MODEL_NAME", "small")

# Language for transcription: "en" or None for auto-detection
LANGUAGE = os.getenv("LANGUAGE", "en")

# Whisper device preference: "cuda" for GPU or "cpu"
WHISPER_DEVICE_PREFERENCE = os.getenv("WHISPER_DEVICE_PREFERENCE", "cpu")


# =========================
# AUDIO DEVICE SELECTION
# =========================

# Optional filters to select specific devices by name substring
# Examples: "Focusrite USB Audio", "DELL S2721Q", "Speakers"
SPEAKER_NAME_FILTER = os.getenv("SPEAKER_NAME_FILTER", None)

# Microphone filter - defaults to "Yeti", falls back to system default
MIC_NAME_FILTER = os.getenv("MIC_NAME_FILTER", "Yeti")


# =========================
# TOPIC SUMMARIZATION CONFIGURATION
# =========================

# Polling interval for checking new transcripts (seconds)
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))

# Interval for generating topic summaries (seconds)
SUMMARY_INTERVAL_SECONDS = float(os.getenv("SUMMARY_INTERVAL_SECONDS", "60.0"))

# Model to use for topic extraction
TOPIC_MODEL = os.getenv("TOPIC_MODEL", "Qwen/Qwen3-0.6GB")

# Parallax LLM endpoint configuration
PARALLAX_ENDPOINT = os.getenv("PARALLAX_ENDPOINT", "http://localhost:3001/v1")
PARALLAX_API_KEY = os.getenv("PARALLAX_API_KEY", "parallax")

# Topic extraction prompt
TOPIC_PROMPT = (
    "You are a concise topic spotter. Given up to one minute of transcript, "
    "return a very short title (max 5 words) for the main topic. If the text "
    "is empty or noise, return 'no speech'.\n\nTranscript:\n{transcript}\n\nTitle:"
)


# =========================
# FLASK CONFIGURATION
# =========================

FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
