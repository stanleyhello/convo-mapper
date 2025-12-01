# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Convo-mapper is a real-time audio transcription system that captures both system audio (via loopback) and microphone input, transcribes them using faster-whisper, and provides live topic summarization through a local LLM. It features a Flask-based web UI for viewing transcripts in real-time.

**Key Technologies:**
- Audio capture: soundcard (with NumPy 2.x compatibility patch)
- Speech-to-text: faster-whisper (local, offline)
- Topic summarization: OpenAI-compatible API (expects Parallax at localhost:3001)
- Web UI: Flask with embedded HTML/JS
- Python threading for concurrent audio capture and processing

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Note: NumPy 2.x compatibility
# The codebase includes a patch in main.py (lines 12-28) that wraps np.fromstring
# to maintain compatibility with soundcard, which uses the deprecated binary mode.
```

## Running the Application

### Standard Mode (Web UI Only)
```bash
python main.py
```
Opens web interface at http://127.0.0.1:5000 with live transcription from system audio and microphone.

### Controller Mode (Web UI + Topic Summarization)
```bash
python controller.py
```
Runs both the transcription web app and topic summarization loop in parallel. Requires a Parallax instance or OpenAI-compatible API running at http://localhost:3001/v1.

## Configuration

All configuration is done via constants at the top of each file:

### main.py Audio & Transcription Config
- `SAMPLE_RATE`: 16000 Hz (optimal for speech)
- `REC_BLOCKSIZE`: 512 frames (macOS CoreAudio limit)
- `SYSTEM_CHUNK_SECONDS`: 3.0s for system audio chunks
- `MIC_CHUNK_SECONDS`: 4.0s for mic chunks (longer for better accuracy)
- `MODEL_NAME`: "small" (tiny/base/small/medium/large-v3)
- `LANGUAGE`: "en" or None for auto-detection
- `WHISPER_DEVICE_PREFERENCE`: "cpu" or "cuda"
- `SPEAKER_NAME_FILTER`: None or substring to match specific speaker
- `MIC_NAME_FILTER`: "Yeti" or None

### loop.py Topic Summarization Config
- `POLL_SECONDS`: 1.0s polling interval for new transcripts
- `SUMMARY_INTERVAL_SECONDS`: 60.0s topic summarization interval
- `TOPIC_MODEL`: "Qwen/Qwen3-0.6GB" (model name for topic extraction)
- OpenAI client points to http://localhost:3001/v1 with API key "parallax"

## Architecture

### Core Components

**main.py** - Primary transcription engine with Flask web UI
- Audio capture threads: `system_audio_loop()` and `mic_audio_loop()` continuously record from loopback and mic
- Chunker workers: `chunker_worker()` accumulates audio blocks into configurable-length chunks with 200ms overlap
- Transcriber worker: `transcriber_worker()` runs faster-whisper inference with source-specific settings (mic: beam_size=5, no VAD; system: beam_size=1, with VAD)
- Flask routes: `/` (web UI), `/transcripts` (JSON API), `/clear` (reset transcripts)
- Global state: `system_text` and `mic_text` protected by `text_lock`

**loop.py** - Topic summarization daemon
- `poll_transcripts()`: Monitors main.py's shared text variables, prints new content, accumulates into buffers
- `topic_worker(source)`: Every 60s, sends buffered transcript to local LLM for topic extraction
- Runs as separate threads alongside main.py when launched via controller.py

**controller.py** - Orchestrator for running both main and loop in parallel
- Spawns `main.start_audio_and_model()` and `loop.loop()` as threads
- Keeps main thread alive so worker threads can spawn subthreads

**main_test.py** - Simplified version of main.py (older implementation without chunking overlap or advanced Whisper settings)

### Data Flow

```
Audio Sources → Capture Loops → Raw Audio Queues (system_q/mic_q)
                                        ↓
                            Chunker Workers (overlap + accumulation)
                                        ↓
                                Transcribe Queue
                                        ↓
                            Transcriber Worker (faster-whisper)
                                        ↓
                    Global Text Variables (system_text/mic_text)
                            ↓                       ↓
                    Flask /transcripts API    loop.py Poller
                            ↓                       ↓
                        Web UI              Topic Summarization
```

### Threading Model

All audio/transcription threads are daemon threads, meaning they terminate when the main process exits. The controller.py joins threads to keep the main thread alive.

**Thread Safety:**
- `text_lock` (threading.Lock) protects `system_text` and `mic_text`
- `buffer_lock` in loop.py protects topic summarization buffers
- Queues (queue.Queue) are thread-safe by design

## NumPy 2.x Compatibility

The project includes a critical compatibility shim in main.py:12-28 that patches `np.fromstring` to support soundcard's use of the deprecated binary mode. This wraps the removed functionality by delegating to `np.frombuffer` for binary data while preserving the original behavior for text parsing.

## Device Selection

The system supports filtering audio devices by name substring:
- System audio: Uses loopback microphone from the selected speaker (macOS may require permissions)
- Microphone: Defaults to "Yeti" but falls back to system default if not found
- Falls back gracefully when loopback is unavailable (e.g., some macOS configurations)

## Web UI Features

- Two-column live view (system audio vs mic audio)
- Auto-refresh every 250ms for low latency
- Clear button to reset both UI and backend transcripts (POST to /clear)
- Dark theme optimized for readability
