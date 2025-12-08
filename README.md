# ConvoMapper (local AI transcription and analysis)

A local Flask app that records mic + system audio, runs Whisper for live transcripts, and keeps layered memory (STM/MTM/LTM) to build timelines, insights, and optional therapist-style interjections. UI runs at `http://127.0.0.1:5001`.

## What it does
- Captures system audio (via BlackHole on macOS) and microphone simultaneously.
- Transcribes both streams with faster-whisper.
- Maintains three memory tiers in `loop_therapy.py`:
  - **STM**: rolling ~5 minutes of raw transcript snippets (mic + system) for recent dialogue.
  - **MTM**: per-minute summaries (default every 60s), capped to ~50 entries.
  - **LTM**: distilled recurring patterns from recent MTM items (refreshed every N chunks).
- Groups summaries into topics (titles) for the Timeline/History views.
- Can propose interjections using a configured LLM (local Parallax or remote) with guardrails.
- “Reset” button on Timeline wipes today’s memory file and in-memory buffers.

## How it thinks (logic flow)
- **Summaries**: every `SUMMARY_INTERVAL_SECONDS` (default 60s), the latest transcript delta is summarized to MTM. Titles are generated after a warmup; if two summaries look like the same topic, the title is reused.
- **LTM refresh**: every `LTM_REFRESH_CHUNKS` summaries (default 8), recent MTM items are distilled into long-term bullets (max 8).
- **Interjections**: every `INTERJECT_INTERVAL_SECONDS` (default 60s), the loop builds a prompt with LTM + recent MTM + STM. It will only surface an interjection if:
  - Confidence ≥ `INTERJECT_CONFIDENCE_THRESHOLD`
  - `interventionValue` ≥ 2 and reasons are provided
  - The intervention type is allowed
  - Cooldown since the last interjection ≥ `INTERJECT_COOLDOWN_SECONDS`
  Otherwise it logs a skipped decision for visibility.

## LLM backend options (Parallax/local vs remote)
- Default mode is OpenAI-compatible, but you can run fully local with Parallax or any OpenAI-compatible server.
- Parallax/local setup:
  ```
  TOPIC_API_MODE=local
  LOCAL_LLM_BASE=http://localhost:3001/v1   # Parallax OpenAI-compatible endpoint
  LOCAL_LLM_API_KEY=parallax                # your key/token
  TOPIC_MODEL=Qwen/Qwen3-0.6GB              # or whatever model you serve locally
  ```
- Remote (if you choose to use it):
  ```
  TOPIC_API_MODE=openai
  OPENAI_API_KEY=...
  TOPIC_MODEL=gpt-4o
  ```
- The client is created via the OpenAI SDK with `base_url` pointing at your chosen server, so any compatible endpoint works (local GPU or remote).

## Requirements
- Python 3.9+
- macOS recommended with BlackHole 2ch for system audio loopback
- A working microphone
- If using a remote LLM, network access; for Parallax/local, just the local server running

## Setup
```bash
cd convo-mapper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repo root (edit as needed):
```
TOPIC_API_MODE=local
LOCAL_LLM_BASE=http://localhost:3001/v1
LOCAL_LLM_API_KEY=parallax
TOPIC_MODEL=Qwen/Qwen3-0.6GB
# Optional:
# INTERJECT_CONFIDENCE_THRESHOLD=0.75
# INTERJECT_COOLDOWN_SECONDS=180
# SPEAKER_NAME_FILTER=BlackHole 2ch
# MIC_NAME_FILTER=Your Mic Name
```

## Running
```bash
source .venv/bin/activate
python controller.py
```

Starts:
- `main.py`: Flask on `127.0.0.1:5001` + audio capture threads.
- `loop_therapy.py`: memory + interjection loop writing to `data/`.

Open `http://127.0.0.1:5001` and hit Start Recording.

## UI map
- **Live**: mic/system transcripts (autoscroll).
- **Timeline**: grouped topics. Buttons: Reset (clears today’s memory + buffers) and Refresh.
- **Insights**: recent summaries, interjections, LTM bullets.
- **Report**: quick stats + session summary + patterns.
- **History**: past days from `data/memory-*.jsonl`.
- Toggles: system audio, mic, therapy mode. Clear = wipe live transcript buffers only.

## Where data lives
- Transcripts (raw): `data/transcript-YYYY-MM-DD.jsonl` (if enabled).
- Memory (summaries/interjections/LTM): `data/memory-YYYY-MM-DD.jsonl`.
- Audio is not persisted, only text.

## Device tips (macOS)
- Install BlackHole 2ch and set it as system output (add Multi-Output Device if you need local playback).
- Set `SPEAKER_NAME_FILTER="BlackHole 2ch"` and `MIC_NAME_FILTER` to your mic to avoid BlackHole-as-mic.

## Troubleshooting
- Timeline/Insights empty: ensure recording is on and the LLM endpoint responds. Check `data/memory-<today>.jsonl` for `"type": "chunk"` entries.
- State stuck: use Timeline “Reset” (POST `/api/timeline/reset`) to clear buffers and delete today’s memory file.
- Audio device errors: confirm BlackHole is installed/selected; adjust `SPEAKER_NAME_FILTER`/`MIC_NAME_FILTER`.

## Notes
- Whisper model/device is set in `main.py` (`MODEL_NAME`, `WHISPER_DEVICE_PREFERENCE`).
- Summaries every `SUMMARY_INTERVAL_SECONDS` (default 60s); interjections every `INTERJECT_INTERVAL_SECONDS`.
- Interjections are filtered by confidence, value, allowed types, and cooldown to reduce noise. 
