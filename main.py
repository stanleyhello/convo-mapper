import json
import os
import time
import queue
import threading
import datetime
import glob

import numpy as np
try:
    import noisereduce as nr
except ImportError:
    nr = None
    print("Warning: noisereduce not installed; audio denoising disabled.")

try:
    import webrtcvad
except ImportError:
    webrtcvad = None
    print("Warning: webrtcvad not installed; VAD filtering disabled.")

# =========================
# NUMPY / SOUNDCard PATCH
# =========================
# Work around NumPy 2.x removing the old binary mode of fromstring,
# which soundcard still uses internally.
try:
    _orig_fromstring = np.fromstring

    def _fromstring_compat(string, dtype=float, count=-1, sep=''):
        # soundcard passes a raw buffer object from _ffi.buffer(...)
        # with sep == '' → this is the deprecated "binary" mode.
        if sep == '' and not isinstance(string, str):
            # treat anything non-string here as a raw bytes-like buffer
            return np.frombuffer(string, dtype=dtype, count=count)

        # for normal (text) use-cases, defer to the original implementation
        return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)

    np.fromstring = _fromstring_compat
except Exception:
    # if anything goes wrong, leave numpy as-is
    pass

import soundcard as sc
from faster_whisper import WhisperModel
from flask import Flask, jsonify, render_template_string, request, send_from_directory


# =========================
# CONFIG
# =========================

SAMPLE_RATE = 16000          # target sample rate for STT

# recorder buffer sizes. macOS CoreAudio has per-device limits.
REC_BLOCKSIZE_SYSTEM = 512   # frames per read for system audio (BlackHole supports 512)
REC_BLOCKSIZE_MIC = 256      # frames per read for mic (some mics limit to 480)

# separate chunk sizes
SYSTEM_CHUNK_SECONDS = 3.0   # system audio chunks
MIC_CHUNK_SECONDS = 4.0      # mic audio chunks (more context for accuracy)

MODEL_NAME = "small"         # "tiny"/"base"/"small" are better for CPU
LANGUAGE = "en"              # or None for auto

# Preferred device for Whisper: "cuda" (GPU) or "cpu"
WHISPER_DEVICE_PREFERENCE = "cpu"
# Guiding prompt to reduce hallucinations during silence
GUIDING_PROMPT = (
    "The audio may have silence or background noise. "
    "If no one is speaking, return nothing. Do not make up words or add conversational fillers."
)
# Phrases to strip unconditionally from ASR output (prompt echoes)
DROP_PHRASES = [
    "do not make up words",
    "conversational fillers",
    "if no one is speaking",
]
# Phrases to detect if Whisper echoes the prompt; any match will be canonicalized
GUIDING_PROMPT_SNIPPETS = [
    "the audio may have silence or background noise",
    "if no one is speaking, return nothing",
]
# Canonical marker to make silence-echo insertions easy to strip in post
SILENCE_ECHO_MARKER = "[[SILENCE]]"
# Toggle false-positive phrase filtering (set to False to disable)
ENABLE_FALSE_POSITIVE_FILTER = False

# Toggle sources
ENABLE_SYSTEM_CAPTURE = True
ENABLE_MIC_CAPTURE = True

# Minimum volume (dBFS) required for a chunk to be considered speech.
# Set closer to 0 for more sensitivity; more negative for stricter filtering.
MIN_VOLUME_DB = -50.0  # Lowered to capture quieter speech

# Voice activity detection configuration
ENABLE_VAD = True
VAD_AGGRESSIVENESS = 2  # 0-3; 2 = medium strictness (was 3, too strict)
VAD_MIN_SPEECH_RATIO = 0.15  # Minimum fraction of frames that must contain speech (lowered from 0.3)

# Optional filters to force a specific device by name substring
# Set to None to use default/any available device
SPEAKER_NAME_FILTER = "BlackHole 2ch"   # For system audio capture via BlackHole virtual device
MIC_NAME_FILTER = "Akif's AirPods Pro"  # Use AirPods mic (no speaker bleed)

# Optional JSONL logging (rotated daily)
ENABLE_TRANSCRIPT_LOG = True
LOG_PATH_TEMPLATE = "data/transcript-{date}.jsonl"


# =========================
# QUEUES & STATE
# =========================

system_q = queue.Queue()
mic_q = queue.Queue()
transcribe_q = queue.Queue()

system_text = ""
mic_text = ""
text_lock = threading.Lock()
log_lock = threading.Lock()
noise_lock = threading.Lock()
noise_profiles = {"system": None, "mic": None}

# Recording state controls
recording_enabled = False
system_capture_enabled = True
mic_capture_enabled = True
therapy_mode_enabled = False
state_lock = threading.Lock()


def _log_entry(source: str, text: str):
    """
    Append a transcript entry to the current day's JSONL file.
    """
    if not ENABLE_TRANSCRIPT_LOG or not text:
        return

    path = LOG_PATH_TEMPLATE.format(date=datetime.date.today().isoformat())
    entry = {
        "ts": time.time(),
        "source": source,
        "text": text,
    }
    try:
        with log_lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Avoid spamming; best-effort logging
        print(f"Warning: failed to write transcript log to {path}: {e}")


def _condense_summaries(summaries: list, max_chars: int = 400) -> str:
    """
    Condense multiple summaries into a single concise paragraph.
    If only 1 summary or total is short, return as-is.
    Otherwise, extract key sentences to create a condensed version.
    """
    if not summaries:
        return ""
    
    if len(summaries) == 1:
        return summaries[0]
    
    # Join all summaries
    combined = " ".join(summaries)
    
    # If short enough, return as-is
    if len(combined) <= max_chars:
        return combined
    
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', combined)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 2:
        return combined[:max_chars] + "..."
    
    # Take first sentence, and try to include more if space allows
    result = []
    current_len = 0
    
    # Always include first sentence
    result.append(sentences[0])
    current_len += len(sentences[0])
    
    # Try to add middle/end sentences if space allows
    remaining = max_chars - current_len - 50  # Leave room for "..." indicator
    
    # Add last sentence if it fits
    if len(sentences[-1]) <= remaining:
        result.append(sentences[-1])
        remaining -= len(sentences[-1])
    
    # Add a middle sentence if space
    if len(sentences) > 3:
        mid_idx = len(sentences) // 2
        if len(sentences[mid_idx]) <= remaining:
            result.insert(1, sentences[mid_idx])
    
    # If we couldn't fit much, just truncate the combined text
    if len(result) == 1 and len(combined) > max_chars:
        return combined[:max_chars].rsplit(' ', 1)[0] + "..."
    
    return " ".join(result)


# =========================
# DEVICE HELPERS
# =========================

def get_system_speaker():
    """
    Pick a speaker based on SPEAKER_NAME_FILTER, otherwise default speaker.
    """
    speakers = sc.all_speakers()
    if SPEAKER_NAME_FILTER:
        for spk in speakers:
            if SPEAKER_NAME_FILTER in spk.name:
                print(f"System audio: {spk.name}")
                return spk
        print(f"System audio (filter '{SPEAKER_NAME_FILTER}' not found) → using default speaker")
    default = sc.default_speaker()
    print(f"System audio: {default.name}")
    return default


def get_system_loopback_mic():
    """
    On macOS, loopback doesn't work natively. Instead, we look for 
    BlackHole (or similar virtual audio device) as a microphone input
    that receives audio from Multi-Output Device.
    """
    # On macOS, we need to use BlackHole as a microphone input directly
    mics = sc.all_microphones(include_loopback=True)
    
    # First, try to find BlackHole as a microphone
    if SPEAKER_NAME_FILTER:
        for mic in mics:
            if SPEAKER_NAME_FILTER.lower() in mic.name.lower():
                print(f"System audio input: {mic.name}")
                return mic
    
    # Fallback: try traditional loopback (works on Windows/Linux)
    try:
        speaker = get_system_speaker()
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        print(f"System audio (loopback): {loopback.name}")
        return loopback
    except Exception as e:
        print(f"Loopback not available: {e}")
    
    print("No system audio capture device found. Check BlackHole setup.")
    return None


def get_mic():
    """
    Pick a microphone based on MIC_NAME_FILTER, otherwise default mic.
    Excludes BlackHole from mic selection (that's for system audio).
    """
    mics = sc.all_microphones()
    
    # If a filter is specified, try to find matching mic
    if MIC_NAME_FILTER:
        for m in mics:
            if MIC_NAME_FILTER.lower() in m.name.lower():
                print(f"Mic input: {m.name}")
                return m
        print(f"Mic input (filter '{MIC_NAME_FILTER}' not found) → using default mic")
    
    # Use default mic, but avoid BlackHole (that's for system audio)
    default = sc.default_microphone()
    if default and "blackhole" not in default.name.lower():
        print(f"Mic input: {default.name}")
        return default
    
    # If default is BlackHole, find another mic
    for m in mics:
        if "blackhole" not in m.name.lower():
            print(f"Mic input: {m.name}")
            return m
    
    print("Warning: No suitable microphone found (only BlackHole available)")
    return default


# =========================
# AUDIO CAPTURE LOOPS
# =========================

def system_audio_loop():
    """Capture system audio via BlackHole."""
    try:
        dev = get_system_loopback_mic()
        if dev is None:
            print("Skipping system audio (no device).", flush=True)
            return
        print(f"[SYSTEM] Ready: {dev.name}", flush=True)
        with dev.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=REC_BLOCKSIZE_SYSTEM) as rec:
            while True:
                data = rec.record(numframes=REC_BLOCKSIZE_SYSTEM)
                with state_lock:
                    enabled = recording_enabled and system_capture_enabled
                if enabled:
                    system_q.put((time.time(), data.astype(np.float32)))
    except Exception as e:
        print(f"[SYSTEM] Error: {e}", flush=True)


def mic_audio_loop():
    """Capture mic audio from the chosen mic."""
    try:
        mic = get_mic()
        if mic is None:
            print("[MIC] Skipping mic capture (no microphone found).", flush=True)
            return
        print(f"[MIC] Ready: {mic.name}", flush=True)
        with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=REC_BLOCKSIZE_MIC) as rec:
            while True:
                data = rec.record(numframes=REC_BLOCKSIZE_MIC)
                with state_lock:
                    enabled = recording_enabled and mic_capture_enabled
                if enabled:
                    mic_q.put((time.time(), data.astype(np.float32)))
    except Exception as e:
        print(f"[MIC] Error: {e}", flush=True)


# =========================
# CHUNKER WORKERS
# =========================

def chunker_worker(name, in_q, out_q, sample_rate, chunk_seconds):
    """
    Collect blocks from in_q, convert to mono, accumulate until
    chunk_seconds of audio, then push (name, chunk) to out_q.

    Uses a small overlap between successive chunks so words at
    boundaries are less likely to be cut in half.
    """
    samples_needed = int(sample_rate * chunk_seconds)
    # 200 ms overlap
    overlap_samples = int(sample_rate * 0.2)
    # never overlap more than the chunk itself
    overlap_samples = min(overlap_samples, samples_needed // 2)

    buffer = np.zeros(0, dtype=np.float32)

    while True:
        timestamp, data = in_q.get()
        if data is None:
            break

        # data: (frames, channels)
        if data.ndim == 2:
            mono = data.mean(axis=1)
        else:
            mono = data

        buffer = np.concatenate([buffer, mono])

        while len(buffer) >= samples_needed:
            # take a full chunk
            chunk = buffer[:samples_needed]
            out_q.put((name, chunk))

            # keep a bit of the end of this chunk as overlap
            buffer = buffer[samples_needed - overlap_samples:]


# =========================
# TRANSCRIBER WORKER
# =========================
def denoise_audio(source: str, audio: np.ndarray) -> np.ndarray:
    """
    Apply lightweight noise reduction using a running noise profile per source.
    Falls back to raw audio if noisereduce is unavailable.
    """
    if nr is None or audio.size == 0:
        return audio

    with noise_lock:
        noise_profile = noise_profiles.get(source)
        if noise_profile is None:
            noise_profiles[source] = audio.copy()
            return audio
        noise_snapshot = noise_profile.copy()

    try:
        cleaned = nr.reduce_noise(
            y=audio,
            y_noise=noise_snapshot,
            sr=SAMPLE_RATE,
            stationary=False,
            prop_decrease=0.85,
        ).astype(np.float32)

        # Update noise profile using exponential moving average
        with noise_lock:
            current = noise_profiles[source]
            trim = min(len(current), len(audio))
            current[:trim] = 0.9 * current[:trim] + 0.1 * audio[:trim]
        return cleaned
    except Exception as exc:
        print(f"Noise reduction failed for {source}: {exc}")
        return audio


def is_chunk_loud(audio: np.ndarray, threshold_db: float = MIN_VOLUME_DB) -> bool:
    """
    Determine whether the audio chunk has sufficient volume to be sent to STT.
    """
    if audio.size == 0:
        return False

    rms = np.sqrt(np.mean(np.square(audio)))
    rms = max(rms, 1e-8)  # avoid log(0)
    db = 20 * np.log10(rms)
    return db >= threshold_db


# Common false positive phrases that Whisper often generates from silence/noise
FALSE_POSITIVE_PHRASES = {
    # Keep only single-word fillers to avoid blocking legit phrases like "thank you"
    "okay", "ok",
    "yeah", "yes", "yep", "yup",
    "uh", "um",
    "mm", "hmm",
    "hello", "hi", "hey",
    "goodbye", "bye",
    "sorry",
    "please",
    "alright",
    "sure",
    "right",
    "well",
}


def is_likely_false_positive(text: str) -> bool:
    """
    Check if transcribed text is likely a false positive from silence/noise.
    Filters out common short phrases that Whisper often hallucinates.
    Made less aggressive to avoid filtering legitimate speech.
    """
    if not text:
        return True
    if text == SILENCE_ECHO_MARKER:
        return False
    if not ENABLE_FALSE_POSITIVE_FILTER:
        return False
    
    text_lower = text.lower().strip()

    # Drop anything that echoes the guiding prompt (or snippets of it)
    if any(snippet in text_lower for snippet in GUIDING_PROMPT_SNIPPETS):
        return True
    
    # Filter out very short texts (likely noise) - but be more lenient
    if len(text_lower) < 2:
        return True
    
    # Only filter exact matches of known false positive phrases
    # This is more conservative - won't filter partial matches
    if text_lower in FALSE_POSITIVE_PHRASES:
        return True
    
    # Only filter if text is just repeated single words (e.g., "okay okay okay")
    # and it's a known false positive
    words = text_lower.split()
    if len(words) >= 3 and len(set(words)) == 1:
        # Check if the repeated word is a known false positive
        if words[0] in FALSE_POSITIVE_PHRASES:
            return True
    
    # Don't filter 2-word phrases - they might be legitimate speech
    # Only filter single-word exact matches
    if len(words) == 1 and text_lower in FALSE_POSITIVE_PHRASES:
        return True
    
    return False


def has_voice_activity(audio: np.ndarray) -> bool:
    """
    Use WebRTC VAD to determine if the chunk contains speech.
    Requires a minimum percentage of frames to contain speech to reduce false positives.
    Returns True when VAD is disabled or unavailable.
    """
    if not ENABLE_VAD or webrtcvad is None:
        return True
    if audio.size == 0:
        return False

    try:
        vad = webrtcvad.Vad(int(np.clip(VAD_AGGRESSIVENESS, 0, 3)))
        frame_duration_ms = 30  # 10/20/30ms supported
        frame_length = int(SAMPLE_RATE * frame_duration_ms / 1000)
        if frame_length == 0:
            return False

        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        bytes_per_frame = frame_length * 2  # int16 -> 2 bytes
        total_bytes = len(pcm16) * 2
        usable_bytes = total_bytes - (total_bytes % bytes_per_frame)
        if usable_bytes <= 0:
            return False

        pcm_bytes = pcm16.tobytes()
        speech_frames = 0
        total_frames = 0
        
        for offset in range(0, usable_bytes, bytes_per_frame):
            frame = pcm_bytes[offset:offset + bytes_per_frame]
            total_frames += 1
            if vad.is_speech(frame, SAMPLE_RATE):
                speech_frames += 1
        
        if total_frames == 0:
            return False
        
        # Require minimum percentage of frames to contain speech
        speech_ratio = speech_frames / total_frames
        return speech_ratio >= VAD_MIN_SPEECH_RATIO
    except Exception as exc:
        print(f"VAD processing failed, passing audio through: {exc}")
        return True


def transcriber_worker(model, in_q, language=None):
    """
    Pull (source, chunk) from in_q and run faster-whisper locally.
    Append text to in-memory transcripts.
    """
    global system_text, mic_text

    prompt_kwargs = {"initial_prompt": GUIDING_PROMPT} if GUIDING_PROMPT else {}

    while True:
        source, chunk = in_q.get()
        if chunk is None:
            break

        raw_audio = chunk.astype(np.float32)

        # Hard energy gate before denoise
        if not is_chunk_loud(raw_audio):
            denoise_audio(source, raw_audio)  # keep profile up to date
            continue

        # Require VAD speech before transcribing
        if not has_voice_activity(raw_audio):
            denoise_audio(source, raw_audio)
            continue

        audio = denoise_audio(source, raw_audio)

        # Post-denoise energy gate to avoid boosted noise passing through
        if not is_chunk_loud(audio):
            continue
        
        if source == "mic":
            # Mic: prioritize quality/stability
            # Disable Whisper's VAD since we're doing our own filtering
            # This prevents double-filtering that might miss speech
            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=5,      # more search → better accuracy
                vad_filter=False,  # Disabled - we already filtered with our VAD
                
                **prompt_kwargs,
            )
        else:
            # System: keep fast, it's already good
            # System audio is usually cleaner, so keep VAD enabled
            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=True,
                **prompt_kwargs,
            )

        text = "".join(seg.text for seg in segments).strip()
        if not text:
            continue

        # If Whisper echoes our guiding prompt, replace with a canonical marker
        text_lower = text.lower()
        # Strip known prompt-echo phrases at the ASR layer
        if any(phrase in text_lower for phrase in DROP_PHRASES):
            continue

        if any(snippet in text_lower for snippet in GUIDING_PROMPT_SNIPPETS):
            # skip adding it to transcripts; treat as silence
            text = SILENCE_ECHO_MARKER
            continue
        
        # Filter out likely false positives
        if is_likely_false_positive(text):
            continue

        with text_lock:
            if source == "system":
                system_text += text + "\n"
            elif source == "mic":
                mic_text += text + "\n"
        _log_entry(source, text)


# =========================
# FLASK WEB APP
# =========================

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ConvoMapper</title>
  <link rel="icon" type="image/x-icon" href="/assets/favicon.ico">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/assets/favicon-16x16.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0f1a;
      --bg-secondary: #0f1629;
      --bg-card: #141b2d;
      --bg-card-hover: #1a2340;
      --border: #1e293b;
      --border-accent: #2563eb;
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent: #3b82f6;
      --accent-soft: rgba(59, 130, 246, 0.15);
      --success: #22c55e;
      --warning: #f59e0b;
      --error: #ef4444;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Outfit', system-ui, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.5;
    }
    
    /* Header */
    header {
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 0.875rem 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    
    .logo {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .logo-icon {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      object-fit: contain;
    }
    
    .logo h1 {
      font-size: 1.125rem;
      font-weight: 600;
      letter-spacing: -0.025em;
    }
    
    .status-group {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    
    .status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      color: var(--text-secondary);
    }
    
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 2s infinite;
    }
    
    .status-dot.offline { background: var(--text-muted); animation: none; }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    /* Tabs */
    .tabs {
      display: flex;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 0 1.5rem;
      gap: 0.25rem;
    }
    
    .tab {
      padding: 0.875rem 1.25rem;
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      background: none;
      border-top: none;
      border-left: none;
      border-right: none;
    }
    
    .tab:hover { color: var(--text-secondary); }
    .tab.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }
    
    /* Main content */
    main {
      padding: 1.25rem 1.5rem;
      max-width: 1600px;
      margin: 0 auto;
    }
    
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    /* Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    
    .card-header {
      padding: 0.625rem 0.75rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .card-title {
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--text-primary);
    }
    
    .badge {
      font-size: 0.7rem;
      font-weight: 500;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    
    .badge-live {
      background: rgba(34, 197, 94, 0.15);
      color: var(--success);
    }
    
    .badge-mic {
      background: var(--accent-soft);
      color: var(--accent);
    }
    
    .card-body {
      padding: 0.5rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      line-height: 1.5;
      color: var(--text-secondary);
      max-height: 400px;
      overflow-y: auto;
    }
    
    .card-body.transcript { white-space: pre-wrap; }
    
    .card-body::-webkit-scrollbar { width: 6px; }
    .card-body::-webkit-scrollbar-track { background: transparent; }
    .card-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    
    .source-badges {
      display: flex;
      gap: 0.5rem;
      font-size: 0.7rem;
    }
    
    .source-badge {
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 500;
    }
    
    .source-badge.active {
      background: rgba(34, 197, 94, 0.15);
      color: var(--success);
    }
    
    .source-badge.inactive {
      background: rgba(100, 116, 139, 0.15);
      color: var(--text-muted);
    }
    
    /* Toggle switch */
    .toggle-switch {
      position: relative;
      width: 40px;
      height: 22px;
      background: var(--border);
      border-radius: 11px;
      cursor: pointer;
      transition: background 0.2s;
    }
    
    .toggle-switch.active {
      background: var(--success);
    }
    
    .toggle-switch::after {
      content: '';
      position: absolute;
      top: 2px;
      left: 2px;
      width: 18px;
      height: 18px;
      background: white;
      border-radius: 50%;
      transition: transform 0.2s;
    }
    
    .toggle-switch.active::after {
      transform: translateX(18px);
    }
    
    .controls-bar {
      display: flex;
      align-items: center;
      gap: 1.5rem;
      padding: 0.75rem 1.5rem;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
    }
    
    .control-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      color: var(--text-secondary);
    }
    
    .control-label {
      font-weight: 500;
    }
    
    .btn-record {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1.25rem;
      font-weight: 500;
      border-radius: 8px;
      transition: all 0.2s;
    }
    
    .btn-record.stopped {
      background: var(--success);
      color: white;
    }
    
    .btn-record.stopped:hover {
      background: #16a34a;
    }
    
    .btn-record.recording {
      background: var(--error);
      color: white;
      animation: pulse-record 1.5s infinite;
    }
    
    .btn-record.recording:hover {
      background: #dc2626;
    }
    
    @keyframes pulse-record {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }
    
    .record-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }
    
    /* Live tab layout */
    .live-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      height: calc(100vh - 180px);
    }
    
    .live-grid .card { display: flex; flex-direction: column; }
    .live-grid .card-body { flex: 1; max-height: none; }
    
    /* Insights tab */
    .insights-grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 0.75rem;
    }
    
    .insights-main { display: flex; flex-direction: column; gap: 0.75rem; }
    .insights-sidebar { display: flex; flex-direction: column; gap: 0.75rem; }
    
    /* Summary items */
    .summary-item {
      padding: 0.35rem 0.5rem;
      border-bottom: 1px solid var(--border);
      transition: background 0.2s;
    }
    
    .summary-item:last-child { border-bottom: none; }
    .summary-item:hover { background: var(--bg-card-hover); }
    
    .summary-time {
      font-size: 0.65rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      margin-bottom: 0;
    }
    
    .summary-title {
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-primary);
      margin-bottom: 0.1rem;
    }
    
    .summary-text {
      font-size: 0.75rem;
      color: var(--text-secondary);
      line-height: 1.35;
      margin: 0;
    }
    
    /* LTM bullets */
    .ltm-list {
      list-style: none;
      padding: 0.5rem 0.75rem;
    }
    
    .ltm-item {
      padding: 0.4rem 0;
      font-size: 0.75rem;
      color: var(--text-secondary);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
      line-height: 1.4;
    }
    
    .ltm-item:last-child { border-bottom: none; }
    
    .ltm-bullet {
      width: 5px;
      height: 5px;
      background: var(--accent);
      border-radius: 50%;
      margin-top: 0.4rem;
      flex-shrink: 0;
    }
    
    /* Interjection cards */
    .interject-item {
      padding: 0.35rem 0.5rem;
      border-bottom: 1px solid var(--border);
    }
    
    .interject-item:last-child { border-bottom: none; }
    
    .interject-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.1rem;
    }
    
    .interject-type {
      font-size: 0.7rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--warning);
    }
    
    .interject-conf {
      font-size: 0.65rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }
    
    .interject-msg {
      font-size: 0.8rem;
      color: var(--text-primary);
      line-height: 1.35;
      font-style: italic;
      margin: 0;
    }
    
    /* Report tab */
    .report-section {
      margin-bottom: 1.5rem;
    }
    
    .report-section h3 {
      font-size: 0.9rem;
      font-weight: 500;
      color: var(--text-primary);
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .report-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    
    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
      text-align: center;
    }
    
    .stat-value {
      font-size: 2rem;
      font-weight: 600;
      color: var(--accent);
      line-height: 1;
    }
    
    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 0.375rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    
    /* Timeline */
    .timeline {
      padding: 1rem;
    }
    
    .timeline-item {
      border-left: 2px solid var(--border);
      margin-left: 1.5rem;
      padding-left: 1.5rem;
      padding-bottom: 1rem;
      position: relative;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .timeline-item:hover {
      border-left-color: var(--accent);
    }
    
    .timeline-item:last-child {
      border-left-color: transparent;
      padding-bottom: 0;
    }
    
    .timeline-item::before {
      content: '';
      position: absolute;
      left: -7px;
      top: 0;
      width: 12px;
      height: 12px;
      background: var(--bg-primary);
      border: 2px solid var(--accent);
      border-radius: 50%;
      transition: all 0.2s ease;
    }
    
    .timeline-item:hover::before {
      background: var(--accent);
    }
    
    .timeline-item.expanded::before {
      background: var(--accent);
    }
    
    .timeline-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .timeline-time {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      color: var(--text-muted);
      background: var(--bg-secondary);
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
    }
    
    .timeline-title {
      font-size: 0.9rem;
      font-weight: 500;
      color: var(--text-primary);
    }
    
    .timeline-badge {
      font-size: 0.65rem;
      color: var(--text-muted);
      background: var(--bg-tertiary);
      padding: 0.1rem 0.4rem;
      border-radius: 3px;
      margin-left: auto;
    }
    
    .timeline-summary {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.3s ease, padding 0.3s ease;
      font-size: 0.8rem;
      color: var(--text-secondary);
      line-height: 1.6;
      background: var(--bg-tertiary);
      border-radius: 6px;
      margin-top: 0;
    }
    
    .timeline-item.expanded .timeline-summary {
      max-height: 500px;
      padding: 0.75rem;
      margin-top: 0.75rem;
    }
    
    .timeline-expand-icon {
      font-size: 0.7rem;
      color: var(--text-muted);
      transition: transform 0.2s ease;
    }
    
    .timeline-item.expanded .timeline-expand-icon {
      transform: rotate(90deg);
    }
    
    /* Buttons */
    .btn {
      font-family: 'Outfit', sans-serif;
      font-size: 0.8rem;
      font-weight: 500;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
      border: none;
    }
    
    .btn-primary {
      background: var(--accent);
      color: white;
    }
    
    .btn-primary:hover { background: #2563eb; }
    
    .btn-ghost {
      background: transparent;
      color: var(--text-secondary);
      border: 1px solid var(--border);
    }
    
    .btn-ghost:hover {
      background: var(--bg-card);
      color: var(--text-primary);
    }
    
    /* Empty states */
    .empty-state {
      text-align: center;
      padding: 1.5rem 1rem;
      color: var(--text-muted);
    }
    
    .empty-state-icon {
      font-size: 1.5rem;
      margin-bottom: 0.5rem;
      opacity: 0.5;
    }
    
    .empty-state-text {
      font-size: 0.8rem;
    }
    
    /* Responsive */
    @media (max-width: 900px) {
      .live-grid, .insights-grid {
        grid-template-columns: 1fr;
      }
      .report-stats {
        grid-template-columns: 1fr;
      }
    }
    
    /* Therapy Toast Notifications */
    .toast-container {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      max-width: 360px;
    }
    
    .toast {
      background: var(--bg-card);
      border: 1px solid var(--warning);
      border-left: 3px solid var(--warning);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      animation: slideIn 0.3s ease;
      cursor: pointer;
      transition: opacity 0.3s, transform 0.3s;
    }
    
    .toast:hover { transform: translateX(-4px); }
    .toast.hiding { opacity: 0; transform: translateX(100%); }
    
    @keyframes slideIn {
      from { opacity: 0; transform: translateX(100%); }
      to { opacity: 1; transform: translateX(0); }
    }
    
    .toast-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.25rem;
    }
    
    .toast-type {
      font-size: 0.65rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--warning);
    }
    
    .toast-close {
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 1rem;
      padding: 0;
      line-height: 1;
    }
    
    .toast-message {
      font-size: 0.8rem;
      color: var(--text-primary);
      line-height: 1.4;
    }
    
    .toast-conf {
      font-size: 0.6rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }
    
    /* Therapy toggle special styling */
    .therapy-toggle {
      margin-left: auto;
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(168, 85, 247, 0.05));
      padding: 0.5rem 1rem;
      border-radius: 20px;
      border: 1px solid rgba(139, 92, 246, 0.2);
      transition: all 0.3s ease;
      gap: 0.6rem;
    }
    
    .therapy-toggle:hover {
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(168, 85, 247, 0.1));
      border-color: rgba(139, 92, 246, 0.4);
    }
    
    .therapy-toggle.active { 
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.15));
      border-color: rgba(139, 92, 246, 0.5);
      box-shadow: 0 0 12px rgba(139, 92, 246, 0.2);
    }
    
    .therapy-toggle .control-label { 
      color: rgb(167, 139, 250);
      font-weight: 500;
      font-size: 0.8rem;
      letter-spacing: 0.02em;
    }
    
    /* History */
    .history-container {
      padding: 1rem;
    }
    
    .history-date-group {
      margin-bottom: 0.5rem;
    }
    
    .history-date-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1rem;
      background: var(--bg-secondary);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      border: 1px solid var(--border);
    }
    
    .history-date-header:hover {
      background: var(--bg-tertiary);
      border-color: var(--accent);
    }
    
    .history-date-header.expanded {
      border-radius: 8px 8px 0 0;
      border-bottom: none;
      background: var(--bg-tertiary);
    }
    
    .history-date-left {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .history-date-icon {
      font-size: 1.2rem;
    }
    
    .history-date-text {
      font-weight: 500;
      color: var(--text-primary);
    }
    
    .history-date-label {
      font-size: 0.7rem;
      color: var(--text-muted);
      background: var(--bg-primary);
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      margin-left: 0.5rem;
    }
    
    .history-date-right {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .history-session-count {
      font-size: 0.75rem;
      color: var(--text-muted);
      background: var(--bg-primary);
      padding: 0.25rem 0.6rem;
      border-radius: 12px;
    }
    
    .history-expand-icon {
      font-size: 0.8rem;
      color: var(--text-muted);
      transition: transform 0.2s ease;
    }
    
    .history-date-header.expanded .history-expand-icon {
      transform: rotate(90deg);
    }
    
    .history-sessions {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.4s ease;
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-top: none;
      border-radius: 0 0 8px 8px;
    }
    
    .history-date-group.expanded .history-sessions {
      max-height: 5000px;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <img src="/assets/android-chrome-192x192.png" alt="ConvoMapper" class="logo-icon">
      <h1>ConvoMapper</h1>
    </div>
    <div class="status-group">
      <div class="status">
        <div class="status-dot" id="statusDot"></div>
        <span id="statusText">Connecting...</span>
      </div>
      <button class="btn btn-ghost" onclick="clearTranscripts()">Clear</button>
    </div>
  </header>
  
  <nav class="tabs">
    <button class="tab active" data-tab="live">Live</button>
    <button class="tab" data-tab="timeline">Timeline</button>
    <button class="tab" data-tab="insights">Insights</button>
    <button class="tab" data-tab="report">Report</button>
    <button class="tab" data-tab="history">History</button>
  </nav>
  
  <div class="controls-bar">
    <button id="recordBtn" class="btn btn-record stopped" onclick="toggleRecording()">
      <span class="record-dot"></span>
      <span id="recordBtnText">Start Recording</span>
    </button>
    
    <div class="control-group">
      <span class="control-label">System Audio</span>
      <div id="toggleSystem" class="toggle-switch active" onclick="toggleSystem()"></div>
      </div>
    
    <div class="control-group">
      <span class="control-label">Microphone</span>
      <div id="toggleMic" class="toggle-switch active" onclick="toggleMic()"></div>
    </div>
    
    <div class="control-group therapy-toggle" id="therapyToggleGroup">
      <span class="control-label">Therapy Mode</span>
      <div id="toggleTherapy" class="toggle-switch" onclick="toggleTherapy()"></div>
      </div>
  </div>
  
  <div class="toast-container" id="toastContainer"></div>
  
  <main>
    <!-- LIVE TAB -->
    <div id="live" class="tab-content active">
      <div class="live-grid">
        <div class="card">
          <div class="card-header">
            <span class="card-title">System Audio</span>
            <span class="badge badge-live">Live</span>
      </div>
          <div id="systemText" class="card-body transcript">Waiting for audio...</div>
    </div>
        <div class="card">
          <div class="card-header">
            <span class="card-title">Microphone</span>
            <span class="badge badge-mic">Mic</span>
      </div>
          <div id="micText" class="card-body transcript">Waiting for audio...</div>
        </div>
    </div>
  </div>
    
    <!-- INSIGHTS TAB -->
    <div id="insights" class="tab-content">
      <div class="insights-grid">
        <div class="insights-main">
          <div class="card">
            <div class="card-header">
              <span class="card-title">Recent Summaries</span>
              <button class="btn btn-ghost" onclick="fetchInsights()">Refresh</button>
            </div>
            <div id="summariesList" class="card-body" style="padding: 0; font-family: inherit;">
              <div class="empty-state">
                <div class="empty-state-icon">📝</div>
                <div class="empty-state-text">No summaries yet. Start a conversation!</div>
              </div>
            </div>
          </div>
          
          <div class="card">
            <div class="card-header">
              <span class="card-title">AI Interjections</span>
            </div>
            <div id="interjectionsList" class="card-body" style="padding: 0; font-family: inherit;">
              <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <div class="empty-state-text">No interjections yet</div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="insights-sidebar">
          <div class="card">
            <div class="card-header">
              <span class="card-title">Long-Term Patterns</span>
            </div>
            <ul id="ltmList" class="ltm-list">
              <div class="empty-state">
                <div class="empty-state-icon">🧠</div>
                <div class="empty-state-text">Patterns emerge over time</div>
              </div>
            </ul>
          </div>
        </div>
      </div>
    </div>
    
    <!-- TIMELINE TAB -->
    <div id="timeline" class="tab-content">
      <div class="card" style="height: calc(100vh - 180px);">
        <div class="card-header">
          <span class="card-title">Conversation Timeline</span>
          <button class="btn btn-ghost" onclick="fetchTimeline()">Refresh</button>
        </div>
        <div id="timelineView" class="timeline" style="overflow-y: auto; max-height: calc(100vh - 260px);">
          <div class="empty-state">
            <div class="empty-state-icon">🕐</div>
            <div class="empty-state-text">Topics will appear here as the conversation progresses</div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- REPORT TAB -->
    <div id="report" class="tab-content">
      <div class="report-stats" id="reportStats">
        <div class="stat-card">
          <div class="stat-value" id="statChunks">0</div>
          <div class="stat-label">Topics Discussed</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="statInterjections">0</div>
          <div class="stat-label">AI Interjections</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="statPatterns">0</div>
          <div class="stat-label">Patterns Found</div>
        </div>
      </div>
      
      <div class="card report-section">
        <div class="card-header">
          <span class="card-title">Session Summary</span>
          <button class="btn btn-primary" onclick="generateReport()">Generate Report</button>
        </div>
        <div id="sessionSummary" class="card-body" style="font-family: inherit;">
          <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-text">Click "Generate Report" to see the session summary</div>
          </div>
        </div>
      </div>
      
      <div class="card report-section">
        <div class="card-header">
          <span class="card-title">Key Patterns</span>
        </div>
        <ul id="reportPatterns" class="ltm-list">
          <div class="empty-state">
            <div class="empty-state-text">Report data will appear here</div>
          </div>
        </ul>
      </div>
    </div>
    
    <!-- HISTORY TAB -->
    <div id="history" class="tab-content">
      <div class="card" style="height: calc(100vh - 180px);">
        <div class="card-header">
          <span class="card-title">Session History</span>
          <button class="btn btn-ghost" onclick="fetchHistory()">Refresh</button>
        </div>
        <div id="historyView" class="history-container" style="overflow-y: auto; max-height: calc(100vh - 260px);">
          <div class="empty-state">
            <div class="empty-state-icon">📚</div>
            <div class="empty-state-text">Loading session history...</div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
        
        // Auto-fetch data when switching tabs
        if (tab.dataset.tab === 'insights') fetchInsights();
        if (tab.dataset.tab === 'timeline') fetchTimeline();
        if (tab.dataset.tab === 'history') fetchHistory();
      });
    });
    
    // Live transcripts
    async function fetchTranscripts() {
      try {
        const res = await fetch('/transcripts');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        const sysEl = document.getElementById('systemText');
        const micEl = document.getElementById('micText');
        
        sysEl.textContent = data.system || 'Waiting for audio...';
        micEl.textContent = data.mic || 'Waiting for audio...';
        
        // Auto-scroll to bottom
        sysEl.scrollTop = sysEl.scrollHeight;
        micEl.scrollTop = micEl.scrollHeight;
        
        document.getElementById('statusDot').classList.remove('offline');
        document.getElementById('statusText').textContent = 'Live';
      } catch (e) {
        document.getElementById('statusDot').classList.add('offline');
        document.getElementById('statusText').textContent = 'Disconnected';
        console.error(e);
      }
    }

    async function clearTranscripts() {
      document.getElementById('systemText').textContent = 'Waiting for audio...';
      document.getElementById('micText').textContent = 'Waiting for audio...';
      try {
        await fetch('/clear', { method: 'POST' });
      } catch (e) {
        console.error('Failed to clear:', e);
      }
    }
    
    // Insights
    async function fetchInsights() {
      try {
        const res = await fetch('/api/insights');
        const data = await res.json();
        
        // Render summaries
        const summariesEl = document.getElementById('summariesList');
        if (data.chunks && data.chunks.length > 0) {
          summariesEl.innerHTML = data.chunks.map(c => 
            `<div class="summary-item"><div class="summary-time">${new Date(c.ts * 1000).toLocaleTimeString()}</div><div class="summary-title">${escapeHtml(c.title)}</div><div class="summary-text">${escapeHtml(c.summary)}</div></div>`
          ).join('');
        }
        
        // Render interjections
        const interjectEl = document.getElementById('interjectionsList');
        if (data.interjections && data.interjections.length > 0) {
          interjectEl.innerHTML = data.interjections.map(i => 
            `<div class="interject-item"><div class="interject-type">${escapeHtml(i.type.replace(/_/g, ' '))}</div><div class="interject-msg">"${escapeHtml(i.message)}"</div></div>`
          ).join('');
        }
        
        // Render LTM
        const ltmEl = document.getElementById('ltmList');
        if (data.ltm && data.ltm.length > 0) {
          ltmEl.innerHTML = data.ltm.map(b => 
            `<li class="ltm-item"><span class="ltm-bullet"></span><span>${escapeHtml(b)}</span></li>`
          ).join('');
        }
      } catch (e) {
        console.error('Failed to fetch insights:', e);
      }
    }
    
    // Timeline
    async function fetchTimeline() {
      try {
        const res = await fetch('/api/timeline');
        const data = await res.json();
        
        const timelineEl = document.getElementById('timelineView');
        if (data.topics && data.topics.length > 0) {
          timelineEl.innerHTML = data.topics.map((t, idx) => `
            <div class="timeline-item" onclick="toggleTimelineItem(this)" data-idx="${idx}">
              <div class="timeline-header">
                <span class="timeline-time">${t.time}</span>
                <span class="timeline-title">${escapeHtml(t.title)}</span>
                ${t.chunk_count > 1 ? `<span class="timeline-badge">${t.chunk_count} parts</span>` : ''}
                <span class="timeline-expand-icon">▶</span>
              </div>
              <div class="timeline-summary">${escapeHtml(t.summary)}</div>
            </div>
          `).join('');
        } else {
          timelineEl.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🕐</div><div class="empty-state-text">No topics recorded yet</div></div>';
        }
      } catch (e) {
        console.error('Failed to fetch timeline:', e);
      }
    }
    
    function toggleTimelineItem(el) {
      el.classList.toggle('expanded');
    }
    
    // History
    async function fetchHistory() {
      try {
        const res = await fetch('/api/history');
        const data = await res.json();
        const historyEl = document.getElementById('historyView');
        
        if (data.dates && data.dates.length > 0) {
          historyEl.innerHTML = data.dates.map((dateGroup, dateIdx) => `
            <div class="history-date-group" data-date="${dateGroup.date}">
              <div class="history-date-header" onclick="toggleHistoryDate(this)">
                <div class="history-date-left">
                  <span class="history-date-text">${escapeHtml(dateGroup.date_display)}</span>
                  ${dateGroup.date_label ? `<span class="history-date-label">${escapeHtml(dateGroup.date_label)}</span>` : ''}
                </div>
                <div class="history-date-right">
                  <span class="history-session-count">${dateGroup.topic_count} topics</span>
                  <span class="history-expand-icon">▶</span>
                </div>
              </div>
              <div class="history-sessions">
                <div class="timeline" style="padding: 1rem;">
                  ${dateGroup.topics.map((t, idx) => `
                    <div class="timeline-item" onclick="toggleTimelineItem(this)" data-idx="${idx}">
                      <div class="timeline-header">
                        <span class="timeline-time">${t.time}</span>
                        <span class="timeline-title">${escapeHtml(t.title)}</span>
                        ${t.parts > 1 ? `<span class="timeline-badge">${t.parts} parts</span>` : ''}
                        <span class="timeline-expand-icon">▶</span>
                      </div>
                      <div class="timeline-summary">${escapeHtml(t.summary)}</div>
                    </div>
                  `).join('')}
                </div>
              </div>
            </div>
          `).join('');
        } else {
          historyEl.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📚</div><div class="empty-state-text">No session history found</div></div>';
        }
      } catch (e) {
        console.error('Failed to fetch history:', e);
        document.getElementById('historyView').innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div><div class="empty-state-text">Failed to load history</div></div>';
      }
    }
    
    function toggleHistoryDate(el) {
      const group = el.closest('.history-date-group');
      group.classList.toggle('expanded');
      el.classList.toggle('expanded');
    }
    
    // Report
    async function generateReport() {
      try {
        const res = await fetch('/api/report');
        const data = await res.json();
        
        // Stats
        document.getElementById('statChunks').textContent = data.duration_chunks || 0;
        document.getElementById('statInterjections').textContent = data.total_interjections || 0;
        document.getElementById('statPatterns').textContent = (data.key_patterns || []).length;
        
        // Session summary
        const summaryEl = document.getElementById('sessionSummary');
        if (data.summaries && data.summaries.length > 0) {
          summaryEl.innerHTML = data.summaries.map(s => `<p style="margin-bottom: 0.75rem; color: var(--text-secondary);">${escapeHtml(s)}</p>`).join('');
        } else {
          summaryEl.innerHTML = '<div class="empty-state"><div class="empty-state-text">No summaries available yet</div></div>';
        }
        
        // Patterns
        const patternsEl = document.getElementById('reportPatterns');
        if (data.key_patterns && data.key_patterns.length > 0) {
          patternsEl.innerHTML = data.key_patterns.map(p => `
            <li class="ltm-item">
              <span class="ltm-bullet"></span>
              <span>${escapeHtml(p)}</span>
            </li>
          `).join('');
        }
      } catch (e) {
        console.error('Failed to generate report:', e);
      }
    }
    
    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text || '';
      return div.innerHTML;
    }
    
    // Recording controls
    let isRecording = false;
    let systemEnabled = true;
    let micEnabled = true;
    let therapyEnabled = false;
    let lastInterjectionTs = 0;
    
    async function toggleRecording() {
      const btn = document.getElementById('recordBtn');
      const btnText = document.getElementById('recordBtnText');
      
      if (isRecording) {
        await fetch('/api/recording/stop', { method: 'POST' });
        btn.classList.remove('recording');
        btn.classList.add('stopped');
        btnText.textContent = 'Start Recording';
        isRecording = false;
      } else {
        await fetch('/api/recording/start', { method: 'POST' });
        btn.classList.remove('stopped');
        btn.classList.add('recording');
        btnText.textContent = 'Stop Recording';
        isRecording = true;
      }
    }
    
    async function toggleSystem() {
      const res = await fetch('/api/toggle/system', { method: 'POST' });
      const data = await res.json();
      systemEnabled = data.system_enabled;
      const toggle = document.getElementById('toggleSystem');
      toggle.classList.toggle('active', systemEnabled);
    }
    
    async function toggleMic() {
      const res = await fetch('/api/toggle/mic', { method: 'POST' });
      const data = await res.json();
      micEnabled = data.mic_enabled;
      const toggle = document.getElementById('toggleMic');
      toggle.classList.toggle('active', micEnabled);
    }
    
    async function toggleTherapy() {
      const res = await fetch('/api/toggle/therapy', { method: 'POST' });
      const data = await res.json();
      therapyEnabled = data.therapy_enabled;
      const toggle = document.getElementById('toggleTherapy');
      const group = document.getElementById('therapyToggleGroup');
      toggle.classList.toggle('active', therapyEnabled);
      group.classList.toggle('active', therapyEnabled);
    }
    
    // Toast notifications
    function showToast(type, message) {
      const container = document.getElementById('toastContainer');
      const toast = document.createElement('div');
      toast.className = 'toast';
      toast.innerHTML = `
        <div class="toast-header">
          <span class="toast-type">${escapeHtml(type.replace(/_/g, ' '))}</span>
          <button class="toast-close" onclick="dismissToast(this.parentElement.parentElement)">&times;</button>
        </div>
        <div class="toast-message">"${escapeHtml(message)}"</div>
      `;
      container.appendChild(toast);
      
      // Auto-dismiss after 12 seconds
      setTimeout(() => dismissToast(toast), 12000);
    }
    
    function dismissToast(toast) {
      if (!toast || !toast.parentElement) return;
      toast.classList.add('hiding');
      setTimeout(() => toast.remove(), 300);
    }
    
    // Poll for new interjections when therapy mode is on
    async function checkInterjections() {
      if (!therapyEnabled) return;
      
      try {
        const res = await fetch('/api/insights');
        const data = await res.json();
        
        if (data.interjections && data.interjections.length > 0) {
          // Show new interjections as toasts
          for (const i of data.interjections) {
            if (i.ts > lastInterjectionTs && !i.skipped) {
              showToast(i.type, i.message);
              lastInterjectionTs = i.ts;
            }
          }
        }
      } catch (e) {
        console.error('Failed to check interjections:', e);
      }
    }
    
    // Fetch initial recording status
    async function fetchRecordingStatus() {
      try {
        const res = await fetch('/api/recording/status');
        const data = await res.json();
        isRecording = data.recording;
        systemEnabled = data.system_enabled;
        micEnabled = data.mic_enabled;
        therapyEnabled = data.therapy_enabled;
        
        const btn = document.getElementById('recordBtn');
        const btnText = document.getElementById('recordBtnText');
        if (isRecording) {
          btn.classList.remove('stopped');
          btn.classList.add('recording');
          btnText.textContent = 'Stop Recording';
        }
        
        document.getElementById('toggleSystem').classList.toggle('active', systemEnabled);
        document.getElementById('toggleMic').classList.toggle('active', micEnabled);
        document.getElementById('toggleTherapy').classList.toggle('active', therapyEnabled);
        document.getElementById('therapyToggleGroup').classList.toggle('active', therapyEnabled);
      } catch (e) {
        console.error('Failed to fetch recording status:', e);
      }
    }
    
    // Polling
    setInterval(fetchTranscripts, 300);
    setInterval(checkInterjections, 5000);
    fetchTranscripts();
    fetchRecordingStatus();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/transcripts")
def transcripts():
    with text_lock:
        return jsonify({
            "system": system_text,
            "mic": mic_text,
        })

@app.route("/clear", methods=["POST"])
def clear_all():
    global system_text, mic_text
    with text_lock:
        system_text = ""
        mic_text = ""
    return jsonify({"status": "cleared"})


@app.route("/api/recording/start", methods=["POST"])
def start_recording():
    global recording_enabled
    with state_lock:
        recording_enabled = True
    return jsonify({"recording": True})


@app.route("/api/recording/stop", methods=["POST"])
def stop_recording():
    global recording_enabled
    with state_lock:
        recording_enabled = False
    return jsonify({"recording": False})


@app.route("/api/recording/status")
def get_recording_status():
    with state_lock:
        return jsonify({
            "recording": recording_enabled,
            "system_enabled": system_capture_enabled,
            "mic_enabled": mic_capture_enabled,
            "therapy_enabled": therapy_mode_enabled
        })


@app.route("/api/toggle/system", methods=["POST"])
def toggle_system():
    global system_capture_enabled
    with state_lock:
        system_capture_enabled = not system_capture_enabled
        return jsonify({"system_enabled": system_capture_enabled})


@app.route("/api/toggle/mic", methods=["POST"])
def toggle_mic():
    global mic_capture_enabled
    with state_lock:
        mic_capture_enabled = not mic_capture_enabled
        return jsonify({"mic_enabled": mic_capture_enabled})


@app.route("/api/toggle/therapy", methods=["POST"])
def toggle_therapy():
    global therapy_mode_enabled
    with state_lock:
        therapy_mode_enabled = not therapy_mode_enabled
        return jsonify({"therapy_enabled": therapy_mode_enabled})


@app.route("/api/therapy/status")
def get_therapy_status():
    with state_lock:
        return jsonify({"therapy_enabled": therapy_mode_enabled})


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    """Serve static files from assets folder."""
    return send_from_directory("assets", filename)


def _read_memory_log():
    """
    Read today's memory log file and parse entries.
    Returns chunks, interjections, and ltm bullets.
    """
    date_str = datetime.date.today().isoformat()
    path = f"data/memory-{date_str}.jsonl"
    
    chunks = []
    interjections = []
    ltm_bullets = []
    
    if not os.path.exists(path):
        return {"chunks": chunks, "interjections": interjections, "ltm": ltm_bullets}
    
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
            # Split by double newlines (each entry ends with \n\n)
            entries = content.strip().split("\n\n")
            for entry_str in entries:
                if not entry_str.strip():
                    continue
                try:
                    entry = json.loads(entry_str.strip())
                    entry_type = entry.get("type", "")
                    
                    if entry_type == "chunk":
                        chunks.append({
                            "ts": entry.get("ts", 0),
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", ""),
                        })
                    elif entry_type == "interjection":
                        decision = entry.get("decision", {})
                        if not entry.get("skipped", False):
                            interjections.append({
                                "ts": entry.get("ts", 0),
                                "type": decision.get("interventionType", ""),
                                "confidence": decision.get("confidence", 0),
                                "message": decision.get("candidateMessage", ""),
                                "reasons": decision.get("reasons", []),
                            })
                    elif entry_type == "ltm":
                        ltm_bullets = entry.get("bullets", [])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading memory log: {e}")
    
    return {"chunks": chunks, "interjections": interjections, "ltm": ltm_bullets}


@app.route("/api/insights")
def get_insights():
    """Return memory insights from the therapy loop."""
    data = _read_memory_log()
    # Sort chunks by timestamp descending (most recent first)
    data["chunks"] = sorted(data["chunks"], key=lambda x: x["ts"], reverse=True)[:20]
    data["interjections"] = sorted(data["interjections"], key=lambda x: x["ts"], reverse=True)[:10]
    return jsonify(data)


@app.route("/api/timeline")
def get_timeline():
    """Get grouped topic timeline - combines consecutive same-topic chunks."""
    data = _read_memory_log()
    chunks = sorted(data["chunks"], key=lambda x: x["ts"])
    
    # Group consecutive chunks with same title
    grouped = []
    for chunk in chunks:
        title = chunk.get("title", "").strip()
        summary = chunk.get("summary", "").strip()
        ts = chunk.get("ts", 0)
        
        # Skip pending or empty
        if not title or title == "(pending)" or title.lower() == "no speech":
            continue
        if not summary or summary.lower() == "no speech":
            continue
            
        # Check if same topic as previous
        if grouped and grouped[-1]["title"] == title:
            # Append to existing topic
            grouped[-1]["summaries"].append(summary)
            grouped[-1]["end_ts"] = ts
        else:
            # New topic
            grouped.append({
                "title": title,
                "start_ts": ts,
                "end_ts": ts,
                "summaries": [summary]
            })
    
    # Format for frontend (reversed - latest first)
    timeline = []
    for g in reversed(grouped):
        condensed = _condense_summaries(g["summaries"])
        timeline.append({
            "time": time.strftime("%H:%M", time.localtime(g["start_ts"])),
            "title": g["title"],
            "summary": condensed,
            "chunk_count": len(g["summaries"])
        })
    
    return jsonify({"topics": timeline})


@app.route("/api/report")
def generate_report():
    """Generate a conversation report based on all available data."""
    data = _read_memory_log()
    
    # Get transcript text
    with text_lock:
        transcript = system_text + "\n" + mic_text
    
    # Group topics for timeline
    chunks = sorted(data["chunks"], key=lambda x: x["ts"])
    grouped = []
    for chunk in chunks:
        title = chunk.get("title", "").strip()
        if not title or title == "(pending)":
            continue
        if grouped and grouped[-1]["title"] == title:
            grouped[-1]["summaries"].append(chunk.get("summary", ""))
            grouped[-1]["end_ts"] = chunk["ts"]
        else:
            grouped.append({
                "title": title,
                "start_ts": chunk["ts"],
                "end_ts": chunk["ts"],
                "summaries": [chunk.get("summary", "")]
            })
    
    # Build report
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_chunks": len(data["chunks"]),
        "total_interjections": len(data["interjections"]),
        "key_patterns": data["ltm"],
        "topic_timeline": [
            {
                "time": time.strftime("%H:%M", time.localtime(g["start_ts"])),
                "title": g["title"],
                "summary": _condense_summaries(g["summaries"])
            }
            for g in grouped
        ],
        "summaries": [_condense_summaries(g["summaries"]) for g in grouped],
        "transcript_preview": transcript[:2000] if transcript else "No transcript available",
    }
    return jsonify(report)


@app.route("/api/history")
def get_history():
    """Get all session history from the data folder, grouped by date with timeline format."""
    import glob
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        return jsonify({"dates": []})
    
    # Find all memory files
    memory_files = sorted(glob.glob(os.path.join(data_dir, "memory-*.jsonl")), reverse=True)
    
    dates = []
    
    for memory_file in memory_files:
        basename = os.path.basename(memory_file)
        # Extract date from filename: memory-2025-12-04.jsonl -> 2025-12-04
        date_str = basename.replace("memory-", "").replace(".jsonl", "")
        
        # Parse the memory file
        chunks = []
        
        try:
            with open(memory_file, "r", encoding="utf-8") as fh:
                content = fh.read()
                entries = content.strip().split("\n\n")
                
                for entry_str in entries:
                    if not entry_str.strip():
                        continue
                    try:
                        entry = json.loads(entry_str.strip())
                        if entry.get("type") == "chunk":
                            chunks.append({
                                "ts": entry.get("ts", 0),
                                "title": entry.get("title", ""),
                                "summary": entry.get("summary", ""),
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {memory_file}: {e}")
            continue
        
        if not chunks:
            continue  # Skip empty dates
        
        # Group consecutive same-topic chunks (same as Timeline)
        chunks = sorted(chunks, key=lambda x: x["ts"], reverse=True)  # Latest first
        grouped = []
        for chunk in chunks:
            title = chunk.get("title", "").strip()
            if not title or title == "(pending)":
                continue
            if grouped and grouped[-1]["title"] == title:
                grouped[-1]["summaries"].append(chunk.get("summary", ""))
                grouped[-1]["start_ts"] = chunk["ts"]  # Since reversed, this extends backwards
            else:
                grouped.append({
                    "title": title,
                    "start_ts": chunk["ts"],
                    "end_ts": chunk["ts"],
                    "summaries": [chunk.get("summary", "")]
                })
        
        if not grouped:
            continue
        
        # Build timeline topics (same format as /api/timeline)
        topics = []
        for g in grouped:
            condensed = _condense_summaries(g["summaries"])
            topics.append({
                "time": time.strftime("%H:%M", time.localtime(g["start_ts"])),
                "title": g["title"],
                "summary": condensed,
                "parts": len(g["summaries"])
            })
        
        # Format date nicely
        try:
            from datetime import datetime as dt
            date_obj = dt.strptime(date_str, "%Y-%m-%d")
            date_display = date_obj.strftime("%A, %B %d, %Y")
            days_ago = (dt.now() - date_obj).days
            if days_ago == 0:
                date_label = "Today"
            elif days_ago == 1:
                date_label = "Yesterday"
            else:
                date_label = f"{days_ago} days ago"
        except:
            date_display = date_str
            date_label = ""
        
        dates.append({
            "date": date_str,
            "date_display": date_display,
            "date_label": date_label,
            "topic_count": len(topics),
            "topics": topics,
        })
    
    return jsonify({"dates": dates})


# =========================
# STARTUP
# =========================

def start_audio_and_model():
    print("Loading faster-whisper model...")

    # Try preferred device first, fall back gracefully
    model = None
    if WHISPER_DEVICE_PREFERENCE == "cuda":
        try:
            model = WhisperModel(
                MODEL_NAME,
                device="cuda",
                compute_type="float16",
            )
            print(f"Loaded model '{MODEL_NAME}' on CUDA (float16)")
        except Exception as e:
            print(f"CUDA failed ({e}); falling back to CPU int8")

    if model is None:
        model = WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
        print(f"Loaded model '{MODEL_NAME}' on CPU (int8)")

    print("Model loaded. Starting capture and transcription threads...")

    if ENABLE_SYSTEM_CAPTURE:
        threading.Thread(target=system_audio_loop, daemon=True).start()
        threading.Thread(
            target=chunker_worker,
            args=("system", system_q, transcribe_q, SAMPLE_RATE, SYSTEM_CHUNK_SECONDS),
            daemon=True
        ).start()
    else:
        print("System audio capture disabled by config.")

    if ENABLE_MIC_CAPTURE:
        threading.Thread(target=mic_audio_loop, daemon=True).start()
        threading.Thread(
            target=chunker_worker,
            args=("mic", mic_q, transcribe_q, SAMPLE_RATE, MIC_CHUNK_SECONDS),
            daemon=True
        ).start()
    else:
        print("Mic audio capture disabled by config.")

    threading.Thread(
        target=transcriber_worker,
        args=(model, transcribe_q, LANGUAGE),
        daemon=True
    ).start()


if __name__ == "__main__":
    print("Starting local transcriber web app on http://127.0.0.1:5001")
    start_audio_and_model()
    # Disable reloader so threads aren’t started twice
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

