import json
import os
import time
import queue
import threading
import datetime

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
from flask import Flask, jsonify, render_template_string


# =========================
# CONFIG
# =========================

SAMPLE_RATE = 16000          # target sample rate for STT

# recorder buffer sizes (larger = fewer underruns). macOS CoreAudio limits blocksize <= 512.
REC_BLOCKSIZE = 480          # frames per read for both system and mic

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
# e.g. "Focusrite USB Audio" or "DELL S2721Q"
SPEAKER_NAME_FILTER = "BlackHole"   # e.g. "BlackHole 2ch" virtual loopback
MIC_NAME_FILTER = "Yeti"            # prefer Yeti; falls back to default mic

# Optional JSONL logging (rotated daily)
ENABLE_TRANSCRIPT_LOG = True
LOG_PATH_TEMPLATE = "transcript-{date}.jsonl"


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
    For the chosen speaker, get a loopback 'microphone' that captures
    what is being played to that speaker.
    """
    speaker = get_system_speaker()
    try:
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        return loopback
    except Exception as e:
        print(f"Loopback not available for '{speaker.name}': {e}")
        return None


def get_mic():
    """
    Pick a microphone based on MIC_NAME_FILTER, otherwise default mic.
    """
    mics = sc.all_microphones()
    if MIC_NAME_FILTER:
        for m in mics:
            if MIC_NAME_FILTER in m.name:
                print(f"Mic input : {m.name}")
                return m
        print(f"Mic input (filter '{MIC_NAME_FILTER}' not found) → using default mic")
    default = sc.default_microphone()
    print(f"Mic input : {default.name}")
    return default


# =========================
# AUDIO CAPTURE LOOPS
# =========================

def system_audio_loop():
    """
    Capture system audio (what you hear) via loopback microphone.
    Runs in its own thread.
    """
    loopback_mic = get_system_loopback_mic()
    if loopback_mic is None:
        print("Skipping system audio capture (no loopback device on macOS).")
        return

    with loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=REC_BLOCKSIZE) as rec:
        while True:
            data = rec.record(numframes=REC_BLOCKSIZE)  # (frames, channels)
            timestamp = time.time()
            system_q.put((timestamp, data.astype(np.float32)))


def mic_audio_loop():
    """
    Capture mic audio from the chosen mic.
    Runs in its own thread.
    """
    mic = get_mic()

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=REC_BLOCKSIZE) as rec:
        while True:
            data = rec.record(numframes=REC_BLOCKSIZE)  # (frames, channels)
            timestamp = time.time()
            mic_q.put((timestamp, data.astype(np.float32)))


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
  <title>Local Transcriber</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: #e5e7eb;
    }
    header {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid #1f2937;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 {
      font-size: 1.1rem;
      margin: 0;
    }
    header span {
      font-size: 0.85rem;
      color: #9ca3af;
    }
    .container {
      display: flex;
      gap: 1rem;
      padding: 1rem;
      height: calc(100vh - 64px);
      box-sizing: border-box;
    }
    .column {
      flex: 1;
      background: #020617;
      border-radius: 0.75rem;
      border: 1px solid #1f2937;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .column-header {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid #1f2937;
      font-size: 0.9rem;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .badge {
      font-size: 0.75rem;
      padding: 0.1rem 0.45rem;
      border-radius: 999px;
      border: 1px solid #4b5563;
      color: #9ca3af;
    }
    .content {
      padding: 0.75rem 1rem;
      overflow-y: auto;
      font-size: 0.9rem;
      line-height: 1.4;
      white-space: pre-wrap;
    }
    .controls {
      display: flex;
      gap: 0.5rem;
      align-items: center;
      font-size: 0.8rem;
    }
    button {
      background: #111827;
      color: #e5e7eb;
      border-radius: 999px;
      border: 1px solid #374151;
      padding: 0.3rem 0.75rem;
      font-size: 0.8rem;
      cursor: pointer;
    }
    button:hover {
      background: #1f2937;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Local Transcriber</h1>
      <span>Offline · System audio + Mic (Yeti)</span>
    </div>
    <div class="controls">
      <button type="button" onclick="clearTranscripts()">Clear view</button>
      <span id="status">Updating…</span>
    </div>
  </header>
  <div class="container">
    <div class="column">
      <div class="column-header">
        <span>System audio</span>
        <span class="badge">OUTPUT</span>
      </div>
      <div id="system" class="content"></div>
    </div>
    <div class="column">
      <div class="column-header">
        <span>Mic audio (Yeti)</span>
        <span class="badge">MIC</span>
      </div>
      <div id="mic" class="content"></div>
    </div>
  </div>

  <script>
    let lastUpdate = 0;

    async function fetchTranscripts() {
      try {
        const res = await fetch('/transcripts');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        document.getElementById('system').textContent = data.system || '';
        document.getElementById('mic').textContent = data.mic || '';

        lastUpdate = Date.now();
        document.getElementById('status').textContent = 'Live';
      } catch (e) {
        document.getElementById('status').textContent = 'Disconnected';
        console.error(e);
      }
    }

    async function clearTranscripts() {
      // Clear UI immediately
      document.getElementById('system').textContent = '';
      document.getElementById('mic').textContent = '';

      // Clear backend transcripts
      try {
        await fetch('/clear', { method: 'POST' });
      } catch (e) {
        console.error('Failed to clear backend:', e);
      }
    }

    // faster polling for lower perceived latency
    setInterval(fetchTranscripts, 250);
    window.onload = fetchTranscripts;
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
    print("Starting local transcriber web app on http://127.0.0.1:5000")
    start_audio_and_model()
    # Disable reloader so threads aren’t started twice
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
