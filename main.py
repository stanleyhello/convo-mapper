import time
import queue
import threading

import numpy as np

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

# recorder buffer sizes (larger = fewer underruns)
REC_BLOCKSIZE = 1024         # frames per read for both system and mic

# separate chunk sizes
SYSTEM_CHUNK_SECONDS = 3.0   # system audio chunks
MIC_CHUNK_SECONDS = 4.0      # mic audio chunks (more context for accuracy)

MODEL_NAME = "medium"        # "tiny"/"base" recommended for low latency
LANGUAGE = "en"              # or None for auto

# Preferred device for Whisper: "cuda" (GPU) or "cpu"
WHISPER_DEVICE_PREFERENCE = "cuda"

# Optional filters to force a specific device by name substring
# e.g. "Focusrite USB Audio" or "DELL S2721Q"
SPEAKER_NAME_FILTER = None          # e.g. "Speakers (Focusrite USB Audio)"
MIC_NAME_FILTER = "Yeti"            # prefer Yeti; falls back to default mic


# =========================
# QUEUES & STATE
# =========================

system_q = queue.Queue()
mic_q = queue.Queue()
transcribe_q = queue.Queue()

system_text = ""
mic_text = ""
text_lock = threading.Lock()


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
    loopback = sc.get_microphone(speaker.name, include_loopback=True)
    return loopback


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

def transcriber_worker(model, in_q, language=None):
    """
    Pull (source, chunk) from in_q and run faster-whisper locally.
    Append text to in-memory transcripts.
    """
    global system_text, mic_text

    while True:
        source, chunk = in_q.get()
        if chunk is None:
            break

        audio = chunk.astype(np.float32)

        if source == "mic":
            # Mic: prioritize quality/stability
            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=5,      # more search → better accuracy
                vad_filter=False  # avoid chopping speech on noisy mic
            )
        else:
            # System: keep fast, it’s already good
            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=True
            )

        text = "".join(seg.text for seg in segments).strip()
        if not text:
            continue

        with text_lock:
            if source == "system":
                system_text += text + "\n"
            elif source == "mic":
                mic_text += text + "\n"


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

    threading.Thread(target=system_audio_loop, daemon=True).start()
    threading.Thread(target=mic_audio_loop, daemon=True).start()

    threading.Thread(
        target=chunker_worker,
        args=("system", system_q, transcribe_q, SAMPLE_RATE, SYSTEM_CHUNK_SECONDS),
        daemon=True
    ).start()

    threading.Thread(
        target=chunker_worker,
        args=("mic", mic_q, transcribe_q, SAMPLE_RATE, MIC_CHUNK_SECONDS),
        daemon=True
    ).start()

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
