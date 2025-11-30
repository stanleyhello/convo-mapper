import time
import queue
import threading

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel
from flask import Flask, jsonify, render_template_string


# =========================
# CONFIG
# =========================

SAMPLE_RATE = 16000          # target sample rate for STT
CHUNK_SECONDS = 5.0          # length of audio per STT call
MODEL_NAME = "small"         # tiny/base/small/medium/large-v3
LANGUAGE = "en"              # or None for auto

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

    with loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=1024) as rec:
        while True:
            data = rec.record(numframes=1024)  # (frames, channels)
            timestamp = time.time()
            system_q.put((timestamp, data.astype(np.float32)))


def mic_audio_loop():
    """
    Capture mic audio from the chosen mic.
    Runs in its own thread.
    """
    mic = get_mic()

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=1024) as rec:
        while True:
            data = rec.record(numframes=1024)  # (frames, channels)
            timestamp = time.time()
            mic_q.put((timestamp, data.astype(np.float32)))


# =========================
# CHUNKER WORKERS
# =========================

def chunker_worker(name, in_q, out_q, sample_rate, chunk_seconds):
    """
    Collect blocks from in_q, convert to mono, accumulate until
    chunk_seconds of audio, then push (name, chunk) to out_q.
    """
    samples_needed = int(sample_rate * chunk_seconds)
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
            chunk = buffer[:samples_needed]
            buffer = buffer[samples_needed:]
            out_q.put((name, chunk))


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

        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=True
        )

        text = "".join(seg.text for seg in segments).strip()
        if not text:
            continue
        stream=open("transcript.txt","w")
        stream.write(text+"\n")
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
      <button onclick="clearTranscripts()">Clear view</button>
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

    function clearTranscripts() {
      document.getElementById('system').textContent = '';
      document.getElementById('mic').textContent = '';
    }

    setInterval(fetchTranscripts, 1000);
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


# =========================
# STARTUP
# =========================

def start_audio_and_model():
    print("Loading faster-whisper model...")
    model = WhisperModel(
        MODEL_NAME,
        device="cpu",       # set to "cuda" if you have a GPU
        compute_type="int8" # good for CPU
    )
    print("Model loaded. Starting capture and transcription threads...")

    threading.Thread(target=system_audio_loop, daemon=True).start()
    threading.Thread(target=mic_audio_loop, daemon=True).start()

    threading.Thread(
        target=chunker_worker,
        args=("system", system_q, transcribe_q, SAMPLE_RATE, CHUNK_SECONDS),
        daemon=True
    ).start()

    threading.Thread(
        target=chunker_worker,
        args=("mic", mic_q, transcribe_q, SAMPLE_RATE, CHUNK_SECONDS),
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
