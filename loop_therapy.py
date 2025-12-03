import json
import os
import threading
import time
from collections import deque

from dotenv import load_dotenv
from openai import OpenAI

import main

# Load environment variables from a .env file if present
load_dotenv()

POLL_SECONDS = 1.0
SUMMARY_INTERVAL_SECONDS = 60.0
STM_WINDOW_SECONDS = 300  # keep last 5 minutes of raw text
MTM_MAX_LEN = 50          # keep last 30-50 summaries
LTM_REFRESH_CHUNKS = 5    # refresh LTM every N chunks
LTM_RECENT_SUMMARIES = 10 # how many recent summaries to feed into LTM refresh

MODEL_NAME = os.getenv("TOPIC_MODEL", "Qwen/Qwen3-0.6GB")
API_MODE = os.getenv("TOPIC_API_MODE", "local")  # "local" or "openai"
LOCAL_LLM_BASE = os.getenv("LOCAL_LLM_BASE", "http://localhost:3001/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "parallax")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY_FILE = os.getenv("OPENAI_API_KEY_FILE", ".openai_key")
MEMORY_LOG_ENABLED = True
MEMORY_LOG_TEMPLATE = "memory-{date}.jsonl"

SUMMARY_PROMPT = (
    "Summarize this transcript chunk in one paragraph (2-3 sentences, max 80 words). "
    "Keep it concise and factual. If empty or noise, return 'no speech'.\n\n"
    "Transcript:\n{transcript}\n\nSummary:"
)

LTM_PROMPT = (
    "You maintain long-term therapeutic memory for this couple.\n\n"
    "Your job:\n"
    "• Treat CURRENT bullets as durable patterns that should be preserved unless contradicted by strong new evidence.\n"
    "• Use RECENT summaries only to update, refine, or very selectively add patterns that appear repeatedly or carry significant emotional weight.\n"
    "• NEVER overwrite CURRENT bullets with short-lived or one-time themes.\n"
    "• Only store patterns that are persistent, repeated, or emotionally core (attachment needs, stable communication styles, enduring conflict cycles).\n"
    "• Merge similar items instead of duplicating them.\n"
    "• Remove items only if RECENT strongly contradicts them.\n"
    "• Output a maximum of 8 refined bullets representing long-term themes.\n\n"
    "Return ONLY valid JSON:\n{{\"bullets\": [\"...\", \"...\"]}}\n\n"
    "RECENT:\n{recent}\n\nCURRENT:\n{current}"
)

TITLE_PROMPT = (
    "You are a concise topic spotter. Given this summary, return a very short "
    "title (max 5 words) for the main topic. If empty or noise, return 'no speech'.\n\n"
    "Summary:\n{summary}\n\nTitle:"
)

COMPARE_PROMPT = (
    "Decide if the CURRENT summary is roughly the same topic as the PREVIOUS summary. "
    "Answer with exactly 'yes' or 'no'.\n\nPREVIOUS:\n{previous}\n\nCURRENT:\n{current}\n\nSame topic?"
)

if API_MODE.lower() == "openai":
    key = OPENAI_API_KEY
    if not key and os.path.exists(OPENAI_API_KEY_FILE):
        with open(OPENAI_API_KEY_FILE, "r") as fh:
            key = fh.read().strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when TOPIC_API_MODE=openai "
            f"(set env var or write to {OPENAI_API_KEY_FILE})"
        )
    client = OpenAI(api_key=key)
    model_name = MODEL_NAME
else:
    client = OpenAI(base_url=LOCAL_LLM_BASE, api_key=LOCAL_LLM_API_KEY)
    model_name = MODEL_NAME

memory_lock = threading.Lock()
stm = deque()  # list of {"ts", "source", "text"}
mtm = deque(maxlen=MTM_MAX_LEN)  # list of {"ts", "summary"}
ltm = []  # list of bullet strings
chunk_buffer = []  # raw strings collected since last summary
log_lock = threading.Lock()


def log_memory(entry: dict):
    """
    Append a memory entry to the daily JSONL log.
    """
    if not MEMORY_LOG_ENABLED:
        return
    path = MEMORY_LOG_TEMPLATE.format(date=time.strftime("%Y-%m-%d"))
    try:
        with log_lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n\n")
    except Exception as e:
        print(f"[memory][log] failed to write {path}: {e}")


def call_chat(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def summarize_transcript(transcript: str) -> str:
    prompt = SUMMARY_PROMPT.format(transcript=transcript)
    return call_chat(prompt)


def generate_title_from_summary(summary: str) -> str:
    prompt = TITLE_PROMPT.format(summary=summary)
    return call_chat(prompt)


def is_same_topic(prev_summary: str, current_summary: str) -> bool:
    prompt = COMPARE_PROMPT.format(previous=prev_summary, current=current_summary)
    try:
        reply = call_chat(prompt).lower()
        return reply.startswith("yes")
    except Exception as e:
        print(f"[topic][compare] error: {e}")
        return False


def distill_ltm(recent_summaries, current_bullets):
    prompt = LTM_PROMPT.format(
        recent="\n".join(recent_summaries) or "none",
        current="\n".join(current_bullets) or "none",
    )
    raw = call_chat(prompt)

    payload = {}
    try:
        payload = json.loads(raw)
    except Exception:
        if "{" in raw and "}" in raw:
            candidate = raw[raw.index("{") : raw.rindex("}") + 1]
            try:
                payload = json.loads(candidate)
            except Exception:
                payload = {}

    bullets = []
    if isinstance(payload, dict) and isinstance(payload.get("bullets"), list):
        bullets = [str(b).strip() for b in payload["bullets"] if str(b).strip()]
    else:
        # fallback: split lines
        bullets = [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]

    # cap size
    return bullets[:8]


def purge_stm(now_ts):
    cutoff = now_ts - STM_WINDOW_SECONDS
    while stm and stm[0]["ts"] < cutoff:
        stm.popleft()


def poll_transcripts():
    """
    Poll the shared transcripts in main.py and accumulate new mic/system text.
    Update STM window and per-chunk buffer.
    """
    last_lengths = {"mic": 0, "system": 0}

    while True:
        with main.text_lock:
            mic_text = main.mic_text
            system_text = main.system_text

        now = time.time()
        for source, text in (("mic", mic_text), ("system", system_text)):
            if len(text) > last_lengths[source]:
                new = text[last_lengths[source]:].strip()
                if new:
                    with memory_lock:
                        stm.append({"ts": now, "source": source, "text": new})
                        chunk_buffer.append(new)
                        purge_stm(now)
                last_lengths[source] = len(text)

        time.sleep(POLL_SECONDS)


def memory_worker():
    """
    Every SUMMARY_INTERVAL_SECONDS, summarize the buffered text, update MTM,
    and periodically refresh LTM.
    """
    chunk_count = 0
    prev_summary = None
    prev_title = None

    while True:
        time.sleep(SUMMARY_INTERVAL_SECONDS)

        with memory_lock:
            collected = chunk_buffer.copy()
            chunk_buffer.clear()

        chunk_text = " ".join(collected).strip()
        if not chunk_text:
            continue

        # Keep prompt size reasonable
        chunk_text = chunk_text[-2000:]

        try:
            summary = summarize_transcript(chunk_text)
            if not summary or summary.lower() == "no speech":
                continue

            # Decide title and topic continuity
            same = False
            if prev_summary:
                same = is_same_topic(prev_summary, summary)

            if same and prev_title:
                title = prev_title
            else:
                title = generate_title_from_summary(summary)

            ts = time.time()
            with memory_lock:
                mtm.append({"ts": ts, "summary": summary, "title": title})
                chunk_count += 1
                prev_summary = summary
                prev_title = title

                # Refresh LTM periodically
                if chunk_count % LTM_REFRESH_CHUNKS == 0:
                    recent_items = list(mtm)[-LTM_RECENT_SUMMARIES:]
                    recent = [item["summary"] for item in recent_items]
                    new_ltm = distill_ltm(recent, ltm)
                    if new_ltm:
                        ltm.clear()
                        ltm.extend(new_ltm)

            # Log STM chunk + MTM summary
            log_memory(
                {
                    "type": "chunk",
                    "ts": ts,
                    "summary": summary,
                    "title": title,
                }
            )

            # Log LTM when updated
            if chunk_count % LTM_REFRESH_CHUNKS == 0 and ltm:
                log_memory({"type": "ltm", "ts": ts, "bullets": list(ltm)})

            timestamp = time.strftime("%H:%M:%S")
            print(
                f"\n[{timestamp}] TITLE: {title}\n"
                f"           SUMMARY: {summary}\n"
                f"           MTM size: {len(mtm)} | LTM bullets: {len(ltm)}"
            )
            if chunk_count % LTM_REFRESH_CHUNKS == 0 and ltm:
                print("           LTM update:")
                for i, b in enumerate(ltm, 1):
                    print(f"             {i}. {b}")
        except Exception as e:
            print(f"[memory] error: {e}")


def loop():
    """
    Therapy memory loop: builds STM/MTM/LTM while main.py captures audio.
    """
    print(
        f"Therapy memory loop using {model_name} via {API_MODE} | "
        f"STM window {STM_WINDOW_SECONDS//60}m | MTM max {MTM_MAX_LEN} | "
        f"LTM refresh every {LTM_REFRESH_CHUNKS} chunks"
    )

    threading.Thread(target=poll_transcripts, daemon=True).start()
    threading.Thread(target=memory_worker, daemon=True).start()

    # Keep the loop alive
    while True:
        time.sleep(1)
