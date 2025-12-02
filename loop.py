import json
import os
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

import main

# Load environment variables from a .env file if present
load_dotenv()

POLL_SECONDS = 1.0
SUMMARY_INTERVAL_SECONDS = 60.0
TOPIC_MODEL = os.getenv("TOPIC_MODEL", "Qwen/Qwen3-0.6GB")
TOPIC_API_MODE = os.getenv("TOPIC_API_MODE", "local")  # "local" or "openai"
LOCAL_LLM_BASE = os.getenv("LOCAL_LLM_BASE", "http://localhost:3001/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "parallax")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY_FILE = os.getenv("OPENAI_API_KEY_FILE", ".openai_key")

COMBINED_PROMPT = (
    "You are a concise topic spotter. For the CURRENT transcript:\n"
    "1) Write a summary of the current transcript only (1 paragraph).\n"
    "2) Compare that summary to the PREVIOUS summary and decide if they are roughly the same topic.\n"
    "3) If same topic → reuse the previous title. If different → generate a new title (max 5 words).\n"
    "Return ONLY valid JSON with keys: "
    "\"summary\" (the current summary), "
    "\"same_topic\" ('yes' or 'no'), "
    "\"title\" (reuse previous title if same_topic=yes; otherwise new title). "
    "If transcript is empty or noise, use summary 'no speech', same_topic 'no', title 'no speech'.\n\n"
    "PREVIOUS SUMMARY:\n{prev_summary}\n"
    "PREVIOUS TITLE:\n{prev_title}\n"
    "CURRENT TRANSCRIPT:\n{transcript}"
)

if TOPIC_API_MODE.lower() == "openai":
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
    topic_model = TOPIC_MODEL
else:
    client = OpenAI(base_url=LOCAL_LLM_BASE, api_key=LOCAL_LLM_API_KEY)
    topic_model = TOPIC_MODEL

buffer_lock = threading.Lock()
buffers = {"mic": [], "system": []}


def call_chat(prompt):
    resp = client.chat.completions.create(
        model=topic_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def analyze_transcript(prev_summary: str, prev_title: str, transcript: str):
    """
    One model call to summarize, classify same/different topic, and propose a title.
    Returns (summary, same_topic_bool, title_candidate).
    """
    prompt = COMBINED_PROMPT.format(
        prev_summary=prev_summary or "none",
        prev_title=prev_title or "none",
        transcript=transcript,
    )

    raw = call_chat(prompt)
    summary = ""
    same_topic = False
    title = ""

    # Try to parse JSON from the response
    try:
        payload = json.loads(raw)
    except Exception:
        # try to salvage JSON inside fences
        if "{" in raw and "}" in raw:
            candidate = raw[raw.index("{") : raw.rindex("}") + 1]
            try:
                payload = json.loads(candidate)
            except Exception:
                payload = {}
        else:
            payload = {}

    if payload:
        summary = str(payload.get("summary", "")).strip()
        title = str(payload.get("title", "")).strip()
        same_topic = str(payload.get("same_topic", "")).strip().lower().startswith("y")
    else:
        # fallback: use raw as summary
        summary = raw.strip()

    return summary, same_topic, title


def poll_transcripts():
    """
    Poll the shared transcripts in main.py and accumulate new mic/system text
    into per-source buffers for topic summarization.
    """
    last_lengths = {"mic": 0, "system": 0}

    while True:
        with main.text_lock:
            mic_text = main.mic_text
            system_text = main.system_text

        for source, text in (("mic", mic_text), ("system", system_text)):
            if len(text) > last_lengths[source]:
                new = text[last_lengths[source]:].strip()
                if new:
                    with buffer_lock:
                        buffers[source].append(new)
                last_lengths[source] = len(text)

        time.sleep(POLL_SECONDS)


def topic_worker(source):
    """
    Every SUMMARY_INTERVAL_SECONDS, summarize the buffered text for `source`,
    decide if it is the same topic as the previous chunk, and print the topic.
    """
    prev_title = None
    prev_summary = None

    while True:
        time.sleep(SUMMARY_INTERVAL_SECONDS)

        with buffer_lock:
            collected = buffers[source]
            buffers[source] = []

        transcript = " ".join(collected).strip()
        if not transcript:
            continue

        # Keep prompt size reasonable
        transcript = transcript[-2000:]

        try:
            summary, same, title_candidate = analyze_transcript(
                prev_summary, prev_title, transcript
            )
            if not summary or summary.lower() == "no speech":
                continue

            if same and prev_title:
                title = prev_title
            else:
                title = title_candidate or " ".join(summary.split()[:5])

            prev_summary = summary
            prev_title = title

            timestamp = time.strftime("%H:%M:%S")
            print(
                f"\n[{timestamp}] {source.upper()} TOPIC: {title}\n"
                f"           SUMMARY: {summary}\n"
            )
        except Exception as e:
            print(f"[topic][{source}] error: {e}")


def loop():
    """
    Designed to run alongside main.start_audio_and_model from controller.py.
    Starts polling for new transcript text and periodic topic summaries.
    """
    print(
        f"Polling transcripts every {POLL_SECONDS}s; "
        f"summarizing every {SUMMARY_INTERVAL_SECONDS}s using {topic_model} via {TOPIC_API_MODE}"
    )

    threading.Thread(target=poll_transcripts, daemon=True).start()
    threading.Thread(target=topic_worker, args=("mic",), daemon=True).start()
    threading.Thread(target=topic_worker, args=("system",), daemon=True).start()

    # Keep the loop alive
    while True:
        time.sleep(1)
