import threading
import time

from openai import OpenAI

import main

# Import centralized configuration
from config import (
    POLL_SECONDS,
    SUMMARY_INTERVAL_SECONDS,
    TOPIC_MODEL,
    TOPIC_PROMPT,
    PARALLAX_ENDPOINT,
    PARALLAX_API_KEY,
)

client = OpenAI(
    base_url=PARALLAX_ENDPOINT,
    api_key=PARALLAX_API_KEY,
)

buffer_lock = threading.Lock()
buffers = {"mic": [], "system": []}


def poll_transcripts():
    """
    Poll the shared transcripts in main.py and print any new mic/system text.
    Also accumulate text into per-source buffers for topic summarization.
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
                    print(f"[{source}] {new}")
                    with buffer_lock:
                        buffers[source].append(new)
                last_lengths[source] = len(text)

        time.sleep(POLL_SECONDS)


def topic_worker(source):
    """
    Every SUMMARY_INTERVAL_SECONDS, summarize the buffered text for `source`
    via the local chat model and print the topic.
    """
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
            resp = client.chat.completions.create(
                model=TOPIC_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": TOPIC_PROMPT.format(transcript=transcript),
                    }
                ],
            )
            topic = resp.choices[0].message.content.strip()
            print(f"[topic][{source}] {topic}")
        except Exception as e:
            print(f"[topic][{source}] error: {e}")


def loop():
    """
    Designed to run alongside main.start_audio_and_model from controller.py.
    Starts polling for new transcript text and periodic topic summaries.
    """
    print(
        f"Polling transcripts every {POLL_SECONDS}s; "
        f"summarizing every {SUMMARY_INTERVAL_SECONDS}s using {TOPIC_MODEL}"
    )

    threading.Thread(target=poll_transcripts, daemon=True).start()
    threading.Thread(target=topic_worker, args=("mic",), daemon=True).start()
    threading.Thread(target=topic_worker, args=("system",), daemon=True).start()

    # Keep the loop alive
    while True:
        time.sleep(1)
