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
LTM_REFRESH_CHUNKS = 8    # refresh LTM every N chunks (less churn)
LTM_RECENT_SUMMARIES = 10 # how many recent summaries to feed into LTM refresh
INTERJECT_INTERVAL_SECONDS = 60.0  # how often to consider interjecting
INTERJECT_CONFIDENCE_THRESHOLD = float(os.getenv("INTERJECT_CONFIDENCE_THRESHOLD", 0.75))
INTERJECT_COOLDOWN_SECONDS = float(os.getenv("INTERJECT_COOLDOWN_SECONDS", 180))
INTERJECT_START_DELAY_SECONDS = 300  # no interjections before 5 minutes
TITLE_WARMUP_SECONDS = 120           # wait 2 minutes before first title

MODEL_NAME = os.getenv("TOPIC_MODEL", "Qwen/Qwen3-0.6GB")
API_MODE = os.getenv("TOPIC_API_MODE", "local")  # "local" or "openai"
LOCAL_LLM_BASE = os.getenv("LOCAL_LLM_BASE", "http://localhost:3001/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "parallax")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY_FILE = os.getenv("OPENAI_API_KEY_FILE", ".openai_key")
MEMORY_LOG_ENABLED = True
MEMORY_LOG_TEMPLATE = "memory-{date}.jsonl"

SUMMARY_PROMPT = (
    "Summarize this transcript chunk in one paragraph (max 4 sentences). "
    "Use plain, concrete language; include specific details; avoid fluff. If empty or noise, return 'no speech'.\n\n"
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
    "• Keep bullets concrete and specific to recurring issues (e.g., late-night money fights, repeated timing conflicts), avoid generic communication advice.\n"
    "• If no new recurring issue appears, keep bullets unchanged.\n"
    "• Output a maximum of 8 refined bullets representing long-term themes.\n\n"
    "Return ONLY valid JSON:\n{{\"bullets\": [\"...\", \"...\"]}}\n\n"
    "RECENT:\n{recent}\n\nCURRENT:\n{current}"
)

TITLE_PROMPT = (
    "You are a concise topic spotter. Given this summary, return a short, concrete "
    "title (max 5 words) that mentions the specific issue (e.g., 'Late-night money fights'). "
    "Avoid generic labels. If empty or noise, return 'no speech'.\n\n"
    "Summary:\n{summary}\n\nTitle:"
)

COMPARE_PROMPT = (
    "Decide if the CURRENT summary is roughly the same topic as the PREVIOUS summary. "
    "Answer with exactly 'yes' or 'no'.\n\nPREVIOUS:\n{previous}\n\nCURRENT:\n{current}\n\nSame topic?"
)

INTERJECT_PROMPT = (
    "You are an AI couples therapist assistant.\n"
    "You are given:\n"
    "- LongTermPatterns: persistent themes for this couple\n"
    "- RecentSummaries: recent 60s summaries\n"
    "- RecentDialogue: the last few minutes of raw dialogue\n\n"
    "Your job has TWO steps:\n\n"
    "STEP 1 – ANALYZE:\n"
    "Infer the following about the RecentDialogue:\n"
    "- escalationLevel: 0–3 (0=calm, 1=tense, 2=escalating, 3=very heated)\n"
    "- blameLevel: 0–3 (0=none, 1=some blame, 2=strong blame, 3=attacks)\n"
    "- nameCallingPresent: true/false (insults, contempt, character attacks)\n"
    "- globalCriticismPresent: true/false (phrases like 'you always', 'you never',\n"
    "  or global negative judgments about the person)\n"
    "- youStatementsHeavy: true/false (dominant use of 'you' accusations vs 'I' statements)\n"
    "- misunderstandingLikely: true/false (partners talking past each other,\n"
    "  responding to different issues or meanings)\n"
    "- cycleDescription: short description (if any recurring interaction pattern\n"
    "  appears, e.g., 'one pursues, the other shuts down', or 'mutual escalation')\n"
    "- underlyingFeelingCandidate: short phrase for the deeper vulnerable feeling\n"
    "  if any (e.g., 'feeling unimportant', 'fear of rejection'); else 'none'\n"
    "- unspokenNeedCandidate: short phrase for an underlying need if any\n"
    "  (e.g., 'need for reassurance', 'need for appreciation'); else 'none'\n\n"
    "STEP 2 – DECIDE WHETHER TO INTERJECT:\n"
    "You may choose exactly one of the following interventionType values:\n"
    '- \"none\"\n'
    '- \"reflect_deeper_emotion\"      (Highlighting underlying feelings)\n'
    '- \"highlight_pattern\"           (Naming the interaction pattern/cycle)\n'
    '- \"de_escalate\"                 (Slowing down escalation)\n'
    '- \"repair_attempt\"              (Suggest brief reset or small repair)\n'
    '- \"teach_I_statements\"          (Guiding I-statements)\n'
    '- \"clarify_meaning\"             (Addressing misunderstandings)\n'
    '- \"identify_unspoken_need\"      (Identifying unspoken needs)\n'
    '- \"boundary_setting\"            (Promoting respectful communication)\n\n'
    "Use these guidelines:\n"
    "- Choose \"boundary_setting\" when nameCallingPresent or globalCriticismPresent\n"
    "  is true, or blameLevel >= 2 with strong personal attacks.\n"
    "- Choose \"teach_I_statements\" when youStatementsHeavy is true and blameLevel >= 1\n"
    "  but there is no severe name-calling.\n"
    "- Choose \"de_escalate\" when escalationLevel >= 2 (heated) even if there is\n"
    "  no name-calling yet.\n"
    "- Choose \"repair_attempt\" when escalationLevel is 1–2 and there has been\n"
    "  tension, but the conversation is still workable. Use language like\n"
    "  'Would now be a good time for a brief reset?'.\n"
    "- Choose \"reflect_deeper_emotion\" when underlyingFeelingCandidate is not 'none'\n"
    "  and there is repeated frustration or hurt.\n"
    "- Choose \"identify_unspoken_need\" when unspokenNeedCandidate is not 'none'\n"
    "  and the partners are arguing about behavior instead of the underlying need.\n"
    "- Choose \"highlight_pattern\" when a clear cycleDescription is present and is\n"
    "  contributing to the conflict.\n"
    "- Choose \"clarify_meaning\" when misunderstandingLikely is true.\n"
    "- If multiple apply, pick the one that best promotes safety and understanding\n"
    "  in this moment (boundary_setting > de_escalate > repair_attempt > others).\n"
    "- If nothing clearly applies or your confidence is low, use interventionType=\"none\".\n\n"
    "When you DO interject:\n"
    "- Use 1 short, plain sentence (max ~25 words).\n"
    "- Be neutral and non-judgmental. Do not take sides.\n"
    "- For repair_attempt, you may use language like: \"Would now be a good time for a brief reset?\".\n\n"
    "Return ONLY valid JSON in this exact structure:\n"
    "{{\n"
    "  \"analysis\": {{\n"
    "    \"escalationLevel\": 0,\n"
    "    \"blameLevel\": 0,\n"
    "    \"nameCallingPresent\": false,\n"
    "    \"globalCriticismPresent\": false,\n"
    "    \"youStatementsHeavy\": false,\n"
    "    \"misunderstandingLikely\": false,\n"
    "    \"cycleDescription\": \"\",\n"
    "    \"underlyingFeelingCandidate\": \"\",\n"
    "    \"unspokenNeedCandidate\": \"\"\n"
    "  }},\n"
    "  \"intervention\": {{\n"
    "    \"shouldInterject\": true/false,\n"
    "    \"interventionType\": \"none or one of the types above\",\n"
    "    \"confidence\": 0.0,\n"
    "    \"reasons\": [\"short reason 1\", \"short reason 2\"],\n"
    "    \"candidateMessage\": \"...\"\n"
    "  }}\n"
    "}}\n\n"
    "LongTermPatterns:\n{ltm}\n\n"
    "RecentSummaries:\n{mtm}\n\n"
    "RecentDialogue:\n{stm}\n"
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
start_time = time.time()
warmup_summaries = []


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


def decide_interjection(stm_text: str, mtm_items, ltm_bullets):
    """
    Run the interjection classifier/generator and return parsed decision dict.
    Now expects the model to return a nested JSON with 'analysis' and 'intervention'.
    """
    mtm_lines = []
    for item in mtm_items:
        title = item.get("title") or ""
        summary = item.get("summary") or ""
        mtm_lines.append(f"- {title}: {summary}")
    mtm_block = "\n".join(mtm_lines) or "none"

    stm_block = stm_text or "none"
    ltm_block = "\n".join(ltm_bullets) or "none"

    prompt = INTERJECT_PROMPT.format(
        ltm=ltm_block,
        mtm=mtm_block,
        stm=stm_block,
    )

    raw = call_chat(prompt)
    data = {}

    try:
        data = json.loads(raw)
    except Exception:
        # Best-effort bracket extraction if LLM wrapped JSON in extra text
        if "{" in raw and "}" in raw:
            candidate = raw[raw.index("{") : raw.rindex("}") + 1]
            try:
                data = json.loads(candidate)
            except Exception:
                data = {}

    analysis = data.get("analysis", {}) or {}
    intervention = data.get("intervention", {}) or {}

    allowed_types = {
        "none",
        "de_escalate",
        "highlight_pattern",
        "clarify_meaning",
        "reflect_deeper_emotion",
        "reflect_attachment_need",  # legacy
        "teach_communication_skill",  # legacy
        "structure_turn_taking",      # legacy
        "repair_attempt",
        "teach_I_statements",
        "identify_unspoken_need",
        "boundary_setting",
    }

    should_interject = bool(intervention.get("shouldInterject", False))
    intervention_type = str(intervention.get("interventionType", "none"))
    try:
        confidence = float(intervention.get("confidence", 0))
    except Exception:
        confidence = 0.0
    reasons = intervention.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r).strip() for r in reasons if str(r).strip()]
    candidate_message = str(intervention.get("candidateMessage", "")).strip()

    # enforce constraints
    if intervention_type not in allowed_types:
        should_interject = False
        intervention_type = "none"
    if confidence < INTERJECT_CONFIDENCE_THRESHOLD:
        should_interject = False
    if not reasons:
        should_interject = False
    if should_interject and intervention_type == "none":
        should_interject = False
    # keep message short
    if len(candidate_message.split()) > 30:
        candidate_message = " ".join(candidate_message.split()[:30])

    return {
        "shouldInterject": should_interject,
        "interventionType": intervention_type,
        "confidence": confidence,
        "reasons": reasons,
        "candidateMessage": candidate_message,
        "raw": raw,
        "analysis": analysis,
    }


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


def build_stm_block():
    """
    Construct a short text block from STM entries for prompting.
    """
    with memory_lock:
        entries = list(stm)
    lines = []
    for item in entries:
        ts = time.strftime("%H:%M:%S", time.localtime(item["ts"]))
        lines.append(f"{ts} [{item['source']}]: {item['text']}")
    return "\n".join(lines[-50:])  # cap lines to keep prompt small


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

            # Decide title with warmup
            elapsed = time.time() - start_time
            title = ""
            if prev_title is None:
                warmup_summaries.append(summary)
                if elapsed >= TITLE_WARMUP_SECONDS:
                    warmup_text = " ".join(warmup_summaries)
                    title = generate_title_from_summary(warmup_text)
                    prev_title = title
                    prev_summary = summary
                else:
                    title = "(pending)"
            else:
                # normal flow
                same = False
                if prev_summary:
                    same = is_same_topic(prev_summary, summary)
                if same and prev_title:
                    title = prev_title
                else:
                    title = generate_title_from_summary(summary)
                prev_summary = summary
                prev_title = title

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


def interject_worker():
    """
    Periodically decide whether to interject, based on STM/MTM/LTM.
    """
    last_interject_ts = 0

    while True:
        time.sleep(INTERJECT_INTERVAL_SECONDS)

        if (time.time() - start_time) < INTERJECT_START_DELAY_SECONDS:
            continue

        stm_block = build_stm_block()
        with memory_lock:
            recent_mtm = list(mtm)[-LTM_RECENT_SUMMARIES:]
            ltm_bullets = list(ltm)

        decision = decide_interjection(stm_block, recent_mtm, ltm_bullets)

        should = decision["shouldInterject"]
        conf = decision["confidence"]
        now = time.time()
        cooldown_ok = (now - last_interject_ts) >= INTERJECT_COOLDOWN_SECONDS

        if should and conf >= INTERJECT_CONFIDENCE_THRESHOLD and cooldown_ok:
            last_interject_ts = now
            # Log decision
            log_memory(
                {
                    "type": "interjection",
                    "ts": now,
                    "decision": decision,
                }
            )
            print(
                f"\n[INTERJECT] type={decision['interventionType']} "
                f"conf={conf:.2f} reasons={decision['reasons']}\n"
                f"Message: {decision['candidateMessage']}"
            )
        else:
            # Still log the decision for observability
            log_memory(
                {
                    "type": "interjection",
                    "ts": now,
                    "decision": decision,
                    "skipped": True,
                    "cooldown_ok": cooldown_ok,
                }
            )


def loop():
    """
    Therapy memory loop: builds STM/MTM/LTM while main.py captures audio.
    """
    print(
        f"Therapy memory loop using {model_name} via {API_MODE} | "
        f"STM window {STM_WINDOW_SECONDS//60}m | MTM max {MTM_MAX_LEN} | "
        f"LTM refresh every {LTM_REFRESH_CHUNKS} chunks | "
        f"Interject every {INTERJECT_INTERVAL_SECONDS}s (cooldown {INTERJECT_COOLDOWN_SECONDS}s, "
        f"threshold {INTERJECT_CONFIDENCE_THRESHOLD})"
    )

    threading.Thread(target=poll_transcripts, daemon=True).start()
    threading.Thread(target=memory_worker, daemon=True).start()
    threading.Thread(target=interject_worker, daemon=True).start()

    # Keep the loop alive
    while True:
        time.sleep(1)
