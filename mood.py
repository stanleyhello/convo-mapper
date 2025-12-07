import json
import time
import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

"""
Lightweight mood detection worker that turns recent transcript text into a
smoothed mood/energy signal for the music engine.
"""

MOOD_LABELS = ["calm", "happy", "playful", "tense", "angry", "sad", "anxious", "neutral"]
MOOD_ALIASES = {
    "reflective": "calm",
    "thoughtful": "calm",
    "pensive": "calm",
}


@dataclass
class MoodConfig:
    poll_interval_s: float = 30.0        # cadence for mood checks (target ~30s latency)
    primary_window_s: float = 30.0       # primary recent window (last 30 seconds)
    context_window_s: float = 120.0      # broader STM context (last 2 minutes)
    mtm_tail: int = 5                    # how many recent MTM summaries to include
    history_len: int = 3                 # smoothing window (shorter for responsiveness)
    flip_patience: int = 1               # fewer consecutive hits before switching
    min_duration_s: float = 5.0          # minimum time to hold a mood before switching
    confidence_gate: float = 0.6         # allow immediate flip if confidence above this
    model_name: Optional[str] = None     # optional override for mood model


class MoodStateHolder:
    """
    Thread-safe holder for the current mood state.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: Optional[Dict] = None

    def get(self) -> Optional[Dict]:
        with self._lock:
            return self._state.copy() if self._state else None

    def set(self, state: Dict):
        with self._lock:
            self._state = state.copy()


def _default_state(now: Optional[float] = None) -> Dict:
    ts = now if now is not None else time.time()
    return {"timestamp": ts, "mood": "neutral", "energy": 1, "confidence": 0.5}


MOOD_PROMPT_TEMPLATE = (
    "You are a mood classifier for a live conversation.\n"
    "Read the recent transcript and return JSON only.\n"
    'Mood must be one of: ["calm","happy","playful","tense","angry","sad","anxious","neutral"].\n'
    "Energy must be an integer 0 (very low) to 3 (very high).\n"
    'Return: {{"mood": "<mood>", "energy": <0-3>, "confidence": <0.0-1.0>}}.\n'
    "Recent (last minute):\n{recent}\n\n"
    "Context (last 5 minutes):\n{context}\n\n"
    "JSON:"
)


def build_prompt(recent: str, context: str) -> str:
    return MOOD_PROMPT_TEMPLATE.format(
        recent=recent.strip() or "(no recent text)",
        context=context.strip() or "(no context)",
    )


def _parse_response(raw: str) -> Optional[Dict]:
    try:
        payload = json.loads(raw)
    except Exception:
        if "{" in raw and "}" in raw:
            try:
                payload = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            except Exception:
                return None
        else:
            return None

    if not isinstance(payload, dict):
        return None

    mood_raw = str(payload.get("mood", "")).strip().lower()
    mood = mood_raw
    if mood not in MOOD_LABELS and mood in MOOD_ALIASES:
        mood = MOOD_ALIASES[mood]
    energy = payload.get("energy", 1)
    confidence = payload.get("confidence", 0.5)

    if mood not in MOOD_LABELS:
        return None

    try:
        energy_int = int(energy)
    except Exception:
        energy_int = 1
    energy_int = max(0, min(3, energy_int))

    try:
        conf_f = float(confidence)
    except Exception:
        conf_f = 0.5
    conf_f = max(0.0, min(1.0, conf_f))

    return {
        "timestamp": time.time(),
        "mood": mood,
        "energy": energy_int,
        "confidence": conf_f,
    }


def analyze_mood(prompt_text: str, call_chat_fn: Callable[[str], str]) -> Dict:
    """
    Call the LLM to classify mood/energy; fall back to neutral on failure.
    """
    prompt = prompt_text
    raw = ""
    try:
        raw = call_chat_fn(prompt)
        parsed = _parse_response(raw)
        if parsed:
            return parsed
    except Exception as exc:
        print(f"[mood] analyze error: {exc}")

    print(f"[mood] fallback to neutral; raw response: {raw!r}")
    return _default_state()


def _smooth_history(history: Deque[Dict]) -> Dict:
    """
    Majority vote for mood, average energy/confidence. Uses latest item to break ties.
    """
    if not history:
        return _default_state()

    mood_counts = Counter(item["mood"] for item in history)
    top_mood, _ = mood_counts.most_common(1)[0]

    energy_avg = sum(item["energy"] for item in history) / len(history)
    conf_avg = sum(item["confidence"] for item in history) / len(history)

    smoothed = {
        "timestamp": history[-1]["timestamp"],
        "mood": top_mood,
        "energy": int(round(energy_avg)),
        "confidence": round(conf_avg, 3),
    }
    smoothed["energy"] = max(0, min(3, smoothed["energy"]))
    return smoothed


def mood_worker(
    fetch_stm_fn: Callable[[float], List[str]],
    fetch_mtm_fn: Callable[[int], List[str]],
    state_holder: MoodStateHolder,
    cfg: MoodConfig,
    call_chat_fn: Callable[[str], str],
):
    """
    Periodically analyze recent STM text (plus recent MTM summaries) and update shared mood state.
    """
    history: Deque[Dict] = deque(maxlen=cfg.history_len)
    last_state = None
    pending_flip = 0

    while True:
        time.sleep(cfg.poll_interval_s)

        now = time.time()
        recent_stm = fetch_stm_fn(cfg.primary_window_s) if fetch_stm_fn else []
        context_stm = fetch_stm_fn(cfg.context_window_s) if fetch_stm_fn else []
        mtm_summaries = fetch_mtm_fn(cfg.mtm_tail) if fetch_mtm_fn else []

        recent_block = "\n".join(recent_stm).strip()
        context_block = "\n".join(context_stm + mtm_summaries).strip()

        if not recent_block and not context_block:
            continue

        prompt_text = build_prompt(
            recent=recent_block[-3000:],
            context=context_block[-5000:],
        )

        candidate = analyze_mood(prompt_text, call_chat_fn)
        history.append(candidate)
        smoothed = _smooth_history(history)

        if last_state is None:
            last_state = smoothed
            state_holder.set(smoothed)
            print(f"[mood] init mood={smoothed['mood']} energy={smoothed['energy']} conf={smoothed['confidence']}")
            continue

        if smoothed["mood"] != last_state["mood"]:
            pending_flip += 1
            dwell_ok = now - last_state["timestamp"] >= cfg.min_duration_s
            confident = smoothed["confidence"] >= cfg.confidence_gate
            if (pending_flip >= cfg.flip_patience and dwell_ok) or confident:
                last_state = {**smoothed, "timestamp": now}
                state_holder.set(last_state)
                pending_flip = 0
                print(f"[mood] mood change -> mood={last_state['mood']} energy={last_state['energy']} conf={last_state['confidence']}")
        else:
            pending_flip = 0
            # allow energy/confidence drift while keeping mood stable
            if smoothed["energy"] != last_state["energy"] or smoothed["confidence"] != last_state["confidence"]:
                last_state = {**smoothed, "timestamp": now}
                state_holder.set(last_state)
                print(f"[mood] update -> mood={last_state['mood']} energy={last_state['energy']} conf={last_state['confidence']}")
