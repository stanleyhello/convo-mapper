import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import pathlib

try:
    import fluidsynth
except ImportError:
    fluidsynth = None
else:
    if not hasattr(fluidsynth, "Synth"):
        # Wrong package installed (needs pyfluidsynth); disable and warn.
        print(
            "[music] fluidsynth module lacks Synth; "
            "install pyfluidsynth (pip install pyfluidsynth) and libfluidsynth."
        )
        fluidsynth = None

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_SF2 = BASE_DIR / "piano.sf2"  # or "soundfonts/salamander.sf2"
DEFAULT_SF2_STR = str(DEFAULT_SF2)

def _default_soundfont() -> Optional[str]:
    # env var can override, but bundled sf2 is the default
    return os.getenv("MUSIC_SOUNDFONT", DEFAULT_SF2_STR)


try:
    import mido
except ImportError:
    mido = None

# Simple generative piano engine tuned by mood/energy from mood.py.


@dataclass
class MusicConfig:
    enabled: bool = True
    midi_port: Optional[str] = None           # None → default output (used if no fluidsynth)
    soundfont_path: Optional[str] = field(default_factory=_default_soundfont)      # if provided and fluidsynth available, use in-process synth
    tick_ms: int = 50                         # scheduler granularity
    base_tempo_by_energy: Dict[int, int] = field(
        default_factory=lambda: {0: 55, 1: 70, 2: 90, 3: 110}
    )
    velocity_by_energy: Dict[int, int] = field(
        default_factory=lambda: {0: 50, 1: 70, 2: 85, 3: 100}
    )
    density_by_energy: Dict[int, float] = field(
        default_factory=lambda: {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.0}
    )
    swing_by_mood: Dict[str, float] = field(
        default_factory=lambda: {
            "calm": 0.52,
            "happy": 0.55,
            "playful": 0.6,
            "tense": 0.5,
            "angry": 0.48,
            "sad": 0.5,
            "anxious": 0.47,
            "neutral": 0.5,
        }
    )
    register_by_mood: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "calm": (52, 72),
            "happy": (55, 76),
            "playful": (57, 79),
            "tense": (50, 70),
            "angry": (48, 68),
            "sad": (50, 69),
            "anxious": (51, 71),
            "neutral": (53, 73),
        }
    )
    progressions_by_mood: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=lambda: {
            # smoother, open voicings
            "calm": [("C", "maj7"), ("G", "sus2"), ("Am", "min7"), ("F", "maj7")],
            # bright, resolving
            "happy": [("C", "maj"), ("G", "maj"), ("Em", "min7"), ("F", "maj")],
            # light tension and release
            "playful": [("C", "6"), ("Em", "min7"), ("A", "7"), ("Dm", "min7")],
            # modal mixture with leading tension
            "tense": [("Dm", "min"), ("G", "7"), ("Bm", "dim"), ("C", "maj7")],
            # darker colors, V of iv
            "angry": [("Cm", "min"), ("Ab", "maj"), ("G", "7"), ("Fm", "min")],
            # descending/plagal motions
            "sad": [("Am", "min7"), ("F", "maj7"), ("C", "maj7"), ("G", "sus2")],
            # restless shifts between minor and sus
            "anxious": [("Dm", "min7"), ("Bb", "maj"), ("C", "sus2"), ("G", "7")],
            # balanced diatonic loop
            "neutral": [("C", "maj"), ("Am", "min"), ("F", "maj"), ("G", "maj")],
        }
    )
    patterns_by_energy: Dict[int, List[float]] = field(
        default_factory=lambda: {
            0: [0.0, 2.0],          # sparse
            1: [0.0, 1.5, 3.0],     # light syncopation
            2: [0.0, 1.0, 2.0, 3.0],
            3: [0.0, 0.75, 1.5, 2.25, 3.0],  # denser
        }
    )


NOTE_OFFSETS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

QUALITY_INTERVALS = {
    "maj": [0, 4, 7],
    "maj7": [0, 4, 7, 11],
    "6": [0, 4, 7, 9],
    "sus2": [0, 2, 7],
    "7": [0, 4, 7, 10],
    "min": [0, 3, 7],
    "min7": [0, 3, 7, 10],
    "dim": [0, 3, 6],
}


class FluidSynthOut:
    def __init__(self, soundfont_path: str):
        self.soundfont_path = soundfont_path
        self.synth = None
        self.available = fluidsynth is not None
        if not self.available:
            print("[music] fluidsynth not installed; skipping in-process synth.")
            return

        if not self.soundfont_path or not pathlib.Path(self.soundfont_path).exists():
            print(f"[music] soundfont not found: {self.soundfont_path}")
            return
        try:
            self.synth = fluidsynth.Synth()
            self.synth.start()  # default driver
            sfid = self.synth.sfload(self.soundfont_path)
            if sfid == -1:
                print(f"[music] failed to load soundfont: {self.soundfont_path}")
                self.synth.delete()
                self.synth = None
                return
            try:
                self.synth.program_select(0, sfid, 0, 0)
            except Exception as exc:
                print(f"[music] failed to select program for soundfont: {exc}")
                self.synth.delete()
                self.synth = None
                return
            print(f"[music] fluidsynth loaded soundfont: {self.soundfont_path}")
        except Exception as exc:
            print(f"[music] failed to init fluidsynth: {exc}")
            self.synth = None

    def note_on(self, note: int, velocity: int):
        if self.synth:
            self.synth.noteon(0, note, velocity)

    def note_off(self, note: int):
        if self.synth:
            self.synth.noteoff(0, note)

    def close(self):
        if self.synth:
            try:
                self.synth.delete()
            except Exception:
                pass


class MidiOut:
    def __init__(self, cfg: MusicConfig):
        self.cfg = cfg
        self.outport = None
        self.available = mido is not None
        if not self.available:
            print("[music] mido not installed; running in silent mode.")
            return

        try:
            if cfg.midi_port:
                self.outport = mido.open_output(cfg.midi_port)
            else:
                self.outport = mido.open_output()
            print(f"[music] MIDI out: {self.outport.name}")
        except Exception as exc:
            print(f"[music] failed to open MIDI output: {exc}")
            self.outport = None

    def note_on(self, note: int, velocity: int):
        if self.outport:
            self.outport.send(mido.Message("note_on", note=note, velocity=velocity))

    def note_off(self, note: int):
        if self.outport:
            self.outport.send(mido.Message("note_off", note=note, velocity=0))

    def close(self):
        if self.outport:
            self.outport.close()


def make_backend(cfg: MusicConfig):
    # Prefer in-process synth if soundfont provided
    if cfg.soundfont_path:
        fs = FluidSynthOut(cfg.soundfont_path)
        if fs.synth:
            return fs
    # Fallback to MIDI out
    return MidiOut(cfg)


def _clamp_note(note: int, register: Tuple[int, int]) -> int:
    low, high = register
    return max(low, min(high, note))


def _chord_notes(root: str, quality: str, register: Tuple[int, int]) -> List[int]:
    intervals = QUALITY_INTERVALS.get(quality, QUALITY_INTERVALS["maj"])
    base = NOTE_OFFSETS.get(root, NOTE_OFFSETS["C"])
    notes = []
    octave_base = 60  # middle C
    for iv in intervals:
        note = octave_base + base + iv
        notes.append(_clamp_note(note, register))
    return notes


def choose_progression(mood: str, cfg: MusicConfig) -> List[Tuple[str, str]]:
    return cfg.progressions_by_mood.get(mood, cfg.progressions_by_mood["neutral"])


def choose_pattern(energy: int, cfg: MusicConfig) -> List[float]:
    return cfg.patterns_by_energy.get(energy, cfg.patterns_by_energy[1])


def generate_bar(
    chord: Tuple[str, str],
    pattern_beats: List[float],
    register: Tuple[int, int],
    density: float,
) -> List[Dict]:
    """
    Return a list of chord events with offsets within the bar.
    """
    root, quality = chord
    chord_tones = _chord_notes(root, quality, register)
    events = []
    chord_voicing_size = 3 if len(chord_tones) < 4 else random.choice([3, 4])

    for beat in sorted(pattern_beats):
        if random.random() > density:
            continue
        duration_beats = random.choice([0.6, 0.8, 1.0])

        # chord stab: choose triad/7th voicing, optionally invert and double bass
        notes = chord_tones.copy()
        random.shuffle(notes)
        notes = notes[:chord_voicing_size]

        # simple inversion: move one note up/down an octave if within register
        if random.random() < 0.6 and notes:
            idx = random.randrange(len(notes))
            shift = random.choice([-12, 12])
            notes[idx] = _clamp_note(notes[idx] + shift, register)

        # occasional bass doubling
        if random.random() < 0.4:
            bass = _clamp_note(notes[0] - 12, register)
            notes.insert(0, bass)

        events.append({"offset_beats": beat, "notes": notes, "duration_beats": duration_beats})
    return events


def music_engine_loop(state_holder, cfg: MusicConfig):
    """
    Main loop that reads mood state and renders MIDI bars.
    """
    backend = make_backend(cfg)
    bar_index = 0

    try:
        while True:
            if not cfg.enabled:
                time.sleep(0.5)
                continue

            mood_state = state_holder.get() if state_holder else None
            if not mood_state:
                time.sleep(0.25)
                continue

            mood = mood_state["mood"]
            if mood == "neutral":
                # idle when mood is neutral; keep loop timing gentle
                time.sleep(0.5)
                continue
            energy = int(mood_state.get("energy", 1))
            tempo = cfg.base_tempo_by_energy.get(energy, 80)
            velocity = cfg.velocity_by_energy.get(energy, 70)
            density = cfg.density_by_energy.get(energy, 0.7)
            swing = cfg.swing_by_mood.get(mood, 0.5)
            register = cfg.register_by_mood.get(mood, cfg.register_by_mood["neutral"])

            progression = choose_progression(mood, cfg)
            chord = progression[bar_index % len(progression)]
            pattern = choose_pattern(energy, cfg)

            events = generate_bar(chord, pattern, register, density)

            sec_per_beat = 60.0 / max(tempo, 1)
            bar_start = time.time()
            for ev in events:
                beat_time = ev["offset_beats"] * sec_per_beat
                # basic swing: delay off-beats
                if ev["offset_beats"] % 1 != 0:
                    beat_time *= swing * 2
                target = bar_start + beat_time
                now = time.time()
                sleep_time = target - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                for n in ev["notes"]:
                    backend.note_on(n, velocity)
                time.sleep(ev["duration_beats"] * sec_per_beat)
                for n in ev["notes"]:
                    backend.note_off(n)

            # keep bar length consistent even if sparse
            elapsed = time.time() - bar_start
            bar_len = 4 * sec_per_beat
            if elapsed < bar_len:
                time.sleep(bar_len - elapsed)

            bar_index += 1
    except Exception as exc:
        print(f"[music] engine error: {exc}")
    finally:
        backend.close()
