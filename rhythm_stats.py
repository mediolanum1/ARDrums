"""
rhythm_stats.py  —  Two-phase metronome rhythm-accuracy session for AR Drum.

Overview
--------
Plays a metronome at BPM_SLOW for PHASE_DURATION_S seconds, then at BPM_FAST
for another PHASE_DURATION_S seconds.  The user is expected to strike the
Hi-Hat on every beat with one hand.

Stats collected per phase
-------------------------
  expected_beats      — total metronome ticks in the phase
  triggers_2d         — times the 2-D state machine decided to attempt a hit
                        (speed + downward-motion thresholds met → just before
                        check_line_intersection is called)
  hits_3d             — times check_line_intersection confirmed "Hi-Hat"
  mismatches_no_hit   — 2D triggered, 3D found NO drum at all (None)
  mismatches_wrong_drum — 2D triggered, 3D hit a different drum (not Hi-Hat)
  hit_rate            — hits_3d / expected_beats
  on_beat_rate        — fraction of 3D hits within ±BEAT_WINDOW_MS of a beat
  mean_offset_ms      — mean signed timing error (+late, −early)
  std_offset_ms       — std-dev of timing errors (consistency)
  avg_2d_trigger_px   — mean screen-space coords when 2D fires
  avg_3d_hit_px       — mean screen-space coords on confirmed Hi-Hat hits
  avg_3d_hit_world    — mean Kalman-filtered world coords on confirmed hits

Integration (minimal changes)
------------------------------
1.  In GestureWristProcessor.process(), add an optional `rhythm_session=None`
    parameter and insert two hook calls inside the DOWN-state hit block:

        # >>> BEFORE check_line_intersection:
        if rhythm_session is not None:
            rhythm_session.on_2d_trigger(wrist_px, curr_3d_coords, cur_time_ms)

        hit_detected = kit.check_line_intersection(...)

        # >>> AFTER check_line_intersection:
        if rhythm_session is not None:
            rhythm_session.on_3d_result(hit_detected, wrist_px,
                                        curr_3d_coords, cur_time_ms)

2.  In ARDrumApp, create and start a session, then pass it to the processor:

        self.rhythm_session = RhythmSession()
        self.rhythm_session.start()

        # in main_render_loop, pass to left_arm.process(..., rhythm_session=self.rhythm_session)

3.  When the session ends:

        if not self.rhythm_session.is_active:
            self.rhythm_session.print_summary()
            self.rhythm_session.save("results/rhythm_session.json")
"""

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pygame


# ── Tuneable constants ────────────────────────────────────────────────────────

TARGET_DRUM      = "Hi-Hat"      # the only drum that counts toward hits
PHASE_DURATION_S = 15.0          # seconds per phase
BPM_SLOW         = 60            # phase-1 tempo
BPM_FAST         = 90            # phase-2 tempo
BEAT_WINDOW_MS   = 150           # ±ms: "on beat" tolerance window
COUNTDOWN_S      = 5.0           # warmup countdown before slow phase
FAST_COUNTDOWN_S = 4.0           # pre-fast countdown before 90 BPM
BREAK_S          = 3.0           # rest break between phases in seconds

# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class HitEvent:
    """Single timestamped wrist event (2D trigger or 3D confirmed hit)."""
    time_ms:        float
    px:             float          # screen x (pixels)
    py:             float          # screen y (pixels)
    wx:             float          # Kalman world x (metres)
    wy:             float          # Kalman world y (metres)
    wz:             float          # Kalman world z (metres)
    beat_offset_ms: Optional[float] = None   # None for 2D-only events


@dataclass
class PhaseStats:
    """All data for one metronome phase."""
    label:          str
    bpm:            float
    duration_s:     float
    expected_beats: int = 0

    # raw event lists (append-only, written from main thread)
    triggers_2d:      List[HitEvent] = field(default_factory=list)
    hits_3d:          List[HitEvent] = field(default_factory=list)
    # mismatch counters
    mismatches_no_hit:     int = 0   # 2D fired, 3D returned None
    mismatches_wrong_drum: int = 0   # 2D fired, 3D hit a *different* drum

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def n_2d(self) -> int:
        return len(self.triggers_2d)

    @property
    def n_3d(self) -> int:
        return len(self.hits_3d)

    @property
    def hit_rate(self) -> float:
        return self.n_3d / self.expected_beats if self.expected_beats else 0.0

    @property
    def avg_2d_px(self) -> Optional[Tuple[float, float]]:
        if not self.triggers_2d:
            return None
        return (
            sum(e.px for e in self.triggers_2d) / len(self.triggers_2d),
            sum(e.py for e in self.triggers_2d) / len(self.triggers_2d),
        )

    @property
    def avg_3d_px(self) -> Optional[Tuple[float, float]]:
        if not self.hits_3d:
            return None
        return (
            sum(e.px for e in self.hits_3d) / len(self.hits_3d),
            sum(e.py for e in self.hits_3d) / len(self.hits_3d),
        )

    @property
    def avg_3d_world(self) -> Optional[Tuple[float, float, float]]:
        if not self.hits_3d:
            return None
        return (
            sum(e.wx for e in self.hits_3d) / len(self.hits_3d),
            sum(e.wy for e in self.hits_3d) / len(self.hits_3d),
            sum(e.wz for e in self.hits_3d) / len(self.hits_3d),
        )

    @property
    def beat_offsets(self) -> List[float]:
        return [e.beat_offset_ms for e in self.hits_3d
                if e.beat_offset_ms is not None]

    @property
    def mean_offset_ms(self) -> Optional[float]:
        offs = self.beat_offsets
        return sum(offs) / len(offs) if offs else None

    @property
    def std_offset_ms(self) -> Optional[float]:
        offs = self.beat_offsets
        if len(offs) < 2:
            return None
        mean = sum(offs) / len(offs)
        return math.sqrt(sum((x - mean) ** 2 for x in offs) / len(offs))

    @property
    def on_beat_count(self) -> int:
        return sum(1 for x in self.beat_offsets if abs(x) <= BEAT_WINDOW_MS)

    @property
    def on_beat_rate(self) -> float:
        offs = self.beat_offsets
        return self.on_beat_count / len(offs) if offs else 0.0


# ── Main session class ────────────────────────────────────────────────────────

class RhythmSession:
    """
    Two-phase metronome timing-accuracy test focused on Hi-Hat hits.

    Thread model
    ------------
    • A daemon background thread (_run) drives the metronome clock and updates
      _beat_times_ms + _current_phase under _lock.
    • The main render thread calls on_2d_trigger / on_3d_result sequentially
      (they are never concurrent with each other), so _pending_2d needs no
      extra guard beyond the normal Python GIL — but the lock is still used
      consistently to read shared state.
    """

    def __init__(
        self,
        bpm_slow:         float = BPM_SLOW,
        bpm_fast:         float = BPM_FAST,
        phase_duration_s: float = PHASE_DURATION_S,
        beat_window_ms:   float = BEAT_WINDOW_MS,
        click_sound_path: Optional[str] = None,
        countdown_s:      float = COUNTDOWN_S,
        fast_countdown_s: float = FAST_COUNTDOWN_S,
        break_s:          float = BREAK_S,
    ):
        self._phases_cfg = [
            ("Slow", float(bpm_slow),  float(phase_duration_s)),
            ("Fast", float(bpm_fast),  float(phase_duration_s)),
        ]
        self._beat_window_ms = beat_window_ms
        self._countdown_s = countdown_s
        self._fast_countdown_s = fast_countdown_s
        self._break_s     = break_s

        # Populated as each phase begins
        self._phases: List[PhaseStats] = []
        self._current_phase: Optional[PhaseStats] = None
        self._beat_times_ms: List[float] = []   # beats fired so far in this phase
        self._phase_end_time: float      = 0.0
        self._countdown_end_time: float = 0.0
        self._in_countdown: bool        = False

        self._session_active = False
        self._lock           = threading.Lock()

        # Between on_2d_trigger → on_3d_result; always overwritten before read
        self._pending_2d: Optional[HitEvent] = None

        self._click_sound = self._make_click(click_sound_path)

    # ── Sound synthesis ───────────────────────────────────────────────────────

    @staticmethod
    def _make_click(path: Optional[str]) -> Optional[pygame.mixer.Sound]:
        """Load a click from file, or synthesise a short 1 kHz tone if absent."""
        if path and os.path.exists(path):
            try:
                return pygame.mixer.Sound(path)
            except Exception as exc:
                print(f"[RHYTHM] Could not load click '{path}': {exc}")

        # Synthesise: 22 ms sine burst with exponential decay
        try:
            sr  = 44100
            dur = 0.022
            n   = int(sr * dur)
            t   = np.linspace(0, dur, n, endpoint=False)
            wave = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 180) * 32767).astype(np.int16)
            stereo = np.column_stack([wave, wave])
            return pygame.sndarray.make_sound(stereo)
        except Exception as exc:
            print(f"[RHYTHM] Could not synthesise click: {exc}")
            return None

    # ── Public lifecycle ──────────────────────────────────────────────────────

    def start(self):
        """Start the two-phase session.  Returns immediately; runs in background."""
        if self._session_active:
            return
        self._session_active = True
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def is_active(self) -> bool:
        return self._session_active

    # ── Status helpers (safe to call every frame) ─────────────────────────────

    def phase_label(self) -> str:
        with self._lock:
            return self._current_phase.label if self._current_phase else ""

    def current_bpm(self) -> float:
        with self._lock:
            return self._current_phase.bpm if self._current_phase else 0.0

    def time_remaining_s(self) -> float:
        return max(0.0, self._phase_end_time - time.time())

    def overlay_text(self) -> str:
        """One-liner for camera overlay: e.g. 'RHYTHM  Slow 60 BPM  12.3 s'"""
        if not self._session_active:
            return ""

        with self._lock:
            in_countdown = self._in_countdown
            countdown_rem = max(0.0, self._countdown_end_time - time.time())

        if in_countdown:
            count_int = max(1, int(countdown_rem) + 1)
            return f"RHYTHM  Starting in {count_int}..."

        label = self.phase_label()
        bpm   = self.current_bpm()
        rem   = self.time_remaining_s()
        return f"RHYTHM  {label} {int(bpm)} BPM  {rem:.1f} s"

    def _run_countdown(self, bpm: float, duration_s: float, description: str):
        """Run a click countdown for the specified tempo and exact duration."""
        interval_s = 60.0 / bpm
        start = time.time()
        end = start + duration_s

        print(f"[RHYTHM] ⏱ {description} ({bpm:.0f} BPM) for {duration_s:.0f} s...")

        # Schedule all exact-multiple clicks, then ensure a final click at countdown end.
        click_times = []
        i = 1
        while True:
            target_t = start + i * interval_s
            if target_t >= end:
                break
            click_times.append(target_t)
            i += 1
        click_times.append(end)

        for target_t in click_times:
            sleep_t = target_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

            if self._click_sound:
                self._click_sound.play()

        with self._lock:
            self._countdown_end_time = end

        return end

    # ── Background metronome thread ───────────────────────────────────────────

    def _run(self):
        # Initial warmup countdown at slow tempo before the first phase.
        with self._lock:
            self._in_countdown = True
            self._countdown_end_time = time.time() + self._countdown_s
        self._run_countdown(BPM_SLOW, self._countdown_s, "Warmup countdown")
        with self._lock:
            self._in_countdown = False

        for phase_idx, (label, bpm, dur) in enumerate(self._phases_cfg):
            if label == "Fast":
                # Rest break before the fast phase.
                print(f"[RHYTHM] ⏸ Rest break {self._break_s:.0f} seconds...")
                with self._lock:
                    self._current_phase = None
                time.sleep(self._break_s)

                # Countdown at the fast tempo so the first fast beat starts immediately.
                with self._lock:
                    self._in_countdown = True
                    self._countdown_end_time = time.time() + self._fast_countdown_s
                self._run_countdown(bpm, self._fast_countdown_s, "Pre-fast countdown")
                with self._lock:
                    self._in_countdown = False

            beat_interval_s = 60.0 / bpm
            n_beats         = int(dur / beat_interval_s)

            phase_start = self._countdown_end_time if phase_idx in (0, 1) else time.time()

            phase = PhaseStats(
                label          = label,
                bpm            = bpm,
                duration_s     = dur,
                expected_beats = n_beats,
            )

            with self._lock:
                self._phases.append(phase)
                self._current_phase = phase
                self._beat_times_ms = []
                self._phase_end_time = phase_start + dur

            print(f"[RHYTHM] ▶ Phase '{label}' — {bpm:.0f} BPM  "
                  f"({n_beats} beats over {dur:.0f} s)")

            for beat_idx in range(n_beats):
                target_t = phase_start + beat_idx * beat_interval_s
                sleep_t  = target_t - time.time()
                if sleep_t > 0:
                    time.sleep(sleep_t)

                beat_ms = time.time() * 1000.0
                with self._lock:
                    self._beat_times_ms.append(beat_ms)

                if self._click_sound:
                    # Skip playing duplicate click for the first beat immediately after a countdown.
                    if not (beat_idx == 0 and phase_idx in (0, 1)):
                        self._click_sound.play()

            # Drain remaining phase time (last beat may fire slightly early)
            leftover = self._phase_end_time - time.time()
            if leftover > 0:
                time.sleep(leftover)

        print("[RHYTHM] ✓ Session complete.")
        with self._lock:
            self._session_active = False
            self._current_phase  = None

    # ── Event hooks — call from GestureWristProcessor ────────────────────────

    def on_2d_trigger(
        self,
        wrist_px: Tuple[float, float],
        wrist_3d: Tuple[float, float, float],
        time_ms:  float,
    ):
        """
        Call this just *before* kit.check_line_intersection().
        Records that the 2-D state machine decided to attempt a hit.
        """
        with self._lock:
            phase = self._current_phase
        if phase is None:
            return

        evt = HitEvent(
            time_ms = time_ms,
            px      = wrist_px[0],
            py      = wrist_px[1],
            wx      = wrist_3d[0],
            wy      = wrist_3d[1],
            wz      = wrist_3d[2],
        )
        phase.triggers_2d.append(evt)   # main thread only — no lock needed
        self._pending_2d = evt

    def on_3d_result(
        self,
        drum_name: Optional[str],   # return value of check_line_intersection
        wrist_px:  Tuple[float, float],
        wrist_3d:  Tuple[float, float, float],
        time_ms:   float,
    ):
        """
        Call this immediately *after* kit.check_line_intersection() returns.
        Pass its return value as drum_name (None if no drum was hit).

        Outcome logic:
          drum_name == TARGET_DRUM  → confirmed hit; compute beat offset
          drum_name is None         → mismatch_no_hit  (3D found nothing)
          drum_name != TARGET_DRUM  → mismatch_wrong_drum (hit elsewhere)
        """
        with self._lock:
            phase      = self._current_phase
            beat_times = list(self._beat_times_ms)  # snapshot for offset calc

        # Guard: session may have just ended between trigger and result
        if phase is None or self._pending_2d is None:
            self._pending_2d = None
            return

        self._pending_2d = None   # always consumed

        if drum_name == TARGET_DRUM:
            # Signed offset to nearest beat: positive = late, negative = early
            offset = None
            if beat_times:
                nearest = min(beat_times, key=lambda b: abs(b - time_ms))
                offset  = time_ms - nearest

            evt = HitEvent(
                time_ms        = time_ms,
                px             = wrist_px[0],
                py             = wrist_px[1],
                wx             = wrist_3d[0],
                wy             = wrist_3d[1],
                wz             = wrist_3d[2],
                beat_offset_ms = offset,
            )
            phase.hits_3d.append(evt)

        elif drum_name is None:
            phase.mismatches_no_hit += 1

        else:
            # A real drum was hit, just not the Hi-Hat
            phase.mismatches_wrong_drum += 1

    # ── Results ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a fully serialisable dict of all collected stats."""
        with self._lock:
            phases = list(self._phases)

        out = {
            "target_drum":    TARGET_DRUM,
            "beat_window_ms": self._beat_window_ms,
            "phases": [],
        }

        for p in phases:
            a2 = p.avg_2d_px
            a3 = p.avg_3d_px
            aw = p.avg_3d_world
            mo = p.mean_offset_ms
            so = p.std_offset_ms

            out["phases"].append({
                "label":                p.label,
                "bpm":                  p.bpm,
                "duration_s":           p.duration_s,
                # ── counts ──────────────────────────────────────────────
                "expected_beats":       p.expected_beats,
                "triggers_2d":          p.n_2d,
                "hits_3d":              p.n_3d,
                "mismatches_no_hit":    p.mismatches_no_hit,
                "mismatches_wrong_drum":p.mismatches_wrong_drum,
                # ── rates ───────────────────────────────────────────────
                "hit_rate":             round(p.hit_rate, 4),
                "on_beat_count":        p.on_beat_count,
                "on_beat_rate":         round(p.on_beat_rate, 4),
                # ── timing accuracy ──────────────────────────────────────
                "mean_offset_ms":       round(mo, 2) if mo is not None else None,
                "std_offset_ms":        round(so, 2) if so is not None else None,
                # ── spatial averages ──────────────────────────────────────
                "avg_2d_trigger_px":    {"x": round(a2[0], 1), "y": round(a2[1], 1)} if a2 else None,
                "avg_3d_hit_px":        {"x": round(a3[0], 1), "y": round(a3[1], 1)} if a3 else None,
                "avg_3d_hit_world":     {
                    "x": round(aw[0], 4),
                    "y": round(aw[1], 4),
                    "z": round(aw[2], 4),
                } if aw else None,
            })

        return out

    def print_summary(self):
        """Pretty-print session results to stdout."""
        data = self.summary()
        BAR  = "─" * 56
        print(f"\n{'═' * 56}")
        print(f"  RHYTHM SESSION  —  target: {data['target_drum']}")
        print(f"  Beat window: ±{data['beat_window_ms']} ms")
        print(f"{'═' * 56}")

        for p in data["phases"]:
            print(f"\n  ▸ Phase: {p['label']}  ({p['bpm']:.0f} BPM, {p['duration_s']:.0f} s)")
            print(BAR)
            print(f"  {'Expected beats':<28}: {p['expected_beats']}")
            print(f"  {'2D triggers':<28}: {p['triggers_2d']}")
            print(f"  {'3D confirmed hits':<28}: {p['hits_3d']}")
            print(f"  {'2D-yes / 3D-nothing':<28}: {p['mismatches_no_hit']}")
            print(f"  {'2D-yes / 3D-wrong drum':<28}: {p['mismatches_wrong_drum']}")
            print(BAR)
            print(f"  {'Hit rate':<28}: {p['hit_rate']*100:.1f}%")
            print(f"  {'On-beat hits':<28}: {p['on_beat_count']}  "
                  f"({p['on_beat_rate']*100:.1f}%)")

            mo = p["mean_offset_ms"]
            so = p["std_offset_ms"]
            if mo is not None:
                direction = "late" if mo >= 0 else "early"
                so_str    = f"  σ={so:.1f} ms" if so is not None else ""
                print(f"  {'Mean beat offset':<28}: {abs(mo):.1f} ms {direction}{so_str}")
            else:
                print(f"  {'Mean beat offset':<28}: n/a")

            if p["avg_2d_trigger_px"]:
                c = p["avg_2d_trigger_px"]
                print(f"  {'Avg 2D trigger (px)':<28}: ({c['x']:.0f}, {c['y']:.0f})")
            if p["avg_3d_hit_px"]:
                c = p["avg_3d_hit_px"]
                print(f"  {'Avg 3D hit (px)':<28}: ({c['x']:.0f}, {c['y']:.0f})")
            if p["avg_3d_hit_world"]:
                w = p["avg_3d_hit_world"]
                print(f"  {'Avg 3D hit (world m)':<28}: "
                      f"x={w['x']:.3f}  y={w['y']:.3f}  z={w['z']:.3f}")

        print(f"\n{'═' * 56}\n")

    def save(self, path: str = "rhythm_session.json"):
        os.makedirs("results", exist_ok=True)
        """Persist summary as JSON.  Creates parent directories if needed."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"[RHYTHM] Results saved → {path}")