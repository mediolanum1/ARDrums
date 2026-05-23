"""
stats_collector.py
──────────────────
Collects per-frame and per-session statistics for the AR Drum Kit app so
results can be compared across machines with different cameras/placements.

Usage (in main.py / ARDrumApp):
    from stats_collector import StatsCollector

    # In __init__:
    self.stats = StatsCollector()

    # After calibration (_calibrate):
    self.stats.record_calibration(...)

    # In main_render_loop, each processed frame:
    self.stats.record_frame(...)

    # On hit:
    self.stats.record_hit(...)

    # In start(), before cleanup:
    self.stats.save()
"""

import json
import time
import math
import platform
import subprocess
import os
import statistics
from collections import defaultdict
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_mean(lst):
    return statistics.mean(lst) if lst else None

def _safe_stdev(lst):
    return statistics.stdev(lst) if len(lst) >= 2 else None

def _safe_median(lst):
    return statistics.median(lst) if lst else None

def _percentile(lst, p):
    """Simple percentile without numpy."""
    if not lst:
        return None
    s = sorted(lst)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class StatsCollector:
    """
    Thread-safe-ish stats collector (writes are done from the render thread only).
    Call record_* methods from ARDrumApp; call save() at shutdown.
    """

    # How many raw frame records to keep in RAM before compacting.
    # Raise if you want full per-frame data; lower to save memory.
    MAX_RAW_FRAMES = 10_000

    def __init__(self, output_dir: str = "."):
        self.output_dir   = output_dir
        self.session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()

        # ── Environment (populated immediately) ───────────────
        self.env = self._collect_env()

        # ── Calibration snapshot ──────────────────────────────
        self.calibration: dict = {}

        # ── Per-frame raw lists (compacted periodically) ───────
        self._frame_times:          list[float] = []   # wall-clock seconds
        self._pipeline_latencies:   list[float] = []   # result_queue age (s)
        self._shoulder_widths_px:   list[float] = []   # _current_sw_px
        self._landmark_vis_l:       list[float] = []   # left wrist visibility
        self._landmark_vis_r:       list[float] = []   # right wrist visibility
        self._queue_drops:          int         = 0    # frames skipped (queue full)
        self._total_frames:         int         = 0
        self._pose_detected_frames: int         = 0

        # ── Hit events ────────────────────────────────────────
        # { drum_name: [hit_timestamp, ...] }
        self._hits: dict[str, list[float]] = defaultdict(list)

        # ── Arm Z statistics ──────────────────────────────────
        self._wrist_z_l: list[float] = []
        self._wrist_z_r: list[float] = []

        # ── Timing between consecutive frames ─────────────────
        self._last_frame_wall: float | None = None
        self._inter_frame_gaps: list[float] = []

        # ── Per-drum hit-to-detection latency ─────────────────
        # (time from wrist crossing threshold to audio trigger)
        self._hit_latencies: list[float] = []

        print(f"[StatsCollector] session={self.session_id}  out={output_dir}")

    # ──────────────────────────────────────────────────────────
    # Public record_* API
    # ──────────────────────────────────────────────────────────

    def record_calibration(
        self,
        *,
        focal_length: float,
        assumed_fov_deg: float,
        frame_width: int,
        frame_height: int,
        fixed_sw_m: float,
        fixed_sw_px: float,
        cam_dist_m: float,
        z_offset_l: float,
        z_offset_r: float,
        drum_depth_scales: dict,
    ):
        """Call once, right after _calibrate() succeeds."""
        self.calibration = {
            "timestamp":         time.time() - self.session_start,
            "focal_length_px":   focal_length,
            "assumed_fov_deg":   assumed_fov_deg,
            "frame_width":       frame_width,
            "frame_height":      frame_height,
            "fixed_sw_m":        fixed_sw_m,
            "fixed_sw_px":       fixed_sw_px,
            "cam_dist_m":        cam_dist_m,
            "z_offset_l":        z_offset_l,
            "z_offset_r":        z_offset_r,
            "drum_depth_scales": drum_depth_scales,
            # Derived: actual per-pixel FOV estimate
            "effective_fov_deg": math.degrees(
                2 * math.atan((frame_width / 2) / focal_length)
            ) if focal_length > 0 else None,
        }
        print(
            f"[StatsCollector] calibration recorded: "
            f"sw={fixed_sw_m:.3f}m  dist={cam_dist_m:.2f}m  "
            f"fov={self.calibration['effective_fov_deg']:.1f}°"
        )

    def record_frame(
        self,
        *,
        cur_time: float,
        pipeline_latency: float,         # seconds from capture → render
        shoulder_width_px: float,
        pose_detected: bool,
        wrist_vis_l: float = 0.0,        # mediapipe visibility score 0-1
        wrist_vis_r: float = 0.0,
        wrist_z_l: float   = 0.0,        # normalised world z
        wrist_z_r: float   = 0.0,
        queue_dropped: bool = False,
    ):
        """Call once per render-loop iteration."""
        self._total_frames += 1
        if pose_detected:
            self._pose_detected_frames += 1
        if queue_dropped:
            self._queue_drops += 1

        # Inter-frame gap
        if self._last_frame_wall is not None:
            gap = cur_time - self._last_frame_wall
            self._inter_frame_gaps.append(gap)
        self._last_frame_wall = cur_time

        # Only store richer data when pose is present
        if pose_detected:
            self._pipeline_latencies.append(pipeline_latency)
            self._shoulder_widths_px.append(shoulder_width_px)
            self._landmark_vis_l.append(wrist_vis_l)
            self._landmark_vis_r.append(wrist_vis_r)
            self._wrist_z_l.append(wrist_z_l)
            self._wrist_z_r.append(wrist_z_r)

        # Compact if RAM is filling up
        if len(self._inter_frame_gaps) > self.MAX_RAW_FRAMES:
            self._compact()

    def record_hit(self, drum_name: str, latency_s: float | None = None):
        """
        Call when a drum hit is confirmed.
        latency_s: optional time (seconds) from gesture crossing threshold
                   to hit confirmation, for responsiveness analysis.
        """
        self._hits[drum_name].append(time.time() - self.session_start)
        if latency_s is not None:
            self._hit_latencies.append(latency_s)

    def record_queue_drop(self):
        """Call from camera_thread or ai_thread when a frame is discarded."""
        self._queue_drops += 1

    # ──────────────────────────────────────────────────────────
    # Snapshot / live summary
    # ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Build a complete summary dict (no file I/O)."""
        elapsed = time.time() - self.session_start

        # Effective FPS from inter-frame gaps
        avg_gap = _safe_mean(self._inter_frame_gaps)
        render_fps = (1.0 / avg_gap) if avg_gap else None

        total_hits = sum(len(v) for v in self._hits.values())

        hit_counts = {k: len(v) for k, v in self._hits.items()}
        hit_timeline = {k: v for k, v in self._hits.items()}  # relative seconds

        return {
            "session_id":   self.session_id,
            "session_duration_s": round(elapsed, 2),

            # ── Environment ───────────────────────────────────
            "environment": self.env,

            # ── Calibration ───────────────────────────────────
            "calibration": self.calibration,

            # ── Frame pipeline ────────────────────────────────
            "frames": {
                "total":              self._total_frames,
                "with_pose":          self._pose_detected_frames,
                "pose_rate":          round(self._pose_detected_frames / max(self._total_frames, 1), 3),
                "queue_drops":        self._queue_drops,
                "render_fps_mean":    round(render_fps, 2) if render_fps else None,
                "render_fps_p5":      round(_percentile(self._inter_frame_gaps and
                                                        [1/g for g in self._inter_frame_gaps if g > 0], 5) or 0, 2),
                "render_fps_p95":     round(_percentile(self._inter_frame_gaps and
                                                        [1/g for g in self._inter_frame_gaps if g > 0], 95) or 0, 2),
                "pipeline_latency_ms": {
                    "mean":   round((_safe_mean(self._pipeline_latencies) or 0) * 1000, 1),
                    "p95":    round((_percentile(self._pipeline_latencies, 95) or 0) * 1000, 1),
                    "max":    round((max(self._pipeline_latencies) if self._pipeline_latencies else 0) * 1000, 1),
                },
            },

            # ── Shoulder width stability ───────────────────────
            # High stdev → user moved a lot or camera is shakier
            "shoulder_width_px": {
                "mean":   round(_safe_mean(self._shoulder_widths_px) or 0, 1),
                "stdev":  round(_safe_stdev(self._shoulder_widths_px) or 0, 1),
                "min":    round(min(self._shoulder_widths_px, default=0), 1),
                "max":    round(max(self._shoulder_widths_px, default=0), 1),
            },

            # ── Landmark quality ──────────────────────────────
            "landmark_visibility": {
                "wrist_l_mean": round(_safe_mean(self._landmark_vis_l) or 0, 3),
                "wrist_r_mean": round(_safe_mean(self._landmark_vis_r) or 0, 3),
                # low visibility → lighting / angle problem
            },

            # ── Wrist Z distribution ──────────────────────────
            # Tells you how far forward/back wrists travel in normalised coords.
            # Compare across sessions to see if depth calibration is consistent.
            "wrist_z_normalised": {
                "left":  {
                    "mean":  round(_safe_mean(self._wrist_z_l) or 0, 4),
                    "stdev": round(_safe_stdev(self._wrist_z_l) or 0, 4),
                    "min":   round(min(self._wrist_z_l, default=0), 4),
                    "max":   round(max(self._wrist_z_l, default=0), 4),
                },
                "right": {
                    "mean":  round(_safe_mean(self._wrist_z_r) or 0, 4),
                    "stdev": round(_safe_stdev(self._wrist_z_r) or 0, 4),
                    "min":   round(min(self._wrist_z_r, default=0), 4),
                    "max":   round(max(self._wrist_z_r, default=0), 4),
                },
            },

            # ── Hit statistics ────────────────────────────────
            "hits": {
                "total": total_hits,
                "per_drum": hit_counts,
                "timeline_s": hit_timeline,  # relative to session start
                "hit_latency_ms": {
                    "mean": round((_safe_mean(self._hit_latencies) or 0) * 1000, 1),
                    "p95":  round((_percentile(self._hit_latencies, 95) or 0) * 1000, 1),
                } if self._hit_latencies else None,
            },
        }

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save(self) -> str:
        """Write JSON to disk. Returns the file path."""
        data   = self.summary()
        fname  = f"drum_stats_{self.session_id}.json"
        fpath  = os.path.join(self.output_dir, fname)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[StatsCollector] saved → {fpath}")
        return fpath

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _compact(self):
        """Trim raw lists to last MAX_RAW_FRAMES//2 entries to free memory."""
        keep = self.MAX_RAW_FRAMES // 2
        self._inter_frame_gaps  = self._inter_frame_gaps[-keep:]
        self._pipeline_latencies = self._pipeline_latencies[-keep:]
        self._shoulder_widths_px = self._shoulder_widths_px[-keep:]
        self._landmark_vis_l    = self._landmark_vis_l[-keep:]
        self._landmark_vis_r    = self._landmark_vis_r[-keep:]
        self._wrist_z_l         = self._wrist_z_l[-keep:]
        self._wrist_z_r         = self._wrist_z_r[-keep:]

    @staticmethod
    def _collect_env() -> dict:
        env = {
            "platform":        platform.system(),
            "platform_release": platform.release(),
            "machine":         platform.machine(),
            "python_version":  platform.python_version(),
            "hostname":        platform.node(),
            "cpu_count":       os.cpu_count(),
            "timestamp_utc":   datetime.utcnow().isoformat(),
        }
        # Try to get camera device info on Linux
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "--list-devices"], stderr=subprocess.DEVNULL, text=True
            )
            env["v4l2_devices"] = out.strip()
        except Exception:
            pass
        # Try on macOS
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPCameraDataType"], stderr=subprocess.DEVNULL, text=True
            )
            env["macos_camera_info"] = out.strip()[:500]
        except Exception:
            pass
        return env