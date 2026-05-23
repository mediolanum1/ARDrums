"""
compare_stats.py
────────────────
Compare drum_stats_*.json files collected on different machines.

Usage:
    python compare_stats.py                          # auto-find all JSONs in .
    python compare_stats.py stats_A.json stats_B.json
    python compare_stats.py --dir /path/to/stats/
"""

import json
import sys
import os
import glob
import argparse
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

W_KEY   = 36   # label column width
W_COL   = 18   # value column width

def _h(label: str) -> str:
    return f"\n{'─'*6}  {label.upper()}  {'─'*(80 - len(label) - 10)}"

def _row(label: str, *values) -> str:
    left = f"  {label:<{W_KEY}}"
    cols = "".join(f"{str(v) if v is not None else 'n/a':>{W_COL}}" for v in values)
    return left + cols

def _flag(val_a, val_b, tol_pct: float = 15.0) -> str:
    """Return ⚠  if values differ by more than tol_pct percent."""
    try:
        a, b = float(val_a), float(val_b)
        if a == 0 and b == 0:
            return ""
        denom = max(abs(a), abs(b))
        if abs(a - b) / denom * 100 > tol_pct:
            return "  ⚠"
    except (TypeError, ValueError):
        pass
    return ""

def _get(d: dict, *keys, default=None) -> Any:
    """Safe nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare(sessions: list[dict], labels: list[str]):
    n = len(sessions)

    def row(label, *paths_and_fmt):
        """
        paths_and_fmt: alternating (key_path_tuple, format_string) pairs.
        key_path_tuple can be a tuple of keys for nested access.
        """
        # Simple usage: row("label", ("key1","key2"), ".1f")
        keys, fmt = paths_and_fmt[0], paths_and_fmt[1]
        vals = [_get(s, *keys) for s in sessions]
        formatted = []
        for v in vals:
            if v is None:
                formatted.append("n/a")
            else:
                try:
                    formatted.append(format(float(v), fmt))
                except (ValueError, TypeError):
                    formatted.append(str(v))
        flag = _flag(vals[0], vals[1]) if n == 2 else ""
        print(_row(label, *formatted) + flag)

    def header(title):
        print(_h(title))
        print(_row("", *[f"[{l}]" for l in labels]))
        print("  " + "─" * (W_KEY + W_COL * n + 2))

    # ── Session overview ───────────────────────────────────────────────────────
    header("Session overview")
    print(_row("Session ID",         *[s.get("session_id","?") for s in sessions]))
    print(_row("Duration (s)",        *[f"{s.get('session_duration_s',0):.1f}" for s in sessions]))
    print(_row("Platform",            *[_get(s,"environment","platform","?") for s in sessions]))
    print(_row("Hostname",            *[_get(s,"environment","hostname","?") for s in sessions]))
    print(_row("Timestamp UTC",       *[_get(s,"environment","timestamp_utc","?") for s in sessions]))

    # ── Camera / calibration ───────────────────────────────────────────────────
    header("Camera & calibration")
    row("Resolution (w×h)",
        ("calibration", "frame_width"), "d")   # quick hack — just show width
    # Show both dimensions properly:
    res_vals = [
        f"{_get(s,'calibration','frame_width','?')}×{_get(s,'calibration','frame_height','?')}"
        for s in sessions
    ]
    print(_row("Resolution",          *res_vals))
    row("Focal length (px)",           ("calibration","focal_length_px"),  ".1f")
    row("Assumed FOV (°)",             ("calibration","assumed_fov_deg"),   ".1f")
    row("Effective FOV (°)",           ("calibration","effective_fov_deg"), ".2f")
    row("Shoulder width (m)",          ("calibration","fixed_sw_m"),        ".4f")
    row("Shoulder width (px) @ calib", ("calibration","fixed_sw_px"),       ".1f")
    row("Camera dist (m)",             ("calibration","cam_dist_m"),        ".3f")
    row("Z offset left wrist",         ("calibration","z_offset_l"),        ".4f")
    row("Z offset right wrist",        ("calibration","z_offset_r"),        ".4f")

    # ── Drum depth scales ──────────────────────────────────────────────────────
    header("Drum depth scales  (1.0 = at camera plane)")
    all_drums = set()
    for s in sessions:
        all_drums |= set(_get(s, "calibration", "drum_depth_scales", default={}).keys())
    for drum in sorted(all_drums):
        vals = [_get(s, "calibration", "drum_depth_scales", drum) for s in sessions]
        formatted = [f"{v:.4f}" if v is not None else "n/a" for v in vals]
        flag = _flag(vals[0], vals[1]) if n == 2 else ""
        print(_row(drum, *formatted) + flag)

    # ── Frame pipeline ─────────────────────────────────────────────────────────
    header("Frame pipeline")
    row("Total frames",                ("frames","total"),              "d")
    row("Pose-detected frames",        ("frames","with_pose"),          "d")
    row("Pose detection rate",         ("frames","pose_rate"),          ".1%")
    row("Queue drops",                 ("frames","queue_drops"),        "d")
    row("Render FPS (mean)",           ("frames","render_fps_mean"),    ".1f")
    row("Render FPS (p5)",             ("frames","render_fps_p5"),      ".1f")
    row("Render FPS (p95)",            ("frames","render_fps_p95"),     ".1f")
    row("Pipeline latency mean (ms)",  ("frames","pipeline_latency_ms","mean"), ".1f")
    row("Pipeline latency p95 (ms)",   ("frames","pipeline_latency_ms","p95"),  ".1f")
    row("Pipeline latency max (ms)",   ("frames","pipeline_latency_ms","max"),  ".1f")

    # ── Shoulder width stability ────────────────────────────────────────────────
    header("Shoulder width (px) — stability proxy")
    row("Mean",  ("shoulder_width_px","mean"),  ".1f")
    row("Stdev", ("shoulder_width_px","stdev"), ".2f")
    row("Min",   ("shoulder_width_px","min"),   ".1f")
    row("Max",   ("shoulder_width_px","max"),   ".1f")

    # ── Landmark visibility ────────────────────────────────────────────────────
    header("Landmark visibility (0=invisible, 1=perfect)")
    row("Wrist left  mean",  ("landmark_visibility","wrist_l_mean"), ".3f")
    row("Wrist right mean",  ("landmark_visibility","wrist_r_mean"), ".3f")

    # ── Wrist Z distribution ───────────────────────────────────────────────────
    header("Wrist Z (normalised by shoulder width) — depth behaviour")
    for side, key in [("Left", "left"), ("Right", "right")]:
        for stat in ("mean", "stdev", "min", "max"):
            row(f"{side} wrist Z {stat}", ("wrist_z_normalised", key, stat), ".4f")

    # ── Hit statistics ─────────────────────────────────────────────────────────
    header("Hit statistics")
    row("Total hits",             ("hits","total"), "d")
    all_drums2 = set()
    for s in sessions:
        all_drums2 |= set(_get(s,"hits","per_drum",default={}).keys())
    for drum in sorted(all_drums2):
        vals = [_get(s,"hits","per_drum",drum) for s in sessions]
        formatted = [str(v) if v is not None else "0" for v in vals]
        print(_row(f"  Hits: {drum}", *formatted))

    # Hit latency (if recorded)
    lat_vals = [_get(s,"hits","hit_latency_ms","mean") for s in sessions]
    if any(v is not None for v in lat_vals):
        header("Hit latency (ms)")
        row("Mean", ("hits","hit_latency_ms","mean"), ".1f")
        row("p95",  ("hits","hit_latency_ms","p95"),  ".1f")

    # ── Legend ────────────────────────────────────────────────────────────────
    print("\n  ⚠  = values differ by >15% between sessions\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare AR Drum Kit statistics across sessions.")
    parser.add_argument("files", nargs="*", help="JSON stat files to compare")
    parser.add_argument("--dir",  default=".", help="Directory to search for drum_stats_*.json")
    args = parser.parse_args()

    if args.files:
        paths = args.files
    else:
        paths = sorted(glob.glob(os.path.join(args.dir, "drum_stats_*.json")))

    if not paths:
        print("No drum_stats_*.json files found. Run the app first.")
        sys.exit(1)

    if len(paths) == 1:
        print(f"Only one session found ({paths[0]}). Showing its summary:\n")
        with open(paths[0]) as f:
            data = json.load(f)
        print(json.dumps(data, indent=2))
        return

    sessions, labels = [], []
    for p in paths:
        with open(p) as f:
            sessions.append(json.load(f))
        labels.append(os.path.basename(p).replace("drum_stats_","").replace(".json",""))

    print(f"\nComparing {len(sessions)} sessions: {', '.join(labels)}\n")
    compare(sessions, labels)


if __name__ == "__main__":
    main()