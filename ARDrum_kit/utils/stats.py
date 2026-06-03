import os
import json
import time

class StatsManager:
    def __init__(self, filename="results/depth_comparison_analysis.json"):
        self.filename = filename
        self.depth_comparison_stats = []

    def add_depth_stats(self, stats_payload):
        self.depth_comparison_stats.append(stats_payload)

    def save_depth_comparison_json(self):
        if not self.depth_comparison_stats:
            print("[STATS] No comparison data gathered during this session (List is empty).")
            return

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        l_wrist_deltas = [entry["left_wrist"]["delta"] for entry in self.depth_comparison_stats]
        r_wrist_deltas = [entry["right_wrist"]["delta"] for entry in self.depth_comparison_stats]
        l_elbow_deltas = [entry["left_elbow"]["delta"] for entry in self.depth_comparison_stats]
        r_elbow_deltas = [entry["right_elbow"]["delta"] for entry in self.depth_comparison_stats]

        num_frames = len(self.depth_comparison_stats)
        
        start_time = self.depth_comparison_stats[0].get("timestamp_ms", 0)
        end_time = self.depth_comparison_stats[-1].get("timestamp_ms", 0)
        session_duration_sec = (end_time - start_time) / 1000.0

        summary = {
            "total_frames_recorded": num_frames,
            "session_duration_seconds": session_duration_sec,
            "average_frames_per_second": num_frames / session_duration_sec if session_duration_sec > 0 else 0,
            "metrics_in_meters": {
                "left_wrist_mean_delta": sum(l_wrist_deltas) / num_frames,
                "left_wrist_max_delta": max(l_wrist_deltas, key=abs),
                "right_wrist_mean_delta": sum(r_wrist_deltas) / num_frames,
                "right_wrist_max_delta": max(r_wrist_deltas, key=abs),
                "left_elbow_mean_delta": sum(l_elbow_deltas) / num_frames,
                "right_elbow_mean_delta": sum(r_elbow_deltas) / num_frames,
            }
        }

        output_payload = {
            "metadata": {
                "description": "Comparison between raw MediaPipe Z-coordinates and Kinematic IK Z-coordinates.",
                "timestamp": time.ctime()
            },
            "summary": summary,
            "timeseries_data": self.depth_comparison_stats
        }

        try:
            with open(self.filename, "w") as f:
                json.dump(output_payload, f, indent=4)
            print(f"\n[STATS] SUCCESS: Depth performance comparison exported to: {self.filename}")
            print(f"[STATS] ├─ Total Frames: {num_frames}")
            print(f"[STATS] ├─ Session Length: {session_duration_sec:.1f}s")
            print(f"[STATS] └─ Avg Left Wrist Delta: {summary['metrics_in_meters']['left_wrist_mean_delta']*100:.2f} cm")
        except Exception as e:
            print(f"[STATS] ERROR writing comparison file: {e}")