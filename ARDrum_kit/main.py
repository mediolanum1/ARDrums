import cv2
import time
import math

# ─── Vision & Threads ───
from vision.camera import CameraManager
from vision.pose_tracker import PoseTracker

# ─── Processing & Math ───
from processing.calibration import CalibrationManager
from processing.depth_processing import DepthManager
from processing.depth_estimator import KinematicDepthEstimator
from processing.wrist_processor import GestureWristProcessor
from processing.foot_processor import GestureFootProcessor
from processing.color_tip_tracker import ColorTipTracker

# ─── Assets & UI ───
from drums.drum import VirtualDrumKit
from ui.ui_renderer import UIRenderer

# ─── Utils ───
# (Assuming your stats.py contains a class or functions to handle JSON dumping)
try:
    from utils.stats import StatsManager
except ImportError:
    # Fallback if you haven't written the StatsManager class yet
    class StatsManager:
        def add_depth_stats(self, stats): pass
        def save_depth_comparison_json(self): print("[STATS] Saved.")


class AppState:
    """Holds global UI toggles and configuration state."""
    def __init__(self):
        self.show_drums = True
        self.show_drum_names = True
        self.show_pov = True
        self.stick_mode = False
        self.show_coords = False
        self.freeze_drums = True
        self.show_flow = False
        
        # Depth configuration
        self.depth_active = False # True = Use Neural Net, False = Use Kinematic
        self.depth_label = "Kinematic Depth Mode"
        self.depth_status_msg = "Running Kinematic IK"
        
        # Rhythm session
        self.rhythm_active = False


class ARDrumApp:
    def __init__(self):
        print("[INIT] Booting AR Drum Kit...")
        
        # 1. State Management
        self.state = AppState()
        self.running = True

        # 2. Hardware & Vision Threads
        self.camera = CameraManager()
        self.pose_tracker = PoseTracker(self.camera.frame_queue)
        
        # Calculate focal length for the UI and tracking
        self.focal_length = (self.camera.frame_width / 2) / math.tan(math.radians(60.0) / 2)
        
        # 3. Processors & Math
        self.calibration = CalibrationManager(self.camera.frame_width, self.camera.frame_height, self.focal_length)
        self.depth_estimator = KinematicDepthEstimator()
        self.depth_manager = DepthManager(self.camera.frame_width, self.camera.frame_height)
        
        self.left_arm = GestureWristProcessor("L")
        self.right_arm = GestureWristProcessor("R")
        self.foot = GestureFootProcessor("RF")
        
        self.tip_tracker_l = ColorTipTracker(color="orange", stick_length_m=0.406, focal_length=self.focal_length, frame_w=self.camera.frame_width, frame_h=self.camera.frame_height)
        self.tip_tracker_r = ColorTipTracker(color="pink", stick_length_m=0.406, focal_length=self.focal_length, frame_w=self.camera.frame_width, frame_h=self.camera.frame_height)
        
        self.stats = StatsManager()

        # 4. Assets & UI
        self.kit = VirtualDrumKit()
        self.ui = UIRenderer(self.camera.frame_width, self.camera.frame_height, self.focal_length)

    def start(self):
        """Starts background threads and enters the main loop."""
        self.camera.start()
        self.pose_tracker.start()
        self.main_loop()

    def main_loop(self):
        """The core application loop."""
        print("[INIT] Entering Main Loop...")
        
        while self.running:
            # 1. Fetch Data
            result_data = self.pose_tracker.get_latest_result()
            if not result_data:
                continue
                
            image, result, cur_time = result_data
            cur_time_ms = int(cur_time * 1000)
            dims = (self.camera.frame_width, self.camera.frame_height)

            # Dictionary to pass required state data to the decoupled UI Renderer
            render_state = self._build_render_state(cur_time)

            if result.pose_landmarks and result.pose_world_landmarks:
                s_lm = result.pose_landmarks[0]
                w_lm = result.pose_world_landmarks[0]

                # ─── A. CALIBRATION PHASE ───
                if not self.calibration.is_calibrated:
                    self.calibration.update(s_lm, w_lm, self.depth_estimator)
                    
                    # Draw countdown UI directly onto the camera image
                    status_txt, err_txt = self.calibration.get_ui_text()
                    if status_txt:
                        cv2.putText(image, status_txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if err_txt:
                        cv2.putText(image, err_txt, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                    self.ui.draw_combined_view(image, None, None)

                # ─── B. DRUMMING PHASE ───
                else:
                    # Update dynamic UI state (like distance)
                    current_sw_px = math.hypot((s_lm[11].x - s_lm[12].x) * dims[0], (s_lm[11].y - s_lm[12].y) * dims[1])
                    render_state["dist_m"] = self.calibration.get_current_distance(current_sw_px)
                    live_metric_to_px_scale = self.calibration.get_live_metric_to_px_scale(current_sw_px)
                    self.calibration.metric_to_px_scale = live_metric_to_px_scale

                    # 1. Depth Processing (Kinematic Override)
                    w_lm_eff, stats_payload = self.depth_manager.process_kinematic_depth(
                        s_lm, w_lm, self.depth_estimator, live_metric_to_px_scale
                    )
                    self.stats.add_depth_stats(stats_payload)

                    # 2. Stick Tracking Update (Optional visual overlay)
                    wrist_l_world = (w_lm_eff[15].x, w_lm_eff[15].y, w_lm_eff[15].z)
                    wrist_r_world = (w_lm_eff[16].x, w_lm_eff[16].y, w_lm_eff[16].z)
                    tip_3d_l, tip_px_l, _ = self.tip_tracker_l.update(image, wrist_l_world)
                    tip_3d_r, tip_px_r, _ = self.tip_tracker_r.update(image, wrist_r_world)

                    if self.state.stick_mode:
                        self.tip_tracker_l.draw_debug(image, tip_px_l, tip_3d_l, (0, 200, 255))
                        self.tip_tracker_r.draw_debug(image, tip_px_r, tip_3d_r, (255, 200, 0))

                    # 3. Hit Detection (Arms & Foot)
                    hit_l, dbg_l = self.left_arm.process(
                        s_lm[15], w_lm_eff[15], s_lm[11], w_lm_eff[11], s_lm[13], 1.0, 
                        self.kit, cur_time_ms, dims, s_lm[12], mediapipe_present=True
                    )
                    hit_r, dbg_r = self.right_arm.process(
                        s_lm[16], w_lm_eff[16], s_lm[12], w_lm_eff[12], s_lm[14], 1.0, 
                        self.kit, cur_time_ms, dims, s_lm[11], mediapipe_present=True
                    )
                    hit_foot, dbg_foot = self.foot.process(
                        s_lm[27], w_lm_eff[27], s_lm[23], s_lm[24], 
                        self.kit, cur_time_ms, dims, mediapipe_present=(s_lm[27].visibility > 0.3)
                    )

                    # 4. Inject hit data into render state
                    render_state["dbg_l"] = dbg_l
                    render_state["dbg_r"] = dbg_r
                    render_state["dbg_foot"] = dbg_foot
                    if hit_l: render_state["last_l_hit_time"] = cur_time
                    if hit_r: render_state["last_r_hit_time"] = cur_time
                    if hit_foot: render_state["last_foot_hit_time"] = cur_time

                    # 5. Render final frames
                    pov_canvas = None
                    if self.state.show_pov:
                        pov_canvas = self.ui.render_pov_canvas(
                            self.kit, w_lm_eff, cur_time, dbg_l, dbg_r, dbg_foot, 
                            self.calibration.fixed_sw_m, render_state
                        )
                    
                    controls_panel = self.ui.build_controls_panel(render_state)
                    self.ui.draw_combined_view(image, pov_canvas, controls_panel)
            else:
                # No landmarks detected
                self.ui.draw_combined_view(image, None, None)

            # 2. Input Handling
            self._handle_input()

    def _build_render_state(self, cur_time):
        """Packages application state into a dictionary for the UI Renderer."""
        # Note: getattr handles mapping your state attributes safely
        return {
            "cur_time": cur_time,
            "show_drums": self.state.show_drums,
            "show_drum_names": self.state.show_drum_names,
            "show_pov": self.state.show_pov,
            "stick_mode": self.state.stick_mode,
            "show_coords": self.state.show_coords,
            "freeze_drums": self.state.freeze_drums,
            "show_flow": self.state.show_flow,
            "depth_active": self.state.depth_active,
            "depth_label": self.state.depth_label,
            "depth_state": self.state.depth_active,
            "depth_status_msg": self.state.depth_status_msg,
            "rhythm_active": self.state.rhythm_active,
            
            # These will be updated dynamically in the main loop if hits occur
            "last_l_hit_time": getattr(self, "_last_l_hit", 0),
            "last_r_hit_time": getattr(self, "_last_r_hit", 0),
            "last_foot_hit_time": getattr(self, "_last_foot_hit", 0),
        }

    def _handle_input(self):
        """Keyboard event listener."""
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            self.running = False
        elif key == ord("f"): self.state.freeze_drums = not self.state.freeze_drums
        elif key == ord("d"): self.state.show_drums = not self.state.show_drums
        elif key == ord("c"): self.state.show_coords = not self.state.show_coords
        elif key == ord("n"): self.state.show_drum_names = not self.state.show_drum_names
        elif key == ord("p"): self.state.show_pov = not self.state.show_pov
        elif key == ord("o"): self.state.show_flow = not self.state.show_flow
        elif key == ord("s"): 
            self.state.stick_mode = not self.state.stick_mode
            # Inform the drum kit if stick mode changed (optional based on your kit logic)
            if hasattr(self.kit, 'use_sticks'):
                self.kit.use_sticks = self.state.stick_mode

    def cleanup(self):
        """Closes threads, saves data, and releases hardware."""
        print("[CLEANUP] Saving statistics and closing threads...")
        self.camera.stop()
        self.pose_tracker.stop()
        
        # Save your depth analysis JSON file!
        if hasattr(self.stats, 'save_depth_comparison_json'):
            self.stats.save_depth_comparison_json()
            
        self.kit.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = ARDrumApp()
    try:
        app.start()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()