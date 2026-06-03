import cv2
import time
import math
from ARDrum_kit.vision.camera import CameraManager
from ARDrum_kit.vision.pose_tracker import PoseTracker
from ARDrum_kit.processing.calibration import CalibrationManager
from ARDrum_kit.processing.depth_processing import DepthManager
from ARDrum_kit.processing.depth_estimator import KinematicDepthEstimator
from ARDrum_kit.processing.wrist_processor import GestureWristProcessor
from ARDrum_kit.processing.foot_processor import GestureFootProcessor
from ARDrum_kit.processing.color_tip_tracker import ColorTipTracker
from ARDrum_kit.drums.drum import VirtualDrumKit
from ARDrum_kit.ui.ui_renderer import UIRenderer
try:
    from ARDrum_kit.utils.stats import StatsManager
except ImportError:
    class StatsManager:
        def add_depth_stats(self, stats): pass
        def save_depth_comparison_json(self): print("[STATS] Saved.")


class AppState:
    # this is app state for UI 
    def __init__(self):
        self.show_drums = True
        self.show_drum_names = True
        self.show_pov = True
        self.stick_mode = False
        self.show_coords = False
        self.freeze_drums = True
        self.show_flow = False
        self.show_latency = False
        self.enabled_drums = set()
        
        # Depth configuration
        self.depth_active = False 
        self.depth_label = "Kinematic Depth Mode"
        self.depth_status_msg = "Running Kinematic IK"
        
        self.rhythm_active = False


class ARDrumApp:
    def __init__(self):
        print("[INIT] Booting AR Drum Kit...")
        self.state = AppState()
        self.running = True

        self.camera = CameraManager()
        self.dims = (self.camera.frame_width, self.camera.frame_height)
        self.pose_tracker = PoseTracker(self.camera.frame_queue)
        
        self.focal_length = (self.camera.frame_width / 2) / math.tan(math.radians(60.0) / 2)
        
        self.calibration = CalibrationManager(self.camera.frame_width, self.camera.frame_height, self.focal_length)
        self.depth_estimator = KinematicDepthEstimator()
        self.depth_manager = DepthManager(self.camera.frame_width, self.camera.frame_height)
        
        self.left_arm = GestureWristProcessor("L")
        self.right_arm = GestureWristProcessor("R")
        self.foot = GestureFootProcessor("RF")
        
        self.tip_tracker_l = ColorTipTracker(color="orange", stick_length_m=0.406, focal_length=self.focal_length, frame_w=self.camera.frame_width, frame_h=self.camera.frame_height)
        self.tip_tracker_r = ColorTipTracker(color="pink", stick_length_m=0.406, focal_length=self.focal_length, frame_w=self.camera.frame_width, frame_h=self.camera.frame_height)
        
        self.stats = StatsManager()

        self.kit = VirtualDrumKit()
        self.state.enabled_drums = self.kit.enabled_drums
        self.ui = UIRenderer(self.camera.frame_width, self.camera.frame_height, self.focal_length)

        self._last_l_hit = 0
        self._last_r_hit = 0
        self._last_foot_hit = 0
        
        self._smoothed_sw_px = None

    def start(self):
        self.camera.start()
        self.pose_tracker.start()
        self.main_loop()

    def main_loop(self):
        print("[INIT] Entering Main Loop...")
        
        while self.running:
            result_data = self.pose_tracker.get_latest_result()
            if not result_data:
                self._handle_input()
                continue
                
            image, result, cur_time = result_data
            cur_time_ms = int(cur_time * 1000)
            

            render_state = self._build_render_state(cur_time)

            if result.pose_landmarks and result.pose_world_landmarks:
                s_lm = result.pose_landmarks[0]
                w_lm = result.pose_world_landmarks[0]

                # calibration phase 
                if not self.calibration.is_calibrated:
                    self.calibration.update(s_lm, w_lm, self.depth_estimator)
                    
                   # get the text to display depedning on calibration state 
                    status_txt, err_txt = self.calibration.get_ui_text()
                    if status_txt:
                        cv2.putText(image, status_txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if err_txt:
                        cv2.putText(image, err_txt, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                    self.ui.draw_combined_view(image, None, None)

                # main running code aftrer calibration
                else:
                    raw_sw_px = max(1.0, math.hypot((s_lm[11].x - s_lm[12].x) * self.dims[0], (s_lm[11].y - s_lm[12].y) * self.dims[1]))
     
                    if self._smoothed_sw_px is None:
                        self._smoothed_sw_px = raw_sw_px
                    else:
                       # self._smoothed_sw_px = (self._smoothed_sw_px * 0.8) + (raw_sw_px * 0.2)
                        self._smoothed_sw_px = (self._smoothed_sw_px * 0.85) + (raw_sw_px * 0.15)

                    render_state["dist_m"] = self.calibration.get_current_distance(self._smoothed_sw_px)
                    self.calibration.metric_to_px_scale = (
                        self._smoothed_sw_px / self.calibration.fixed_sw_m if self.calibration.fixed_sw_m > 0 else 1.0
                    )
                    
                    # this is only if drums are not frozen 
                    if not self.state.freeze_drums or not self.kit.pixel_positions:
                        torso_cx = int(((s_lm[23].x + s_lm[24].x) / 2) * self.dims[0])
                        torso_cy = int(((s_lm[23].y + s_lm[24].y) / 2) * self.dims[1])
                        
                        # only check ankle visibility if we are actually updating the layout
                        ankle_pos = None
                        if s_lm[27].visibility > 0.3:
                            ankle_pos = (s_lm[27].x * self.dims[0], s_lm[27].y * self.dims[1])

                        self.kit.update_layout(torso_cx, torso_cy, self.calibration.metric_to_px_scale, ankle_pos)
                 
                    # getting depth estimation from anatomic estimator 
                    w_lm_eff, stats_payload = self.depth_manager.process_kinematic_depth(
                        s_lm, w_lm, self.depth_estimator, self.calibration.metric_to_px_scale
                    )

                    # since we collect every N seconds to compare the depth models 
                    if stats_payload is not None:
                        self.stats.add_depth_stats(stats_payload)

                    # optional for if stick mode 
                    wrist_l_world = (w_lm_eff[15].x, w_lm_eff[15].y, w_lm_eff[15].z)
                    wrist_r_world = (w_lm_eff[16].x, w_lm_eff[16].y, w_lm_eff[16].z)
                    tip_3d_l, tip_px_l, _ = self.tip_tracker_l.update(image, wrist_l_world)
                    tip_3d_r, tip_px_r, _ = self.tip_tracker_r.update(image, wrist_r_world)

                    if self.state.stick_mode:
                        self.tip_tracker_l.draw_debug(image, tip_px_l, tip_3d_l, (0, 200, 255))
                        self.tip_tracker_r.draw_debug(image, tip_px_r, tip_3d_r, (255, 200, 0))

                    # hit detection 
                    hit_l, dbg_l = self.left_arm.process(
                        s_lm[15], w_lm_eff[15], s_lm[11], w_lm_eff[11], s_lm[13], 1.0, 
                        self.kit, cur_time_ms, self.dims, s_lm[12], mediapipe_present=True
                    )
                    hit_r, dbg_r = self.right_arm.process(
                        s_lm[16], w_lm_eff[16], s_lm[12], w_lm_eff[12], s_lm[14], 1.0, 
                        self.kit, cur_time_ms, self.dims, s_lm[11], mediapipe_present=True
                    )
                    hit_foot, dbg_foot = self.foot.process(
                        s_lm[27], w_lm_eff[27], s_lm[23], s_lm[24], 
                        self.kit, cur_time_ms, self.dims, mediapipe_present=(s_lm[27].visibility > 0.3)
                    )
                    
                    # updating render state 
                    render_state["dbg_l"] = dbg_l
                    render_state["dbg_r"] = dbg_r
                    render_state["dbg_foot"] = dbg_foot
                    if hit_l: 
                        self._last_l_hit = cur_time
                        render_state["last_l_hit_time"] = cur_time
                    if hit_r: 
                        self._last_r_hit = cur_time
                        render_state["last_r_hit_time"] = cur_time
                    if hit_foot: 
                        self._last_foot_hit = cur_time
                        render_state["last_foot_hit_time"] = cur_time

                    if self.state.show_drums:
                        self.ui.draw_drums_2d(image, self.kit, cur_time, self.state.show_drum_names)
                    # draw hand points on UI
                    self.ui.draw_2d_overlays(image, dbg_l, dbg_r, dbg_foot, self.state.show_coords)
                
                    # drawing UI canvases and passing to comined view to render
                    pov_canvas = None
                    if self.state.show_pov:
                        pov_canvas = self.ui.render_pov_canvas(
                            self.kit, w_lm_eff, cur_time, dbg_l, dbg_r, dbg_foot, 
                            self.calibration.fixed_sw_m, render_state
                        )
                    else:
                        pov_canvas = None

                    controls_panel = self.ui.build_controls_panel(render_state)
                    latency_text = self._format_latency_text() if self.state.show_latency else None
                    self.ui.draw_combined_view(image, pov_canvas, controls_panel, latency_text)
            else:
                # if no output from mediapipe output frame and empty UI 
                self.ui.draw_combined_view(image, None, None)

            # handle keyboard input
            self._handle_input()

    def _build_render_state(self, cur_time):
        # return all dynamic state vars needed for UI 
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
            "show_latency": self.state.show_latency,
            "enabled_drums": self.kit.enabled_drums,
            "drum_order": self.kit.drum_order,
            
            # these will be updated dynamically in the main loop if hits occur
            "last_l_hit_time": getattr(self, "_last_l_hit", 0),
            "last_r_hit_time": getattr(self, "_last_r_hit", 0),
            "last_foot_hit_time": getattr(self, "_last_foot_hit", 0),
        }

    def _handle_input(self):
        # keyboard input handling 
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            self.running = False
        elif key == ord("f"): self.state.freeze_drums = not self.state.freeze_drums
        elif key == ord("d"): self.state.show_drums = not self.state.show_drums
        elif key == ord("c"): self.state.show_coords = not self.state.show_coords
        elif key == ord("n"): self.state.show_drum_names = not self.state.show_drum_names
        elif key == ord("p"): self.state.show_pov = not self.state.show_pov
        elif key == ord("o"): self.state.show_flow = not self.state.show_flow
        elif key == ord("l"): self.state.show_latency = not self.state.show_latency
        elif key == ord("s"): 
            self.state.stick_mode = not self.state.stick_mode
            
            if hasattr(self.kit, 'use_sticks'):
                self.kit.use_sticks = self.state.stick_mode
        elif key >= ord("1") and key < ord("1") + len(self.kit.drum_order):
            index = key - ord("1")
            drum_name = self.kit.drum_order[index]
            self.kit.toggle_drum(drum_name)

    def cleanup(self):    
        print("[CLEANUP] Saving statistics and closing threads...")
        self.camera.stop()
        self.pose_tracker.stop()
        if hasattr(self.stats, 'save_depth_comparison_json'):
            self.stats.save_depth_comparison_json()
        self.kit.cleanup()
        cv2.destroyAllWindows()

    def _format_latency_text(self):
        latencies = self.kit.last_playback_latency_ms
        left_ms = latencies.get("L", 0.0)
        right_ms = latencies.get("R", 0.0)
        kick_ms = latencies.get("RF", 0.0)
        return f"L: {left_ms:0.2f} ms   R: {right_ms:0.2f} ms   K: {kick_ms:0.2f} ms"

if __name__ == "__main__":
    app = ARDrumApp()
    try:
        app.start()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()