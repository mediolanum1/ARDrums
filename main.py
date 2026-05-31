from email.mime import image
import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import time
import math
from color_tip_tracker import ColorTipTracker
import torch
from processors1 import GestureWristProcessor
from foot_processor import GestureFootProcessor          
import sys
import numpy as np
from drum import VirtualDrumKit
from kalman_wrist import WristKalman
from rhythm_stats import RhythmSession
from typing import Optional
sys.path.append('./Depth-Anything-V2/metric_depth')
from depth_estimator import KinematicDepthEstimator
# 2. Import dpt THROUGH the depth_anything_v2 package
from depth_anything_v2.dpt import DepthAnythingV2

# ── Master flag ────────────────────────────────────────────────────────────────
USE_DEPTH_ANYTHING = False  

_DEPTH_SIGN = 1.0   
# ──────────────────────────────────────────────────────────────────────────────


APP_H     = 950
RIGHT_W   = 640
POV_H     = 540
WIN_NAME  = "AR Drum Kit"

_POV_REN_W = 800
_POV_REN_H = 640


class _LM:
    """Lightweight read/write landmark proxy."""
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, lm, z=None):
        self.x          = lm.x
        self.y          = lm.y
        self.z          = z if z is not None else lm.z
        self.visibility = lm.visibility
        self.presence   = getattr(lm, "presence", 1.0)


class ARDrumApp:
    def __init__(self):
        self.depth_comparison_stats = []
        self.anatomical_depth = KinematicDepthEstimator()
        self.frame_queue  = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running      = True
        self.freeze_drums = True
        self.cap          = cv2.VideoCapture(0)
        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.focal_length = (self.frame_width / 2) / math.tan(math.radians(60.0) / 2)
        self.kit          = VirtualDrumKit()
        self.is_calibrated         = False
        self.fixed_sw_m            = 1.0
        self.cached_drum_positions = None
        self.static_drum_positions = None
        self.metric_to_px_scale = 1.0
        self._current_sw_px     = 120.0

        self._calibrated_distance_m = None 

        self.show_drums        = True
        self.show_coords       = False
        self.show_drum_names   = True
        self.show_flow         = False
        self.show_pov          = True
        self.rhythm_session: Optional[RhythmSession] = None
        self.optical_flow_enabled = True
        self.prev_gray           = None
        self.prev_wrist_px       = {"left": None, "right": None}
        self.TARGET_CALIB_FRAMES   = 5
        self._calib_sw_m_list      = []
        self._calib_sw_px_list     = []
        self._calib_p_forearm_px_list = []
        self._calib_p_upper_px_list   = []
        self.stick_mode   = False
        self.stick_length = 0.1
        self._stick_ext_l = (0.0, 0.0, 0.0)
        self._stick_ext_r = (0.0, 0.0, 0.0)
        self.calibration_error_msg  = ""
        self._last_stick_dir_l = (0.0, 1.0)
        self._last_stick_dir_r = (0.0, 1.0)
        self._STICK_MIN_ARM_PX = 22
        self._STICK_BLEND      = 0.55

        self.last_l_hit_time    = 0
        self.last_r_hit_time    = 0
        self.last_foot_hit_time = 0

        self.program_start_time = time.time()
        self.COUNTDOWN_SECONDS  = 5
        self._depth_boost_weight = 0.0
        self.left_arm  = GestureWristProcessor("L")
        self.right_arm = GestureWristProcessor("R")
        self._tip_tracker_l = ColorTipTracker(
            color="orange",
            stick_length_m=0.406,
            focal_length=self.focal_length,
            frame_w=self.frame_width,
            frame_h=self.frame_height,
        )
        self._tip_tracker_r = ColorTipTracker(
            color="pink",
            stick_length_m=0.406,
            focal_length=self.focal_length,
            frame_w=self.frame_width,
            frame_h=self.frame_height,
        )
  
        self.right_foot = GestureFootProcessor("RF")
        self.depth_anything_loaded = False
        self.depth_active          = False
        self._depth_frame_queue    = queue.Queue(maxsize=2)
        self._latest_depth_map     = None
        self._depth_lock           = threading.Lock()
        self._depth_status_msg     = ""
    
    def camera_thread(self):
        while self.running:
            success, image = self.cap.read()
            if not success:
                continue
            image = cv2.flip(image, 1)

            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put(image)

            if USE_DEPTH_ANYTHING or self.depth_anything_loaded:
                               

                try:
                    self._depth_frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._depth_frame_queue.put(image)

    def _preprocess(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        blurred_l = cv2.GaussianBlur(l, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred_l, cv2.CV_16S, ksize=3)
        sharpened_l = np.int16(l) - laplacian
        l_final = np.clip(sharpened_l, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l_final, a, b]), cv2.COLOR_LAB2BGR)


    def ai_thread(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="./pose_landmarker_models/pose_landmarker_full.task"
            ),
            running_mode=vision.RunningMode.VIDEO,
        )
        with PoseLandmarker.create_from_options(options) as pose_landmarker:
            while self.running:
                raw_image = self.frame_queue.get()
                processed_image = self._preprocess(raw_image)
                rgb   = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                ts_ms = int(time.time() * 1000)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = pose_landmarker.detect_for_video(mp_img, ts_ms)

                if not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put((raw_image, result, time.time()))


    def _load_depth_model(self):
        self._depth_status_msg = "Loading Depth Anything V2 Metric…"
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[DEPTH] Loading Metric Model on {device} …")

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
            }
            pipe = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': 20})
            pipe.load_state_dict(torch.load('pose_landmarker_models/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
            pipe = pipe.to(device).eval()

            self.depth_anything_loaded = True
            self.depth_active          = True
            self._depth_status_msg     = f"Metric Depth ready ({device})"
            print("[DEPTH] Metric Model ready.")
            return pipe

        except Exception as exc:
            self._depth_status_msg = f"Depth model FAILED: {exc}"
            print(f"[DEPTH] {self._depth_status_msg}")
            return None

    def depth_thread(self, pipe):
        while self.running:
            try:
                frame = self._depth_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self.depth_active:
                continue

            try:
                dm = pipe.infer_image(frame)
                with self._depth_lock:
                    self._latest_depth_map = dm
            except Exception as exc:
                print(f"[DEPTH] Inference error: {exc}")


    def _depth_at_screen_lm(self, lm, dm):
        px = int(np.clip(lm.x * self.frame_width,  0, self.frame_width  - 1))
        py = int(np.clip(lm.y * self.frame_height, 0, self.frame_height - 1))
        return float(dm[py, px])

    def _patch_wlm_with_depth(self, w_lm, s_lm):
        with self._depth_lock:
            dm = self._latest_depth_map

        if dm is None:
            return list(w_lm)

        patched = list(w_lm)

        l_hip, r_hip = s_lm[23], s_lm[24]
        if l_hip.visibility < 0.4 or r_hip.visibility < 0.4:
            return patched

        d_hip = (self._depth_at_screen_lm(l_hip, dm) +
                 self._depth_at_screen_lm(r_hip, dm)) / 2.0
        fused_indices = (11, 12, 13, 14, 15, 16,   # arms / wrists
                         25, 26, 27, 28)             # legs / ankles
        for idx in fused_indices:
            if s_lm[idx].visibility < 0.4:
                continue
            d_joint = self._depth_at_screen_lm(s_lm[idx], dm)
            z_new   = _DEPTH_SIGN * (d_joint - d_hip)
            patched[idx] = _LM(w_lm[idx], z=z_new)

        return patched

    def main_render_loop(self):
        while self.running:
            try:
                image, result, cur_time = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                if 'image' in locals():
                    self._show_combined(image, None, None, None, None, time.time())
                continue

            if result.pose_landmarks and result.pose_world_landmarks:
                s_lm = result.pose_landmarks[0]
                w_lm = result.pose_world_landmarks[0]

                if not self.is_calibrated:
                    elapsed = cur_time - self.program_start_time
                    if elapsed < self.COUNTDOWN_SECONDS:
                        cv2.putText(
                            image,
                            f"READY IN: {int(self.COUNTDOWN_SECONDS - elapsed)}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        )
                        if hasattr(self, 'calibration_error_msg') and self.calibration_error_msg:
                            cv2.putText(
                                image,
                                self.calibration_error_msg,
                                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                            )
                    else:
                        self._calibrate(s_lm, w_lm)

                if self.is_calibrated:
                    self._update_drum_positions(s_lm)
                    cur_time_ms = int(time.time() * 1000)
                    dims        = (self.frame_width, self.frame_height)
                    if self.depth_active and self.depth_anything_loaded:
                        w_lm_eff = self._patch_wlm_with_depth(w_lm, s_lm)
                    else:
                        # Convert to list so we can mutate the wrists
                        w_lm_eff = list(w_lm)
                            # ---> NEW: Apply Anatomical Depth Override <---
                    if not self.depth_active: # Only use if Depth Anything isn't overriding
                        # Helper to convert normalized landmark to pixel coordinates
                        def to_px(lm):
                            return (lm.x * self.frame_width, lm.y * self.frame_height)

                        # --- LEFT ARM ---
                        l_shoulder_px = to_px(s_lm[11])
                        l_elbow_px    = to_px(s_lm[13])
                        l_wrist_px    = to_px(s_lm[15])
                        
                        # Store raw MediaPipe Z values before mutation
                        mp_l_elbow_z = w_lm_eff[13].z
                        mp_l_wrist_z = w_lm_eff[15].z
                        
                        # Calculate geometric Z
                        l_elbow_geom_z, l_wrist_geom_z = self.anatomical_depth.estimate_chain_z(
                            l_shoulder_px, l_elbow_px, l_wrist_px, 
                            self.metric_to_px_scale, 
                            shoulder_z=w_lm_eff[11].z # Anchor to MediaPipe's shoulder Z
                        )
                        # Override the world landmarks
                        w_lm_eff[13] = _LM(w_lm_eff[13], z=l_elbow_geom_z)
                        w_lm_eff[15] = _LM(w_lm_eff[15], z=l_wrist_geom_z)

                        # --- RIGHT ARM ---
                        r_shoulder_px = to_px(s_lm[12])
                        r_elbow_px    = to_px(s_lm[14])
                        r_wrist_px    = to_px(s_lm[16])
                        
                        # Store raw MediaPipe Z values before mutation
                        mp_r_elbow_z = w_lm_eff[14].z
                        mp_r_wrist_z = w_lm_eff[16].z
                        
                        r_elbow_geom_z, r_wrist_geom_z = self.anatomical_depth.estimate_chain_z(
                            r_shoulder_px, r_elbow_px, r_wrist_px, 
                            self.metric_to_px_scale, 
                            shoulder_z=w_lm_eff[12].z
                        )
                        w_lm_eff[14] = _LM(w_lm_eff[14], z=r_elbow_geom_z)
                        w_lm_eff[16] = _LM(w_lm_eff[16], z=r_wrist_geom_z)

                        # ─── NEW: Collect Comparison Statistics ───
                        self.depth_comparison_stats.append({
                            "timestamp_ms": int(time.time() * 1000),
                            "left_elbow": {
                                "mediapipe_z": float(mp_l_elbow_z),
                                "anatomical_z": float(l_elbow_geom_z),
                                "delta": float(l_elbow_geom_z - mp_l_elbow_z)
                            },
                            "left_wrist": {
                                "mediapipe_z": float(mp_l_wrist_z),
                                "anatomical_z": float(l_wrist_geom_z),
                                "delta": float(l_wrist_geom_z - mp_l_wrist_z)
                            },
                            "right_elbow": {
                                "mediapipe_z": float(mp_r_elbow_z),
                                "anatomical_z": float(r_elbow_geom_z),
                                "delta": float(r_elbow_geom_z - mp_r_elbow_z)
                            },
                            "right_wrist": {
                                "mediapipe_z": float(mp_r_wrist_z),
                                "anatomical_z": float(r_wrist_geom_z),
                                "delta": float(r_wrist_geom_z - mp_r_wrist_z)
                            }
                        })
                    wl, wr = w_lm_eff[15], w_lm_eff[16]

                    y_shoulder_avg   = (w_lm_eff[11].y + w_lm_eff[12].y) / 2.0
                    y_hip_avg        = (w_lm_eff[23].y + w_lm_eff[24].y) / 2.0
                    torso_height     = y_hip_avg - y_shoulder_avg

                    y_stomach_top    = y_shoulder_avg + (torso_height * 0.4)
                    y_stomach_bottom = y_hip_avg

                    # Screen-space horizontal check: wrists must be within the shoulder column,
                    # not off to either side. Uses s_lm (normalised 0-1 image coords).
                    torso_x_min  = min(s_lm[11].x, s_lm[12].x)
                    torso_x_max  = max(s_lm[11].x, s_lm[12].x)
                    shoulder_w   = torso_x_max - torso_x_min
                    h_padding    = shoulder_w * 0.10          # 10% slack on each edge

                    l_in_torso_x = torso_x_min - h_padding < s_lm[15].x < torso_x_max + h_padding
                    r_in_torso_x = torso_x_min - h_padding < s_lm[16].x < torso_x_max + h_padding

                    condition_met = (
                        y_stomach_top < wl.y < y_stomach_bottom and   # world Y: stomach height
                        y_stomach_top < wr.y < y_stomach_bottom and
                        wl.z < -0.2 and                                # world Z: arms extended forward
                        wr.z < -0.2 and
                        l_in_torso_x and                               # screen X: not off to the side
                        r_in_torso_x
                    )

                    _BOOST_RISE = 0.25   # ~4 frames to fully activate
                    _BOOST_FALL = 0.05   # ~20 frames to fully decay

                    if condition_met:
                        self._depth_boost_weight = min(1.0, self._depth_boost_weight + _BOOST_RISE)
                    else:
                        self._depth_boost_weight = max(0.0, self._depth_boost_weight - _BOOST_FALL)

                    if self._depth_boost_weight > 1e-3:
                        effective_boost = -0.08 * self._depth_boost_weight
                        w_lm_eff[15] = _LM(wl, z=wl.z + effective_boost)
                        w_lm_eff[16] = _LM(wr, z=wr.z + effective_boost)

                    wrist_l_world = (w_lm_eff[15].x, w_lm_eff[15].y, w_lm_eff[15].z)
                    wrist_r_world = (w_lm_eff[16].x, w_lm_eff[16].y, w_lm_eff[16].z)

                    tip_3d_l, tip_px_l, _ = self._tip_tracker_l.update(image, wrist_l_world)
                    tip_3d_r, tip_px_r, _ = self._tip_tracker_r.update(image, wrist_r_world)

                    if self.stick_mode:
                        self._tip_tracker_l.draw_debug(image, tip_px_l, tip_3d_l, (0, 200, 255))
                        self._tip_tracker_r.draw_debug(image, tip_px_r, tip_3d_r, (255, 200, 0))

                    left_wrist_px = (s_lm[15].x * self.frame_width, s_lm[15].y * self.frame_height)
                    right_wrist_px = (s_lm[16].x * self.frame_width, s_lm[16].y * self.frame_height)

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    if self.optical_flow_enabled:
                        left_flow, right_flow = self._compute_wrist_optical_flow(gray, left_wrist_px, right_wrist_px)
                    else:
                        left_flow, right_flow = None, None

                 
                    self.kit.active_stick_ext = self._stick_ext_l
                    hit_l, dbg_l = self.left_arm.process(
                        s_lm[15], w_lm_eff[15], s_lm[11], w_lm_eff[11], s_lm[13],
                        1.0, self.kit, cur_time_ms, dims, s_lm[12], left_flow
                    )


                    self.kit.active_stick_ext = self._stick_ext_r
                    hit_r, dbg_r = self.right_arm.process(
                        s_lm[16], w_lm_eff[16], s_lm[12], w_lm_eff[12], s_lm[14],
                        1.0, self.kit, cur_time_ms, dims, s_lm[11], right_flow,
                        rhythm_session=self.rhythm_session
                        
                    )
                    if (self.rhythm_session is not None and
                            not self.rhythm_session.is_active):
                        self.rhythm_session.print_summary()
                        self.rhythm_session.save("results/rhythm_session_with_feet_5.json")
                        self.rhythm_session = None 
                    self.kit.active_stick_ext = (0.0, 0.0, 0.0)
                    self.prev_gray = gray
                    self.prev_wrist_px["left"]  = left_wrist_px
                    self.prev_wrist_px["right"] = right_wrist_px

                   
                    hit_foot, dbg_foot = self.right_foot.process(
                        ankle_scr     = s_lm[27],         
                        ankle_wrl     = w_lm_eff[27],
                        hip_scr       = s_lm[23],          
                        other_hip_scr = s_lm[24],          
                        kit           = self.kit,
                        cur_time_ms   = cur_time_ms,
                        frame_dims    = dims,
                        mediapipe_present = (s_lm[27].visibility > 0.3),
                    )

                    if hit_l:    self.last_l_hit_time    = cur_time
                    if hit_r:    self.last_r_hit_time    = cur_time
                    if hit_foot: self.last_foot_hit_time = cur_time

                    self._draw_arm_debug(image, dbg_l, (255, 0, 0))
                    self._draw_arm_debug(image, dbg_r, (0, 0, 255))
                    self._draw_foot_debug(image, dbg_foot)

                    if self.show_drums and self.cached_drum_positions:
                        self._draw_drums(image, cur_time)
                    if self.show_flow:
                        self._draw_optical_flow(image, left_wrist_px, left_flow, (0, 255, 255))
                        self._draw_optical_flow(image, right_wrist_px, right_flow, (255, 255, 0))
                    pov_canvas = (
                        self._render_pov_canvas(dbg_l, dbg_r, dbg_foot, cur_time, w_lm_eff)
                        if self.show_pov else None
                    )
                    self._show_combined(image, pov_canvas, dbg_l, dbg_r, dbg_foot, cur_time)
                else:
                    self._show_combined(image, None, None, None, None, cur_time)
            else:
                self._show_combined(image, None, None, None, None, cur_time)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.running = False
                self._save_depth_comparison_json()
            elif key == ord("f"): self.freeze_drums      = not self.freeze_drums
            elif key == ord("d"): self.show_drums        = not self.show_drums
            elif key == ord("c"): self.show_coords       = not self.show_coords
            elif key == ord("n"): self.show_drum_names   = not self.show_drum_names
            elif key == ord("o"): self.show_flow         = not self.show_flow
            elif key == ord("p"): self.show_pov          = not self.show_pov
            elif key == ord("s"):
                self.stick_mode     = not self.stick_mode
                self.kit.use_sticks = self.stick_mode
            elif key == ord("q"):
                if self.depth_anything_loaded:
                    self.depth_active = not self.depth_active
                    print(f"[DEPTH] → {'ON' if self.depth_active else 'OFF'}")
                elif self._depth_status_msg.startswith("Loading"):
                    print("[DEPTH] Still loading, please wait…")
                else:
                    pipe = self._load_depth_model()
                    if pipe is not None:
                        threading.Thread(target=self.depth_thread, args=(pipe,), daemon=True).start()
            elif key == ord("r"):
                if self.rhythm_session is None or not self.rhythm_session.is_active:
                    self.rhythm_session = RhythmSession(
                        bpm_slow         = 60,
                        bpm_fast         = 90,
                        phase_duration_s = 15,
                        beat_window_ms   = 150,
                    )
                    self.rhythm_session.start()
                    print("[RHYTHM] Session started — hit the Hi-Hat to the beat!")
                else:
                    print("[RHYTHM] Session already running.")

    def _estimate_distance_m(self):
        """Estimate distance to user (metres) from calibrated shoulder width
        and the current pixel shoulder width using the pinhole camera model:
            distance = focal_length * real_width / pixel_width
        """
        if not self.is_calibrated or self._current_sw_px <= 0 or self.fixed_sw_m <= 0:
            return None
        return (self.focal_length * self.fixed_sw_m) / self._current_sw_px

    def _show_combined(self, cam_img, pov_canvas, dbg_l, dbg_r, dbg_foot, cur_time):
        cam_w   = int(self.frame_width * (APP_H / self.frame_height))
        total_w = cam_w + RIGHT_W
        ctrl_h  = APP_H - POV_H

        combined = np.zeros((APP_H, total_w, 3), dtype=np.uint8)

        cam = cv2.resize(cam_img, (cam_w, APP_H))
        if self.rhythm_session and self.rhythm_session.is_active:
         overlay = self.rhythm_session.overlay_text()
         cv2.putText(cam, overlay, (12, APP_H - 16),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 200), 2)
 

        combined[:, :cam_w] = cam
        cv2.line(combined, (cam_w, 0), (cam_w, APP_H), (40, 40, 40), 2)

        if self.show_pov and pov_canvas is not None:
            pov_resized = cv2.resize(pov_canvas, (RIGHT_W, POV_H))
        else:
            pov_resized = np.full((POV_H, RIGHT_W, 3), (12, 14, 22), dtype=np.uint8)
            cv2.putText(pov_resized, "POV hidden  [P] to show",
                        (RIGHT_W // 2 - 130, POV_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 80), 1)

        combined[:POV_H, cam_w:] = pov_resized
        cv2.line(combined, (cam_w, POV_H), (total_w, POV_H), (40, 40, 40), 2)

        ctrl_panel = self._build_controls_panel(RIGHT_W, ctrl_h, dbg_l, dbg_r, dbg_foot, cur_time)
        combined[POV_H:, cam_w:] = ctrl_panel

        
        cv2.imshow(WIN_NAME, combined)

    def _build_controls_panel(self, w, h, dbg_l, dbg_r, dbg_foot, cur_time):
        panel = np.full((h, w, 3), (14, 16, 24), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (w, 32), (24, 28, 42), -1)
        cv2.putText(panel, "CONTROLS", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 210), 1)

    
        dist_m = self._estimate_distance_m()
        if dist_m is not None:
            dist_txt = f"Distance: {dist_m:.2f} m"
            dist_col = (0, 210, 210)
        else:
            dist_txt = "Distance: calibrating…"
            dist_col = (100, 100, 120)

        cv2.rectangle(panel, (8, 36), (w - 8, 62), (24, 32, 50), -1)
        cv2.rectangle(panel, (8, 36), (w - 8, 62), (50, 60, 90), 1)
        ts = cv2.getTextSize(dist_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(panel, dist_txt, ((w - ts[0]) // 2, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_col, 1)
       
        if not USE_DEPTH_ANYTHING and not self.depth_anything_loaded:
            depth_state = None
            depth_label = "DepthAnyV2 (disabled)"
        elif self._depth_status_msg.startswith("Loading"):
            depth_state = None
            depth_label = "DepthAnyV2 (loading…)"
        elif self._depth_status_msg.startswith("Depth model FAILED"):
            depth_state = None
            depth_label = "DepthAnyV2 (FAILED)"
        else:
            depth_state = self.depth_active
            depth_label = "DepthAnyV2 depth"

        keys = [
            ("[D]",   "Toggle drums",   self.show_drums),
            ("[N]",   "Drum names",     self.show_drum_names),
            ("[P]",   "POV window",     self.show_pov),
            ("[S]",   "Stick mode",     self.stick_mode),
            ("[C]",   "Coords overlay", self.show_coords),
            ("[F]",   "Freeze drums",   self.freeze_drums),
            ("[O]",   "Flow debug",     self.show_flow),
            ("[Q]",   depth_label,      depth_state),
            ("[R]",   "Rhythm test",    self.rhythm_session is not None and self.rhythm_session.is_active),
            ("[ESC]", "Quit",           None),
        ]


        DIST_BOX_BOTTOM = 68
        row_h = max(26, (h - DIST_BOX_BOTTOM - 40) // len(keys))
        for i, (key, label, state) in enumerate(keys):
            y  = DIST_BOX_BOTTOM + 12 + i * row_h
            kw = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
            cv2.rectangle(panel, (10, y - 14), (10 + kw + 10, y + 6), (34, 38, 60), -1)
            cv2.rectangle(panel, (10, y - 14), (10 + kw + 10, y + 6), (60, 65, 100), 1)
            cv2.putText(panel, key,   (15, y),           cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 1)
            cv2.putText(panel, label, (10 + kw + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 180), 1)

            if state is not None:
                dot_col = (0, 220, 110) if state else (80, 80, 100)
                dot_x   = w - 22
                cv2.circle(panel, (dot_x, y - 4), 6, dot_col, -1)
                lbl = "ON" if state else "off"
                cv2.putText(panel, lbl, (dot_x - 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_col, 1)

      
        base_y = h - 18
        col_hit = (0, 255, 120)
        if dbg_l and (cur_time - self.last_l_hit_time) < 0.4:
            cv2.putText(panel, "LEFT  HIT!", (14, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_hit, 2)
        if dbg_r and (cur_time - self.last_r_hit_time) < 0.4:
            cv2.putText(panel, "RIGHT HIT!", (w // 2 - 60, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_hit, 2)
        if dbg_foot and (cur_time - self.last_foot_hit_time) < 0.4:
            cv2.putText(panel, "KICK!", (w // 2 - 22, base_y - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 130, 255), 2)

        if self._depth_status_msg:
            col = (0, 200, 100) if self.depth_active else (120, 120, 140)
            cv2.putText(panel, self._depth_status_msg, (10, h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        return panel


    def _calibrate(self, s_lm, w_lm):
            l_sh_w, r_sh_w = w_lm[11], w_lm[12]
            l_sh_s, r_sh_s = s_lm[11], s_lm[12]
            
            # If shoulders aren't clearly visible, skip this frame
            if l_sh_s.visibility <= 0.5 or r_sh_s.visibility <= 0.5:
                return

            # 1. Calculate current frame measurements
            cur_sw_m = math.sqrt(
                (l_sh_w.x - r_sh_w.x) ** 2 +
                (l_sh_w.y - r_sh_w.y) ** 2 +
                (l_sh_w.z - r_sh_w.z) ** 2
            )
            cur_sw_px = math.hypot(
            (l_sh_s.x - r_sh_s.x) * self.frame_width,
            (l_sh_s.y - r_sh_s.y) * self.frame_height,
        )
            # Inside your _calibrate loop where you average frames:
            # Left Arm px measurements
            l_shoulder_px = (s_lm[11].x * self.frame_width, s_lm[11].y * self.frame_height)
            l_elbow_px    = (s_lm[13].x * self.frame_width, s_lm[13].y * self.frame_height)
            l_wrist_px    = (s_lm[15].x * self.frame_width, s_lm[15].y * self.frame_height)

            # Right Arm px measurements
            r_shoulder_px = (s_lm[12].x * self.frame_width, s_lm[12].y * self.frame_height)
            r_elbow_px    = (s_lm[14].x * self.frame_width, s_lm[14].y * self.frame_height)
            r_wrist_px    = (s_lm[16].x * self.frame_width, s_lm[16].y * self.frame_height)

            # Calculate lengths for left
            l_upper_px   = math.hypot(l_elbow_px[0] - l_shoulder_px[0], l_elbow_px[1] - l_shoulder_px[1])
            l_forearm_px = math.hypot(l_wrist_px[0] - l_elbow_px[0], l_wrist_px[1] - l_elbow_px[1])

            # Calculate lengths for right
            r_upper_px   = math.hypot(r_elbow_px[0] - r_shoulder_px[0], r_elbow_px[1] - r_shoulder_px[1])
            r_forearm_px = math.hypot(r_wrist_px[0] - r_elbow_px[0], r_wrist_px[1] - r_elbow_px[1])

            # Average them to remove asymmetric bias
            p_upper_px   = (l_upper_px + r_upper_px) / 2.0
            p_forearm_px = (l_forearm_px + r_forearm_px) / 2.0
            # 2. Store them
            self._calib_p_upper_px_list.append(p_upper_px)
            self._calib_p_forearm_px_list.append(p_forearm_px)
            self._calib_sw_m_list.append(cur_sw_m)
            self._calib_sw_px_list.append(cur_sw_px)
            print(f"[CAL] Frame {len(self._calib_sw_m_list)}/{self.TARGET_CALIB_FRAMES} captured.")

            # 3. Check if we have enough frames to finalize
            if len(self._calib_sw_m_list) >= self.TARGET_CALIB_FRAMES:
                # Average the collected data
                self.fixed_sw_m = sum(self._calib_sw_m_list) / self.TARGET_CALIB_FRAMES
                avg_sw_px       = sum(self._calib_sw_px_list) / self.TARGET_CALIB_FRAMES
                avg_upper_px    = sum(self._calib_p_upper_px_list) / self.TARGET_CALIB_FRAMES
                avg_forearm_px  = sum(self._calib_p_forearm_px_list) / self.TARGET_CALIB_FRAMES
                # Establish the baseline scale and distance
                self.metric_to_px_scale = avg_sw_px / self.fixed_sw_m if self.fixed_sw_m > 0 else 1.0

                # Convert to meters using your established metric_to_px_scale
                avg_upper_arm_m = avg_upper_px / self.metric_to_px_scale
                avg_forearm_m = avg_forearm_px / self.metric_to_px_scale

                # Store these in lists, average them over the calibration frames, and pass to the estimator:
                self.anatomical_depth.calibrate_exact_lengths(avg_upper_arm_m, avg_forearm_m)

                if avg_sw_px > 0 and self.fixed_sw_m > 0:
                    self._calibrated_distance_m = (self.focal_length * self.fixed_sw_m) / avg_sw_px
                    if self._calibrated_distance_m < 1.0:
                        print(f"[CAL] FAILED: Distance {self._calibrated_distance_m:.2f}m is less than 1.0m. Restarting...")
                        self.calibration_error_msg = "please stand at least 1 meter away from camera"
                        self._calib_sw_m_list.clear()
                        self._calib_sw_px_list.clear()
                        # NEW: Clear the kinematic lists too!
                        self._calib_p_upper_px_list.clear()
                        self._calib_p_forearm_px_list.clear()
                        self.program_start_time = time.time() # Reset the countdown timer
                        return
                print(f"[CAL] DONE. Avg sw={self.fixed_sw_m:.3f} m  dist≈{self._calibrated_distance_m:.2f} m")
                self.is_calibrated = True

    def _update_drum_positions(self, s_lm):
        l_sh_s, r_sh_s = s_lm[11], s_lm[12]
        self._current_sw_px = max(1, math.hypot(
            (l_sh_s.x - r_sh_s.x) * self.frame_width,
            (l_sh_s.y - r_sh_s.y) * self.frame_height,
        ))
        self.metric_to_px_scale = (
            self._current_sw_px / self.fixed_sw_m if self.fixed_sw_m > 0 else 1.0
        )

        if self.freeze_drums and self.static_drum_positions is not None:
            self.cached_drum_positions = self.static_drum_positions
            return

        anchor_x = int(((s_lm[23].x + s_lm[24].x) / 2) * self.frame_width)
        anchor_y = int(((s_lm[23].y + s_lm[24].y) / 2) * self.frame_height)

        positions = {}
        for name, props in self.kit.drums.items():
            if name == "Bass Drum":
                # Bass drum anchored at the LEFT ankle (landmark 27).
                l_ankle = s_lm[27]
                if l_ankle.visibility > 0.3:
                    rx_m, ry_m, _ = props["radii"]
                    positions[name] = {
                        "cx": int(l_ankle.x * self.frame_width),
                        "cy": int(l_ankle.y * self.frame_height),
                        "rx": int(rx_m * self.metric_to_px_scale * 0.9),
                        "ry": int(ry_m * self.metric_to_px_scale * 0.45),
                    }
                continue

            cx_m, cy_m, _ = props["center"]
            rx_m, ry_m, _ = props["radii"]
            positions[name] = {
                "cx": int(anchor_x + cx_m * self.metric_to_px_scale),
                "cy": int(anchor_y + cy_m * self.metric_to_px_scale),
                "rx": int(rx_m * self.metric_to_px_scale),
                "ry": int(ry_m * self.metric_to_px_scale),
            }

        self.cached_drum_positions = positions
        if self.freeze_drums:
            self.static_drum_positions = positions
        self.kit.pixel_positions = self.cached_drum_positions

    # ── Stick / hand drawing ──────────────────────────────────────────────────
    def _save_depth_comparison_json(self, filename="results/depth_comparison_analysis.json"):
            """Compiles overall run metrics and dumps tracking history to JSON."""
            import os
            import json
            
            if not self.depth_comparison_stats:
                print("[STATS] No comparison data gathered during this session.")
                return

            # Ensure directory structure exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Compute helpful statistical summaries over the session
            l_wrist_deltas = [entry["left_wrist"]["delta"] for entry in self.depth_comparison_stats]
            r_wrist_deltas = [entry["right_wrist"]["delta"] for entry in self.depth_comparison_stats]
            
            summary = {
                "total_frames_recorded": len(self.depth_comparison_stats),
                "session_duration_seconds": (self.depth_comparison_stats[-1]["timestamp_ms"] - self.depth_comparison_stats[0]["timestamp_ms"]) / 1000.0,
                "metrics": {
                    "left_wrist_mean_delta": sum(l_wrist_deltas) / len(l_wrist_deltas),
                    "left_wrist_max_delta": max(l_wrist_deltas, key=abs),
                    "right_wrist_mean_delta": sum(r_wrist_deltas) / len(r_wrist_deltas),
                    "right_wrist_max_delta": max(r_wrist_deltas, key=abs)
                }
            }

            output_payload = {
                "summary": summary,
                "timeseries_data": self.depth_comparison_stats
            }

            try:
                with open(filename, "w") as f:
                    json.dump(output_payload, f, indent=4)
                print(f"[STATS] Depth performance comparison successfully exported to: {filename}")
            except Exception as e:
                print(f"[STATS] Error writing comparison file: {e}")


    def _compute_stick_ext(self, finger_w, wrist_w):
        dx = finger_w.x - wrist_w.x
        dy = finger_w.y - wrist_w.y
        dz = finger_w.z - wrist_w.z
        mag = math.sqrt(dx**2 + dy**2 + dz**2)
        if mag < 1e-6:
            return (0.0, 0.0, 0.0)
        return (dx / mag * self.stick_length,
                dy / mag * self.stick_length,
                dz / mag * self.stick_length)

    def _draw_stick_ar(self, image, elbow_px, wrist_s, color, side='r'):
        W, H = self.frame_width, self.frame_height
        ew   = (int(elbow_px[0]), int(elbow_px[1]))
        wr   = (int(wrist_s.x * W), int(wrist_s.y * H))
        dx   = wr[0] - ew[0]
        dy   = wr[1] - ew[1]
        arm_len_px = math.hypot(dx, dy)
        cached = self._last_stick_dir_l if side == 'l' else self._last_stick_dir_r

        if arm_len_px >= self._STICK_MIN_ARM_PX:
            ndx, ndy = dx / arm_len_px, dy / arm_len_px
            bx = cached[0] * (1 - self._STICK_BLEND) + ndx * self._STICK_BLEND
            by = cached[1] * (1 - self._STICK_BLEND) + ndy * self._STICK_BLEND
            mag = math.hypot(bx, by)
            if mag > 1e-6: bx, by = bx / mag, by / mag
            new_dir = (bx, by)
        else:
            bx = cached[0] * 0.97 + 0.0 * 0.03
            by = cached[1] * 0.97 + 1.0 * 0.03
            mag = math.hypot(bx, by)
            new_dir = (bx / mag, by / mag) if mag > 1e-6 else cached

        if side == 'l': self._last_stick_dir_l = new_dir
        else:           self._last_stick_dir_r = new_dir

        ndx, ndy = new_dir
        stick_px_len = self.stick_length * self.metric_to_px_scale
        tip = (int(wr[0] + ndx * stick_px_len), int(wr[1] + ndy * stick_px_len))
        cv2.line(image, wr, tip, color, 4)
        cv2.circle(image, tip, 7, (255, 240, 80), -1)

    def _draw_optical_flow(self, image, wrist_px, flow, color):
        if flow is None:
            return
        start = (int(wrist_px[0]), int(wrist_px[1]))
        end = (int(wrist_px[0] + flow[0] * 3), int(wrist_px[1] + flow[1] * 3))
        cv2.arrowedLine(image, start, end, color, 2, tipLength=0.3)

    def _compute_wrist_optical_flow(self, gray, left_px, right_px):
        if self.prev_gray is None:
            return None, None

        prev_left  = self.prev_wrist_px.get("left")
        prev_right = self.prev_wrist_px.get("right")
        if prev_left is None or prev_right is None:
            return None, None

        prev_pts = np.array([prev_left, prev_right], dtype=np.float32).reshape(-1, 1, 2)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(31, 31), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        left_flow  = None
        right_flow = None
        if curr_pts is not None and status is not None:
            if status.shape[0] > 0 and status[0][0] == 1:
                left_flow  = (float(curr_pts[0, 0, 0] - prev_left[0]),  float(curr_pts[0, 0, 1] - prev_left[1]))
            if status.shape[0] > 1 and status[1][0] == 1:
                right_flow = (float(curr_pts[1, 0, 0] - prev_right[0]), float(curr_pts[1, 0, 1] - prev_right[1]))

        if left_flow  is None and left_px  is not None:
            left_flow  = (left_px[0]  - prev_left[0],  left_px[1]  - prev_left[1])
        if right_flow is None and right_px is not None:
            right_flow = (right_px[0] - prev_right[0], right_px[1] - prev_right[1])

        return left_flow, right_flow

    def _draw_arm_debug(self, image, dbg, color):
        px = dbg["pos_px"]
        cv2.circle(image, px, 15, (0, 255, 0) if dbg["hit"] else color, -1)
        if self.show_coords:
            cv2.putText(
                image,
                f"STATE:{dbg['state']} Z:{dbg['z']:.2f}",
                (px[0] - 40, px[1] - 40),
                0, 1.2, (0, 0, 255), 2,
            )

    def _draw_foot_debug(self, image, dbg):
        """Draw left-ankle overlay for bass-drum state visualisation."""
        if dbg is None:
            return
        px = dbg["pos_px"]
        is_hit      = bool(dbg.get("hit"))
        is_pressing = dbg.get("state") == "DOWN"

        ring_col = (0, 130, 255) if is_pressing else (80, 60, 30)
        cv2.circle(image, px, 20, ring_col, 2)

        dot_col = (0, 200, 255) if is_hit else (180, 100, 30)
        cv2.circle(image, px, 10, dot_col, -1)

        if is_hit:
            cv2.putText(image, "KICK", (px[0] - 20, px[1] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        if self.show_coords:
            cv2.putText(
                image,
                f"FOOT:{dbg['state']} spd:{dbg['debug_speed']:.3f}",
                (px[0] - 50, px[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1,
            )

    def _draw_drums(self, image, cur_time):
        if not self.cached_drum_positions: return
        overlay = image.copy()
        for name, pos in self.cached_drum_positions.items():
            is_hit = (
                cur_time - self.kit.last_hit_time["L"][name] < 0.2 or
                cur_time - self.kit.last_hit_time["R"][name] < 0.2
            )
            color = (0, 255, 0) if is_hit else self.kit.drums[name]["color_idle"]
            cv2.ellipse(overlay, (pos["cx"], pos["cy"]),
                        (max(pos["rx"], 4), max(pos["ry"], 2)), 0, 0, 360, color, -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        if self.show_drum_names:
            for name, pos in self.cached_drum_positions.items():
                label = name.upper()
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                tx, ty = pos["cx"] - ts[0] // 2, pos["cy"] + ts[1] // 2
                cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── POV canvas ────────────────────────────────────────────────────────────

    def _render_pov_canvas(self, dbg_l, dbg_r, dbg_foot, cur_time, w_lm=None):
        canvas = np.full((_POV_REN_H, _POV_REN_W, 3), (8, 10, 18), dtype=np.uint8)

        SCALE  = 170
        cx_c   = _POV_REN_W // 2
        cy_c   = (_POV_REN_H // 2) - 100

        CAM_X  = 0.0
        CAM_Y  = 0.85
        CAM_Z  = 0.0

        PITCH  = math.radians(22)
        cos_p  = math.cos(PITCH)
        sin_p  = math.sin(PITCH)

        def project(x, y, z):
            x += CAM_X;  y += CAM_Y;  z += CAM_Z
            y_rot = y * cos_p + z * sin_p
            z_rot = z * cos_p - y * sin_p
            dist  = max(abs(z_rot), 0.1)
            px    = int(cx_c + (x / dist) * SCALE)
            py    = int(cy_c + (y_rot / dist) * SCALE)
            return px, py, z_rot, dist

        rq = []

        XS     = [-1.1, -0.7, -0.28, 0, 0.28, 0.7, 1.1]
        ZS     = [-0.15, -0.35, -0.55, -0.75, -0.95, -1.15]
        GRID_Y = 0.65

        for gx in XS:
            p1 = project(gx, GRID_Y, ZS[0]);  p2 = project(gx, GRID_Y, ZS[-1])
            rq.append({"t":"line","depth":(p1[2]+p2[2])/2,"p1":p1[:2],"p2":p2[:2],"color":(30,42,58),"w":1})
        for gz in ZS:
            p1 = project(XS[0], GRID_Y, gz);  p2 = project(XS[-1], GRID_Y, gz)
            rq.append({"t":"line","depth":(p1[2]+p2[2])/2,"p1":p1[:2],"p2":p2[:2],"color":(30,42,58),"w":1})

        for name, props in self.kit.drums.items():
            cx, cy, cz = props["center"]
            is_hit = (
                cur_time - self.kit.last_hit_time["L"].get(name, 0) < 0.20 or
                cur_time - self.kit.last_hit_time["R"].get(name, 0) < 0.20 or
                cur_time - self.kit.last_hit_time["RF"].get(name, 0) < 0.20
            )
            col = (0, 240, 60) if is_hit else props["color_idle"]
            ppx, ppy, depth, dist = project(cx, cy, cz)
            rx_m, ry_m, rz_m = props["radii"]

            if name == "Bass Drum":
                visual_thickness_m = 0.20
                rx    = max(int((rx_m * SCALE) / dist), 4)
                ry    = max(int((rx_m * cos_p * SCALE) / dist), 4)
                thick = max(int((visual_thickness_m * SCALE) / dist), 2)
                rq.append({"t":"bass_drum","depth":depth,"name":"BD",
                           "px":ppx,"py":ppy,"rx":rx,"ry":ry,"thick":thick,"col":col})
            else:
                visual_thickness_m = 0.02 if "Cymbal" in name or "Hi-Hat" in name else 0.12
                rx    = max(int((rx_m * SCALE) / dist), 4)
                ry    = max(int((ry_m * SCALE) / dist), 2)
                thick = max(int((visual_thickness_m * SCALE) / dist), 2)
                rq.append({"t":"drum","depth":depth,"name":name[:3].upper(),
                           "px":ppx,"py":ppy,"rx":rx,"ry":ry,"thick":thick,"col":col})

        sw = self.fixed_sw_m
        if w_lm and sw > 0:
            arm_defs = [
                (w_lm[11], w_lm[13], w_lm[15], w_lm[19], dbg_l, (100, 220, 255)),
                (w_lm[12], w_lm[14], w_lm[16], w_lm[20], dbg_r, ( 80,  80, 255)),
            ]
            for sh_w, el_w, wr_w, fi_w, dbg, arm_col in arm_defs:
                if sh_w.visibility < 0.3 or el_w.visibility < 0.3 or wr_w.visibility < 0.3:
                    continue
                sh3 = (sh_w.x, sh_w.y, sh_w.z)
                el3 = (el_w.x, el_w.y, el_w.z)
                wr3 = (wr_w.x, wr_w.y, wr_w.z)
                fi3 = (fi_w.x, fi_w.y, fi_w.z)
                sh_p, el_p, wr_p = project(*sh3), project(*el3), project(*wr3)
                line_w = max(2, int(8 / el_p[3]))
                rq.append({"t":"line","depth":(sh_p[2]+el_p[2])/2,"p1":sh_p[:2],"p2":el_p[:2],"color":arm_col,"w":line_w})
                rq.append({"t":"line","depth":(el_p[2]+wr_p[2])/2,"p1":el_p[:2],"p2":wr_p[:2],"color":arm_col,"w":line_w})
                rad_sh = max(3, int(10 / sh_p[3]))
                rad_el = max(2, int( 8 / el_p[3]))
                rq.append({"t":"dot","depth":sh_p[2],"px":sh_p[0],"py":sh_p[1],"col":arm_col,"r":rad_sh})
                rq.append({"t":"dot","depth":el_p[2],"px":el_p[0],"py":el_p[1],"col":arm_col,"r":rad_el})

                fw_x = fi3[0]-wr3[0]; fw_y = fi3[1]-wr3[1]; fw_z = fi3[2]-wr3[2]
                fw_mag = math.sqrt(fw_x**2 + fw_y**2 + fw_z**2)
                if self.stick_mode and fw_mag > 1e-3:
                    ext_len = self.stick_length
                    tip3 = (wr3[0]+(fw_x/fw_mag)*ext_len,
                            wr3[1]+(fw_y/fw_mag)*ext_len,
                            wr3[2]+(fw_z/fw_mag)*ext_len)
                    tp_p    = project(*tip3)
                    stick_w = max(1, int(6 / tp_p[3]))
                    tip_r   = max(2, int(10 / tp_p[3]))
                    rq.append({"t":"line","depth":(wr_p[2]+tp_p[2])/2,"p1":wr_p[:2],"p2":tp_p[:2],"color":(255,220,50),"w":stick_w})
                    rq.append({"t":"dot","depth":tp_p[2],"px":tp_p[0],"py":tp_p[1],"col":(255,220,50),"r":tip_r})

                is_hit_wrist = dbg.get("hit", False)
                wrist_col    = (0, 255, 60) if is_hit_wrist else arm_col
                rad_wr = max(4, int(14 / wr_p[3]))
                rq.append({"t":"hand","depth":wr_p[2],"px":wr_p[0],"py":wr_p[1],
                           "col":wrist_col,"state":dbg.get("state",""),"r":rad_wr})

        rq.sort(key=lambda i: i["depth"])
        for item in rq:
            t = item["t"]
            if   t == "line":
                cv2.line(canvas, item["p1"], item["p2"], item["color"], item["w"])
            elif t == "dot":
                cv2.circle(canvas, (item["px"],item["py"]), item["r"], item["col"], -1)
            elif t == "bass_drum":
                ppx, ppy = item["px"], item["py"]
                rx, ry, thick, c = item["rx"], item["ry"], item["thick"], item["col"]
                shade  = (int(c[0]*0.4), int(c[1]*0.4), int(c[2]*0.4))
                back_y = ppy - thick
                cv2.ellipse(canvas, (ppx, back_y), (rx, ry), 0, 0, 360, shade, -1)
                pts = np.array([(ppx-rx,ppy),(ppx+rx,ppy),(ppx+rx,back_y),(ppx-rx,back_y)], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], shade)
                head_col = (200, 255, 200) if c == (0, 240, 60) else (170, 170, 170)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, head_col, -1)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, c, max(3, int(rx*0.15)))
                cv2.ellipse(canvas, (ppx, ppy), (rx+2, ry+2), 0, 0, 360, (40,40,40), 1)
                lbl = item["name"]
                ts  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            elif t == "drum":
                ppx, ppy = item["px"], item["py"]
                rx, ry, thick, c = item["rx"], item["ry"], item["thick"], item["col"]
                shade = (int(c[0]*.45), int(c[1]*.45), int(c[2]*.45))
                cv2.ellipse(canvas, (ppx, ppy+thick), (rx, ry), 0, 0, 360, shade, -1)
                pts = np.array([(ppx-rx,ppy),(ppx+rx,ppy),(ppx+rx,ppy+thick),(ppx-rx,ppy+thick)], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], shade)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, c, -1)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, (170,170,170), 1)
                lbl = item["name"]
                ts  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 3)
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
            elif t == "hand":
                ppx, ppy, r, c = item["px"], item["py"], item["r"], item["col"]
                if item["state"] == "DOWN": cv2.circle(canvas, (ppx,ppy), r+8, c, 2)
                cv2.circle(canvas, (ppx,ppy), r, c, -1)
                cv2.circle(canvas, (ppx,ppy), r, (255,255,255), 1)

        if self.depth_active and self.depth_anything_loaded:
            cv2.putText(canvas, "DEPTH-ANYTHING ON", (12, _POV_REN_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 120), 1)

        cv2.putText(canvas, "TRUE 1ST PERSON POV", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 210), 1)

        if dbg_foot is not None:
            foot_label = f"KICK: {dbg_foot.get('state','UP')}"
            foot_col   = (0, 200, 255) if dbg_foot.get("state") == "DOWN" else (80, 80, 100)
            cv2.putText(canvas, foot_label, (_POV_REN_W - 160, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, foot_col, 1)

        return canvas

    # ── Entry point ───────────────────────────────────────────────────────────

    def start(self):
        threading.Thread(target=self.camera_thread, daemon=True).start()
        threading.Thread(target=self.ai_thread,     daemon=True).start()

        if USE_DEPTH_ANYTHING:
            pipe = self._load_depth_model()
            if pipe is not None:
                threading.Thread(target=self.depth_thread, args=(pipe,), daemon=True).start()
        else:
            self._depth_status_msg = ""

        self.main_render_loop()
        self.cap.release()
        cv2.destroyAllWindows()
        self.kit.cleanup()
      

if __name__ == "__main__":
    ARDrumApp().start()